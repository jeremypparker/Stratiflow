#pragma once

#include "Stratiflow.h"

#include "Field.h"
#include "Differentiation.h"
#include "Integration.h"
#include "Graph.h"
#include "OSUtils.h"
#include "Tridiagonal.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <dirent.h>
#include <map>

#include <omp.h>

#include <cstdio>
#include <iostream>
#include <string>

// will become unnecessary with C++17
#define MatMulDim1 Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, gridParams.N2, gridParams.N3>
#define MatMulDim2 Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, gridParams.N2, gridParams.N3>
#define MatMulDim3 Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, gridParams.N2, gridParams.N3>
#define MatMulDim3Nodal Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>, stratifloat, stratifloat, gridParams.N1, gridParams.N2, gridParams.N3>
#define MatMul1D Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>, stratifloat, stratifloat, gridParams.N1, gridParams.N2, gridParams.N3>

class IMEXRK
{
public:
    stratifloat deltaT = 0.01f;

public:
    IMEXRK()
    : solveLaplacian(M1*gridParams.N2)
    , implicitSolveVelocityNeumann{std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2), std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2), std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2)}
    , implicitSolveVelocityDirichlet{std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2), std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2), std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2)}
    , implicitSolveBuoyancyNeumann{std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2), std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2), std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>>(M1*gridParams.N2)}
    {
        assert(gridParams.ThirdDimension() || gridParams.N2 == 1);

        std::cout << "Evaluating derivative matrices..." << std::endl;

        dim1Derivative2 = FourierSecondDerivativeMatrix(flowParams.L1, gridParams.N1, 1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(flowParams.L2, gridParams.N2, 2);
        dim3Derivative2Neumann = VerticalSecondDerivativeMatrix(flowParams.L3, gridParams.N3, BoundaryCondition::Neumann);
        dim3Derivative2Dirichlet = VerticalSecondDerivativeMatrix(flowParams.L3, gridParams.N3, BoundaryCondition::Dirichlet);

        MatrixX laplacian;

        // we solve each vetical line separately, so N1*gridParams.N2 total solves
        for (int j1=0; j1<M1; j1++)
        {
            for (int j2=0; j2<gridParams.N2; j2++)
            {
                laplacian = dim3Derivative2Neumann;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(gridParams.N3, gridParams.N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(gridParams.N3, gridParams.N3);

                Neumannify(laplacian);

                // correct for singularity
                if (j1==0 && j2==0)
                {
                    laplacian.row(0).setZero();
                    laplacian(0,0) = 1;
                    laplacian.row(1).setZero();
                    laplacian(1,1) = 1;
                }

                solveLaplacian[j1*gridParams.N2+j2].compute(laplacian);
            }
        }

        UpdateForTimestep();
    }

    void TimeStep();
    void TimeStepLinear();

    void TimeStepAdjoint(stratifloat time,
                         const NeumannModal& u1Below,
                         const NeumannModal& u2Below,
                         const DirichletModal& u3Below,
                         const NeumannModal& bBelow,
                         const NeumannModal& u1Above,
                         const NeumannModal& u2Above,
                         const DirichletModal& u3Above,
                         const NeumannModal& bAbove)
    {
        stratifloat interpFrac = 0;
        for (int k=0; k<s; k++)
        {
            // interpolate the direct state at the RK substep
            u1_tot = (1-interpFrac)*u1Below + interpFrac*u1Above;
            u2_tot = (1-interpFrac)*u2Below + interpFrac*u2Above;
            u3_tot = (1-interpFrac)*u3Below + interpFrac*u3Above;
            b_tot = (1-interpFrac)*bBelow + interpFrac*bAbove;

            // todo: add on background in modal?
            u1_tot.ToNodal(U1_tot);

            U1_tot += U_;

            U1_tot.ToModal(u1_tot);

            UpdateAdjointVariables(u1_tot, u2_tot, u3_tot, b_tot);

            ExplicitRK(k);
            BuildRHSAdjoint();
            FinishRHS(k);

            CrankNicolson(k);

            RemoveDivergence(1/h[k]);
            FilterAll();

            PopulateNodalVariables();

            time -= h[k];
            interpFrac += h[k]/deltaT;
        }
    }

    void FilterAll()
    {
        // To prevent anything dodgy accumulating in the unused coefficients
        u1.Filter(gridParams.dimensionality==Dimensionality::ThreeDimensional);
        if(gridParams.ThirdDimension())
        {
            u2.Filter(gridParams.dimensionality==Dimensionality::ThreeDimensional);
        }
        u3.Filter(gridParams.dimensionality==Dimensionality::ThreeDimensional);
        b.Filter(gridParams.dimensionality==Dimensionality::ThreeDimensional);
        p.Filter(gridParams.dimensionality==Dimensionality::ThreeDimensional);
    }

    void PopulateNodalVariables()
    {
        u1.ToNodal(U1);
        if (gridParams.ThirdDimension())
        {
            u2.ToNodal(U2);
        }
        u3.ToNodal(U3);
        b.ToNodal(B);

        // U1.Antisymmetrise();
        // if (gridParams.ThirdDimension())
        // {
        //     U2.Antisymmetrise();
        // }
        // U3.Antisymmetrise();
        // B.Antisymmetrise();

        // U1.ToModal(u1);
        // U3.ToModal(u3);
        // B.ToModal(b);
    }

    void PrepareRun(std::string imageDir, bool makeDirs = true)
    {
        imageDirectory = imageDir;

        PopulateNodalVariables();

        if (makeDirs)
        {
            MakeCleanDir(imageDirectory+"/u1");
            MakeCleanDir(imageDirectory+"/u2");
            MakeCleanDir(imageDirectory+"/u3");
            MakeCleanDir(imageDirectory+"/buoyancy");
            MakeCleanDir(imageDirectory+"/buoyancyBG");
            MakeCleanDir(imageDirectory+"/pressure");
            MakeCleanDir(imageDirectory+"/vorticity");
            MakeCleanDir(imageDirectory+"/perturbvorticity");
        }
    }

    void PrepareRunAdjoint(std::string imageDir)
    {
        imageDirectory = imageDir;

        p.Zero();

        PopulateNodalVariables();

        MakeCleanDir(imageDirectory+"/u1");
        MakeCleanDir(imageDirectory+"/u2");
        MakeCleanDir(imageDirectory+"/u3");
        MakeCleanDir(imageDirectory+"/buoyancy");
        MakeCleanDir(imageDirectory+"/pressure");
        MakeCleanDir(imageDirectory+"/vorticity");
        MakeCleanDir(imageDirectory+"/perturbvorticity");
        MakeCleanDir(imageDirectory+"/buoyancyBG");
    }

    void PrepareRunLinear(std::string imageDir, bool makeDirs = true)
    {
        imageDirectory = imageDir;

        PopulateNodalVariables();


        if (makeDirs)
        {
            MakeCleanDir(imageDirectory+"/u1");
            MakeCleanDir(imageDirectory+"/u2");
            MakeCleanDir(imageDirectory+"/u3");
            MakeCleanDir(imageDirectory+"/buoyancy");
            MakeCleanDir(imageDirectory+"/pressure");
            MakeCleanDir(imageDirectory+"/vorticity");
            MakeCleanDir(imageDirectory+"/perturbvorticity");
            MakeCleanDir(imageDirectory+"/buoyancyBG");
        }
    }

    void PlotBuoyancy(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
        {
            Neumann1D B_;
            B_.SetValue([](stratifloat z){return z;}, flowParams.L3);

            nnTemp = B_ + B;
            nnTemp.ToModal(neumannTemp);
            HeatPlot(neumannTemp, flowParams.L1, flowParams.L3, j2, filename);
        }
        else
        {
            HeatPlot(b, flowParams.L1, flowParams.L3, j2, filename);
        }
    }

    void PlotPressure(std::string filename, int j2) const
    {
        HeatPlot(p, flowParams.L1, flowParams.L3, j2, filename);
    }

    void PlotVerticalVelocity(std::string filename, int j2) const
    {
        HeatPlot(u3, flowParams.L1, flowParams.L3, j2, filename);
    }

    void PlotSpanwiseVelocity(std::string filename, int j2) const
    {
        if(gridParams.ThirdDimension())
        {
            HeatPlot(u2, flowParams.L1, flowParams.L3, j2, filename);
        }
    }

    void PlotSpanwiseVorticity(std::string filename, int j2) const
    {
        nnTemp = U1 + U_;
        nnTemp.ToModal(neumannTemp);

        dirichletTemp = -1.0*ddz(neumannTemp)+ddx(u3);
        HeatPlot(dirichletTemp, flowParams.L1, flowParams.L3, j2, filename);
    }

    void PlotPerturbationVorticity(std::string filename, int j2) const
    {
        dirichletTemp = ddz(u1)+-1.0*ddx(u3);
        HeatPlot(dirichletTemp, flowParams.L1, flowParams.L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
        {
            nnTemp = U1 + U_;
            nnTemp.ToModal(neumannTemp);
            HeatPlot(neumannTemp, flowParams.L1, flowParams.L3, j2, filename);
        }
        else
        {
            HeatPlot(u1, flowParams.L1, flowParams.L3, j2, filename);
        }
    }

    void PlotAll(std::string filename, bool includeBackground) const
    {
        PlotPressure(imageDirectory+"/pressure/"+filename, gridParams.N2/2);
        PlotBuoyancy(imageDirectory+"/buoyancy/"+filename, gridParams.N2/2, includeBackground);
        PlotVerticalVelocity(imageDirectory+"/u3/"+filename, gridParams.N2/2);
        PlotSpanwiseVelocity(imageDirectory+"/u2/"+filename, gridParams.N2/2);
        PlotStreamwiseVelocity(imageDirectory+"/u1/"+filename, gridParams.N2/2, includeBackground);
        PlotPerturbationVorticity(imageDirectory+"/perturbvorticity/"+filename, gridParams.N2/2);

        if (includeBackground)
        {
            PlotSpanwiseVorticity(imageDirectory+"/vorticity/"+filename, gridParams.N2/2);
        }
        else
        {
            //PlotPerturbationVorticity(imageDirectory+"/perturbvorticity/"+filename, gridParams.N2/2);
            //PlotBuoyancyBG(imageDirectory+"/buoyancyBG/"+filename, gridParams.N2/2);
        }
    }

    void SetInitial(const NeumannNodal& velocity1, const NeumannNodal& velocity2, const DirichletNodal& velocity3, const NeumannNodal& buoyancy)
    {
        velocity1.ToModal(u1);
        velocity2.ToModal(u2);
        velocity3.ToModal(u3);
        buoyancy.ToModal(b);
    }

    void SetInitial(const NeumannModal& velocity1, const NeumannModal& velocity2, const DirichletModal& velocity3, const NeumannModal& buoyancy)
    {
        u1 = velocity1;
        u2 = velocity2;
        u3 = velocity3;
        b = buoyancy;
    }

    void SetBackground(const Neumann1D& velocity)
    {
        U_ = velocity;
    }

    void SetBackground(std::function<stratifloat(stratifloat)> velocity)
    {
        Neumann1D Ubar;
        Ubar.SetValue(velocity, flowParams.L3);

        SetBackground(Ubar);
    }

    void SetBackground(const NeumannModal& velocity1, const NeumannModal& velocity2, const DirichletModal& velocity3, const NeumannModal& buoyancy)
    {
        u1_tot = velocity1;
        if ((gridParams.dimensionality == Dimensionality::ThreeDimensional))
        {
            u2_tot = velocity2;
        }
        u3_tot = velocity3;
        b_tot = buoyancy;

        u1_tot.ToNodal(U1_tot);

        if ((gridParams.dimensionality == Dimensionality::ThreeDimensional))
        {
            u2_tot.ToNodal(U2_tot);
        }
        u3_tot.ToNodal(U3_tot);
        b_tot.ToNodal(B_tot);
    }

    // gives an upper bound on cfl number - also updates timestep
    stratifloat CFL()
    {
        static ArrayX z = VerticalPoints(flowParams.L3, gridParams.N3);

        U1_tot = U1 + U_;

        stratifloat delta1 = flowParams.L1/gridParams.N1;
        stratifloat delta2 = flowParams.L2/gridParams.N2;
        stratifloat delta3 = z(gridParams.N3/2+1) - z(gridParams.N3/2); // smallest gap in middle

        stratifloat cfl = U1_tot.Max()/delta1 + U2.Max()/delta2 + U3.Max()/delta3;
        cfl *= deltaT;

        // update timestep for target cfl
        constexpr stratifloat targetCFL = 0.8;
        deltaT *= targetCFL / cfl;
        UpdateForTimestep();

        return cfl;
    }

    stratifloat CFLlinear()
    {
        static ArrayX z = VerticalPoints(flowParams.L3, gridParams.N3);

        stratifloat delta1 = flowParams.L1/gridParams.N1;
        stratifloat delta2 = flowParams.L2/gridParams.N2;
        stratifloat delta3 = z(gridParams.N3/2+1) - z(gridParams.N3/2); // smallest gap in middle

        stratifloat cfl = (1+U1_tot.Max())/delta1 + U2_tot.Max()/delta2 + U3_tot.Max()/delta3;
        cfl *= deltaT;

        // update timestep for target cfl
        constexpr stratifloat targetCFL = 0.4;
        deltaT *= targetCFL / cfl;
        UpdateForTimestep();

        return cfl;
    }

    stratifloat KE() const
    {
        stratifloat energy = 0.5f*(InnerProd(u1, u1, flowParams.L3) + InnerProd(u3, u3, flowParams.L3));

        if(gridParams.ThirdDimension())
        {
            energy += 0.5f*InnerProd(u2, u2, flowParams.L3);
        }

        return energy;

    }

    stratifloat PE() const
    {
        return flowParams.Ri*0.5f*InnerProd(b, b, flowParams.L3);
    }

    void RemoveDivergence(stratifloat pressureMultiplier=1.0);

    void SaveFlow(const std::string& filename) const
    {
        std::ofstream filestream(filename, std::ios::out | std::ios::binary);

        U1.Save(filestream);
        U2.Save(filestream);
        U3.Save(filestream);
        B.Save(filestream);
    }

    void LoadFlow(const std::string& filename, bool twoDimensional)
    {
        std::ifstream filestream(filename, std::ios::in | std::ios::binary);

        U1.Load(filestream, twoDimensional);
        U2.Load(filestream, twoDimensional);
        U3.Load(filestream, twoDimensional);
        B.Load(filestream, twoDimensional);

        if (twoDimensional)
        {
            U2.Zero(); // just to be sure
        }

        U1.ToModal(u1);
        U2.ToModal(u2);
        U3.ToModal(u3);
        B.ToModal(b);
    }

    stratifloat JoverK()
    {
        static NeumannModal u1_total;
        static NeumannModal b_total;

        nnTemp = U1 + U_;
        nnTemp.ToModal(u1_total);

        nnTemp = B;
        nnTemp.ToModal(b_total);

        UpdateAdjointVariables(u1_total, u2, u3, b_total);

        return J/K;
    }

    void UpdateAdjointVariables(const NeumannModal& u1_total,
                                const NeumannModal& u2_total,
                                const DirichletModal& u3_total,
                                const NeumannModal& b_total)
    {
        // todo: remove some of these
        u1_total.ToNodal(U1_tot);
        u2_total.ToNodal(U2_tot);
        u3_total.ToNodal(U3_tot);
        b_total.ToNodal(B_tot);

        // work out variation of buoyancy from average
        static Neumann1D bAve;
        HorizontalAverage(b_total, bAve);

        static Dirichlet1D wAve;
        HorizontalAverage(u3_total, wAve);

        nnTemp = B_tot + -1*bAve;

        // (<b>-b)*w
        ndTemp = -1*nnTemp*U3_tot;
        ndTemp.ToModal(dirichletTemp);

        // construct integrand for J
        static Dirichlet1D Jintegrand;
        HorizontalAverage(dirichletTemp, Jintegrand);
        J = IntegrateVertically(Jintegrand, flowParams.L3);

        K = 2;

        // forcing term for u3
        u3Forcing = (1/K)*(B_tot+(-1)*bAve);

        // forcing term for b
        bForcing = (1/K)*(U3_tot+(-1)*wAve);

        u1Forcing.Zero();
        u2Forcing.Zero();
    }

    void UpdateForTimestep()
    {
        std::cout << "Solving matices..." << std::endl;

        h[0] = deltaT*8.0/15.0;
        h[1] = deltaT*2.0/15.0;
        h[2] = deltaT*5.0/15.0;


        #pragma omp parallel for
        for (int j1=0; j1<M1; j1++)
        {
            MatrixX laplacian;
            MatrixX solve;

            for (int j2=0; j2<gridParams.N2; j2++)
            {
                for (int k=0; k<s; k++)
                {
                    laplacian = dim3Derivative2Neumann;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(gridParams.N3, gridParams.N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(gridParams.N3, gridParams.N3);

                    solve = (MatrixX::Identity(gridParams.N3, gridParams.N3)-0.5*h[k]*laplacian/flowParams.Re);
                    Neumannify(solve);
                    implicitSolveVelocityNeumann[k][j1*gridParams.N2+j2].compute(solve);

                    solve = (MatrixX::Identity(gridParams.N3, gridParams.N3)-0.5*h[k]*laplacian/flowParams.Re/flowParams.Pr);
                    Neumannify(solve);
                    implicitSolveBuoyancyNeumann[k][j1*gridParams.N2+j2].compute(solve);


                    laplacian = dim3Derivative2Dirichlet;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(gridParams.N3, gridParams.N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(gridParams.N3, gridParams.N3);

                    solve = (MatrixX::Identity(gridParams.N3, gridParams.N3)-0.5*h[k]*laplacian/flowParams.Re);
                    Dirichlify(solve);
                    implicitSolveVelocityDirichlet[k][j1*gridParams.N2+j2].compute(solve);
                }

            }
        }
    }

private:
    void CNSolve(NeumannModal& solve, NeumannModal& into, int k)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveVelocityNeumann[k], into);
    }

    void CNSolve(DirichletModal& solve, DirichletModal& into, int k)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveVelocityDirichlet[k], into);
    }

    void CNSolveBuoyancy(NeumannModal& solve, NeumannModal& into, int k)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveBuoyancyNeumann[k], into);
    }

    void CNSolve(NeumannNodal& solve, NeumannNodal& into, int k)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveVelocityNeumann[k][0], into);
    }

    void CNSolve(DirichletNodal& solve, DirichletNodal& into, int k)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveVelocityDirichlet[k][0], into);
    }

    void CNSolveBuoyancy(NeumannNodal& solve, NeumannNodal& into, int k)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveBuoyancyNeumann[k][0], into);
    }

    void CNSolve1D(Neumann1D& solve, Neumann1D& into, int k, bool buoyancy = false)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveVelocityNeumann[k][0], into);
    }

    void CNSolveBuoyancy1D(Neumann1D& solve, Neumann1D& into, int k, bool buoyancy = false)
    {
        solve.ZeroEnds();
        solve.Solve(implicitSolveBuoyancyNeumann[k][0], into);
    }

    void CrankNicolson(int k, bool evolveBackground = false);
    void FinishRHS(int k);
    void ExplicitRK(int k, bool evolveBackground = false);
    void BuildRHS();
    void BuildRHSLinear();
    void BuildRHSAdjoint();

public:
    // these are the actual variables we care about
    NeumannModal u1, u2, b, p;
    DirichletModal u3;

    // background flow
    Neumann1D U_;
private:


    // direct flow (used for adjoint evolution)
    NeumannModal u1_tot, u2_tot, b_tot;
    DirichletModal u3_tot;

    // Nodal versions of variables
    mutable NeumannNodal U1, U2, B;
    mutable DirichletNodal U3;


    // extra variables required for adjoint forcing
    stratifloat J, K;

    NeumannNodal u1Forcing, u2Forcing;
    DirichletNodal u3Forcing, bForcing;

    // parameters for the scheme
    static constexpr int s = 3;
    stratifloat h[3];
    static constexpr stratifloat beta[3] = {1.0, 25.0/8.0, 9.0/4.0};
    static constexpr stratifloat zeta[3] = {0, -17.0/8.0, -5.0/4.0};

    // these are intermediate variables used in the computation, preallocated for efficiency
    NeumannModal R1, R2, RB;
    DirichletModal R3;

    Neumann1D RU_, RB_;

    NeumannModal r1, r2, rB;
    DirichletModal r3;

    mutable NeumannNodal U1_tot, U2_tot, B_tot;
    mutable DirichletNodal U3_tot;

    mutable NeumannNodal nnTemp, nnTemp2;
    mutable DirichletNodal ndTemp, ndTemp2;

    mutable NeumannModal neumannTemp;
    mutable DirichletModal dirichletTemp;

    mutable Dirichlet1D ndTemp1D;
    mutable Neumann1D nnTemp1D;

    NeumannModal divergence;
    NeumannModal q;

    // these are precomputed matrices for performing and solving derivatives
    DiagonalMatrix<stratifloat, -1> dim1Derivative2;
    DiagonalMatrix<stratifloat, -1> dim2Derivative2;
    MatrixX dim3Derivative2Neumann;
    MatrixX dim3Derivative2Dirichlet;

    std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>> implicitSolveVelocityNeumann[3];
    std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>> implicitSolveVelocityDirichlet[3];
    std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>> implicitSolveBuoyancyNeumann[3];
    std::vector<Tridiagonal<stratifloat, gridParams.N3>, aligned_allocator<Tridiagonal<stratifloat, gridParams.N3>>> solveLaplacian;

    std::string imageDirectory;
};
