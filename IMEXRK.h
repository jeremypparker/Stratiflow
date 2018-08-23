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
#define MatMulDim1 Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMulDim2 Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMulDim3 Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMul1D Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>, stratifloat, stratifloat, N1, N2, N3>

class IMEXRK
{
public:
    stratifloat deltaT = 0.01f;

public:
    IMEXRK()
    : solveLaplacian(M1*N2)
    , implicitSolveVelocityNeumann{std::vector<Tridiagonal<stratifloat, N3>>(M1*N2), std::vector<Tridiagonal<stratifloat, N3>>(M1*N2), std::vector<Tridiagonal<stratifloat, N3>>(M1*N2)}
    , implicitSolveVelocityDirichlet{std::vector<Tridiagonal<stratifloat, N3>>(M1*N2), std::vector<Tridiagonal<stratifloat, N3>>(M1*N2), std::vector<Tridiagonal<stratifloat, N3>>(M1*N2)}
    , implicitSolveBuoyancyNeumann{std::vector<Tridiagonal<stratifloat, N3>>(M1*N2), std::vector<Tridiagonal<stratifloat, N3>>(M1*N2), std::vector<Tridiagonal<stratifloat, N3>>(M1*N2)}
    {
        assert(ThreeDimensional || N2 == 1);

        std::cout << "Evaluating derivative matrices..." << std::endl;

        dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1, 1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2, 2);
        dim3Derivative2Neumann = VerticalSecondDerivativeMatrix(L3, N3, BoundaryCondition::Neumann);
        dim3Derivative2Dirichlet = VerticalSecondDerivativeMatrix(L3, N3, BoundaryCondition::Dirichlet);

        MatrixX laplacian;

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<M1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                laplacian = dim3Derivative2Neumann;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                Neumannify(laplacian);

                // correct for singularity
                if (j1==0 && j2==0)
                {
                    laplacian.row(0).setZero();
                    laplacian(0,0) = 1;
                    laplacian.row(1).setZero();
                    laplacian(1,1) = 1;
                }

                solveLaplacian[j1*N2+j2].compute(laplacian);
            }
        }

        UpdateForTimestep();
    }

    void TimeStep();
    void TimeStepLinear()
    {
        // see Numerical Renaissance
        for (int k=0; k<s; k++)
        {
            ExplicitCN(k);
            //BuildRHSLinear();
            FinishRHS(k);

            ImplicitUpdate(k);
            RemoveDivergence(1/h[k]);
            //if (k==s-1)
            //{
                FilterAll();
            //}
            PopulateNodalVariables();
        }
    }

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
            b_tot.ToNodal(B_tot);

            U1_tot += U_;
            B_tot += B_;

            U1_tot.ToModal(u1_tot);
            B_tot.ToModal(b_tot);

            UpdateAdjointVariables(u1_tot, u2_tot, u3_tot, b_tot);

            ExplicitCN(k);
            //BuildRHSAdjoint();
            FinishRHS(k);

            ImplicitUpdate(k);

            RemoveDivergence(1/h[k]);

            if (k==s-1)
            {
                FilterAll();
            }

            PopulateNodalVariablesAdjoint();

            time -= h[k];
            interpFrac += h[k]/deltaT;
        }
    }

    void FilterAll()
    {
        // To prevent anything dodgy accumulating in the unused coefficients
        u1.Filter();
        if(ThreeDimensional)
        {
            u2.Filter();
        }
        u3.Filter();
        b.Filter();
        p.Filter();
    }

    void PopulateNodalVariables()
    {
        dB_dz = ddz(B_);

        u1.ToNodal(U1);
        if (ThreeDimensional)
        {
            u2.ToNodal(U2);
        }
        u3.ToNodal(U3);
        b.ToNodal(B);
    }

    void PopulateNodalVariablesAdjoint()
    {
        u1.ToNodal(U1);
        if (ThreeDimensional)
        {
            u2.ToNodal(U2);
        }
        u3.ToNodal(U3);
        b.ToNodal(B);
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

        PopulateNodalVariablesAdjoint();

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
            nnTemp = B_ + B;
            nnTemp.ToModal(neumannTemp);
            HeatPlot(neumannTemp, L1, L3, j2, filename);
        }
        else
        {
            HeatPlot(b, L1, L3, j2, filename);
        }
    }

    void PlotBuoyancyBG(std::string filename, int j2) const
    {
        nnTemp = B_;
        nnTemp.ToModal(neumannTemp);
        HeatPlot(neumannTemp, L1, L3, j2, filename);
    }

    void PlotPressure(std::string filename, int j2) const
    {
        HeatPlot(p, L1, L3, j2, filename);
    }

    void PlotVerticalVelocity(std::string filename, int j2) const
    {
        HeatPlot(u3, L1, L3, j2, filename);
    }

    void PlotSpanwiseVelocity(std::string filename, int j2) const
    {
        if(ThreeDimensional)
        {
            HeatPlot(u2, L1, L3, j2, filename);
        }
    }

    void PlotSpanwiseVorticity(std::string filename, int j2) const
    {
        nnTemp = U1 + U_;
        nnTemp.ToModal(neumannTemp);

        dirichletTemp = ddz(neumannTemp)+-1.0*ddx(u3);
        HeatPlot(dirichletTemp, L1, L3, j2, filename);
    }

    void PlotPerturbationVorticity(std::string filename, int j2) const
    {
        dirichletTemp = ddz(u1)+-1.0*ddx(u3);
        HeatPlot(dirichletTemp, L1, L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
        {
            nnTemp = U1 + U_;
            nnTemp.ToModal(neumannTemp);
            HeatPlot(neumannTemp, L1, L3, j2, filename);
        }
        else
        {
            HeatPlot(u1, L1, L3, j2, filename);
        }
    }

    void PlotAll(std::string filename, bool includeBackground) const
    {
        PlotPressure(imageDirectory+"/pressure/"+filename, N2/2);
        PlotBuoyancy(imageDirectory+"/buoyancy/"+filename, N2/2, includeBackground);
        PlotVerticalVelocity(imageDirectory+"/u3/"+filename, N2/2);
        PlotSpanwiseVelocity(imageDirectory+"/u2/"+filename, N2/2);
        PlotStreamwiseVelocity(imageDirectory+"/u1/"+filename, N2/2, includeBackground);
        PlotPerturbationVorticity(imageDirectory+"/perturbvorticity/"+filename, N2/2);

        if (includeBackground)
        {
            PlotSpanwiseVorticity(imageDirectory+"/vorticity/"+filename, N2/2);
        }
        else
        {
            //PlotPerturbationVorticity(imageDirectory+"/perturbvorticity/"+filename, N2/2);
            //PlotBuoyancyBG(imageDirectory+"/buoyancyBG/"+filename, N2/2);
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

    void SetBackground(const Neumann1D& velocity, const Neumann1D& buoyancy)
    {
        U_ = velocity;
        B_ = buoyancy;
    }

    void SetBackground(std::function<stratifloat(stratifloat)> velocity,
                       std::function<stratifloat(stratifloat)> buoyancy)
    {
        Neumann1D Ubar;
        Neumann1D Bbar;
        Ubar.SetValue(velocity, L3);
        Bbar.SetValue(buoyancy, L3);

        SetBackground(Ubar, Bbar);
    }

    void SetBackground(const NeumannModal& velocity1, const NeumannModal& velocity2, const DirichletModal& velocity3, const NeumannModal& buoyancy)
    {
        u1_tot = velocity1;
        if (ThreeDimensional)
        {
            u2_tot = velocity2;
        }
        u3_tot = velocity3;
        b_tot = buoyancy;

        u1_tot.ToNodal(U1_tot);

        if (ThreeDimensional)
        {
            u2_tot.ToNodal(U2_tot);
        }
        u3_tot.ToNodal(U3_tot);
        b_tot.ToNodal(B_tot);
    }

    // gives an upper bound on cfl number - also updates timestep
    stratifloat CFL()
    {
        static ArrayX z = VerticalPoints(L3, N3);

        U1_tot = U1 + U_;

        stratifloat delta1 = L1/N1;
        stratifloat delta2 = L2/N2;
        stratifloat delta3 = z(N3/2) - z(N3/2+1); // smallest gap in middle

        stratifloat cfl = U1_tot.Max()/delta1 + U2.Max()/delta2 + U3.Max()/delta3;
        cfl *= deltaT;

        // update timestep for target cfl
        constexpr stratifloat targetCFL = 0.4;
        deltaT *= targetCFL / cfl;
        UpdateForTimestep();

        return cfl;
    }

    stratifloat CFLlinear()
    {
        static ArrayX z = VerticalPoints(L3, N3);

        stratifloat delta1 = L1/N1;
        stratifloat delta2 = L2/N2;
        stratifloat delta3 = z(N3/2) - z(N3/2+1); // smallest gap in middle

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
        stratifloat energy = 0.5f*(InnerProd(u1, u1, L3) + InnerProd(u3, u3, L3));

        if(ThreeDimensional)
        {
            energy += 0.5f*InnerProd(u2, u2, L3);
        }

        return energy;

    }

    stratifloat PE() const
    {
        return Ri*0.5f*InnerProd(b, b, L3);
    }

    void RemoveDivergence(stratifloat pressureMultiplier=1.0f);

    void SaveFlow(const std::string& filename) const
    {
        std::ofstream filestream(filename, std::ios::out | std::ios::binary);

        U1.Save(filestream);
        U2.Save(filestream);
        U3.Save(filestream);
        B.Save(filestream);
    }

    void LoadFlow(const std::string& filename)
    {
        std::ifstream filestream(filename, std::ios::in | std::ios::binary);

        U1.Load(filestream);
        U2.Load(filestream);
        U3.Load(filestream);
        B.Load(filestream);

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

        nnTemp = B + B_;
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
        //u1_total.ToNodal(U1_tot);
        u2_total.ToNodal(U2_tot);
        u3_total.ToNodal(U3_tot);
        b_total.ToNodal(B_tot);

        // work out variation of buoyancy from average
        static Neumann1D bAve;
        HorizontalAverage(b_total, bAve);

        static Dirichlet1D wAve;
        HorizontalAverage(u3_total, wAve);

        nnTemp = B_tot + -1*bAve;

        // (b-<b>)*w
        ndTemp = nnTemp*U3_tot;
        ndTemp.ToModal(dirichletTemp);

        // construct integrand for J
        static Dirichlet1D Jintegrand;
        HorizontalAverage(dirichletTemp, Jintegrand);
        J = IntegrateVertically(Jintegrand, L3);

        K = 2;

        // forcing term for u3
        u3Forcing = (-1/K)*(B_tot+(-1)*bAve);

        // forcing term for b
        bForcing = (-1/K)*(U3_tot+(-1)*wAve);

        u1Forcing.Zero();
        u2Forcing.Zero();
    }

    void UpdateForTimestep()
    {
        std::cout << "Solving matices..." << std::endl;

        h[0] = deltaT*8.0f/15.0f;
        h[1] = deltaT*2.0f/15.0f;
        h[2] = deltaT*5.0f/15.0f;


        #pragma omp parallel for
        for (int j1=0; j1<M1; j1++)
        {
            MatrixX laplacian;
            MatrixX solve;

            for (int j2=0; j2<N2; j2++)
            {
                for (int k=0; k<s; k++)
                {
                    laplacian = dim3Derivative2Neumann;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Re);
                    Neumannify(solve);
                    implicitSolveVelocityNeumann[k][j1*N2+j2].compute(solve);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Pe);
                    Neumannify(solve);
                    implicitSolveBuoyancyNeumann[k][j1*N2+j2].compute(solve);


                    laplacian = dim3Derivative2Dirichlet;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Re);
                    Dirichlify(solve);
                    implicitSolveVelocityDirichlet[k][j1*N2+j2].compute(solve);
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

    void ImplicitUpdate(int k, bool evolveBackground = false);
    void FinishRHS(int k);
    void ExplicitCN(int k, bool evolveBackground = false);
    void BuildRHS();
    void BuildRHSLinear();
    void BuildRHSAdjoint();

public:
    // these are the actual variables we care about
    NeumannModal u1, u2, b, p;
    DirichletModal u3;
private:
    // background flow
    Neumann1D U_, B_;
    Dirichlet1D dB_dz;

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
    static constexpr stratifloat beta[3] = {1.0f, 25.0f/8.0f, 9.0f/4.0f};
    static constexpr stratifloat zeta[3] = {0, -17.0f/8.0f, -5.0f/4.0f};

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

    std::vector<Tridiagonal<stratifloat, N3>> implicitSolveVelocityNeumann[3];
    std::vector<Tridiagonal<stratifloat, N3>> implicitSolveVelocityDirichlet[3];
    std::vector<Tridiagonal<stratifloat, N3>> implicitSolveBuoyancyNeumann[3];
    std::vector<Tridiagonal<stratifloat, N3>> solveLaplacian;

    std::string imageDirectory;
};
