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
#define MatMulDim1 Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, gridParams.N1, gridParams.N2, M3>
#define MatMulDim2 Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, gridParams.N1, gridParams.N2, M3>
#define MatMulDim3 Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, gridParams.N1, gridParams.N2, M3>

class IMEXRK
{
public:
    stratifloat deltaT = 0.01f;

public:
    IMEXRK()
    : solveLaplacian(gridParams.N1*gridParams.N2)
    , implicitSolveVelocity{std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>>(gridParams.N1*gridParams.N2), std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>>(gridParams.N1*gridParams.N2), std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>>(gridParams.N1*gridParams.N2)}
    , implicitSolveBuoyancy{std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>>(gridParams.N1*gridParams.N2), std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>>(gridParams.N1*gridParams.N2), std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>>(gridParams.N1*gridParams.N2)}
    {
        assert(gridParams.ThreeDimensional || gridParams.N2 == 1);

        std::cout << "Evaluating derivative matrices..." << std::endl;

        dim1Derivative2 = FourierSecondDerivativeMatrix(flowParams.L1, gridParams.N1, 1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(flowParams.L2, gridParams.N2, 2);
        dim3Derivative2 = FourierSecondDerivativeMatrix(flowParams.L3, gridParams.N3, 3);

        MatrixX laplacian;

        // we solve each vetical line separately, so N1*gridParams.N2 total solves
        for (int j1=0; j1<gridParams.N1; j1++)
        {
            for (int j2=0; j2<gridParams.N2; j2++)
            {
                laplacian = dim3Derivative2;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(M3, M3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(M3, M3);

                // correct for singularity
                if (dim1Derivative2.diagonal()(j1)==0 && dim2Derivative2.diagonal()(j2)==0)
                {
                    for (int j3=0;j3<M3; j3++)
                    {
                        if (laplacian(j3,j3)==0)
                        {
                            laplacian(j3,j3)=1;
                        }
                    }
                }

                solveLaplacian[j1*gridParams.N2+j2].compute(laplacian);
            }
        }

        UpdateForTimestep();
    }

    void TimeStep();
    void TimeStepLinear();

    void TimeStepAdjoint(const Modal& u1Below,
                         const Modal& u2Below,
                         const Modal& u3Below,
                         const Modal& bBelow,
                         const Modal& u1Above,
                         const Modal& u2Above,
                         const Modal& u3Above,
                         const Modal& bAbove)
    {
        stratifloat interpFrac = 0;
        for (int k=0; k<s; k++)
        {
            // interpolate the direct state at the RK substep
            u1_tot = (1-interpFrac)*u1Above + interpFrac*u1Below;
            u2_tot = (1-interpFrac)*u2Above + interpFrac*u2Below;
            u3_tot = (1-interpFrac)*u3Above + interpFrac*u3Below;
            b_tot =  (1-interpFrac)*bAbove  + interpFrac *bBelow;

            UpdateAdjointVariables(u1_tot, u2_tot, u3_tot, b_tot);

            ExplicitRK(k);
            BuildRHSAdjoint();
            FinishRHS(k);

            CrankNicolson(k);

            RemoveDivergence(1/h[k]);
            FilterAll();

            PopulateNodalVariables();

            interpFrac += h[k]/deltaT;
        }
    }

    void TimeStepLinear(const Modal& u1Below,
                        const Modal& u2Below,
                        const Modal& u3Below,
                        const Modal& bBelow,
                        const Modal& u1Above,
                        const Modal& u2Above,
                        const Modal& u3Above,
                        const Modal& bAbove)
    {
        stratifloat interpFrac = 0;
        for (int k=0; k<s; k++)
        {
            // interpolate the direct state at the RK substep
            u1_tot = (1-interpFrac)*u1Below + interpFrac*u1Above;
            u2_tot = (1-interpFrac)*u2Below + interpFrac*u2Above;
            u3_tot = (1-interpFrac)*u3Below + interpFrac*u3Above;
            b_tot =  (1-interpFrac) *bBelow + interpFrac *bAbove;

            UpdateAdjointVariables(u1_tot, u2_tot, u3_tot, b_tot);

            ExplicitRK(k);
            BuildRHSLinear();
            FinishRHS(k);

            CrankNicolson(k);

            RemoveDivergence(1/h[k]);
            FilterAll();

            PopulateNodalVariables();

            interpFrac += h[k]/deltaT;
        }
    }


    void FilterAll()
    {
        // To prevent anything dodgy accumulating in the unused coefficients
        u1.Filter();
        if(gridParams.ThreeDimensional)
        {
            u2.Filter();
        }
        u3.Filter();
        b.Filter();
        p.Filter();
    }

    void PopulateNodalVariables()
    {
        u1.ToNodal(U1);
        if (gridParams.ThreeDimensional)
        {
            u2.ToNodal(U2);
        }
        u3.ToNodal(U3);
        b.ToNodal(B);

        // U1.Antisymmetrise();
        // if (gridParams.ThreeDimensional)
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
            MakeCleanDir(imageDirectory+"/pressure");
            MakeCleanDir(imageDirectory+"/streamwisevorticity");
            MakeCleanDir(imageDirectory+"/spanwisevorticity");
            MakeCleanDir(imageDirectory+"/verticalvorticity");
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
        MakeCleanDir(imageDirectory+"/streamwisevorticity");
        MakeCleanDir(imageDirectory+"/spanwisevorticity");
        MakeCleanDir(imageDirectory+"/verticalvorticity");
    }

    void PrepareRunLinear(std::string imageDir, bool makeDirs = true)
    {
        imageDirectory = imageDir;

        PopulateNodalVariables();

        p.Zero();

        if (makeDirs)
        {
            MakeCleanDir(imageDirectory+"/u1");
            MakeCleanDir(imageDirectory+"/u2");
            MakeCleanDir(imageDirectory+"/u3");
            MakeCleanDir(imageDirectory+"/buoyancy");
            MakeCleanDir(imageDirectory+"/pressure");
            MakeCleanDir(imageDirectory+"/streamwisevorticity");
            MakeCleanDir(imageDirectory+"/spanwisevorticity");
            MakeCleanDir(imageDirectory+"/verticalvorticity");
        }
    }

    void PlotBuoyancy(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
	{

	    Nodal nodalB;
	    nodalB.SetValue([](stratifloat x, stratifloat y, stratifloat z){return z;}, flowParams.L1, flowParams.L2, flowParams.L3);
	    Modal modalB;
	    nodalB.ToModal(modalB);
	    modalB += b;
        HeatPlot(modalB, flowParams.L1, flowParams.L3, j2, filename);
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
        if(gridParams.ThreeDimensional)
        {
            HeatPlot(u2, flowParams.L1, flowParams.L3, j2, filename);
        }
    }

    void PlotSpanwiseVorticity(std::string filename, int j2) const
    {
        modalTemp2 = ddz(u1)+-1.0*ddx(u3);
        HeatPlot(modalTemp2, flowParams.L1, flowParams.L3, j2, filename);
    }

    void PlotStreamwiseVorticity(std::string filename, int j1) const
    {
        modalTemp2 = ddy(u3)+-1.0*ddz(u2);
        HeatPlotSide(modalTemp2, flowParams.L2, flowParams.L3, j1, filename);
    }

    void PlotVerticalVorticity(std::string filename, int j3) const
    {
        modalTemp2 = ddx(u2)+-1.0*ddy(u1);
        HeatPlotTop(modalTemp2, flowParams.L1, flowParams.L2, j3, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
        {
            nnTemp = U1;
            nnTemp.ToModal(modalTemp1);
            HeatPlot(modalTemp1, flowParams.L1, flowParams.L3, j2, filename);
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
        PlotSpanwiseVorticity(imageDirectory+"/spanwisevorticity/"+filename, gridParams.N2/2);

        if (gridParams.ThreeDimensional)
        {
            PlotStreamwiseVorticity(imageDirectory+"/streamwisevorticity/"+filename, gridParams.N1/2);
            PlotVerticalVorticity(imageDirectory+"/verticalvorticity/"+filename, gridParams.N3/2);
        }
    }

    void SetInitial(const Nodal& velocity1, const Nodal& velocity2, const Nodal& velocity3, const Nodal& buoyancy)
    {
        velocity1.ToModal(u1);
        velocity2.ToModal(u2);
        velocity3.ToModal(u3);
        buoyancy.ToModal(b);
    }

    void SetInitial(const Modal& velocity1, const Modal& velocity2, const Modal& velocity3, const Modal& buoyancy)
    {
        u1 = velocity1;
        u2 = velocity2;
        u3 = velocity3;
        b = buoyancy;
    }

    void SetBackground(const Modal& velocity1, const Modal& velocity2, const Modal& velocity3, const Modal& buoyancy)
    {
        u1_tot = velocity1;
        if (gridParams.ThreeDimensional)
        {
            u2_tot = velocity2;
        }
        u3_tot = velocity3;
        b_tot = buoyancy;

        u1_tot.ToNodal(U1_tot);

        if (gridParams.ThreeDimensional)
        {
            u2_tot.ToNodal(U2_tot);
        }
        u3_tot.ToNodal(U3_tot);
        b_tot.ToNodal(B_tot);
    }

    // gives an upper bound on cfl number - also updates timestep
    stratifloat CFL()
    {
        stratifloat delta1 = flowParams.L1/gridParams.N1;
        stratifloat delta2 = flowParams.L2/gridParams.N2;
        stratifloat delta3 = flowParams.L3/gridParams.N3;

        stratifloat cfl = U1.Max()/delta1 + U2.Max()/delta2 + U3.Max()/delta3;
        cfl *= deltaT;

        // update timestep for target cfl
        constexpr stratifloat targetCFL = 0.8;
        deltaT *= targetCFL / cfl;
        UpdateForTimestep();

        return cfl;
    }

    stratifloat CFLlinear()
    {
        stratifloat delta1 = flowParams.L1/gridParams.N1;
        stratifloat delta2 = flowParams.L2/gridParams.N2;
        stratifloat delta3 = flowParams.L3/gridParams.N3;

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
        stratifloat energy = 0.5f*(InnerProd(u1, u1) + InnerProd(u3, u3));

        if(gridParams.ThreeDimensional)
        {
            energy += 0.5f*InnerProd(u2, u2);
        }

        return energy;

    }

    stratifloat PE() const
    {
        return flowParams.Ri*0.5f*InnerProd(b, b);
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

    void UpdateAdjointVariables(const Modal& u1_total,
                                const Modal& u2_total,
                                const Modal& u3_total,
                                const Modal& b_total)
    {
        // todo: remove some of these
        u1_total.ToNodal(U1_tot);
        u2_total.ToNodal(U2_tot);
        u3_total.ToNodal(U3_tot);
        b_total.ToNodal(B_tot);

        u1Forcing.Zero();
        u2Forcing.Zero();
        u3Forcing.Zero();
        bForcing.Zero();
    }

    void UpdateForTimestep()
    {
        std::cout << "Solving matices..." << std::endl;

        h[0] = deltaT*8.0/15.0;
        h[1] = deltaT*2.0/15.0;
        h[2] = deltaT*5.0/15.0;


        #pragma omp parallel for
        for (int j1=0; j1<gridParams.N1; j1++)
        {
            MatrixX laplacian;
            MatrixX solve;

            for (int j2=0; j2<gridParams.N2; j2++)
            {
                for (int k=0; k<s; k++)
                {
                    laplacian = dim3Derivative2;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(M3, M3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(M3, M3);

                    solve = (MatrixX::Identity(M3, M3)-0.5*h[k]*laplacian/flowParams.Re);
                    implicitSolveVelocity[k][j1*gridParams.N2+j2].compute(solve);

                    solve = (MatrixX::Identity(M3, M3)-0.5*h[k]*laplacian/flowParams.Re/flowParams.Pr);
                    implicitSolveBuoyancy[k][j1*gridParams.N2+j2].compute(solve);

                }

            }
        }
    }

private:
    void CNSolve(Modal& solve, Modal& into, int k)
    {
        solve.Solve(implicitSolveVelocity[k], into);
    }

    void CNSolveBuoyancy(Modal& solve, Modal& into, int k)
    {
        solve.Solve(implicitSolveBuoyancy[k], into);
    }

    void CrankNicolson(int k);
    void FinishRHS(int k);
    void ExplicitRK(int k);
    void BuildRHS();
    void BuildRHSLinear();
    void BuildRHSAdjoint();

public:
    // these are the actual variables we care about
    Modal u1, u2, b, p;
    Modal u3;
private:


    // direct flow (used for adjoint evolution)
    Modal u1_tot, u2_tot, b_tot;
    Modal u3_tot;

    // Nodal versions of variables
    mutable Nodal U1, U2, B;
    mutable Nodal U3;


    // extra variables required for adjoint forcing
    stratifloat J, K;

    Nodal u1Forcing, u2Forcing;
    Nodal u3Forcing, bForcing;

    // parameters for the scheme
    static constexpr int s = 3;
    stratifloat h[3];
    static constexpr stratifloat beta[3] = {1.0, 25.0/8.0, 9.0/4.0};
    static constexpr stratifloat zeta[3] = {0, -17.0/8.0, -5.0/4.0};

    // these are intermediate variables used in the computation, preallocated for efficiency
    Modal R1, R2, RB;
    Modal R3;

    Modal r1, r2, rB;
    Modal r3;

    mutable Nodal U1_tot, U2_tot, B_tot;
    mutable Nodal U3_tot;

    mutable Nodal nnTemp, nnTemp2;
    mutable Nodal ndTemp, ndTemp2;

    mutable Modal modalTemp1;
    mutable Modal modalTemp2;

    Modal divergence;
    Modal q;

    // these are precomputed matrices for performing and solving derivatives
    DiagonalMatrix<stratifloat, -1> dim1Derivative2;
    DiagonalMatrix<stratifloat, -1> dim2Derivative2;
    DiagonalMatrix<stratifloat, -1> dim3Derivative2;

    std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>> implicitSolveVelocity[3];
    std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>> implicitSolveBuoyancy[3];
    std::vector<Tridiagonal<stratifloat, M3>, aligned_allocator<Tridiagonal<stratifloat, M3>>> solveLaplacian;

    std::string imageDirectory;
};
