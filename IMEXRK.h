#pragma once

#include "Stratiflow.h"

#include "Field.h"
#include "Differentiation.h"
#include "Integration.h"
#include "Graph.h"
#include "OSUtils.h"

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
#define MatMul1D Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>, stratifloat, stratifloat, 1, 1, N3>

class IMEXRK
{
public:
    stratifloat deltaT = 0.01f;

    long totalForcing = 0;
    long totalExplicit = 0;
    long totalImplicit = 0;
    long totalDivergence = 0;

    struct State
    {
        State()
        : U1(BoundaryCondition::Bounded)
        , U2(BoundaryCondition::Bounded)
        , U3(BoundaryCondition::Decaying)
        , B(BoundaryCondition::Bounded)
        {}

        NField U1;
        NField U2;
        NField U3;
        NField B;
    };

public:
    IMEXRK()
    : u1(BoundaryCondition::Bounded)
    , u2(BoundaryCondition::Bounded)
    , u3(BoundaryCondition::Decaying)
    , p(BoundaryCondition::Bounded)
    , b(BoundaryCondition::Bounded)

    , u_(BoundaryCondition::Bounded)
    , b_(BoundaryCondition::Bounded)
    , db_dz(BoundaryCondition::Decaying)

    , u1_tot(BoundaryCondition::Bounded)
    , u2_tot(BoundaryCondition::Bounded)
    , u3_tot(BoundaryCondition::Decaying)
    , b_tot(BoundaryCondition::Bounded)

    , U1_tot(BoundaryCondition::Bounded)
    , U2_tot(BoundaryCondition::Bounded)
    , U3_tot(BoundaryCondition::Decaying)
    , B_tot(BoundaryCondition::Bounded)


    , R1(u1), R2(u2), R3(u3), RB(b)
    , RU_(u_), RB_(b_)
    , r1(u1), r2(u2), r3(u3), rB(b)
    , U1(BoundaryCondition::Bounded)
    , U2(BoundaryCondition::Bounded)
    , U3(BoundaryCondition::Decaying)
    , B(BoundaryCondition::Bounded)
    , U_(BoundaryCondition::Bounded)
    , B_(BoundaryCondition::Bounded)
    , dB_dz(BoundaryCondition::Decaying)
    , decayingTemp(BoundaryCondition::Decaying)
    , boundedTemp(BoundaryCondition::Bounded)
    , decayingTemp1D(BoundaryCondition::Decaying)
    , boundedTemp1D(BoundaryCondition::Bounded)
    , ndTemp1D(BoundaryCondition::Decaying)
    , nnTemp1D(BoundaryCondition::Bounded)
    , ndTemp(BoundaryCondition::Decaying)
    , nnTemp(BoundaryCondition::Bounded)
    , ndTemp2(BoundaryCondition::Decaying)
    , nnTemp2(BoundaryCondition::Bounded)
    , divergence(boundedTemp)
    , q(boundedTemp)

    , u1Forcing(U1)
    , u2Forcing(U2)
    , u3Forcing(U3)
    , bForcing(B)

    , solveLaplacian(M1*N2)
    , implicitSolveBounded{std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2)}
    , implicitSolveDecaying{std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2)}
    , implicitSolveBuoyancy{std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SparseLU<SparseMatrix<stratifloat>>>(M1*N2)}
    {
        assert(ThreeDimensional || N2 == 1);

        std::cout << "Evaluating derivative matrices..." << std::endl;

        dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1, 1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2, 2);
        dim3Derivative2Bounded = VerticalSecondDerivativeMatrix(BoundaryCondition::Bounded, L3, N3);
        dim3Derivative2Decaying = VerticalSecondDerivativeMatrix(BoundaryCondition::Decaying, L3, N3);

        MatrixX laplacian;
        SparseMatrix<stratifloat> solve;

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<M1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                laplacian = dim3Derivative2Bounded;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                // prevent singularity - first row gives average value
                if (j1 == 0 && j2 == 0)
                {
                    laplacian.row(0).setConstant(0);
                    laplacian(0,0) = 1;
                }

                solve = laplacian.sparseView();
                solve.makeCompressed();

                solveLaplacian[j1*N2+j2].compute(solve);
            }
        }

        UpdateForTimestep();
    }


    void TimeStep()
    {
        // see Numerical Renaissance
        for (int k=0; k<s; k++)
        {
            auto t0 = std::chrono::high_resolution_clock::now();

            ExplicitUpdate(k);

            auto t1 = std::chrono::high_resolution_clock::now();

            ImplicitUpdate(k);

            auto t2 = std::chrono::high_resolution_clock::now();

            RemoveDivergence(1/h[k]);
            //SolveForPressure();

            auto t3 = std::chrono::high_resolution_clock::now();

            if (k==s-1)
            {
                FilterAll();
            }

            PopulateNodalVariables();

            totalExplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
            totalImplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            totalDivergence += std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
        }
    }

    void TimeStepLinear(stratifloat time)
    {
        // see Numerical Renaissance
        for (int k=0; k<s; k++)
        {
            LoadAtTime(time, false);
            ExplicitUpdateLinear(k);
            ImplicitUpdate(k);
            RemoveDivergence(1/h[k]);
            if (k==s-1)
            {
                FilterAllLinear();
            }
            PopulateNodalVariables();

            time += h[k];
        }
    }

    void TimeStepAdjoint(stratifloat time)
    {
        for (int k=0; k<s; k++)
        {
            auto t4 = std::chrono::high_resolution_clock::now();

            LoadAtTime(time);
            UpdateAdjointVariables(u1_tot, u2_tot, u3_tot, b_tot);

            auto t0 = std::chrono::high_resolution_clock::now();

            ExplicitUpdateAdjoint(k);

            auto t1 = std::chrono::high_resolution_clock::now();

            ImplicitUpdateAdjoint(k);

            auto t2 = std::chrono::high_resolution_clock::now();

            RemoveDivergence(1/h[k]);

            auto t3 = std::chrono::high_resolution_clock::now();

            if (k==s-1)
            {
                FilterAllAdjoint();
            }

            PopulateNodalVariablesAdjoint();

            totalForcing += std::chrono::duration_cast<std::chrono::milliseconds>(t0-t4).count();
            totalExplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
            totalImplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            totalDivergence += std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();

            time -= h[k];
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

        // correct for instability in scheme towards infinity
        u_.ToNodal(U_);
        b_.ToNodal(B_);
        for (int j=0; j<30; j++)
        {
            U_.Get()(j) = 1;
            B_.Get()(j) = -1;
            U_.Get()(N3-1-j) = -1;
            B_.Get()(N3-1-j) = 1;
        }
        U_.ToModal(u_);
        B_.ToModal(b_);

        u_.Filter();
        b_.Filter();
    }

    void FilterAllLinear()
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

    void FilterAllAdjoint()
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
        u_.ToNodal(U_);
        b_.ToNodal(B_);

        db_dz = ddz(b_);
        db_dz.Filter();
        db_dz.ToNodal(dB_dz);

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

    void PrepareRun(std::string imageDir)
    {
        imageDirectory = imageDir;

        totalExplicit = 0;
        totalImplicit = 0;
        totalDivergence = 0;
        totalForcing = 0;

        PopulateNodalVariables();

        MakeCleanDir(imageDirectory+"/u1");
        MakeCleanDir(imageDirectory+"/u2");
        MakeCleanDir(imageDirectory+"/u3");
        MakeCleanDir(imageDirectory+"/buoyancy");
        MakeCleanDir(imageDirectory+"/buoyancyBG");
        MakeCleanDir(imageDirectory+"/pressure");
        MakeCleanDir(imageDirectory+"/vorticity");
        MakeCleanDir(imageDirectory+"/perturbvorticity");
        MakeCleanDir(snapshotdir);
    }

    void PrepareRunAdjoint(std::string imageDir)
    {
        imageDirectory = imageDir;

        totalExplicit = 0;
        totalImplicit = 0;
        totalDivergence = 0;
        totalForcing = 0;

        u1.Zero();
        u2.Zero();
        u3.Zero();
        b.Zero();
        p.Zero();

        PopulateNodalVariablesAdjoint();

        BuildFilenameMap();

        MakeCleanDir(imageDirectory+"/u1");
        MakeCleanDir(imageDirectory+"/u2");
        MakeCleanDir(imageDirectory+"/u3");
        MakeCleanDir(imageDirectory+"/buoyancy");
        MakeCleanDir(imageDirectory+"/pressure");
        MakeCleanDir(imageDirectory+"/vorticity");
        MakeCleanDir(imageDirectory+"/perturbvorticity");
        MakeCleanDir(imageDirectory+"/buoyancyBG");
    }

    void PrepareRunLinear(std::string imageDir)
    {
        imageDirectory = imageDir;

        totalExplicit = 0;
        totalImplicit = 0;
        totalDivergence = 0;
        totalForcing = 0;

        PopulateNodalVariables();

        BuildFilenameMap(false);

        MakeCleanDir(imageDirectory+"/u1");
        MakeCleanDir(imageDirectory+"/u2");
        MakeCleanDir(imageDirectory+"/u3");
        MakeCleanDir(imageDirectory+"/buoyancy");
        MakeCleanDir(imageDirectory+"/pressure");
        MakeCleanDir(imageDirectory+"/vorticity");
        MakeCleanDir(imageDirectory+"/perturbvorticity");
        MakeCleanDir(imageDirectory+"/buoyancyBG");
    }

    void PlotBuoyancy(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
        {
            nnTemp = B_ + B;
            nnTemp.ToModal(boundedTemp);
            HeatPlot(boundedTemp, L1, L3, j2, filename);
        }
        else
        {
            HeatPlot(b, L1, L3, j2, filename);
        }
    }

    void PlotBuoyancyBG(std::string filename, int j2) const
    {
        nnTemp = B_;
        nnTemp.ToModal(boundedTemp);
        HeatPlot(boundedTemp, L1, L3, j2, filename);
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
        nnTemp.ToModal(boundedTemp);

        decayingTemp = ddz(boundedTemp)+-1.0*ddx(u3);
        HeatPlot(decayingTemp, L1, L3, j2, filename);
    }

    void PlotPerturbationVorticity(std::string filename, int j2) const
    {
        decayingTemp = ddz(u1)+-1.0*ddx(u3);
        HeatPlot(decayingTemp, L1, L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2, bool includeBackground = true) const
    {
        if (includeBackground)
        {
            nnTemp = U1 + U_;
            nnTemp.ToModal(boundedTemp);
            HeatPlot(boundedTemp, L1, L3, j2, filename);
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

        if (includeBackground)
        {
            PlotSpanwiseVorticity(imageDirectory+"/vorticity/"+filename, N2/2);
        }
        else
        {
            PlotPerturbationVorticity(imageDirectory+"/perturbvorticity/"+filename, N2/2);
            //PlotBuoyancyBG(imageDirectory+"/buoyancyBG/"+filename, N2/2);
        }
    }

    void SetInitial(NField velocity1, NField velocity2, NField velocity3, NField buoyancy)
    {
        velocity1.ToModal(u1);
        velocity2.ToModal(u2);
        velocity3.ToModal(u3);
        buoyancy.ToModal(b);
    }

    void SetInitial(MField velocity1, MField velocity2, MField velocity3, MField buoyancy)
    {
        u1 = velocity1;
        u2 = velocity2;
        u3 = velocity3;
        b = buoyancy;
    }

    void SetBackground(N1D velocity, N1D buoyancy)
    {
        velocity.ToModal(u_);
        buoyancy.ToModal(b_);

        PopulateNodalVariables();
    }

    void SetBackground(std::function<stratifloat(stratifloat)> velocity,
                       std::function<stratifloat(stratifloat)> buoyancy)
    {
        N1D Ubar(BoundaryCondition::Bounded);
        N1D Bbar(BoundaryCondition::Bounded);
        Ubar.SetValue(velocity, L3);
        Bbar.SetValue(buoyancy, L3);

        SetBackground(Ubar, Bbar);
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

    stratifloat CFLadjoint()
    {
        static ArrayX z = VerticalPoints(L3, N3);

        stratifloat delta1 = L1/N1;
        stratifloat delta2 = L2/N2;
        stratifloat delta3 = z(N3/2) - z(N3/2+1); // smallest gap in middle

        stratifloat cfl = U1_tot.Max()/delta1 + U2_tot.Max()/delta2 + U3_tot.Max()/delta3;
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
        for (int j=0; j<N3; j++)
        {
            if(dB_dz.Get()(j)>-0.001 && dB_dz.Get()(j)<0.001)
            {
                nnTemp1D.Get()(j) = 0;
            }
            else
            {
                nnTemp1D.Get()(j) = 1/dB_dz.Get()(j);
            }
        }

        return 0.5f*InnerProd(b, b, L3, -Ri*nnTemp1D);
    }
    void RemoveDivergence(stratifloat pressureMultiplier=1.0f)
    {
        // construct the diverence of u
        if(ThreeDimensional)
        {
            divergence = ddx(u1) + ddy(u2) + ddz(u3);
        }
        else
        {
            divergence = ddx(u1) + ddz(u3);
        }

        // constant term - set value at infinity to zero
        divergence(0,0,0) = 0;

        // solve Δq = ∇·u as linear system Aq = divergence
        divergence.Solve(solveLaplacian, q);

        // subtract the gradient of this from the velocity
        u1 -= ddx(q);
        if(ThreeDimensional)
        {
            u2 -= ddy(q);
        }
        u3 -= ddz(q);

        // also add it on to p for the next step
        // this is scaled to match the p that was added before
        // effectively we have forward euler
        p += pressureMultiplier*q;
    }

    void SolveForPressure()
    {
        FilterAll();
        PopulateNodalVariables();

        // build up RHS for poisson eqn for p in p itself
        p.Zero();

        // // diagonal terms of ∇u..∇u
        // boundedTemp = ddx(u1);
        // boundedTemp.ToNodal(nnTemp);
        // nnTemp2 = nnTemp*nnTemp;
        // nnTemp2.ToModal(boundedTemp);
        // p -= boundedTemp;


        // if (ThreeDimensional)
        // {
        // boundedTemp = ddy(u2);
        // boundedTemp.ToNodal(nnTemp);
        // nnTemp2 = nnTemp*nnTemp;
        // nnTemp2.ToModal(boundedTemp);
        // p -= boundedTemp;
        // }

        // boundedTemp = ddz(u3);
        // boundedTemp.ToNodal(nnTemp);
        // nnTemp2 = nnTemp*nnTemp;
        // nnTemp2.ToModal(boundedTemp);
        // p -= boundedTemp;

        // // cross terms
        // decayingTemp = ddx(u3);
        // decayingTemp.ToNodal(ndTemp);
        // decayingTemp = ddz(u1);
        // decayingTemp.ToNodal(ndTemp2);
        // nnTemp2 = ndTemp*ndTemp2;
        // nnTemp2.ToModal(boundedTemp);
        // p -= 2.0*boundedTemp;

        // if (ThreeDimensional)
        // {
        // boundedTemp = ddy(u1);
        // boundedTemp.ToNodal(nnTemp);
        // boundedTemp = ddx(u2);
        // boundedTemp.ToNodal(nnTemp2);
        // nnTemp2 = nnTemp*nnTemp2;
        // nnTemp2.ToModal(boundedTemp);
        // p -= 2.0*boundedTemp;

        // decayingTemp = ddy(u3);
        // decayingTemp.ToNodal(ndTemp);
        // decayingTemp = ddz(u2);
        // decayingTemp.ToNodal(ndTemp2);
        // nnTemp2 = ndTemp*ndTemp2;
        // nnTemp2.ToModal(boundedTemp);
        // p -= 2.0*boundedTemp;
        // }

        // // background
        // decayingTemp1D = ddz(u_);
        // decayingTemp = ddx(u3);
        // decayingTemp1D.ToNodal(ndTemp1D);
        // decayingTemp.ToNodal(ndTemp);
        // nnTemp2 = ndTemp*ndTemp1D;
        // nnTemp2.ToModal(boundedTemp);
        // p -= 2.0*boundedTemp;

        // // buoyancy
        // decayingTemp = ddz(b);
        // decayingTemp.ToNodal(ndTemp);
        // nnTemp = ndTemp;
        // nnTemp.ToModal(boundedTemp);
        // p -= Ri*boundedTemp;

        // // Now solve the poisson equation
        // p(0,0,0) = 0; //BC
        // p.Solve(solveLaplacian, p);
    }

    void SaveFlow(const std::string& filename) const
    {
        std::ofstream filestream(filename, std::ios::out | std::ios::binary);

        U1.Save(filestream);
        U2.Save(filestream);
        U3.Save(filestream);
        B.Save(filestream);
    }

    void StoreSnapshot(stratifloat time) const
    {
        U1_tot = U1 + U_;
        B_tot = B + B_;

        if (SnapshotToMemory)
        {

        }
        else
        {
            std::ofstream filestream(snapshotdir+std::to_string(time)+".fields",
                                     std::ios::out | std::ios::binary);

            U1_tot.Save(filestream);
            U2.Save(filestream);
            U3.Save(filestream);
            B_tot.Save(filestream);
        }
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
        static MField u1_total(BoundaryCondition::Bounded);
        static MField b_total(BoundaryCondition::Bounded);

        nnTemp = U1 + U_;
        nnTemp.ToModal(u1_total);

        nnTemp = B + B_;
        nnTemp.ToModal(b_total);

        UpdateAdjointVariables(u1_total, u2, u3, b_total);

        return J/K;
    }

    void UpdateAdjointVariables(const MField& u1_total,
                                       const MField& u2_total,
                                       const MField& u3_total,
                                       const MField& b_total)
    {
        // work out variation of buoyancy from average
        static M1D bAveModal(BoundaryCondition::Bounded);
        HorizontalAverage(b_total, bAveModal);
        static N1D bAveNodal(BoundaryCondition::Bounded);
        bAveModal.ToNodal(bAveNodal);

        static M1D wAveModal(BoundaryCondition::Decaying);
        HorizontalAverage(u3_total, wAveModal);
        static N1D wAveNodal(BoundaryCondition::Decaying);
        wAveModal.ToNodal(wAveNodal);

        b_total.ToNodal(B_tot);
        nnTemp = B_tot + -1*bAveNodal;

        // (b-<b>)*w
        u3_total.ToNodal(U3_tot);
        ndTemp = nnTemp*U3_tot;
        ndTemp.ToModal(decayingTemp);

        // construct integrand for J
        static M1D bwAve(BoundaryCondition::Decaying);
        HorizontalAverage(decayingTemp, bwAve);
        static N1D Jintegrand(BoundaryCondition::Decaying);
        bwAve.ToNodal(Jintegrand);
        J = IntegrateVertically(Jintegrand, L3);

        K = 2;

        // forcing term for u3
        u3Forcing = (-1/K)*(B_tot+(-1)*bAveNodal);

        // forcing term for b
        bForcing = (-1/K)*(U3_tot+(-1)*wAveNodal);

        u1Forcing.Zero();
        u2Forcing.Zero();
    }

    void BuildFilenameMap(bool reverse=true)
    {
        filenames.clear();

        auto dir = opendir(snapshotdir.c_str());
        struct dirent* file = nullptr;
        while((file=readdir(dir)))
        {
            std::string foundfilename(file->d_name);
            int end = foundfilename.find(".fields");
            stratifloat foundtime = strtof(foundfilename.substr(0, end).c_str(), nullptr);

            filenames.insert(std::pair<stratifloat, std::string>(foundtime, snapshotdir+foundfilename));
        }
        closedir(dir);

        if (reverse)
        {
            fileAbove = filenames.end();
            fileAbove--;
        }
        else
        {
            fileAbove = filenames.begin();
        }
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
            SparseMatrix<stratifloat> solve;

            for (int j2=0; j2<N2; j2++)
            {
                for (int k=0; k<s; k++)
                {
                    laplacian = dim3Derivative2Bounded;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Re).sparseView();
                    solve.makeCompressed();

                    implicitSolveBounded[k][j1*N2+j2].compute(solve);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Pe).sparseView();
                    solve.makeCompressed();

                    implicitSolveBuoyancy[k][j1*N2+j2].compute(solve);


                    laplacian = dim3Derivative2Decaying;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Re).sparseView();
                    solve.makeCompressed();

                    implicitSolveDecaying[k][j1*N2+j2].compute(solve);
                }

            }
        }
    }

    stratifloat Optimise(stratifloat& epsilon,
                         stratifloat E_0,
                         MField& oldu1,
                         MField& oldu2,
                         MField& oldu3,
                         MField& oldb,
                         M1D& backgroundB,
                         M1D& backgroundU)
    {
        PopulateNodalVariablesAdjoint();

        stratifloat lambda;

        static MField dLdu1(BoundaryCondition::Bounded);
        static MField dLdu2(BoundaryCondition::Bounded);
        static MField dLdu3(BoundaryCondition::Decaying);
        static MField dLdb(BoundaryCondition::Bounded);

        static MField dEdu1(BoundaryCondition::Bounded);
        static MField dEdu2(BoundaryCondition::Bounded);
        static MField dEdu3(BoundaryCondition::Decaying);
        static MField dEdb(BoundaryCondition::Bounded);

        db_dz = ddz(backgroundB);
        db_dz.Filter();
        db_dz.ToNodal(dB_dz);

        static N1D Bgradientinv(BoundaryCondition::Decaying);
        for (int j=0; j<N3; j++)
        {
            if(dB_dz.Get()(j)>-0.00001 && dB_dz.Get()(j)<0.00001)
            {
                Bgradientinv.Get()(j) = 0;
            }
            else
            {
                Bgradientinv.Get()(j) = 1/dB_dz.Get()(j);
            }
        }

        stratifloat udotu = InnerProd(oldu1, oldu1, L3)
                            + InnerProd(oldu3, oldu3, L3)
                            + (ThreeDimensional?InnerProd(oldu2, oldu2, L3):0);

        stratifloat vdotv = InnerProd(u1, u1, L3)
                            + InnerProd(u3, u3, L3)
                            + (ThreeDimensional?InnerProd(u2, u2, L3):0);

        stratifloat udotv = InnerProd(u1, oldu1, L3)
                            + InnerProd(u3, oldu3, L3)
                            + InnerProd(b, oldb, L3)
                            + (ThreeDimensional?InnerProd(u2, oldu2, L3):0);

        udotu += InnerProd(oldb, oldb, L3, -Ri*Bgradientinv);
        vdotv += InnerProd(b, b, L3, -(1/Ri)*dB_dz);

        while(lambda==0)
        {
            lambda = SolveQuadratic(epsilon*udotu,
                                    2*epsilon*udotv - 2*udotu,
                                    epsilon*vdotv - 2*udotv);
            if (lambda==0)
            {
                std::cout << "Reducing step size" << std::endl;
                epsilon /= 2;
            }
        }

        dLdu1 = u1;
        dLdu2 = u2;
        dLdu3 = u3;

        nnTemp = -(1/Ri)*B*dB_dz;
        nnTemp.ToModal(dLdb);

        dEdu1 = oldu1;
        dEdu2 = oldu2;
        dEdu3 = oldu3;
        dEdb = oldb;

        // perform the gradient descent
        oldu1 = oldu1 + -epsilon*dLdu1 + -epsilon*lambda*dEdu1;
        oldu2 = oldu2 + -epsilon*dLdu2 + -epsilon*lambda*dEdu2;
        oldu3 = oldu3 + -epsilon*dLdu3 + -epsilon*lambda*dEdu3;
        oldb = oldb + -epsilon*dLdb + -epsilon*lambda*dEdb;

        // now actually update the values for the next step
        u1 = oldu1;
        u2 = oldu2;
        u3 = oldu3;
        b = oldb;
        p.Zero();

        // find the residual
        stratifloat residualNumerator = 0;

        boundedTemp = dLdu1 + lambda*dEdu1;
        residualNumerator += InnerProd(boundedTemp, boundedTemp, L3);
        decayingTemp = dLdu3 + lambda*dEdu3;
        residualNumerator += InnerProd(decayingTemp, decayingTemp, L3);
        boundedTemp = dLdb + lambda*dEdb;
        residualNumerator += InnerProd(boundedTemp, boundedTemp, L3);
        if(ThreeDimensional)
        {
            boundedTemp = dLdu2 + lambda*dEdu2;
            residualNumerator += InnerProd(boundedTemp, boundedTemp, L3);
        }

        stratifloat residualDenominator = InnerProd(dLdu1, dLdu1, L3)
                                        + InnerProd(dLdu3, dLdu3, L3)
                                        + InnerProd(dLdb, dLdb, L3)
                                        + (ThreeDimensional?InnerProd(dLdu2, dLdu2, L3):0);

        return residualNumerator/residualDenominator;
    }

    void RescaleForEnergy(stratifloat energy)
    {
        stratifloat scale;

        // energies are entirely quadratic
        // which makes this easy

        stratifloat energyBefore = KE() + PE();

        if (energyBefore!=0.0f)
        {
            scale = sqrt(energy/energyBefore);
        }
        else
        {
            scale = 0.0f;
        }

        u1 *= scale;
        u2 *= scale;
        u3 *= scale;
        b *= scale;
    }

private:
    void LoadAtTime(stratifloat time, bool reverse=true)
    {
        if (reverse)
        {
            while(fileAbove->first >= time && std::prev(fileAbove) != filenames.begin())
            {
                fileAbove--;
            }
            fileAbove++;
        }
        else
        {
            while(fileAbove->first <= time && std::next(std::next(fileAbove)) != filenames.end())
            {
                fileAbove++;
            }
            fileAbove--;
        }

        static stratifloat timeabove = -1;
        static stratifloat timebelow = -1;

        static NField u1Above(BoundaryCondition::Bounded);
        static NField u1Below(BoundaryCondition::Bounded);

        static NField u2Above(BoundaryCondition::Bounded);
        static NField u2Below(BoundaryCondition::Bounded);

        static NField u3Above(BoundaryCondition::Decaying);
        static NField u3Below(BoundaryCondition::Decaying);

        static NField bAbove(BoundaryCondition::Bounded);
        static NField bBelow(BoundaryCondition::Bounded);

        if (timeabove != fileAbove->first)
        {
            if (timebelow == fileAbove->first)
            {
                u1Above = u1Below;
                u2Above = u2Below;
                u3Above = u3Below;
                bAbove = bBelow;
            }
            else
            {
                LoadVariable(fileAbove->second, u1Above, 0);
                LoadVariable(fileAbove->second, u2Above, 1);
                LoadVariable(fileAbove->second, u3Above, 2);
                LoadVariable(fileAbove->second, bAbove, 3);
            }

            if (reverse)
            {
                LoadVariable(std::prev(fileAbove)->second, u1Below, 0);
                LoadVariable(std::prev(fileAbove)->second, u2Below, 1);
                LoadVariable(std::prev(fileAbove)->second, u3Below, 2);
                LoadVariable(std::prev(fileAbove)->second, bBelow, 3);
            }
            else
            {
                LoadVariable(std::next(fileAbove)->second, u1Below, 0);
                LoadVariable(std::next(fileAbove)->second, u2Below, 1);
                LoadVariable(std::next(fileAbove)->second, u3Below, 2);
                LoadVariable(std::next(fileAbove)->second, bBelow, 3);
            }

            timeabove = fileAbove->first;

            if (reverse)
            {
                timebelow = std::prev(fileAbove)->first;
            }
            else
            {
                timebelow = std::next(fileAbove)->first;
            }
        }

        U1_tot = ((time-timebelow)/(timeabove-timebelow))*u1Above + ((timeabove-time)/(timeabove-timebelow))*u1Below;
        U1_tot.ToModal(u1_tot);

        U2_tot = ((time-timebelow)/(timeabove-timebelow))*u2Above + ((timeabove-time)/(timeabove-timebelow))*u2Below;
        U2_tot.ToModal(u2_tot);

        U3_tot = ((time-timebelow)/(timeabove-timebelow))*u3Above + ((timeabove-time)/(timeabove-timebelow))*u3Below;
        U3_tot.ToModal(u3_tot);

        B_tot = ((time-timebelow)/(timeabove-timebelow))*bAbove + ((timeabove-time)/(timeabove-timebelow))*bBelow;
        B_tot.ToModal(b_tot);
    }

    void CNSolve(MField& solve, MField& into, int k, bool buoyancy = false)
    {
        if (buoyancy)
        {
            solve.Solve(implicitSolveBuoyancy[k], into);
        }
        else
        {
            if (solve.BC() == BoundaryCondition::Bounded)
            {
                solve.Solve(implicitSolveBounded[k], into);
            }
            else
            {
                solve.Solve(implicitSolveDecaying[k], into);
            }
        }
    }

    void CNSolve1D(M1D& solve, M1D& into, int k, bool buoyancy = false)
    {
        if (buoyancy)
        {
            solve.Solve(implicitSolveBuoyancy[k][0], into);
        }
        else
        {
            if (solve.BC() == BoundaryCondition::Bounded)
            {
                solve.Solve(implicitSolveBounded[k][0], into);
            }
            else
            {
                solve.Solve(implicitSolveDecaying[k][0], into);
            }
        }
    }

    void ImplicitUpdate(int k)
    {
        CNSolve(R1, u1, k);
        if(ThreeDimensional)
        {
            CNSolve(R2, u2, k);
        }
        CNSolve(R3, u3, k);
        CNSolve(RB, b, k, true);

        if (EvolveBackground)
        {
            CNSolve1D(RU_, u_, k);
            CNSolve1D(RB_, b_, k, true);
        }
    }

    void ImplicitUpdateAdjoint(int k)
    {
        CNSolve(R1, u1, k);
        if(ThreeDimensional)
        {
            CNSolve(R2, u2, k);
        }
        CNSolve(R3, u3, k);
        CNSolve(RB, b, k, true);
    }

    void ExplicitUpdate(int k)
    {
        // build up right hand sides for the implicit solve in R

        //   old      last rk step         pressure         explicit CN
        R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u1)+MatMulDim2(dim2Derivative2, u1)+MatMulDim3(dim3Derivative2Bounded, u1));
        if(ThreeDimensional)
        {
        R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u2)+MatMulDim2(dim2Derivative2, u2)+MatMulDim3(dim3Derivative2Bounded, u2));
        }
        R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u3)+MatMulDim2(dim2Derivative2, u3)+MatMulDim3(dim3Derivative2Decaying, u3));
        RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Pe)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Bounded, b));

        if (EvolveBackground)
        {
            // for the 1D variables u_ and b_ (background flow) we only use vertical derivative matrix
            RU_ = u_ + (0.5f*h[k]/Re)*MatMul1D(dim3Derivative2Bounded, u_);
            RB_ = b_ + (0.5f*h[k]/Pe)*MatMul1D(dim3Derivative2Bounded, b_);
        }

        // now construct explicit terms
        r1.Zero();
        if(ThreeDimensional)
        {
            r2.Zero();
        }
        r3.Zero();
        rB.Zero();

        ndTemp = Ri*B;// buoyancy force
        ndTemp.ToModal(decayingTemp);
        r3 -= decayingTemp;

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space

        // take into account background shear for nonlinear terms
        U1_tot = U1 + U_;

        nnTemp = U1_tot*U1_tot;
        nnTemp.ToModal(boundedTemp);
        r1 -= ddx(boundedTemp);

        ndTemp = U1_tot*U3;
        ndTemp.ToModal(decayingTemp);
        r3 -= ddx(decayingTemp);
        r1 -= ddz(decayingTemp);

        nnTemp = U3*U3;
        nnTemp.ToModal(boundedTemp);
        r3 -= ddz(boundedTemp);

        if(ThreeDimensional)
        {
            nnTemp = U2*U2;
            nnTemp.ToModal(boundedTemp);
            r2 -= ddy(boundedTemp);

            ndTemp = U2*U3;
            ndTemp.ToModal(decayingTemp);
            r3 -= ddy(decayingTemp);
            r2 -= ddz(decayingTemp);

            nnTemp = U1_tot*U2;
            nnTemp.ToModal(boundedTemp);
            r1 -= ddy(boundedTemp);
            r2 -= ddx(boundedTemp);
        }

        // buoyancy nonlinear terms
        nnTemp = U1_tot*B;
        nnTemp.ToModal(boundedTemp);
        rB -= ddx(boundedTemp);

        if(ThreeDimensional)
        {
            nnTemp = U2*B;
            nnTemp.ToModal(boundedTemp);
            rB -= ddy(boundedTemp);
        }

        ndTemp = U3*B;
        ndTemp.ToModal(decayingTemp);
        rB -= ddz(decayingTemp);

        // advection term from background buoyancy
        nnTemp = U3*dB_dz;
        nnTemp.ToModal(boundedTemp);
        rB -= boundedTemp;

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        if(ThreeDimensional)
        {
            R2 += (h[k]*beta[k])*r2;
        }
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;
    }

    void ExplicitUpdateLinear(int k)
    {
        // build up right hand sides for the implicit solve in R

        //   old      last rk step         pressure         explicit CN
        R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u1)+MatMulDim2(dim2Derivative2, u1)+MatMulDim3(dim3Derivative2Bounded, u1));
        if(ThreeDimensional)
        {
        R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u2)+MatMulDim2(dim2Derivative2, u2)+MatMulDim3(dim3Derivative2Bounded, u2));
        }
        R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u3)+MatMulDim2(dim2Derivative2, u3)+MatMulDim3(dim3Derivative2Decaying, u3));
        RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Pe)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Bounded, b));

        // now construct explicit terms
        r1.Zero();
        if(ThreeDimensional)
        {
            r2.Zero();
        }
        r3.Zero();
        rB.Zero();

        ndTemp = Ri*B;// buoyancy force
        ndTemp.ToModal(decayingTemp);
        r3 -= decayingTemp;

        //////// NONLINEAR TERMS ////////
        nnTemp = 2.0*U1_tot*U1;
        nnTemp.ToModal(boundedTemp);
        r1 -= ddx(boundedTemp);

        ndTemp = U1_tot*U3 + U3_tot*U1;
        ndTemp.ToModal(decayingTemp);
        r3 -= ddx(decayingTemp);
        r1 -= ddz(decayingTemp);

        nnTemp = 2.0*U3_tot*U3;
        nnTemp.ToModal(boundedTemp);
        r3 -= ddz(boundedTemp);

        if(ThreeDimensional)
        {
            nnTemp = 2.0*U2_tot*U2;
            nnTemp.ToModal(boundedTemp);
            r2 -= ddy(boundedTemp);

            ndTemp = U2*U3_tot + U3*U2_tot;
            ndTemp.ToModal(decayingTemp);
            r3 -= ddy(decayingTemp);
            r2 -= ddz(decayingTemp);

            nnTemp = U1_tot*U2 + U2_tot*U1;
            nnTemp.ToModal(boundedTemp);
            r1 -= ddy(boundedTemp);
            r2 -= ddx(boundedTemp);
        }

        // buoyancy nonlinear terms
        nnTemp = U1_tot*B + B_tot*U1;
        nnTemp.ToModal(boundedTemp);
        rB -= ddx(boundedTemp);

        if(ThreeDimensional)
        {
            nnTemp = U2_tot*B + U2*B_tot;
            nnTemp.ToModal(boundedTemp);
            rB -= ddy(boundedTemp);
        }

        ndTemp = U3*B_tot + U3_tot*B;
        ndTemp.ToModal(decayingTemp);
        rB -= ddz(decayingTemp);

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        if(ThreeDimensional)
        {
            R2 += (h[k]*beta[k])*r2;
        }
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;
    }

    void ExplicitUpdateAdjoint(int k)
    {
        // build up right hand sides for the implicit solve in R

        //   old      last rk step         pressure         explicit CN
        R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u1)+MatMulDim2(dim2Derivative2, u1)+MatMulDim3(dim3Derivative2Bounded, u1));
        if(ThreeDimensional)
        {
        R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u2)+MatMulDim2(dim2Derivative2, u2)+MatMulDim3(dim3Derivative2Bounded, u2));
        }
        R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u3)+MatMulDim2(dim2Derivative2, u3)+MatMulDim3(dim3Derivative2Decaying, u3));
        RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Pe)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Bounded, b));

        // now construct explicit terms
        r1.Zero();
        if(ThreeDimensional)
        {
            r2.Zero();
        }
        r3.Zero();
        rB.Zero();

        // adjoint buoyancy
        bForcing -= Ri*U3;

        //////// NONLINEAR TERMS ////////
        // advection of adjoint quantities by the direct flow
        nnTemp = U1*U1_tot;
        nnTemp.ToModal(boundedTemp);
        r1 += ddx(boundedTemp);
        if(ThreeDimensional)
        {
            nnTemp = U1*U2_tot;
            nnTemp.ToModal(boundedTemp);
            r1 += ddy(boundedTemp);
        }
        ndTemp = U1*U3_tot;
        ndTemp.ToModal(decayingTemp);
        r1 += ddz(decayingTemp);

        if(ThreeDimensional)
        {
            nnTemp = U2*U1_tot;
            nnTemp.ToModal(boundedTemp);
            r2 += ddx(boundedTemp);
            nnTemp = U2*U2_tot;
            nnTemp.ToModal(boundedTemp);
            r2 += ddy(boundedTemp);
            ndTemp = U2*U3_tot;
            ndTemp.ToModal(decayingTemp);
            r2 += ddz(decayingTemp);
        }

        ndTemp = U3*U1_tot;
        ndTemp.ToModal(decayingTemp);
        r3 += ddx(decayingTemp);
        if(ThreeDimensional)
        {
            ndTemp = U3*U2_tot;
            ndTemp.ToModal(decayingTemp);
            r3 += ddy(decayingTemp);
        }
        nnTemp = U3*U3_tot;
        nnTemp.ToModal(boundedTemp);
        r3 += ddz(boundedTemp);

        nnTemp = B*U1_tot;
        nnTemp.ToModal(boundedTemp);
        rB += ddx(boundedTemp);
        if(ThreeDimensional)
        {
            nnTemp = B*U2_tot;
            nnTemp.ToModal(boundedTemp);
            rB += ddy(boundedTemp);
        }
        ndTemp = B*U3_tot;
        ndTemp.ToModal(decayingTemp);
        rB += ddz(decayingTemp);

        // extra adjoint nonlinear terms
        boundedTemp = ddx(u1_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*U1;
        if(ThreeDimensional)
        {
            boundedTemp = ddx(u2_tot);
            boundedTemp.ToNodal(nnTemp);
            nnTemp2 += nnTemp*U2;
        }
        decayingTemp = ddx(u3_tot);
        decayingTemp.ToNodal(ndTemp);
        u1Forcing -= nnTemp2 + ndTemp*U3;

        if(ThreeDimensional)
        {
            boundedTemp = ddy(u1_tot);
            boundedTemp.ToNodal(nnTemp);
            nnTemp2 = nnTemp*U1;
            boundedTemp = ddy(u2_tot);
            boundedTemp.ToNodal(nnTemp);
            nnTemp2 += nnTemp*U2;
            decayingTemp = ddy(u3_tot);
            decayingTemp.ToNodal(ndTemp);
            u2Forcing -= nnTemp2 + ndTemp*U3;
        }

        decayingTemp = ddz(u1_tot);
        decayingTemp.ToNodal(ndTemp);
        ndTemp2 = ndTemp*U1;
        if(ThreeDimensional)
        {
            decayingTemp = ddz(u2_tot);
            decayingTemp.ToNodal(ndTemp);
            ndTemp2 += ndTemp*U2;
        }
        boundedTemp = ddz(u3_tot);
        boundedTemp.ToNodal(nnTemp);
        u3Forcing -= ndTemp2 + nnTemp*U3;


        boundedTemp = ddx(b_tot);
        boundedTemp.ToNodal(nnTemp);
        u1Forcing -= nnTemp*B;

        if(ThreeDimensional)
        {
            boundedTemp = ddy(b_tot);
            boundedTemp.ToNodal(nnTemp);
            u2Forcing -= nnTemp*B;
        }

        decayingTemp = ddz(b_tot);
        decayingTemp.ToNodal(ndTemp);
        u3Forcing -= ndTemp*B;

        // Now include all the forcing terms
        u1Forcing.ToModal(boundedTemp);
        r1 += boundedTemp;
        if (ThreeDimensional)
        {
            u2Forcing.ToModal(boundedTemp);
            r2 += boundedTemp;
        }
        u3Forcing.ToModal(decayingTemp);
        r3 += decayingTemp;
        bForcing.ToModal(boundedTemp);
        rB += boundedTemp;

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        if(ThreeDimensional)
        {
            R2 += (h[k]*beta[k])*r2;
        }
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;
    }

public:
    // these are the actual variables we care about
    MField u1, u2, u3, b;
    MField p;
private:
    // background flow
    M1D u_, b_;
    mutable M1D db_dz;

    // direct flow (used for adjoint evolution)
    MField u1_tot, u2_tot, u3_tot, b_tot;

    // Nodal versions of variables
    mutable NField U1, U2, U3, B;

    // extra variables required for adjoint forcing
    stratifloat J, K;

    NField u1Forcing, u2Forcing, u3Forcing;
    NField bForcing;

    // parameters for the scheme
    const int s = 3;
    stratifloat h[3];
    const stratifloat beta[3] = {1.0f, 25.0f/8.0f, 9.0f/4.0f};
    const stratifloat zeta[3] = {0, -17.0f/8.0f, -5.0f/4.0f};

    // these are intermediate variables used in the computation, preallocated for efficiency
    MField R1, R2, R3, RB;
    M1D RU_, RB_;
    MField r1, r2, r3, rB;
    mutable NField U1_tot, U2_tot, U3_tot, B_tot;
    mutable N1D U_, B_, dB_dz;
    mutable NField ndTemp, nnTemp;
    mutable NField ndTemp2, nnTemp2;
    mutable MField decayingTemp, boundedTemp;
    mutable M1D decayingTemp1D, boundedTemp1D;
    mutable N1D ndTemp1D, nnTemp1D;
    MField& divergence; // reference to share memory
    MField& q;

    // these are precomputed matrices for performing and solving derivatives
    DiagonalMatrix<stratifloat, -1> dim1Derivative2;
    DiagonalMatrix<stratifloat, -1> dim2Derivative2;
    MatrixX dim3Derivative2Bounded;
    MatrixX dim3Derivative2Decaying;

    std::vector<SparseLU<SparseMatrix<stratifloat>>> implicitSolveBounded[3];
    std::vector<SparseLU<SparseMatrix<stratifloat>>> implicitSolveDecaying[3];
    std::vector<SparseLU<SparseMatrix<stratifloat>>> implicitSolveBuoyancy[3];
    std::vector<SparseLU<SparseMatrix<stratifloat>>> solveLaplacian;

    // for flow saving/loading
    std::map<stratifloat, std::string> filenames;
    std::map<stratifloat, std::string>::iterator fileAbove;
    std::map<stratifloat, State> snapshots;
    std::map<stratifloat, State>::iterator snapshotAbove;

    const std::string snapshotdir = "snapshots/";
    std::string imageDirectory;
};
