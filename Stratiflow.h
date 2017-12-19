#pragma once

#include "Field.h"
#include "Differentiation.h"
#include "Integration.h"
#include "Graph.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <dirent.h>
#include <map>

#include <omp.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
            result += buffer.data();
    }
    return result;
}

// will become unnecessary with C++17
#define MatMulDim1 Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMulDim2 Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMulDim3 Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMul1D Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>, stratifloat, stratifloat, 1, 1, N3>

class IMEXRK
{
public:
    static constexpr int N1 = 384;
    static constexpr int N2 = 1;
    static constexpr int N3 = 440;

    static constexpr int M1 = N1/2 + 1;

    static constexpr stratifloat L1 = 16.0f; // size of domain streamwise
    static constexpr stratifloat L2 = 4.0f;  // size of domain spanwise
    static constexpr stratifloat L3 = 5.0f; // vertical scaling factor

    stratifloat deltaT = 0.01f;
    const stratifloat Re = 1000;
    const stratifloat Ri = 0.1;

    using NField = NodalField<N1,N2,N3>;
    using MField = ModalField<N1,N2,N3>;
    using M1D = Modal1D<N1,N2,N3>;
    using N1D = Nodal1D<N1,N2,N3>;

    long totalForcing = 0;
    long totalExplicit = 0;
    long totalImplicit = 0;
    long totalDivergence = 0;

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

    , u3Forcing(u3)
    , bForcing(b)

    , solveLaplacian(M1*N2)
    , implicitSolveBounded{std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>>(M1*N2)}
    , implicitSolveDecaying{std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>>(M1*N2), std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>>(M1*N2)}

    {
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

                // prevent singularity - first row gives average value at infinity
                if (j1 == 0 && j2 == 0)
                {
                    laplacian.row(0).setConstant(1);

                    // because these terms are different in expansion
                    laplacian(0,0) = 2;
                    laplacian(0,N3-1) = 2;
                }

                solve = laplacian.sparseView();

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

            totalExplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
            totalImplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            totalDivergence += std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();
        }

        // To prevent anything dodgy accumulating in the unused coefficients
        u1.Filter();
        u2.Filter();
        u3.Filter();
        b.Filter();
        p.Filter();

        // correct for instability in scheme towards infinity
        u_.ToNodal(U_);
        b_.ToNodal(B_);
        for (int j=0; j<50; j++)
        {
            U_.Get()(j) = 1;
            B_.Get()(j) = 1;
            U_.Get()(N3-1-j) = -1;
            B_.Get()(N3-1-j) = -1;
        }
        U_.ToModal(u_);
        B_.ToModal(b_);
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

            totalForcing += std::chrono::duration_cast<std::chrono::milliseconds>(t0-t4).count();
            totalExplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
            totalImplicit += std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
            totalDivergence += std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count();

            time -= h[k];
        }

        // To prevent anything dodgy accumulating in the unused coefficients
        u1.Filter();
        u2.Filter();
        u3.Filter();
        b.Filter();
        p.Filter();
    }


    void PlotBuoyancy(std::string filename, int j2) const
    {
        b.ToNodal(B);
        b_.ToNodal(B_);
        nnTemp = B_ + B;
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
        HeatPlot(u2, L1, L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2) const
    {
        u1.ToNodal(U1);
        u_.ToNodal(U_);
        U1 += U_;
        U1.ToModal(boundedTemp);
        HeatPlot(boundedTemp, L1, L3, j2, filename);
    }

    void SetInitial(NField velocity1, NField velocity2, NField velocity3, NField buoyancy)
    {
        velocity1.ToModal(u1);
        velocity2.ToModal(u2);
        velocity3.ToModal(u3);
        buoyancy.ToModal(b);
    }

    void SetBackground(N1D velocity, N1D buoyancy)
    {
        velocity.ToModal(u_);
        buoyancy.ToModal(b_);
    }

    // gives an upper bound on cfl number - also updates timestep
    stratifloat CFL()
    {
        static ArrayX z = VerticalPoints(L3, N3);
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);

        u_.ToNodal(U_);
        U1 += U_;

        stratifloat delta1 = L1/N1;
        stratifloat delta2 = L2/N2;
        stratifloat delta3 = z(N3/2) - z(N3/2+1); // smallest gap in middle

        stratifloat cfl = U1.Max()/delta1 + U2.Max()/delta2 + U3.Max()/delta3;
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
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);

        // // hack for now: perturbation energy relative to frozen bg
        // u_.ToNodal(U_);
        // U1 += U_;
        // NField Uinitial(BoundaryCondition::Bounded);
        // Uinitial.SetValue([](stratifloat z){return tanh(z);}, L3);
        // U1 -= Uinitial;

        ndTemp = 0.5f*(U1*U1 + U2*U2 + U3*U3);

        return IntegrateAllSpace(ndTemp, L1, L2, L3)/L1/L2;
    }

    stratifloat PE() const
    {
        b.ToNodal(B);
        db_dz = ddz(b_);
        db_dz.ToNodal(dB_dz);
        ndTemp = 0.5f*Ri*B*B;

        for (int j=0; j<N3; j++)
        {
            if(dB_dz.Get()(j)>-0.00001 && dB_dz.Get()(j)<0.00001)
            {
                ndTemp.slice(j) = 0;
            }
            else
            {
                ndTemp.slice(j) /= dB_dz.Get()(j);
            }
        }

        return IntegrateAllSpace(ndTemp, L1, L2, L3)/L1/L2;
    }

    void RemoveDivergence(stratifloat pressureMultiplier=1.0f)
    {
        // construct the diverence of u
        divergence = ddx(u1) + ddy(u2) + ddz(u3);

        // constant term - set value at infinity to zero
        divergence(0,0,0) = 0;

        // solve Δq = ∇·u as linear system Aq = divergence
        divergence.Solve(solveLaplacian, q);

        // subtract the gradient of this from the velocity
        u1 -= ddx(q);
        u2 -= ddy(q);
        u3 -= ddz(q);

        // also add it on to p for the next step
        // this is scaled to match the p that was added before
        // effectively we have forward euler
        p += pressureMultiplier*q;
    }

    void SaveFlow(const std::string& filename, bool includeBackground = true) const
    {
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        if (includeBackground)
        {
            u_.ToNodal(U_);
            b_.ToNodal(B_);

            U1 += U_;
            B += B_;
        }

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

    stratifloat I(const MField& b_total) const
    {
        static M1D ave(BoundaryCondition::Bounded);
        HorizontalAverage(b_total, ave);

        static N1D aveN(BoundaryCondition::Bounded);
        ave.ToNodal(aveN);

        static N1D one(BoundaryCondition::Bounded);
        one.SetValue([](stratifloat z){return 1;}, L3);

        static N1D integrand(BoundaryCondition::Decaying);
        integrand = one + (-1)*aveN*aveN;

        return IntegrateVertically(integrand, L3);
    }

    stratifloat JoverK()
    {
        static MField u1_total(BoundaryCondition::Bounded);
        static MField b_total(BoundaryCondition::Bounded);

        u1.ToNodal(U1);
        u_.ToNodal(U_);

        b.ToNodal(B);
        b_.ToNodal(B_);

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
        // calculate length scale of the flow
        stratifloat I = this->I(b_total);

        // calculate weight function for integrals
        static N1D varpi(BoundaryCondition::Decaying);
        varpi.SetValue([I](stratifloat z){return exp(-z*z/I/I);}, L3);

        // work out variation of buoyancy from average
        static M1D bAveModal(BoundaryCondition::Bounded);
        HorizontalAverage(b_total, bAveModal);
        static N1D bAveNodal(BoundaryCondition::Bounded);
        bAveModal.ToNodal(bAveNodal);
        b_total.ToNodal(nnTemp);
        nnTemp = nnTemp + -1*bAveNodal;

        // (b-<b>)*w
        u3_total.ToNodal(U3);
        ndTemp = nnTemp*U3;
        ndTemp.ToModal(decayingTemp);

        // construct integrand for J
        static M1D bwAve(BoundaryCondition::Decaying);
        HorizontalAverage(decayingTemp, bwAve);
        static N1D Jintegrand(BoundaryCondition::Decaying);
        bwAve.ToNodal(Jintegrand);
        Jintegrand = -1*Jintegrand*varpi;

        J = IntegrateVertically(Jintegrand, L3);

        // work out average buoyancy gradient
        static M1D dbdz(BoundaryCondition::Decaying);
        dbdz = ddz(bAveModal);

        // construct integrand for K
        static N1D Kintegrand(BoundaryCondition::Decaying);
        dbdz.ToNodal(Kintegrand);
        Kintegrand = Kintegrand*varpi;

        K = IntegrateVertically(Kintegrand, L3);

        // also calculate other things needed for DAL
        static N1D varpiDerivative(BoundaryCondition::Bounded);
        varpiDerivative.SetValue([I](stratifloat z){return 2*z*z/I/I/I;}, L3);

        Jintegrand = Jintegrand*varpiDerivative;
        stratifloat Jderivative = IntegrateVertically(Jintegrand, L3);

        Kintegrand = Kintegrand*varpiDerivative;
        stratifloat Kderivative = IntegrateVertically(Kintegrand, L3);

        // lagrange multiplier for value of I
        stratifloat lambda = J*Kderivative/K/K - Jderivative/K; // quotient rule for -J/K

        // forcing term for u3
        b_total.ToNodal(nnTemp);
        ndTemp = (1/K)*varpi*(nnTemp+(-1)*bAveNodal);
        ndTemp.ToModal(u3Forcing);

        // forcing term for b
        varpiDerivative.SetValue([I](stratifloat z){return 2*z/I/I;}, L3);
        nnTemp = (-J/K/K)*varpiDerivative*varpi;

        HorizontalAverage(b_total, boundedTemp1D);
        boundedTemp1D.ToNodal(nnTemp1D);
        nnTemp += -2*lambda*nnTemp1D;
        nnTemp.ToModal(bForcing);
    }

    void BuildFilenameMap()
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

        lastFilenameAbove = filenames.end();
        lastFilenameAbove--;
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

                    implicitSolveBounded[k][j1*N2+j2].compute(solve);


                    laplacian = dim3Derivative2Decaying;
                    laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
                    laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);

                    solve = (MatrixX::Identity(N3, N3)-0.5*h[k]*laplacian/Re).sparseView();

                    implicitSolveDecaying[k][j1*N2+j2].compute(solve);
                }

            }
        }
    }

    stratifloat Optimise(stratifloat epsilon,
                         stratifloat E_0,
                         MField& oldu1,
                         MField& oldu2,
                         MField& oldu3,
                         MField& oldb,
                         M1D& backgroundB)
    {
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        MField& scaledvarrho = boundedTemp; // share memory
        db_dz = ddz(backgroundB);
        db_dz.ToNodal(dB_dz);
        nnTemp = (1/Ri)*B*dB_dz;
        nnTemp.ToModal(scaledvarrho);

        ndTemp = U1*U1;
        stratifloat vdotv = IntegrateAllSpace(ndTemp, L1, L2, L3);
        ndTemp = U2*U2;
        vdotv += IntegrateAllSpace(ndTemp, L1, L2, L3);
        ndTemp = U3*U3;
        vdotv += IntegrateAllSpace(ndTemp, L1, L2, L3);
        ndTemp = (1/Ri)*B*B*dB_dz;
        vdotv += IntegrateAllSpace(ndTemp, L1, L2, L3);
        vdotv /= L1*L2;


        oldu1.ToNodal(nnTemp);
        ndTemp = nnTemp*U1;
        stratifloat udotv = IntegrateAllSpace(ndTemp, L1, L2, L3);
        oldu2.ToNodal(nnTemp);
        ndTemp = nnTemp*U2;
        udotv += IntegrateAllSpace(ndTemp, L1, L2, L3);
        oldu3.ToNodal(ndTemp);
        ndTemp = ndTemp*U3;
        udotv += IntegrateAllSpace(ndTemp, L1, L2, L3);
        oldb.ToNodal(nnTemp);
        ndTemp = nnTemp*B;
        udotv += IntegrateAllSpace(ndTemp, L1, L2, L3);
        udotv /= L1*L2;

        stratifloat mag = sqrt(vdotv - udotv*udotv/(2*E_0));
        stratifloat alpha = epsilon * mag;

        std::cout << alpha << " " << udotv << " " << vdotv << " " << E_0 << std::endl;
        stratifloat residual = (mag*mag/vdotv);

        // store the new values in old (which we no longer need after this)
        oldu1 = cos(alpha)*oldu1 + (sqrt(2*E_0)*sin(alpha)/mag)*((udotv/(2*E_0))*oldu1 + -1.0*u1);
        oldu2 = cos(alpha)*oldu2 + (sqrt(2*E_0)*sin(alpha)/mag)*((udotv/(2*E_0))*oldu2 + -1.0*u2);
        oldu3 = cos(alpha)*oldu3 + (sqrt(2*E_0)*sin(alpha)/mag)*((udotv/(2*E_0))*oldu3 + -1.0*u3);
        oldb = cos(alpha)*oldb   + (sqrt(2*E_0)*sin(alpha)/mag)*((udotv/(2*E_0))*oldb + -1.0*scaledvarrho);

        // now actually update the values for the next step
        u1 = oldu1;
        u2 = oldu2;
        u3 = oldu3;
        b = oldb;
        p.Zero();

        return residual;
    }

private:
    void LoadVariable(std::string filename, NField& into, int index)
    {
        std::ifstream filestream(filename, std::ios::in | std::ios::binary);

        filestream.seekg(N1*N2*N3*index*sizeof(stratifloat));
        into.Load(filestream);
    }

    void LoadAtTime(stratifloat time)
    {
        while(lastFilenameAbove->first > time && std::prev(lastFilenameAbove) != filenames.begin())
        {
            lastFilenameAbove--;
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

        if (timeabove != lastFilenameAbove->first)
        {
            if (timebelow == lastFilenameAbove->first)
            {
                bAbove = bBelow;
            }
            else
            {
                LoadVariable(lastFilenameAbove->second, u1Above, 0);
                LoadVariable(lastFilenameAbove->second, u2Above, 1);
                LoadVariable(lastFilenameAbove->second, u3Above, 2);
                LoadVariable(lastFilenameAbove->second, bAbove, 3);
            }
            LoadVariable(std::prev(lastFilenameAbove)->second, u1Below, 0);
            LoadVariable(std::prev(lastFilenameAbove)->second, u2Below, 1);
            LoadVariable(std::prev(lastFilenameAbove)->second, u3Below, 2);
            LoadVariable(std::prev(lastFilenameAbove)->second, bBelow, 3);


            timeabove = lastFilenameAbove->first;
            timebelow = std::prev(lastFilenameAbove)->first;
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

    template<typename T>
    Dim1MatMul<T, complex, complex, M1, N2, N3> ddx(const StackContainer<T, complex, M1, N2, N3>& f) const
    {
        static DiagonalMatrix<complex, -1> dim1Derivative = FourierDerivativeMatrix(L1, N1, 1);

        return Dim1MatMul<T, complex, complex, M1, N2, N3>(dim1Derivative, f);
    }

    template<typename T>
    Dim2MatMul<T, complex, complex, M1, N2, N3> ddy(const StackContainer<T, complex, M1, N2, N3>& f) const
    {
        static DiagonalMatrix<complex, -1> dim2Derivative = FourierDerivativeMatrix(L2, N2, 2);

        return Dim2MatMul<T, complex, complex, M1, N2, N3>(dim2Derivative, f);
    }

    template<typename A, typename T, int K1, int K2, int K3>
    Dim3MatMul<A, stratifloat, T, K1, K2, K3> ddz(const StackContainer<A, T, K1, K2, K3>& f) const
    {
        static MatrixX dim3DerivativeBounded = VerticalDerivativeMatrix(BoundaryCondition::Bounded, L3, N3);
        static MatrixX dim3DerivativeDecaying = VerticalDerivativeMatrix(BoundaryCondition::Decaying, L3, N3);

        if (f.BC() == BoundaryCondition::Decaying)
        {
            return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3DerivativeDecaying, f, BoundaryCondition::Bounded);
        }
        else
        {
            return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3DerivativeBounded, f, BoundaryCondition::Decaying);
        }
    }

    void CNSolve(MField& solve, MField& into, int k)
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

    void CNSolve1D(M1D& solve, M1D& into, int k)
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

    void ImplicitUpdate(int k)
    {
        CNSolve(R1, u1, k);
        CNSolve(R2, u2, k);
        CNSolve(R3, u3, k);
        CNSolve(RB, b, k);

        CNSolve1D(RU_, u_, k);
        CNSolve1D(RB_, b_, k);
    }

    void ImplicitUpdateAdjoint(int k)
    {
        CNSolve(R1, u1, k);
        CNSolve(R2, u2, k);
        CNSolve(R3, u3, k);
        CNSolve(RB, b, k);
    }

    void ExplicitUpdate(int k)
    {
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        // build up right hand sides for the implicit solve in R

        //   old      last rk step         pressure         explicit CN
        R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u1)+MatMulDim2(dim2Derivative2, u1)+MatMulDim3(dim3Derivative2Bounded, u1));
        R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u2)+MatMulDim2(dim2Derivative2, u2)+MatMulDim3(dim3Derivative2Bounded, u2));
        R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u3)+MatMulDim2(dim2Derivative2, u3)+MatMulDim3(dim3Derivative2Decaying, u3));
        RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Bounded, b));

        // // for the 1D variables u_ and b_ (background flow) we only use vertical derivative matrix
        RU_ = u_ + (0.5f*h[k]/Re)*MatMul1D(dim3Derivative2Bounded, u_);
        RB_ = b_ + (0.5f*h[k]/Re)*MatMul1D(dim3Derivative2Bounded, b_);

        // now construct explicit terms
        r1.Zero();
        r2.Zero();
        r3.Zero();
        rB.Zero();

        ndTemp = Ri*B;// buoyancy force
        ndTemp.ToModal(decayingTemp);
        r3 += decayingTemp; // z goes down

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space

        // take into account background shear for nonlinear terms
        u_.ToNodal(U_);
        U1 += U_;

        nnTemp = U1*U1;
        nnTemp.ToModal(boundedTemp);
        r1 -= ddx(boundedTemp);

        ndTemp = U1*U3;
        ndTemp.ToModal(decayingTemp);
        r3 -= ddx(decayingTemp);
        r1 -= ddz(decayingTemp);

        nnTemp = U2*U2;
        nnTemp.ToModal(boundedTemp);
        r2 -= ddy(boundedTemp);

        ndTemp = U2*U3;
        ndTemp.ToModal(decayingTemp);
        r3 -= ddy(decayingTemp);
        r2 -= ddz(decayingTemp);

        nnTemp = U3*U3;
        nnTemp.ToModal(boundedTemp);
        r3 -= ddz(boundedTemp);

        nnTemp = U1*U2;
        nnTemp.ToModal(boundedTemp);
        r1 -= ddy(boundedTemp);
        r2 -= ddx(boundedTemp);

        // buoyancy nonlinear terms
        nnTemp = U1*B;
        nnTemp.ToModal(boundedTemp);
        rB -= ddx(boundedTemp);

        nnTemp = U2*B;
        nnTemp.ToModal(boundedTemp);
        rB -= ddy(boundedTemp);

        ndTemp = U3*B;
        ndTemp.ToModal(decayingTemp);
        rB -= ddz(decayingTemp);

        // advection term from background buoyancy
        db_dz = ddz(b_);
        db_dz.ToNodal(dB_dz);

        nnTemp = U3*dB_dz;
        nnTemp.ToModal(boundedTemp);
        rB -= boundedTemp;

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        R2 += (h[k]*beta[k])*r2;
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;
    }


    void ExplicitUpdateAdjoint(int k)
    {
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        // u1_tot.ToNodal(U1_tot);
        // u2_tot.ToNodal(U2_tot);
        // u3_tot.ToNodal(U3_tot);
        // b_tot.ToNodal(B_tot);

        // build up right hand sides for the implicit solve in R

        //   old      last rk step         pressure         explicit CN
        R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u1)+MatMulDim2(dim2Derivative2, u1)+MatMulDim3(dim3Derivative2Bounded, u1));
        R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u2)+MatMulDim2(dim2Derivative2, u2)+MatMulDim3(dim3Derivative2Bounded, u2));
        R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u3)+MatMulDim2(dim2Derivative2, u3)+MatMulDim3(dim3Derivative2Decaying, u3));
        RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Bounded, b));

        // now construct explicit terms
        r1.Zero();
        r2.Zero();
        r3 = u3Forcing;
        rB = bForcing;

        // adjoint buoyancy
        nnTemp = Ri*U3;
        nnTemp.ToModal(boundedTemp);
        rB -= boundedTemp;

        //////// NONLINEAR TERMS ////////
        // advection of adjoint quantities by the direct flow
        nnTemp = U1*U1_tot;
        nnTemp.ToModal(boundedTemp);
        r1 += ddx(boundedTemp);
        nnTemp = U1*U2_tot;
        nnTemp.ToModal(boundedTemp);
        r1 += ddy(boundedTemp);
        ndTemp = U1*U3_tot;
        ndTemp.ToModal(decayingTemp);
        r1 += ddz(decayingTemp);

        nnTemp = U2*U1_tot;
        nnTemp.ToModal(boundedTemp);
        r2 += ddx(boundedTemp);
        nnTemp = U2*U2_tot;
        nnTemp.ToModal(boundedTemp);
        r2 += ddy(boundedTemp);
        ndTemp = U2*U3_tot;
        ndTemp.ToModal(decayingTemp);
        r2 += ddz(decayingTemp);

        ndTemp = U3*U1_tot;
        ndTemp.ToModal(decayingTemp);
        r3 += ddx(decayingTemp);
        ndTemp = U3*U2_tot;
        ndTemp.ToModal(decayingTemp);
        r3 += ddy(decayingTemp);
        nnTemp = U3*U3_tot;
        nnTemp.ToModal(boundedTemp);
        r3 += ddz(boundedTemp);

        nnTemp = B*U1_tot;
        nnTemp.ToModal(boundedTemp);
        rB += ddx(boundedTemp);
        nnTemp = B*U2_tot;
        nnTemp.ToModal(boundedTemp);
        rB += ddy(boundedTemp);
        ndTemp = B*U3_tot;
        ndTemp.ToModal(decayingTemp);
        rB += ddz(decayingTemp);

        // extra adjoint nonlinear terms
        boundedTemp = ddx(u1_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*U1;
        boundedTemp = ddx(u2_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
        decayingTemp = ddx(u3_tot);
        decayingTemp.ToNodal(ndTemp);
        nnTemp2 = nnTemp2 + ndTemp*U3;
        nnTemp2.ToModal(boundedTemp);
        r1 -= boundedTemp;

        boundedTemp = ddy(u1_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*U1;
        boundedTemp = ddy(u2_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
        decayingTemp = ddy(u3_tot);
        decayingTemp.ToNodal(ndTemp);
        nnTemp2 = nnTemp2 + ndTemp*U3;
        nnTemp2.ToModal(boundedTemp);
        r2 -= boundedTemp;

        decayingTemp = ddz(u1_tot);
        decayingTemp.ToNodal(ndTemp);
        ndTemp2 = ndTemp*U1;
        decayingTemp = ddz(u2_tot);
        decayingTemp.ToNodal(ndTemp);
        ndTemp2 += ndTemp*U2;
        boundedTemp = ddz(u3_tot);
        boundedTemp.ToNodal(nnTemp);
        ndTemp2 = ndTemp2 + nnTemp*U3;
        ndTemp2.ToModal(decayingTemp);
        r3 -= decayingTemp;

        boundedTemp = ddx(b_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*B;
        nnTemp2.ToModal(boundedTemp);
        r1 -= boundedTemp;

        boundedTemp = ddy(b_tot);
        boundedTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*B;
        nnTemp2.ToModal(boundedTemp);
        r2 -= boundedTemp;

        decayingTemp = ddz(b_tot);
        decayingTemp.ToNodal(ndTemp);
        ndTemp2 = ndTemp*B;
        ndTemp2.ToModal(decayingTemp);
        r3 -= decayingTemp;

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        R2 += (h[k]*beta[k])*r2;
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

    // extra variables required for adjoint forcing
    stratifloat J, K;

    MField u3Forcing;
    MField bForcing;

    // parameters for the scheme
    const int s = 3;
    stratifloat h[3];
    const stratifloat beta[3] = {1.0f, 25.0f/8.0f, 9.0f/4.0f};
    const stratifloat zeta[3] = {0, -17.0f/8.0f, -5.0f/4.0f};

    // these are intermediate variables used in the computation, preallocated for efficiency
    MField R1, R2, R3, RB;
    M1D RU_, RB_;
    MField r1, r2, r3, rB;
    mutable NField U1, U2, U3, B;
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

    std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>> implicitSolveBounded[3];
    std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>> implicitSolveDecaying[3];
    std::vector<SimplicialLDLT<SparseMatrix<stratifloat>>> solveLaplacian;

    // for flow saving/loading
    std::map<stratifloat, std::string> filenames;
    std::map<stratifloat, std::string>::iterator lastFilenameAbove;
    const std::string snapshotdir = "/local/scratch/public/jpp39/snapshots/";
};
