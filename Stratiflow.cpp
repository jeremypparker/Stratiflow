#include "Field.h"
#include "Differentiation.h"
#include "Integration.h"
#include "Graph.h"

#include <iostream>
#include <fstream>
#include <chrono>

#include <omp.h>

// will become unnecessary with C++17
#define MatMulDim1 Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMulDim2 Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMulDim3 Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>, stratifloat, complex, M1, N2, N3>
#define MatMul1D Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>, stratifloat, stratifloat, 1, 1, N3>

class IMEXRK
{
public:
    static constexpr int N1 = 320;
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

    long totalExplicit = 0;
    long totalImplicit = 0;
    long totalDivergence = 0;

public:
    IMEXRK()
    : u1(BoundaryCondition::Bounded)
    , u2(BoundaryCondition::Bounded)
    , u3(BoundaryCondition::Decaying)
    , p(BoundaryCondition::Bounded)
    , b(BoundaryCondition::Decaying)

    , u_(BoundaryCondition::Bounded)
    , b_(BoundaryCondition::Bounded)
    , db_dz(BoundaryCondition::Decaying)

    , R1(u1), R2(u2), R3(u3), RB(b)
    , RU_(u_), RB_(b_)
    , r1(u1), r2(u2), r3(u3), rB(b)
    , U1(BoundaryCondition::Bounded)
    , U2(BoundaryCondition::Bounded)
    , U3(BoundaryCondition::Decaying)
    , B(BoundaryCondition::Decaying)
    , U_(BoundaryCondition::Bounded)
    , B_(BoundaryCondition::Bounded)
    , dB_dz(BoundaryCondition::Decaying)
    , decayingTemp(BoundaryCondition::Decaying)
    , boundedTemp(BoundaryCondition::Bounded)
    , ndTemp(BoundaryCondition::Decaying)
    , nnTemp(BoundaryCondition::Bounded)
    , divergence(boundedTemp)
    , q(boundedTemp)

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

    stratifloat KE() const
    {
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);

        // hack for now: perturbation energy relative to frozen bg
        u_.ToNodal(U_);
        U1 += U_;
        NField Uinitial(BoundaryCondition::Bounded);
        Uinitial.SetValue([](stratifloat z){return tanh(z);}, L3);
        U1 -= Uinitial;

        ndTemp = 0.5f*(U1*U1 + U2*U2 + U3*U3);

        return IntegrateAllSpace(ndTemp, L1, L2, L3)/L1/L2;
    }

    stratifloat PE() const
    {
        b.ToNodal(B);

        // hack for now: perturbation energy relative to frozen bg
        b_.ToNodal(B_);
        nnTemp = B_ + B;
        N1D Binitial(BoundaryCondition::Bounded);
        Binitial.SetValue([](stratifloat z){return tanh(2*z);}, L3);
        nnTemp -= Binitial;

        ndTemp = 0.5f*Ri*nnTemp*nnTemp;

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

    void SaveFlow(const std::string& filename)
    {
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        u_.ToNodal(U_);
        b_.ToNodal(B_);

        std::ofstream filestream(filename, std::ios::out | std::ios::binary);
        U1 += U_;
        U1.Save(filestream);
        U2.Save(filestream);
        U3.Save(filestream);

        nnTemp = B + B_;
        nnTemp.Save(filestream);
    }

    stratifloat I(const MField& u_total) const
    {
        static M1D ave(BoundaryCondition::Bounded);
        HorizontalAverage(u_total, ave);

        static N1D aveN(BoundaryCondition::Bounded);
        ave.ToNodal(aveN);

        static N1D one(BoundaryCondition::Bounded);
        one.SetValue([](stratifloat z){return 1;}, L3);

        static N1D integrand(BoundaryCondition::Decaying);
        integrand = one + (-1)*aveN*aveN;

        return IntegrateVertically(integrand, L3);
    }

    stratifloat JoverK() const
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

        return JoverK(u1_total, u2, u3, b_total);
    }

    stratifloat JoverK(const MField& u1_total,
                      const MField& u2_total,
                      const MField& u3_total,
                      const MField& b_total) const
    {
        // calculate length scale of the flow
        stratifloat I = this->I(u1_total);

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

        stratifloat J = IntegrateVertically(Jintegrand, L3);

        // work out average buoyancy gradient
        static M1D dbdz(BoundaryCondition::Decaying);
        dbdz = ddz(bAveModal);

        // construct integrand for K
        static N1D Kintegrand(BoundaryCondition::Decaying);
        dbdz.ToNodal(Kintegrand);
        Kintegrand = Kintegrand*varpi;

        stratifloat K = IntegrateVertically(Kintegrand, L3);

        // also calculate other things needed for DAL
        static N1D varpiDerivative(BoundaryCondition::Bounded);
        varpiDerivative.SetValue([I](stratifloat z){return 2*z*z/I/I/I;}, L3);

        Jintegrand = Jintegrand*varpiDerivative;
        stratifloat Jderivative = IntegrateVertically(Jintegrand);

        Kintegrand = Kintegrand*varpiDerivative;
        stratifloat Kderivative = IntegrateVertically(Kintegrand);

        stratifloat lambda = J*Kderivative/K/K - Jderivative/K; // quotient rule for -J/K

        return J/K;
    }


private:
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

        // CNSolve1D(RU_, u_, k);
        // CNSolve1D(RB_, b_, k);
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
        RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Decaying, b));

        // // for the 1D variables u_ and b_ (background flow) we only use vertical derivative matrix
        // RU_ = u_ + (0.5f*h[k]/Re)*MatMul1D(dim3Derivative2Bounded, u_);
        // RB_ = b_ + (0.5f*h[k]/Re)*MatMul1D(dim3Derivative2Bounded, b_);

        // now construct explicit terms
        r1.Zero();
        r2.Zero();
        r3 = Ri*b; // buoyancy force - z goes down
        rB.Zero();

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
        ndTemp = U1*B;
        ndTemp.ToModal(decayingTemp);
        rB -= ddx(decayingTemp);

        ndTemp = U2*B;
        ndTemp.ToModal(decayingTemp);
        rB -= ddy(decayingTemp);

        nnTemp = U3*B;
        nnTemp.ToModal(boundedTemp);
        rB -= ddz(boundedTemp);

        // advection term from background buoyancy
        db_dz = ddz(b_);
        db_dz.ToNodal(dB_dz);

        ndTemp = U3*dB_dz;
        ndTemp.ToModal(decayingTemp);
        rB -= decayingTemp;

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        R2 += (h[k]*beta[k])*r2;
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;
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

private:
    // these are the actual variables we care about
    MField u1, u2, u3, b;
    MField p;

    // background flow
    M1D u_, b_, db_dz;

    // parameters for the scheme
    const int s = 3;
    stratifloat h[3];
    const stratifloat beta[3] = {1.0f, 25.0f/8.0f, 9.0f/4.0f};
    const stratifloat zeta[3] = {0, -17.0f/8.0f, -5.0f/4.0f};

    // these are intermediate variables used in the computation, preallocated for efficiency
    MField& R1,& R2,& R3,& RB;
    M1D RU_, RB_;
    MField r1, r2, r3, rB;
    mutable NField U1, U2, U3, B;
    mutable N1D U_, B_, dB_dz;
    mutable NField ndTemp, nnTemp;
    mutable MField decayingTemp, boundedTemp;
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
};

int main()
{
    //std::cout << "Initializing fftw..." << std::endl;
    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    std::cout << "Setting ICs..." << std::endl;
    {
        IMEXRK::NField initialU1(BoundaryCondition::Bounded);
        IMEXRK::NField initialU2(BoundaryCondition::Bounded);
        IMEXRK::NField initialU3(BoundaryCondition::Decaying);
        IMEXRK::NField initialB(BoundaryCondition::Decaying);
        auto x3 = VerticalPoints(IMEXRK::L3, IMEXRK::N3);

        // nudge with something like the eigenmode
        initialU3.SetValue([](stratifloat x, stratifloat y, stratifloat z){return 0.1*cos(2*pi*x/16.0f)/cosh(z)/cosh(z);}, IMEXRK::L1, IMEXRK::L2, IMEXRK::L3);

        // add a perturbation to allow secondary instabilities to develop

        stratifloat bandmax = 4;
        for (int j=0; j<IMEXRK::N3; j++)
        {
            if (x3(j) > -bandmax && x3(j) < bandmax)
            {
                initialU1.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
                initialU2.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
                initialU3.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
            }
        }
        solver.SetInitial(initialU1, initialU2, initialU3, initialB);
        solver.RemoveDivergence(0.0f);
    }

    // add background flow
    std::cout << "Setting background..." << std::endl;
    {
        stratifloat R = 2;

        IMEXRK::N1D Ubar(BoundaryCondition::Bounded);
        IMEXRK::N1D Bbar(BoundaryCondition::Bounded);
        IMEXRK::N1D dBdz(BoundaryCondition::Decaying);
        Ubar.SetValue([](stratifloat z){return tanh(z);}, IMEXRK::L3);
        Bbar.SetValue([R](stratifloat z){return tanh(R*z);}, IMEXRK::L3);

        solver.SetBackground(Ubar, Bbar);
    }

    std::ofstream energyFile("energy.dat");

    stratifloat totalTime = 0.0f;
    stratifloat saveEvery = 1.0f;
    int lastFrame = -1;
    for (int step=0; step<500000; step++)
    {
        solver.TimeStep();
        totalTime += solver.deltaT;

        //solver.SaveFlow("snapshots/"+std::to_string(totalTime)+".fields");

        if(step%50==0)
        {
            stratifloat cfl = solver.CFL();
            std::cout << "Step " << step << ", time " << totalTime
                      << ", CFL number: " << cfl << std::endl;

            std::cout << "Average timings: " << solver.totalExplicit / (step+1)
                      << ", " << solver.totalImplicit / (step+1)
                      << ", " << solver.totalDivergence / (step+1)
                      << std::endl;
        }

        int frame = static_cast<int>(totalTime / saveEvery);

        if (frame>lastFrame)
        {
            lastFrame=frame;

            solver.PlotPressure("images/pressure/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
            solver.PlotBuoyancy("images/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
            solver.PlotVerticalVelocity("images/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
            solver.PlotSpanwiseVelocity("images/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
            solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);

            energyFile << totalTime
                       << " " << solver.KE()
                       << " " << solver.PE()
                       << " " << solver.JoverK()
                       << std::endl;
        }
    }

    f3_cleanup_threads();

    return 0;
}
