#include "Field.h"
#include "Differentiation.h"
#include "Integration.h"
#include "Graph.h"

#include <iostream>
#include <chrono>

// will become unnecessary with C++17
#define FullMatMul MatMul<Map<const Array<complex, -1, 1>, Aligned16>, float, complex, M1, N2, N3>

class IMEXRK
{
public:
    static constexpr int N1 = 256;
    static constexpr int N2 = 16;
    static constexpr int N3 = 128;

    static constexpr int M1 = N1/2 + 1;

    static constexpr float L1 = 32; // size of domain streamwise
    static constexpr float L2 = 4.0f;  // size of domain spanwise
    static constexpr float L3 = 4.0f; // vertical scaling factor

    float deltaT = 0.1;
    const float Re = 1000;
    const float Pe = 1000;
    const float Ri = 0.05;

    using NField = NodalField<N1,N2,N3>;
    using MField = ModalField<N1,N2,N3>;

    long totalExplicit = 0;
    long totalImplicit = 0;
    long totalDivergence = 0;

public:
    IMEXRK()
    : u1(BoundaryCondition::Neumann)
    , u2(BoundaryCondition::Neumann)
    , u3(BoundaryCondition::Dirichlet)
    , p(BoundaryCondition::Neumann)
    , b(BoundaryCondition::Dirichlet)

    , U_(BoundaryCondition::Neumann)
    , B_(BoundaryCondition::Neumann)
    , dB_dz(BoundaryCondition::Dirichlet)

    , R1(u1), R2(u2), R3(u3), RB(b)
    , r1(u1), r2(u2), r3(u3), rB(b)
    , U1(BoundaryCondition::Neumann)
    , U2(BoundaryCondition::Neumann)
    , U3(BoundaryCondition::Dirichlet)
    , B(BoundaryCondition::Dirichlet)
    , dirichletTemp(BoundaryCondition::Dirichlet)
    , neumannTemp(BoundaryCondition::Neumann)
    , ndTemp(BoundaryCondition::Dirichlet)
    , nnTemp(BoundaryCondition::Neumann)
    , mdProduct(BoundaryCondition::Dirichlet)
    , mnProduct(BoundaryCondition::Neumann)
    , divergence(mnProduct)
    , q(p)
    {
        dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1, 1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2, 2);
        dim3Derivative2Neumann = VerticalSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        dim3Derivative2Dirichlet = VerticalSecondDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<M1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                MatrixXf laplacian = dim3Derivative2Neumann;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixXf::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixXf::Identity(N3, N3);

                // prevent singularity - first row gives average value at infinity
                if (j1 == 0 && j2 == 0)
                {
                    laplacian.row(0).setConstant(1);

                    // because these terms are different in expansion
                    laplacian(0,0) = 2;
                    laplacian(0,N3-1) = 2;
                }

                solveLaplacian[j1*N2+j2].compute(laplacian);


                // for viscous terms
                explicitSolveDirichlet[j1*N2+j2] = dim3Derivative2Dirichlet;
                explicitSolveDirichlet[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXf::Identity(N3, N3);
                explicitSolveDirichlet[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXf::Identity(N3, N3);
                explicitSolveBuoyancy[j1*N2 + j2] = explicitSolveDirichlet[j1*N2+j2];
                explicitSolveBuoyancy[j1*N2+j2] /= Pe;
                explicitSolveDirichlet[j1*N2+j2] /= Re;

                explicitSolveNeumann[j1*N2+j2] = dim3Derivative2Neumann;
                explicitSolveNeumann[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXf::Identity(N3, N3);
                explicitSolveNeumann[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXf::Identity(N3, N3);
                explicitSolveNeumann[j1*N2+j2] /= Re;
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
        NodalSum(B, B_, nnTemp);
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
        HeatPlot(u2, L1, L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2) const
    {
        u1.ToNodal(U1);
        U1 += U_;
        U1.ToModal(neumannTemp);
        HeatPlot(neumannTemp, L1, L3, j2, filename);
    }

    void SetInitial(NField velocity1, NField velocity2, NField velocity3, NField buoyancy)
    {
        velocity1.ToModal(u1);
        velocity2.ToModal(u2);
        velocity3.ToModal(u3);
        buoyancy.ToModal(b);
    }

    void SetBackground(NField velocity, NField buoyancy, NField buoyancyDerivative)
    {
        U_ = velocity;
        B_ = buoyancy;
        dB_dz = buoyancyDerivative;
    }

    // gives an upper bound on cfl number - also updates if it's too high
    float CFL()
    {
        static ArrayXf z = VerticalPoints(L3, N3);
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);

        float delta1 = L1/N1;
        float delta2 = L2/N2;
        float delta3 = z(N3/2) - z(N3/2+1); // smallest gap in middle

        std::cout << delta3 << std::endl;

        float cfl = U1.Max()/delta1 + U2.Max()/delta2 + U3.Max()/delta3;
        cfl *= deltaT;

        constexpr float targetCFL = 0.4;
        if (cfl>targetCFL)
        {
            deltaT *= targetCFL / cfl;
            UpdateForTimestep();
        }

        return cfl;
    }

    void RemoveDivergence(float pressureMultiplier=1.0f)
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

    template<typename T>
    Dim3MatMul<T, float, complex, M1, N2, N3> ddz(const StackContainer<T, complex, M1, N2, N3>& f) const
    {
        static MatrixXf dim3DerivativeNeumann = VerticalDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        static MatrixXf dim3DerivativeDirichlet = VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        if (f.BC() == BoundaryCondition::Dirichlet)
        {
            return Dim3MatMul<T, float, complex, M1, N2, N3>(dim3DerivativeDirichlet, f, BoundaryCondition::Neumann);
        }
        else
        {
            return Dim3MatMul<T, float, complex, M1, N2, N3>(dim3DerivativeNeumann, f, BoundaryCondition::Dirichlet);
        }
    }

    void ImplicitUpdate(int k)
    {
        R1.Solve(implicitSolveNeumann[k], u1);
        R2.Solve(implicitSolveNeumann[k], u2);
        R3.Solve(implicitSolveDirichlet[k], u3);
        RB.Solve(implicitSolveBuoyancy[k], b);
    }

    void ExplicitUpdate(int k)
    {
        // build up right hand sides for the implicit solve in R

        //   old      last rk step         pressure         explicit CN
        R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + 0.5*h[k]*FullMatMul(explicitSolveNeumann, u1);
        R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + 0.5*h[k]*FullMatMul(explicitSolveNeumann, u2);
        R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + 0.5*h[k]*FullMatMul(explicitSolveDirichlet, u3);
        RB = b  + (h[k]*zeta[k])*rB                  + 0.5*h[k]*FullMatMul(explicitSolveDirichlet, b);

        // now construct explicit terms
        r1.Zero();
        r2.Zero();
        r3 = Ri*b; // buoyancy force - z goes down
        rB.Zero();

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        // take into account background shear for nonlinear terms
        U1 += U_;

        NodalProduct(U1, U1, nnTemp);
        nnTemp.ToModal(mnProduct);
        r1 -= ddx(mnProduct);

        NodalProduct(U1, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        r3 -= ddx(mdProduct);
        r1 -= ddz(mdProduct);

        NodalProduct(U2, U2, nnTemp);
        nnTemp.ToModal(mnProduct);
        r2 -= ddy(mnProduct);

        NodalProduct(U2, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        r3 -= ddy(mdProduct);
        r2 -= ddz(mdProduct);

        NodalProduct(U3, U3, nnTemp);
        nnTemp.ToModal(mnProduct);
        r3 -= ddz(mnProduct);

        NodalProduct(U1, U2, nnTemp);
        nnTemp.ToModal(mnProduct);
        r1 -= ddy(mnProduct);
        r2 -= ddx(mnProduct);

        // buoyancy nonlinear terms
        NodalProduct(U1, B, ndTemp);
        ndTemp.ToModal(mdProduct);
        rB -= ddx(mdProduct);

        NodalProduct(U2, B, ndTemp);
        ndTemp.ToModal(mdProduct);
        rB -= ddy(mdProduct);

        NodalProduct(U3, B, nnTemp);
        nnTemp.ToModal(mnProduct);
        rB -= ddz(mnProduct);

        // advection term from background buoyancy
        NodalProduct(U3, dB_dz, ndTemp);
        ndTemp.ToModal(dirichletTemp);
        rB -= dirichletTemp;

        // now add on explicit terms to RHS
        R1 += (h[k]*beta[k])*r1;
        R2 += (h[k]*beta[k])*r2;
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;
    }

    void UpdateForTimestep()
    {
        h[0] = deltaT*8.0f/15.0f;
        h[1] = deltaT*2.0f/15.0f;
        h[2] = deltaT*5.0f/15.0f;

        for (int j1=0; j1<M1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                for (int k=0; k<s; k++)
                {
                    implicitSolveDirichlet[k][j1*N2+j2].compute(
                        MatrixXf::Identity(N3, N3)-0.5*h[k]*explicitSolveDirichlet[j1*N2+j2]);
                    implicitSolveNeumann[k][j1*N2+j2].compute(
                        MatrixXf::Identity(N3, N3)-0.5*h[k]*explicitSolveNeumann[j1*N2+j2]);
                    implicitSolveBuoyancy[k][j1*N2+j2].compute(
                        MatrixXf::Identity(N3, N3)-0.5*h[k]*explicitSolveBuoyancy[j1*N2+j2]);
                }

            }
        }
    }

private:
    // these are the actual variables we care about
    MField u1, u2, u3, b;
    MField p;

    // background flow
    NField U_;
    NField B_; // used only for plotting
    NField dB_dz;

    // parameters for the scheme
    const int s = 3;
    float h[3];
    const float beta[3] = {1.0f, 25.0f/8.0f, 9.0f/4.0f};
    const float zeta[3] = {0, -17.0f/8.0f, -5.0f/4.0f};

    // these are intermediate variables used in the computation, preallocated for efficiency
    MField R1, R2, R3, RB;
    MField r1, r2, r3, rB;
    mutable NField U1, U2, U3, B;
    mutable NField ndTemp, nnTemp;
    mutable MField mdProduct, mnProduct;
    mutable MField dirichletTemp, neumannTemp;
    MField& divergence; // reference to share memory
    MField q;

    // these are precomputed matrices for performing and solving derivatives
    DiagonalMatrix<float, -1> dim1Derivative2;
    DiagonalMatrix<float, -1> dim2Derivative2;
    MatrixXf dim3Derivative2Neumann;
    MatrixXf dim3Derivative2Dirichlet;

    std::array<MatrixXf, M1*N2> explicitSolveDirichlet;
    std::array<MatrixXf, M1*N2> explicitSolveNeumann;
    std::array<MatrixXf, M1*N2> explicitSolveBuoyancy;
    std::array<PartialPivLU<MatrixXf>, M1*N2> implicitSolveNeumann[3];
    std::array<PartialPivLU<MatrixXf>, M1*N2> implicitSolveDirichlet[3];
    std::array<PartialPivLU<MatrixXf>, M1*N2> implicitSolveBuoyancy[3];
    std::array<PartialPivLU<MatrixXf>, M1*N2> solveLaplacian;
};

int main()
{
    fftwf_init_threads();
    fftwf_plan_with_nthreads(maxthreads);

    IMEXRK solver;

    IMEXRK::NField initialU1(BoundaryCondition::Neumann);
    IMEXRK::NField initialU2(BoundaryCondition::Neumann);
    IMEXRK::NField initialU3(BoundaryCondition::Dirichlet);
    IMEXRK::NField initialB(BoundaryCondition::Dirichlet);
    auto x3 = VerticalPoints(IMEXRK::L3, IMEXRK::N3);

    // nudge with something like the eigenmode
    initialU3.SetValue([](float x, float y, float z){return 0.1*cos(2*pi*x/16.0f)/cosh(z)/cosh(z);}, IMEXRK::L1, IMEXRK::L2, IMEXRK::L3);

    // add a perturbation to allow secondary instabilities to develop

    float bandmax = 4;
    for (int j=0; j<IMEXRK::N3; j++)
    {
        if (x3(j) > -bandmax && x3(j) < bandmax)
        {
            initialU1.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                * Array<float, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
            initialU2.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                * Array<float, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
            initialU3.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                * Array<float, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
        }
    }
    solver.SetInitial(initialU1, initialU2, initialU3, initialB);

    // add background flow
    float alpha = 1;

    IMEXRK::NField Ubar(BoundaryCondition::Neumann);
    IMEXRK::NField Bbar(BoundaryCondition::Neumann);
    IMEXRK::NField dBdz(BoundaryCondition::Dirichlet);
    Ubar.SetValue([](float z){return tanh(z);}, IMEXRK::L3);
    Bbar.SetValue([alpha](float z){return tanh(alpha*z);}, IMEXRK::L3);
    dBdz.SetValue([alpha](float z){return alpha/(cosh(alpha*z)*cosh(alpha*z));}, IMEXRK::L3);

    solver.SetBackground(Ubar, Bbar, dBdz);

    solver.RemoveDivergence(0.0f);

    float totalTime = 0.0f;
    float saveEvery = 1.0f;
    int lastFrame = -1;
    for (int step=0; step<500000; step++)
    {
        solver.TimeStep();
        totalTime += solver.deltaT;

        if(step%50==0)
        {
            float cfl = solver.CFL();
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
        }
    }

    fftwf_cleanup_threads();

    return 0;
}
