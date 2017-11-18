#include "Field.h"
#include "Differentiation.h"
#include "Graph.h"

#include <iostream>
#include <chrono>

class IMEXRK
{
public:
    static constexpr int N1 = 100;
    static constexpr int N2 = 1;
    static constexpr int N3 = 61;

    static constexpr double L1 = 9.44; // size of domain streamwise
    static constexpr double L2 = 15.0;  // size of domain spanwise
    static constexpr double L3 = 2.0; // vertical scaling factor

    const double deltaT = 0.001;
    const double Re = 2000;
    const double Pe = 1000;
    const double Ri = 0.2;

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
    , b(BoundaryCondition::Neumann)

    , R1(u1), R2(u2), R3(u3), RB(b)
    , r1(u1), r2(u2), r3(u3), rB(b)
    , U1(BoundaryCondition::Neumann)
    , U2(BoundaryCondition::Neumann)
    , U3(BoundaryCondition::Dirichlet)
    , B(BoundaryCondition::Neumann)
    , dirichletTemp(BoundaryCondition::Dirichlet)
    , neumannTemp(BoundaryCondition::Neumann)
    , ndTemp(BoundaryCondition::Dirichlet)
    , nnTemp(BoundaryCondition::Neumann)
    , mdProduct(BoundaryCondition::Dirichlet)
    , mnProduct(BoundaryCondition::Neumann)
    , divergence(mnProduct)
    , q(p)
    {
        dim1Derivative = FourierDerivativeMatrix(L1, N1);
        dim2Derivative = FourierDerivativeMatrix(L2, N2);
        dim3DerivativeNeumann = VerticalDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        dim3DerivativeDirichlet = VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2);
        dim3Derivative2Neumann = VerticalSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        dim3Derivative2Dirichlet = VerticalSecondDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                MatrixXd laplacian = dim3Derivative2Neumann;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);

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
                explicitSolveDirichlet[j1*N2+j2] = dim3Derivative2Neumann;
                explicitSolveDirichlet[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                explicitSolveDirichlet[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);
                explicitSolveDirichlet[j1*N2+j2] /= Re;

                explicitSolveNeumann[j1*N2+j2] = dim3Derivative2Neumann;
                explicitSolveNeumann[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                explicitSolveNeumann[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);
                explicitSolveBuoyancy[j1*N2 + j2] = explicitSolveNeumann[j1*N2+j2];
                explicitSolveNeumann[j1*N2+j2] /= Re;
                explicitSolveBuoyancy[j1*N2+j2] /= Pe;

                for (int k=0; k<s; k++)
                {
                    implicitSolveDirichlet[k][j1*N2+j2].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveDirichlet[j1*N2+j2]);
                    implicitSolveNeumann[k][j1*N2+j2].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveNeumann[j1*N2+j2]);
                    implicitSolveBuoyancy[k][j1*N2+j2].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveBuoyancy[j1*N2+j2]);
                }
            }
        }
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
    }

    void PlotBuoyancy(std::string filename, int j2) const
    {
        HeatPlot(b, L1, L3, j2, filename);
    }

    void PlotPressure(std::string filename, int j2) const
    {
        HeatPlot(p, L1, L3, j2, filename);
    }

    void PlotVerticalVelocity(std::string filename, int j2) const
    {
        HeatPlot(u3, L1, L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2) const
    {
        HeatPlot(u1, L1, L3, j2, filename);
    }

    void AddVariables(NField velocity1, NField velocity3, NField buoyancy)
    {
        u1.ToNodal(U1);
        u3.ToNodal(U3);
        b.ToNodal(B);

        U1 += velocity1;
        U3 += velocity3;
        B += buoyancy;

        U1.ToModal(u1);
        U3.ToModal(u3);
        B.ToModal(b);
    }

    // gives an upper bound on cfl number
    double CFL()
    {
        static ArrayXd z = VerticalPoints(L3, N3);
        u1.ToNodal(U1);
        //u2.ToNodal(U2);
        u3.ToNodal(U3);

        double delta1 = L1/N1;
        double delta2 = L2/N2;
        double delta3 = z(N3/2) - z(N3/2+1);

        std::cout << "delta1 = "<<delta1<<std::endl;
        std::cout << "delta3 = "<<delta3<<std::endl;

        double cfl = U1.Max()/delta1 /*+ U2.Max()/delta2*/ + U3.Max()/delta3;
        cfl *= deltaT;

        return cfl;
    }

    void RemoveDivergence(double pressureMultiplier=1.0)
    {
        // construct the diverence of u
        divergence.Zero();

        u1.Dim1MatMul(dim1Derivative, neumannTemp);
        divergence += neumannTemp;
        //u2.Dim2MatMul(dim2Derivative, neumannTemp);
        //divergence += neumannTemp;
        u3.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        divergence += neumannTemp;

        //divergence.ToNodal(B);
        //HeatPlot(B, L1, L3, 0, "images/divergence.png");

        // constant term - set value at infinity to zero
        divergence(0,0,0) = 0;

        // solve Δq = ∇·u as linear system Aq = divergence
        divergence.Dim3Solve(solveLaplacian, q);

        //q.ToNodal(B);
        //HeatPlot(B, L1, L3, 0, "images/pressureupdate.png");

        //std::cout << q.stack(5, 0) << std::endl << std::endl;

        // subtract the gradient of this from the velocity
        q.Dim1MatMul(dim1Derivative, neumannTemp);
        u1 -= neumannTemp;
        //q.Dim2MatMul(dim2Derivative, neumannTemp);
        //u2 -= neumannTemp;
        q.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        u3 -= dirichletTemp;
        //std::cout << dirichletTemp.stack(5, 0) << std::endl << std::endl;

        // also add it on to p for the next step
        // this is scaled to match the p that was added before
        // effectively we have forward euler
        p += pressureMultiplier*q;

        // pressure never gets filtered unless we do it explicitly
        p.Filter();
    }

    void SolveForPressure()
    {
        // solve Δp = -∇u∇u
        // (store rhs in divergence)
        divergence.Zero();


        // todo: cleanup use of temp variables
        u1.Dim1MatMul(dim1Derivative, neumannTemp);
        neumannTemp.ToNodal(U1);
        NodalProduct(U1, U1, nnTemp);
        nnTemp.ToModal(mnProduct);
        divergence -= mnProduct;

        u3.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        neumannTemp.ToNodal(U2);
        NodalProduct(U2, U2, nnTemp);
        nnTemp.ToModal(mnProduct);
        divergence -= mnProduct;

        u1.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        dirichletTemp.ToNodal(ndTemp);
        u3.Dim1MatMul(dim1Derivative, dirichletTemp);
        dirichletTemp.ToNodal(U3);
        NodalProduct(ndTemp, U3, nnTemp);
        nnTemp.ToModal(mnProduct);
        divergence += (-2)*mnProduct;

        // set value at infinity
        divergence(0,0,0) = 0;

        divergence.Dim3Solve(solveLaplacian, p);
    }


private:
    void ImplicitUpdate(int k)
    {
        R1.Dim3Solve(implicitSolveNeumann[k], u1);
        //R2.Dim3Solve(implicitSolveNeumann[k], u2);
        R3.Dim3Solve(implicitSolveDirichlet[k], u3);
        RB.Dim3Solve(implicitSolveBuoyancy[k], b);
    }

    void ExplicitUpdate(int k)
    {
        // calculate rhs terms and accumulate in y
        R1 = u1;
        //R2 = u2;
        R3 = u3;
        RB = b;

        // add term from last rk step
        R1 += (h[k]*zeta[k])*r1;
        //R2 += (h[k]*zeta[k])*r2;
        R3 += (h[k]*zeta[k])*r3;
        RB += (h[k]*zeta[k])*rB;

        // explicit part of CN

        u1.Dim3MatMul(explicitSolveNeumann, neumannTemp);
        R1 += 0.5*h[k]*neumannTemp;

        //  u2.Dim3MatMul(explicitSolveNeumann, neumannTemp);
        // R2 += 0.5*h[k]*neumannTemp;

        u3.Dim3MatMul(explicitSolveDirichlet, dirichletTemp);
        R3 += 0.5*h[k]*dirichletTemp;

        b.Dim3MatMul(explicitSolveBuoyancy, neumannTemp);
        RB += 0.5*h[k]*neumannTemp;

        // now construct explicit terms
        r1.Zero();
        //r2.Zero();
        r3.Zero();
        rB.Zero();

        // add buoyancy force
        //r3 += Ri*b; // z goes down

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space
        u1.ToNodal(U1);
        //u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        // ThreadPool::Get().ExecuteAsync([this](){u1.ToNodal(U1);});
        // //ThreadPool::Get().ExecuteAsync([](){u2.ToNodal(U2);});
        // ThreadPool::Get().ExecuteAsync([this](){u3.ToNodal(U3);});
        // ThreadPool::Get().ExecuteAsync([this](){b.ToNodal(B);});
        // ThreadPool::Get().WaitAll();

        NodalProduct(U1, U1, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        r1 -= neumannTemp;

        // NodalProduct(U1, U2, nnTemp);
        // nnTemp.ToModal(mnProduct);
        // mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        // r2 -= neumannTemp;
        // mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        // r1 -= neumannTemp;

        NodalProduct(U1, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim1MatMul(dim1Derivative, dirichletTemp);
        r3 -= dirichletTemp;
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        r1 -= neumannTemp;

        // NodalProduct(U2, U2, nnTemp);
        // nnTemp.ToModal(mnProduct);
        // mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        // r2 -= neumannTemp;

        // NodalProduct(U2, U3, ndTemp);
        // ndTemp.ToModal(mdProduct);
        // mdProduct.Dim2MatMul(dim2Derivative, dirichletTemp);
        // r3 -= dirichletTemp;
        // mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        // r2 -= neumannTemp;

        NodalProduct(U3, U3, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        r3 -= dirichletTemp;

        // buoyancy nonlinear terms
        NodalProduct(U1, B, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        rB -= neumannTemp;

        // NodalProduct(U2, B, nnTemp);
        // nnTemp.ToModal(mnProduct);
        // mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        // rB -= neumannTemp;

        NodalProduct(U3, B, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        rB -= neumannTemp;

        // now add on explicit terms
        R1 += (h[k]*beta[k])*r1;
        // R2 += (h[k]*beta[k])*r2;
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;

        // now add on pressure term
        p.Dim1MatMul(dim1Derivative, neumannTemp);
        R1 += (-h[k])*neumannTemp;

        // p.Dim2MatMul(dim2Derivative, neumannTemp);
        // R2 += (-h[k])*neumannTemp;

        p.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        R3 += (-h[k])*dirichletTemp;
    }

private:
    // these are the actual variables we care about
    MField u1, u2, u3, b;
    MField p;

    // parameters for the scheme
    const int s = 3;
    const double h[3] = {deltaT*8.0/15.0, deltaT*2.0/15.0, deltaT*5.0/15.0};
    const double beta[3] = {1.0, 25.0/8.0, 9.0/4.0};
    const double zeta[3] = {0, -17.0/8.0, -5.0/4.0};

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
    DiagonalMatrix<complex, -1> dim1Derivative;
    DiagonalMatrix<complex, -1> dim2Derivative;
    MatrixXd dim3DerivativeNeumann;
    MatrixXd dim3DerivativeDirichlet;

    DiagonalMatrix<double, -1> dim1Derivative2;
    DiagonalMatrix<double, -1> dim2Derivative2;
    MatrixXd dim3Derivative2Neumann;
    MatrixXd dim3Derivative2Dirichlet;

    std::array<MatrixXcd, N1*N2> explicitSolveDirichlet;
    std::array<MatrixXcd, N1*N2> explicitSolveNeumann;
    std::array<MatrixXcd, N1*N2> explicitSolveBuoyancy;
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> implicitSolveNeumann[3];
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> implicitSolveDirichlet[3];
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> implicitSolveBuoyancy[3];
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> solveLaplacian;
};

int main()
{
    IMEXRK solver;

    IMEXRK::NField initialU1(BoundaryCondition::Neumann);
    IMEXRK::NField initialU3(BoundaryCondition::Dirichlet);
    IMEXRK::NField initialB(BoundaryCondition::Neumann);
    auto x3 = VerticalPoints(IMEXRK::L3, IMEXRK::N3);

    std::cout << x3 << std::endl;

    double interfaceoffset = 0.0;

    for (int j=0; j<IMEXRK::N3; j++)
    {
        if (j>2*IMEXRK::N3/5 && j<3*IMEXRK::N3/5)
        {
            initialU1.slice(j) += 0.1*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
            initialU3.slice(j) += 0.01*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
        }
    }
    solver.AddVariables(initialU1, initialU3, initialB);

    // add background flow
    initialU1.Zero();
    initialU3.Zero();
    initialB.Zero();
    for (int j=0; j<IMEXRK::N3; j++)
    {
        initialU1.slice(j).setConstant(tanh(x3(j)-interfaceoffset));
        initialB.slice(j).setConstant(tanh(3*(x3(j)-interfaceoffset)));
    }

    solver.AddVariables(initialU1, initialU3, initialB);

    solver.RemoveDivergence(0.0);
    //solver.SolveForPressure();

    //solver.PlotPressure("images/pressure/initial.png", IMEXRK::N2/2);

    for (int step=0; step<50000; step++)
    {
        std::cout << "Step " << step << std::endl;
        solver.TimeStep();

        if(step%400==0)
        {
            //solver.Quiver("images/"+std::to_string(step)+".png", IMEXRK::N2/2);
            //solver.Profile("images/"+std::to_string(step)+"profile.png", 0, 0);
            solver.PlotPressure("images/pressure/"+std::to_string(step)+".png", IMEXRK::N2/2);
            solver.PlotBuoyancy("images/buoyancy/"+std::to_string(step)+".png", IMEXRK::N2/2);
            solver.PlotVerticalVelocity("images/u3/"+std::to_string(step)+".png", IMEXRK::N2/2);
            solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(step)+".png", IMEXRK::N2/2);

            double cfl = solver.CFL();
            std::cout << "Step " << step << ", time " << step*solver.deltaT
                      << ", CFL number: " << cfl << std::endl;

            std::cout << "Average timings: " << solver.totalExplicit / (step+1)
                      << ", " << solver.totalImplicit / (step+1)
                      << ", " << solver.totalDivergence / (step+1)
                      << std::endl;
        }
    }

    return 0;
}