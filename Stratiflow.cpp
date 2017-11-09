#include "Field.h"
#include "Differentiation.h"
#include "Graph.h"

#include <iostream>

class IMEXRK
{
public:
    static constexpr int N1 = 160;
    static constexpr int N2 = 1;
    static constexpr int N3 = 41;

    static constexpr double L1 = 14.0; // size of domain streamwise
    static constexpr double L2 = 3.5;  // size of domain spanwise
    static constexpr double L3 = 15.0; // half size of domain vertically

    const double deltaT = 0.0001;
    const double Re = 2000;
    const double Pe = 1000;

    using NField = NodalField<N1,N2,N3>;
    using MField = ModalField<N1,N2,N3>;

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
        dim3DerivativeNeumann = ChebDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        dim3DerivativeDirichlet = ChebDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2);
        dim3Derivative2Neumann = ChebSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        dim3Derivative2Dirichlet = ChebSecondDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                MatrixXd laplacian = ChebSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);

                // despite the fact we want neumann boundary conditions,
                // we need to impose a boundary value for non-singularity

                // do it at both ends for symmetry
                laplacian.row(0).setZero();
                laplacian.row(N3-1).setZero();
                laplacian(0,0) = 1;
                laplacian(N3-1, N3-1) = 1;

                solveLaplacian[j1*N2+j2].compute(laplacian);


                // for viscous terms
                explicitSolveDirichlet[j1*N2+j2] = ChebSecondDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);
                explicitSolveDirichlet[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                explicitSolveDirichlet[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);
                explicitSolveDirichlet[j1*N2+j2] /= Re;

                explicitSolveNeumann[j1*N2+j2] = ChebSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
                explicitSolveNeumann[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                explicitSolveNeumann[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);
                explicitSolveBuoyancy[j1*N2 + j2] = explicitSolveNeumann[j1*N2+j2];
                explicitSolveNeumann[j1*N2+j2] /= Re;
                explicitSolveBuoyancy[j1*N2+j2] /= Pe;

                for (int k=0; k<s; k++)
                {
                    implicitSolveDirichlet[s*(j1*N2+j2) + k].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveDirichlet[j1*N2+j2]);
                    implicitSolveNeumann[s*(j1*N2+j2) + k].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveNeumann[j1*N2+j2]);
                    implicitSolveBuoyancy[s*(j1*N2+j2) + k].compute(
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
            ExplicitUpdate(k);
            ImplicitUpdate(k);
            RemoveDivergence(k);
        }
    }

    void Quiver(std::string filename, int j2)
    {
        u1.ToNodal(U1);
        u3.ToNodal(U3);

        QuiverPlot(U1, U3, L1, L3, j2, filename);
    }

    void Profile(std::string filename, int j1, int j2)
    {
        u1.ToNodal(U1);
        Interpolate(U1.stack(j1, j2), L3, BoundaryCondition::Neumann, filename);
    }

    void Buoyancy(std::string filename, int j2)
    {
        b.ToNodal(B);
        HeatPlot(B, L1, L3, j2, filename);
    }

    void SetVariables(NField velocity1, NField velocity3, NField buoyancy)
    {
        velocity1.ToModal(u1);
        velocity3.ToModal(u3);
        buoyancy.ToModal(b);
    }

    // gives an upper bound on cfl number
    double CFL()
    {
        ArrayXd x = GaussLobattoNodes(N3);

        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);

        double delta1 = L1/N1;
        double delta2 = L2/N2;
        double delta3 = x(1)-x(0); // minimal delta

        double cfl = U1.Max()/delta1 + U2.Max()/delta2 + U3.Max()/delta3;
        cfl *= deltaT;

        return cfl;
    }

private:
    void ImplicitUpdate(int k)
    {
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                R1.Dim3Solve(implicitSolveNeumann[s*(j1*N2+j2) + k], j1, j2, u1);
                R2.Dim3Solve(implicitSolveNeumann[s*(j1*N2+j2) + k], j1, j2, u2);
                R3.Dim3Solve(implicitSolveDirichlet[s*(j1*N2+j2) + k], j1, j2, u3);
                RB.Dim3Solve(implicitSolveBuoyancy[s*(j1*N2+j2) + k], j1, j2, b);
            }
        }
    }

    void ExplicitUpdate(int k)
    {
        // calculate rhs terms and accumulate in y
        R1 = u1;
        R2 = u2;
        R3 = u3;
        RB = b;

        // add term from last rk step
        R1 += (h[k]*zeta[k])*r1;
        R2 += (h[k]*zeta[k])*r2;
        R3 += (h[k]*zeta[k])*r3;
        RB += (h[k]*zeta[k])*rB;

        // explicit part of CN
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                u1.Dim3MatMul(explicitSolveNeumann[j1*N2+j2], j1, j2, neumannTemp);
            }
        }
        R1 += 0.5*h[k]*neumannTemp;

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                u2.Dim3MatMul(explicitSolveNeumann[j1*N2+j2], j1, j2, neumannTemp);
            }
        }
        R2 += 0.5*h[k]*neumannTemp;

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                u3.Dim3MatMul(explicitSolveDirichlet[j1*N2+j2], j1, j2, dirichletTemp);
            }
        }
        R3 += 0.5*h[k]*dirichletTemp;

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                b.Dim3MatMul(explicitSolveBuoyancy[j1*N2+j2], j1, j2, neumannTemp);
            }
        }
        RB += 0.5*h[k]*neumannTemp;

        // now construct explicit terms
        r1.Zero();
        r2.Zero();
        r3.Zero();
        rB.Zero();

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space
        u1.ToNodal(U1);
        u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        NodalProduct(U1, U1, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        r1 += (-1.0)*neumannTemp;

        NodalProduct(U1, U2, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        r2 += (-1.0)*neumannTemp;
        mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        r1 += (-1.0)*neumannTemp;

        NodalProduct(U1, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim1MatMul(dim1Derivative, dirichletTemp);
        r3 += (-1.0)*dirichletTemp;
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        r1 += (-1.0)*neumannTemp;

        NodalProduct(U2, U2, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        r2 += (-1.0)*neumannTemp;

        NodalProduct(U2, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim2MatMul(dim2Derivative, dirichletTemp);
        r3 += (-1.0)*dirichletTemp;
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        r2 += (-1.0)*neumannTemp;

        NodalProduct(U3, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        r3 += (-1.0)*dirichletTemp;

        // buoyancy nonlinear terms
        NodalProduct(U1, B, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        rB += (-1.0)*neumannTemp;

        NodalProduct(U2, B, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        rB += (-1.0)*neumannTemp;

        NodalProduct(U3, B, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        rB += (-1.0)*neumannTemp;

        // now add on explicit terms
        R1 += (h[k]*beta[k])*r1;
        R2 += (h[k]*beta[k])*r2;
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;

        // now add on pressure term
        p.Dim1MatMul(dim1Derivative, neumannTemp);
        R1 += (-h[k])*neumannTemp;

        p.Dim2MatMul(dim2Derivative, neumannTemp);
        R2 += (-h[k])*neumannTemp;

        p.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        R3 += (-h[k])*dirichletTemp;
    }

    void RemoveDivergence(int k)
    {
        // construct the diverence of u
        divergence.Zero();

        u1.Dim1MatMul(dim1Derivative, neumannTemp);
        divergence += neumannTemp;
        u2.Dim2MatMul(dim2Derivative, neumannTemp);
        divergence += neumannTemp;
        u3.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        divergence += neumannTemp;

        divergence *= 1/h[k];

        // boundary values
        divergence.slice(0).setZero();
        divergence.slice(N3-1).setZero();

        // solve Δq = ∇·u as linear system Aq = divergence
        q.Zero(); // probably not necessary
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                divergence.Dim3Solve(solveLaplacian[j1*N2+j2], j1, j2, q);
            }
        }

        // subtract the gradient of this from the velocity
        q.Dim1MatMul(dim1Derivative, neumannTemp);
        u1 += (-h[k])*neumannTemp;
        q.Dim2MatMul(dim2Derivative, neumannTemp);
        u2 += (-h[k])*neumannTemp;
        q.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        u3 += (-h[k])*dirichletTemp;

        // also add it on to p for the next step
        // this is scaled to match the p that was added before
        // effectively we have forward euler
        p += q;
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
    NField U1, U2, U3, B;
    NField ndTemp, nnTemp;
    MField mdProduct, mnProduct;
    MField dirichletTemp, neumannTemp;
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
    std::array<ColPivHouseholderQR<MatrixXcd>, 3*N1*N2> implicitSolveDirichlet;
    std::array<ColPivHouseholderQR<MatrixXcd>, 3*N1*N2> implicitSolveNeumann;
    std::array<ColPivHouseholderQR<MatrixXcd>, 3*N1*N2> implicitSolveBuoyancy;
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> solveLaplacian;
};

int main()
{
    IMEXRK solver;

    IMEXRK::NField initialU1(BoundaryCondition::Neumann);
    IMEXRK::NField initialU3(BoundaryCondition::Dirichlet);
    IMEXRK::NField initialB(BoundaryCondition::Neumann);
    auto x3 = ChebPoints(IMEXRK::N3, IMEXRK::L3);
    for (int j=0; j<IMEXRK::N3; j++)
    {
        initialU1.slice(j).setConstant(tanh(x3(j)-2));
        //initialU1.slice(j) = exp(-x3(j)*x3(j));

        if (x3(j)-2>0)
        {
            initialB.slice(j).setConstant(-1);
        }
        if (x3(j)-2<0)
        {
            initialB.slice(j).setConstant(1);
        }

        // band to see fluid velocity
        // initialB.slice(j)(0, IMEXRK::N2/2) = 0;
        // initialB.slice(j)(1, IMEXRK::N2/2) = 0;
        // initialB.slice(j)(2, IMEXRK::N2/2) = 0;
        // initialB.slice(j)(3, IMEXRK::N2/2) = 0;
        // initialB.slice(j)(4, IMEXRK::N2/2) = 0;

        if (j!=0 && j!=IMEXRK::N3-1 && j!= IMEXRK::N3/2)
        {
            initialU1.slice(j) += 0.1*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
            initialU3.slice(j) += 0.1*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
        }
    }
    solver.SetVariables(initialU1, initialU3, initialB);

    //solver.Quiver("images/initial.png", IMEXRK::N2/2);
    //solver.Profile("images/profile.png", 0, 0);

    for (int step=0; step<500000; step++)
    {
        solver.TimeStep();

        if(step%200==0)
        {
            //solver.Quiver("images/"+std::to_string(step)+".png", IMEXRK::N2/2);
            //solver.Profile("images/"+std::to_string(step)+"profile.png", 0, 0);
            solver.Buoyancy("images/"+std::to_string(step)+"buoyancy.png", IMEXRK::N2/2);

            std::cout << "Step " << step << ", time " << step*solver.deltaT
                      << ", CFL number: " << solver.CFL() << std::endl;
        }
    }

    return 0;
}