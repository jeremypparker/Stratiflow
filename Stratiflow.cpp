#include "Field.h"
#include "Differentiation.h"

#include <iostream>
#include <matplotlib-cpp.h>

class IMEXRK
{
public:
    static constexpr int N1 = 152;
    static constexpr int N2 = 1;
    static constexpr int N3 = 20;

    static constexpr double L1 = 14.0; // size of domain streamwise
    static constexpr double L2 = 3.5;  // size of domain spanwise
    static constexpr double L3 = 15.0; // half size of domain vertically

    const double deltaT = 0.001;
    const double nu = 0.001;

    using NField = NodalField<N1,N2,N3>;
    using MField = ModalField<N1,N2,N3>;

public:
    IMEXRK()
    : u1(BoundaryCondition::Neumann)
    , u2(BoundaryCondition::Neumann)
    , u3(BoundaryCondition::Dirichlet)
    , p(BoundaryCondition::Neumann)

    , y1(u1), y2(u2), y3(u3)
    , z1(u1), z2(u2), z3(u3)
    , v1(u1), v2(u2), v3(u3)
    , U1(BoundaryCondition::Neumann)
    , U2(BoundaryCondition::Neumann)
    , U3(BoundaryCondition::Dirichlet)
    , dirichletTemp(BoundaryCondition::Dirichlet)
    , neumannTemp(BoundaryCondition::Neumann)
    , ndProduct(BoundaryCondition::Dirichlet)
    , nnProduct(BoundaryCondition::Neumann)
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

        for (int k=0; k<s; k++)
        {
            implicitSolveDirichlet[k].compute(
                MatrixXd::Identity(N3, N3)-a_IM[k][k]*deltaT*dim3Derivative2Dirichlet);
            implicitSolveNeumann[k].compute(
                MatrixXd::Identity(N3, N3)-a_IM[k][k]*deltaT*dim3Derivative2Neumann);
        }

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                MatrixXd laplacian = ChebSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);

                // the laplacian as it is will be singular (with two zero rows)
                // we impose continuity and continuity of derivative at x3 = 0
                // to make it non-singular

                // for continuity (RHS = 0)
                ArrayXd one = ArrayXd::Ones(N3/2);
                laplacian.row(0) << one.transpose(), -one.transpose();

                //for continuous derivative (RHS = 0)
                ArrayXd k = ArrayXd::LinSpaced(N3/2, 0, N3/2-1);
                laplacian.row(N3-1) << (k.reverse()*k.reverse()).transpose(), (k*k).transpose();

                solveLaplacian[j1*N2+j2].compute(laplacian);
            }
        }
    }


    void TimeStep()
    {
        // see (19) in CB15
        for (int k=0; k<s; k++)
        {
            if (k==0)
            {
                y1 = u1;
                y2 = u2;
                y3 = u3;
            }
            else
            {
                double c_IM = deltaT * (a_IM[k][k-1] - b_IM[k-1]);
                double c_EX = deltaT * (a_EX[k][k-1] - b_EX[k-1]);

                y1 *= c_EX;
                y2 *= c_EX;
                y3 *= c_EX;

                y1 += c_IM*z1;
                y2 += c_IM*z2;
                y3 += c_IM*z3;

                y1 += u1;
                y2 += u2;
                y3 += u3;
            }

            ImplicitUpdate(k);
            ExplicitUpdate(k);

            double C_IM = deltaT * b_IM[k];
            double C_EX = deltaT * b_EX[k];

            u1 += C_IM*z1;
            u2 += C_IM*z2;
            u3 += C_IM*z3;

            u1 += C_EX*y1;
            u2 += C_EX*y2;
            u3 += C_EX*y3;

            //////// PRESSURE TERMS ////////
            p.Dim1MatMul(dim1Derivative, neumannTemp);
            u1 += (-1.0)*C_IM*neumannTemp;

            p.Dim2MatMul(dim2Derivative, neumannTemp);
            u2 += (-1.0)*C_IM*neumannTemp;

            p.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
            u3 += (-1.0)*C_IM*dirichletTemp;

            RemoveDivergence(k);
        }
    }

    void Quiver(std::string filename, int j2)
    {
        // plot some arrows
        unsigned int skip1 = 5;
        unsigned int skip3 = 5;

        matplotlibcpp::figure();

        // convert to physical space for this
        u1.ToNodal(U1);
        u3.ToNodal(U3);

        auto x = ChebPoints(N3, L3);

        for (unsigned int j1 = 0; j1 < N1; j1+=skip1)
        {
            for (unsigned int j3 = 0; j3 < N3; j3+=skip3)
            {
                double v1 = 0.2*U1.slice(j3)(j1, j2);
                double v3 = 0.2*U3.slice(j3)(j1, j2);

                double x1 = j1*L1/static_cast<double>(N1);
                double x3 = x(j3);

                matplotlibcpp::plot({x1, x1+v1}, {x3, x3+v3}, "b-");
            }
        }

        //matplotlibcpp::axis("equal");
        matplotlibcpp::save(filename);
        matplotlibcpp::close();
    }

    void SetVelocity(NField velocity1, NField velocity3)
    {
        velocity1.ToModal(u1);
        velocity3.ToModal(u3);
    }

private:
    void ImplicitUpdate(int k)
    {
        // vertical viscous terms

        // z = (I-a A)^-1 A y
        y1.Dim3MatMul(dim3Derivative2Neumann, v1);
        v1.Dim3Solve(implicitSolveNeumann[k], z1);
        y2.Dim3MatMul(dim3Derivative2Neumann, v2);
        v2.Dim3Solve(implicitSolveNeumann[k], z2);
        y3.Dim3MatMul(dim3Derivative2Dirichlet, v3);
        v3.Dim3Solve(implicitSolveDirichlet[k], z3);
    }

    void ExplicitUpdate(int k)
    {
        // explicit terms will be calculated from y + a Δt z as per CB15
        // this value is stored in v
        v1 = y1;
        v2 = y2;
        v3 = y3;

        double c = a_IM[k][k]*deltaT;
        v1 += c*z1;
        v2 += c*z2;
        v3 += c*z3;

        // calculate rhs terms and accumulate in y
        y1.Zero();
        y2.Zero();
        y3.Zero();

        //////// EXPLICIT VISCOUS TERMS ////////
        v1.Dim1MatMul(dim1Derivative2, neumannTemp);
        y1 += nu*neumannTemp;
        v1.Dim2MatMul(dim2Derivative2, neumannTemp);
        y1 += nu*neumannTemp;

        v2.Dim1MatMul(dim1Derivative2, neumannTemp);
        y2 += nu*neumannTemp;
        v2.Dim2MatMul(dim2Derivative2, neumannTemp);
        y2 += nu*neumannTemp;

        v3.Dim1MatMul(dim1Derivative2, dirichletTemp);
        y3 += nu*dirichletTemp;
        v3.Dim2MatMul(dim2Derivative2, dirichletTemp);
        y3 += nu*dirichletTemp;

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space
        v1.ToNodal(U1);
        v2.ToNodal(U2);
        v3.ToNodal(U3);

        NodalProduct(U1, U1, nnProduct);
        nnProduct.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        y1 += (-1.0)*neumannTemp;

        NodalProduct(U1, U2, nnProduct);
        nnProduct.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        y2 += (-1.0)*neumannTemp;
        mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        y1 += (-1.0)*neumannTemp;

        NodalProduct(U1, U3, ndProduct);
        ndProduct.ToModal(mdProduct);
        mdProduct.Dim1MatMul(dim1Derivative, dirichletTemp);
        y3 += (-1.0)*dirichletTemp;
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        y1 += (-1.0)*neumannTemp;

        NodalProduct(U2, U2, nnProduct);
        nnProduct.ToModal(mnProduct);
        mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        y2 += (-1.0)*neumannTemp;

        NodalProduct(U2, U3, ndProduct);
        ndProduct.ToModal(mdProduct);
        mdProduct.Dim2MatMul(dim2Derivative, dirichletTemp);
        y3 += (-1.0)*dirichletTemp;
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        y2 += (-1.0)*neumannTemp;

        NodalProduct(U3, U3, ndProduct);
        ndProduct.ToModal(mdProduct);
        mdProduct.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        y3 += (-1.0)*dirichletTemp;
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

        // set the RHS for the boundary condition solve
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
        u1 += (-1.0)*neumannTemp;
        q.Dim2MatMul(dim2Derivative, neumannTemp);
        u2 += (-1.0)*neumannTemp;
        q.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        u3 += (-1.0)*dirichletTemp;

        // also add it on to p for the next step
        // this is scaled to match the p that was added before
        // effectively we have forward euler
        p += (1/(deltaT * b_IM[k]))*q;
    }

private:
    // these are the actual variables we care about
    MField u1, u2, u3;
    MField p;

    // parameters for the scheme
    const int s = 4;
    const double a_IM[4][4] = {{0,      0,      0,      0},
                               {4/15.0, 4/15.0, 0,      0},
                               {4/15.0, 1/3.0,  1/15.0, 0},
                               {4/15.0, 1/3.0,  7/30.0, 1/6.0}
                              };
    const double b_IM[4]    = {4/15.0, 1/3.0, 7/30.0, 1/6.0};

    const double a_EX[4][4] = {{0,      0,      0,     0},
                               {8/15.0, 0,      0,     0},
                               {1/4.0,  5/12.0, 0,     0},
                               {1/4.0,  0,      3/4.0, 0}
                              };
    const double b_EX[4]    = {1/4.0, 0, 3/4.0, 0};

    // these are intermediate variables used in the computation, preallocated for efficiency
    MField y1, y2, y3;
    MField z1, z2, z3;
    MField v1, v2, v3;
    NField U1, U2, U3;
    NField ndProduct, nnProduct;
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

    std::array<ColPivHouseholderQR<MatrixXcd>, 4> implicitSolveDirichlet;
    std::array<ColPivHouseholderQR<MatrixXcd>, 4> implicitSolveNeumann;
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> solveLaplacian;
};

int main()
{
    IMEXRK solver;

    IMEXRK::NField initialU1(BoundaryCondition::Neumann);
    IMEXRK::NField initialU3(BoundaryCondition::Neumann);
    auto x3 = ChebPoints(IMEXRK::N3, IMEXRK::L3);
    for (int j=0; j<IMEXRK::N3; j++)
    {
        //initialU1.slice(j).setConstant(tanh(x3(j)));
        initialU1.slice(j) = exp(-x3(j)*x3(j));

        initialU1.slice(j) += 0.01*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
        initialU3.slice(j) += 0.01*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
    }
    solver.SetVelocity(initialU1, initialU3);

    solver.Quiver("initial.png", IMEXRK::N2/2);

    for (int step=0; step<10000; step++)
    {
        std::cout << "Step " << step << std::endl;
        solver.TimeStep();

        if(step%4==0)
        {
            solver.Quiver(std::to_string(step)+".png", IMEXRK::N2/2);
        }
    }

    return 0;
}