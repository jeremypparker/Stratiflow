#include <catch.h>

#include "matplotlib-cpp.h"

#include "Differentiation.h"
#include "Field.h"

#include <iostream>

TEST_CASE("Chebyshev derivative matrices")
{
    REQUIRE(ChebSecondDerivativeMatrix(BoundaryCondition::Dirichlet, 1, 7).isApprox(
        ChebDerivativeMatrix(BoundaryCondition::Neumann, 1, 7)*
        ChebDerivativeMatrix(BoundaryCondition::Dirichlet, 1, 7)));

    REQUIRE(ChebSecondDerivativeMatrix(BoundaryCondition::Neumann, 3.14, 11).isApprox(
        ChebDerivativeMatrix(BoundaryCondition::Dirichlet, 3.14, 11)*
        ChebDerivativeMatrix(BoundaryCondition::Neumann, 3.14, 11)));
}

TEST_CASE("Fourier derivative matrices")
{
    REQUIRE(MatrixXcd(FourierSecondDerivativeMatrix(5, 6)).isApprox(
        MatrixXcd(FourierDerivativeMatrix(5, 6))*MatrixXcd(FourierDerivativeMatrix(5, 6))));
}

TEST_CASE("Simple derivatives Neumann")
{
    double L = 16.0;

    constexpr int N1 = 15;
    constexpr int N2 = 20;
    constexpr int N3 = 41;

    auto x = ChebPoints(N3, L);

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2) = tanh(x);
        }
    }

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Dirichlet);
    f2.Dim3MatMul(ChebDerivativeMatrix(BoundaryCondition::Neumann, L, N3), f3);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Dirichlet);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2) = 1/(cosh(x)*cosh(x));
        }
    }

    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim3MatMul(ChebSecondDerivativeMatrix(BoundaryCondition::Neumann, L, N3), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2) = -2*tanh(x)/(cosh(x)*cosh(x));
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Simple derivatives Dirichlet")
{
    constexpr int N1 = 2;
    constexpr int N2 = 2;
    constexpr int N3 = 31;
    double L = 14.0;

    auto x = ChebPoints(N3, L);

    NodalField<N1,N2,N3> f1(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2) = exp(-x*x);
        }
    }

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Dirichlet);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim3MatMul(ChebDerivativeMatrix(BoundaryCondition::Dirichlet, L, N3), f3);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2) = -2*x*exp(-x*x);
        }
    }

    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Dirichlet);
    f2.Dim3MatMul(ChebSecondDerivativeMatrix(BoundaryCondition::Dirichlet, L, N3), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Dirichlet);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2) = (4*x*x-2)*exp(-x*x);
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 1 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 2;
    constexpr int N3 = 5;
    double L = 14.0;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(cos(2*pi*j1/static_cast<double>(N1)));
        }
    }

    ModalField<N1,N2,N3> f2(BoundaryCondition::Dirichlet);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Dirichlet);
    f2.Dim1MatMul(FourierDerivativeMatrix(L, N1), f3);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Dirichlet);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2).setConstant(-2*pi*sin(2*pi*j1/static_cast<double>(N1))/L);
        }
    }

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Dirichlet);
    f2.Dim1MatMul(FourierSecondDerivativeMatrix(L, N1), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Dirichlet);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2).setConstant(-4*pi*pi*cos(2*pi*j1/static_cast<double>(N1))/L/L);
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 2 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 10;
    constexpr int N3 = 5;
    double L = 14.0;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(sin(2*pi*j2/static_cast<double>(N2)) + j1);
        }
    }

    ModalField<N1,N2,N3> f2(BoundaryCondition::Dirichlet);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Dirichlet);
    f2.Dim2MatMul(FourierDerivativeMatrix(L, N2), f3);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Dirichlet);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2).setConstant(2*pi*cos(2*pi*j2/static_cast<double>(N2))/L);
        }
    }

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Dirichlet);
    f2.Dim2MatMul(FourierSecondDerivativeMatrix(L, N2), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Dirichlet);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2).setConstant(-4*pi*pi*sin(2*pi*j2/static_cast<double>(N2))/L/L);
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Inverse Laplacian")
{
    constexpr int N1 = 1;
    constexpr int N2 = 1;
    constexpr int N3 = 21;

    constexpr double L1 = 14.0;
    constexpr double L2 = 3.5;
    constexpr double L3 = 15.0;

    auto dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1);
    auto dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2);

    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> solveLaplacian;

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

            std::cout << laplacian << std::endl << std::endl;

            solveLaplacian[j1*N2+j2].compute(laplacian);
        }
    }

    // create field in physical space
    NodalField<N1, N2, N3> physicalRHS(BoundaryCondition::Neumann);
    auto x = ChebPoints(N3, L3);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            physicalRHS.stack(j1, j2) = (4*x*x - 2/* -4.0*pi*pi/L1/L1*/)*
                                        exp(-x*x);//*sin(2*pi*j1/static_cast<double>(N1));
        }
    }

    ModalField<N1, N2, N3> q(BoundaryCondition::Neumann);

    ModalField<N1, N2, N3> rhs(BoundaryCondition::Neumann);
    physicalRHS.ToModal(rhs);

    rhs.slice(0).setZero();
    rhs.slice(N3-1).setZero();

    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            rhs.Dim3Solve(solveLaplacian[j1*N2+j2], j1, j2, q);
        }
    }

    NodalField<N1, N2, N3> physicalSolution(BoundaryCondition::Neumann);
    q.ToNodal(physicalSolution);

    NodalField<N1, N2, N3> expectedSolution(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expectedSolution.stack(j1, j2)= exp(-x*x);//*sin(2*pi*j1/static_cast<double>(N1));
        }
    }

    std::cout << expectedSolution.stack(0,0) << std::endl << std::endl;

    std::cout << physicalSolution.stack(0,0) << std::endl << std::endl;

    REQUIRE(physicalSolution == expectedSolution);
}

TEST_CASE("GuassLobatto nodes and weights")
{
    {
        ArrayXd x = GaussLobattoNodes(2);
        ArrayXd w = GaussLobattoWeights(x);

        REQUIRE(x(0) == Approx(-1));
        REQUIRE(x(1) == Approx(0));
        REQUIRE(x(2) == Approx(1));

        REQUIRE(w(0) == Approx(1.0/3));
        REQUIRE(w(1) == Approx(4.0/3));
        REQUIRE(w(2) == Approx(1.0/3));
    }

    {
        ArrayXd x = GaussLobattoNodes(4);
        ArrayXd w = GaussLobattoWeights(x);

        REQUIRE(x(0) == Approx(-1));
        REQUIRE(x(1) == Approx(-sqrt(21)/7));
        REQUIRE(x(2) == Approx(0));
        REQUIRE(x(3) == Approx(sqrt(21)/7));
        REQUIRE(x(4) == Approx(1));

        REQUIRE(w(0) == Approx(0.1));
        REQUIRE(w(1) == Approx(0.5444444444444444444));
        REQUIRE(w(2) == Approx(0.7111111111111111111));
        REQUIRE(w(3) == Approx(0.5444444444444444444));
        REQUIRE(w(4) == Approx(0.1));
    }
}