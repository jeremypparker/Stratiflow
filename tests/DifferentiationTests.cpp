#include <catch.h>

#include "matplotlib-cpp.h"

#include "Differentiation.h"
#include "Field.h"

#include <iostream>

TEST_CASE("Chebyshev derivative matrices")
{
    REQUIRE(MatrixXd(VerticalSecondDerivativeMatrix(1, 7)).isApprox(
        MatrixXd(VerticalDerivativeMatrix(BoundaryCondition::Neumann, 1, 7))*
        MatrixXd(VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, 1, 7))));

    REQUIRE(MatrixXd(VerticalSecondDerivativeMatrix(3.14, 11)).isApprox(
        MatrixXd(VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, 3.14, 11))*
        MatrixXd(VerticalDerivativeMatrix(BoundaryCondition::Neumann, 3.14, 11))));
}

TEST_CASE("Fourier derivative matrices")
{
    REQUIRE(MatrixXcd(FourierSecondDerivativeMatrix(5, 6)).isApprox(
        MatrixXcd(FourierDerivativeMatrix(5, 6))*MatrixXcd(FourierDerivativeMatrix(5, 6))));
}

TEST_CASE("Simple derivatives Neumann")
{
    double L = 16.0;

    constexpr int N1 = 1;
    constexpr int N2 = 1;
    constexpr int N3 = 1000;

    auto x = VerticalPoints(L, N3);

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2) = exp(-(x-0.5)*(x-0.5));
        }
    }

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Dirichlet);
    f2.Dim3MatMul(VerticalDerivativeMatrix(BoundaryCondition::Neumann, L, N3), f3);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Dirichlet);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2) = -2*(x-0.5)*exp(-(x-0.5)*(x-0.5));
        }
    }

    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim3MatMul(VerticalSecondDerivativeMatrix(L, N3), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2) = (4*(x-0.5)*(x-0.5)-2)*exp(-(x-0.5)*(x-0.5));
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Simple derivatives Dirichlet")
{
    constexpr int N1 = 2;
    constexpr int N2 = 2;
    constexpr int N3 = 60;
    double L = 14.0;

    auto x = VerticalPoints(L, N3);

    NodalField<N1,N2,N3> f1(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2) = exp(-(x-0.5)*(x-0.5));
        }
    }

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Dirichlet);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim3MatMul(VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L, N3), f3);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2) = -2*(x-0.5)*exp(-(x-0.5)*(x-0.5));
        }
    }

    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Dirichlet);
    f2.Dim3MatMul(VerticalSecondDerivativeMatrix(L, N3), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Dirichlet);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2) = (4*(x-0.5)*(x-0.5)-2)*exp(-(x-0.5)*(x-0.5));
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

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(cos(2*pi*j1/static_cast<double>(N1)));
        }
    }

    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim1MatMul(FourierDerivativeMatrix(L, N1), f3);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2).setConstant(-2*pi*sin(2*pi*j1/static_cast<double>(N1))/L);
        }
    }

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim1MatMul(FourierSecondDerivativeMatrix(L, N1), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
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

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(sin(2*pi*j2/static_cast<double>(N2)) + j1);
        }
    }

    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim2MatMul(FourierDerivativeMatrix(L, N2), f3);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2).setConstant(2*pi*cos(2*pi*j2/static_cast<double>(N2))/L);
        }
    }

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim2MatMul(FourierSecondDerivativeMatrix(L, N2), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
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
    constexpr int N1 = 20;
    constexpr int N2 = 22;
    constexpr int N3 = 40;

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
            MatrixXd laplacian = VerticalSecondDerivativeMatrix(L3, N3);

            // add terms for horizontal derivatives
            laplacian += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
            laplacian += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);

            if (j1==0 && j2==0)
            {
                // despite the fact we want neumann boundary conditions,
                // we need to impose a boundary value for non-singularity
                laplacian.row(0).setConstant(2);
                laplacian(0,0) = 1; // the form of DCT we are using has end coefficients different
                laplacian(0,N3-1) = 1;
            }

            solveLaplacian[j1*N2+j2].compute(laplacian);
        }
    }

    // create field in physical space
    NodalField<N1, N2, N3> physicalRHS(BoundaryCondition::Neumann);
    auto x = VerticalPoints(L3, N3);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            physicalRHS.stack(j1, j2) = (4*x*x - 2 -4.0*pi*pi/L1/L1)*
                                        exp(-x*x)*sin(2*pi*j1/static_cast<double>(N1));
        }
    }

    ModalField<N1, N2, N3> q(BoundaryCondition::Neumann);

    ModalField<N1, N2, N3> rhs(BoundaryCondition::Neumann);
    physicalRHS.ToModal(rhs);

    // for BC
    rhs(0,0,0) = 0;

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
            expectedSolution.stack(j1, j2)= exp(-x*x)*sin(2*pi*j1/static_cast<double>(N1));
        }
    }

    REQUIRE(physicalSolution == expectedSolution);
}