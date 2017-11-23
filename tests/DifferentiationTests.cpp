#include <catch.h>

#include "matplotlib-cpp.h"

#include "Differentiation.h"
#include "Field.h"

#include <iostream>

TEST_CASE("Chebyshev derivative matrices")
{
    REQUIRE(MatrixXf(VerticalSecondDerivativeMatrix(BoundaryCondition::Neumann, 1, 7)).isApprox(
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, 1, 7))*
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Neumann, 1, 7))));

    REQUIRE(MatrixXf(VerticalSecondDerivativeMatrix(BoundaryCondition::Dirichlet, 3.14, 11)).isApprox(
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Neumann, 3.14, 11))*
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, 3.14, 11))));
}

TEST_CASE("Fourier derivative matrices")
{
    REQUIRE(MatrixXcf(FourierSecondDerivativeMatrix(5, 6, 1)).isApprox(
        MatrixXcf(FourierDerivativeMatrix(5, 6, 1))*MatrixXcf(FourierDerivativeMatrix(5, 6, 1))));

    REQUIRE(MatrixXcf(FourierSecondDerivativeMatrix(5, 6, 2)).isApprox(
        MatrixXcf(FourierDerivativeMatrix(5, 6, 2))*MatrixXcf(FourierDerivativeMatrix(5, 6, 2))));
}

TEST_CASE("Simple derivatives Neumann")
{
    float L = 5.0f;

    constexpr int N1 = 1;
    constexpr int N2 = 1;
    constexpr int N3 = 61;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    f1.SetValue([](float z){return tanh(z);}, L);

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Dirichlet);
    f2.Dim3MatMul(VerticalDerivativeMatrix(BoundaryCondition::Neumann, L, N3), f3);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Dirichlet);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Dirichlet);
    expected.SetValue([](float z){return 1/(cosh(z)*cosh(z));}, L);

    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim3MatMul(VerticalSecondDerivativeMatrix(BoundaryCondition::Neumann, L, N3), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
    expected2.SetValue([](float z){return -2*tanh(z)/(cosh(z)*cosh(z));}, L);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Simple derivatives Dirichlet")
{
    constexpr int N1 = 2;
    constexpr int N2 = 2;
    constexpr int N3 = 61;
    float L = 5.0f;

    auto x = VerticalPoints(L, N3);

    NodalField<N1,N2,N3> f1(BoundaryCondition::Dirichlet);
    f1.SetValue([](float z){return exp(-(z-0.5)*(z-0.5));}, L);

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Dirichlet);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim3MatMul(VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L, N3), f3);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    expected.SetValue([](float z){return -2*(z-0.5)*exp(-(z-0.5)*(z-0.5));}, L);


    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Dirichlet);
    f2.Dim3MatMul(VerticalSecondDerivativeMatrix(BoundaryCondition::Dirichlet, L, N3), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Dirichlet);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Dirichlet);
    expected2.SetValue([](float z){return (4*(z-0.5)*(z-0.5)-2)*exp(-(z-0.5)*(z-0.5));}, L);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 1 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 2;
    constexpr int N3 = 5;
    float L = 14.0f;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(cos(2*pi*j1/static_cast<float>(N1)));
        }
    }

    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim1MatMul(FourierDerivativeMatrix(L, N1, 1), f3);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2).setConstant(-2*pi*sin(2*pi*j1/static_cast<float>(N1))/L);
        }
    }

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim1MatMul(FourierSecondDerivativeMatrix(L, N1, 1), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2).setConstant(-4*pi*pi*cos(2*pi*j1/static_cast<float>(N1))/L/L);
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 2 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 10;
    constexpr int N3 = 5;
    float L = 14.0f;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(sin(2*pi*j2/static_cast<float>(N2)) + j1);
        }
    }

    ModalField<N1,N2,N3> f2(BoundaryCondition::Neumann);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Neumann);
    f2.Dim2MatMul(FourierDerivativeMatrix(L, N2, 2), f3);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Neumann);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected.stack(j1, j2).setConstant(2*pi*cos(2*pi*j2/static_cast<float>(N2))/L);
        }
    }

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Neumann);
    f2.Dim2MatMul(FourierSecondDerivativeMatrix(L, N2, 2), f5);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Neumann);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expected2.stack(j1, j2).setConstant(-4*pi*pi*sin(2*pi*j2/static_cast<float>(N2))/L/L);
        }
    }

    REQUIRE(f6 == expected2);
}

TEST_CASE("Inverse Laplacian")
{
    constexpr int N1 = 20;
    constexpr int N2 = 22;
    constexpr int N3 = 41;

    constexpr float L1 = 14.0f;
    constexpr float L2 = 3.5;
    constexpr float L3 = 3.0f;

    auto dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1, 1);
    auto dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2, 2);

    std::array<ColPivHouseholderQR<MatrixXcf>, (N1/2 + 1)*N2> solveLaplacian;

    // we solve each vetical line separately, so N1*N2 total solves
    for (int j1=0; j1<N1/2 + 1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            MatrixXf laplacian = VerticalSecondDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);

            // add terms for horizontal derivatives
            laplacian += dim1Derivative2.diagonal()(j1)*MatrixXf::Identity(N3, N3);
            laplacian += dim2Derivative2.diagonal()(j2)*MatrixXf::Identity(N3, N3);


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
            physicalRHS.stack(j1, j2) = (4*(x+2)*(x+2) - 2 -4.0f*pi*pi/L1/L1)*
                                        exp(-(x+2)*(x+2))*sin(2*pi*j1/static_cast<float>(N1));
        }
    }

    ModalField<N1, N2, N3> q(BoundaryCondition::Neumann);

    ModalField<N1, N2, N3> rhs(BoundaryCondition::Neumann);
    physicalRHS.ToModal(rhs);


    rhs.Solve(solveLaplacian, q);

    NodalField<N1, N2, N3> physicalSolution(BoundaryCondition::Neumann);
    q.ToNodal(physicalSolution);

    NodalField<N1, N2, N3> expectedSolution(BoundaryCondition::Neumann);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            expectedSolution.stack(j1, j2)= exp(-(x+2)*(x+2))*sin(2*pi*j1/static_cast<float>(N1));
        }
    }

    REQUIRE(physicalSolution == expectedSolution);
}