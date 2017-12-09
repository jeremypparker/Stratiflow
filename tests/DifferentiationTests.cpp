#include <catch.h>

#include "Differentiation.h"
#include "Field.h"

#include <iostream>

TEST_CASE("Chebyshev derivative matrices")
{
    REQUIRE(MatrixXf(VerticalSecondDerivativeMatrix(BoundaryCondition::Bounded, 1, 7)).isApprox(
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Decaying, 1, 7))*
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Bounded, 1, 7))));

    REQUIRE(MatrixXf(VerticalSecondDerivativeMatrix(BoundaryCondition::Decaying, 3.14, 11)).isApprox(
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Bounded, 3.14, 11))*
        MatrixXf(VerticalDerivativeMatrix(BoundaryCondition::Decaying, 3.14, 11))));
}

TEST_CASE("Fourier derivative matrices")
{
    REQUIRE(MatrixXcf(FourierSecondDerivativeMatrix(5, 6, 1)).isApprox(
        MatrixXcf(FourierDerivativeMatrix(5, 6, 1))*MatrixXcf(FourierDerivativeMatrix(5, 6, 1))));

    REQUIRE(MatrixXcf(FourierSecondDerivativeMatrix(5, 6, 2)).isApprox(
        MatrixXcf(FourierDerivativeMatrix(5, 6, 2))*MatrixXcf(FourierDerivativeMatrix(5, 6, 2))));
}

TEST_CASE("Simple derivatives Bounded")
{
    float L = 5.0f;

    constexpr int N1 = 1;
    constexpr int N2 = 1;
    constexpr int N3 = 64;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Bounded);
    f1.SetValue([](float z){return tanh(z);}, L);

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Bounded);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Decaying);
    f3 = Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>,float,complex,N1/2+1,N2,N3>(VerticalDerivativeMatrix(BoundaryCondition::Bounded, L, N3), f2);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Decaying);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Decaying);
    expected.SetValue([](float z){return 1/(cosh(z)*cosh(z));}, L);

    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Bounded);
    f5 = Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>,float,complex,N1/2+1,N2,N3>(VerticalSecondDerivativeMatrix(BoundaryCondition::Bounded, L, N3), f2);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Bounded);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Bounded);
    expected2.SetValue([](float z){return -2*tanh(z)/(cosh(z)*cosh(z));}, L);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Simple derivatives Decaying")
{
    constexpr int N1 = 2;
    constexpr int N2 = 2;
    constexpr int N3 = 64;
    float L = 5.0f;

    auto x = VerticalPoints(L, N3);

    NodalField<N1,N2,N3> f1(BoundaryCondition::Decaying);
    f1.SetValue([](float z){return exp(-(z-0.5)*(z-0.5));}, L);

    // convert to modal for differentiation
    ModalField<N1,N2,N3> f2(BoundaryCondition::Decaying);
    f1.ToModal(f2);
    ModalField<N1,N2,N3> f3(BoundaryCondition::Bounded);
    f3 = Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>,float,complex,N1/2+1,N2,N3>(VerticalDerivativeMatrix(BoundaryCondition::Decaying, L, N3), f2);
    NodalField<N1,N2,N3> f4(BoundaryCondition::Bounded);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Bounded);
    expected.SetValue([](float z){return -2*(z-0.5)*exp(-(z-0.5)*(z-0.5));}, L);


    REQUIRE(f4 == expected);

    // also do second derviative
    ModalField<N1,N2,N3> f5(BoundaryCondition::Decaying);
    f5 = Dim3MatMul<Map<const Array<complex, -1, 1>, Aligned16>,float,complex,N1/2+1,N2,N3>(VerticalSecondDerivativeMatrix(BoundaryCondition::Decaying, L, N3), f2);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Decaying);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Decaying);
    expected2.SetValue([](float z){return (4*(z-0.5)*(z-0.5)-2)*exp(-(z-0.5)*(z-0.5));}, L);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 1 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 2;
    constexpr int N3 = 4;
    float L = 14.0f;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Bounded);
    f1.SetValue([L](float x, float y, float z){return cos(2*pi*x/L);}, L, 1, 1);

    ModalField<N1,N2,N3> f2(BoundaryCondition::Bounded);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Bounded);
    f3 = Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>,complex,complex,N1/2+1,N2,N3>(FourierDerivativeMatrix(L, N1, 1), f2);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Bounded);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Bounded);
    expected.SetValue([L](float x, float y, float z){return -2*pi*sin(2*pi*x/L)/L;}, L, 1, 1);


    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Bounded);
    f5 = Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>,float,complex,N1/2+1,N2,N3>(FourierSecondDerivativeMatrix(L, N1, 1), f2);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Bounded);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Bounded);
    expected2.SetValue([L](float x, float y, float z){return -4*pi*pi*cos(2*pi*x/L)/L/L;}, L, 1, 1);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 2 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 10;
    constexpr int N3 = 4;
    float L = 14.0f;

    NodalField<N1,N2,N3> f1(BoundaryCondition::Bounded);
    f1.SetValue([L](float x, float y, float z){return sin(2*pi*y/L) + x;}, 1, L, 1);


    ModalField<N1,N2,N3> f2(BoundaryCondition::Bounded);
    f1.ToModal(f2);

    ModalField<N1,N2,N3> f3(BoundaryCondition::Bounded);
    f3 = Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>,complex,complex,N1/2+1,N2,N3>(FourierDerivativeMatrix(L, N2, 2), f2);

    NodalField<N1,N2,N3> f4(BoundaryCondition::Bounded);
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected(BoundaryCondition::Bounded);
    expected.SetValue([L](float x, float y, float z){return 2*pi*cos(2*pi*y/L)/L;}, 1, L, 1);

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5(BoundaryCondition::Bounded);
    f5 = Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>,float,complex,N1/2+1,N2,N3>(FourierSecondDerivativeMatrix(L, N2, 2), f2);

    NodalField<N1,N2,N3> f6(BoundaryCondition::Bounded);
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2(BoundaryCondition::Bounded);
    expected2.SetValue([L](float x, float y, float z){return -4*pi*pi*sin(2*pi*y/L)/L/L;}, 1, L, 1);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Inverse Laplacian")
{
    constexpr int N1 = 20;
    constexpr int N2 = 22;
    constexpr int N3 = 40;

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
            MatrixXf laplacian = VerticalSecondDerivativeMatrix(BoundaryCondition::Bounded, L3, N3);

            // add terms for horizontal derivatives
            laplacian += dim1Derivative2.diagonal()(j1)*MatrixXf::Identity(N3, N3);
            laplacian += dim2Derivative2.diagonal()(j2)*MatrixXf::Identity(N3, N3);


            solveLaplacian[j1*N2+j2].compute(laplacian);
        }
    }

    // create field in physical space
    NodalField<N1, N2, N3> physicalRHS(BoundaryCondition::Bounded);
    auto x = VerticalPoints(L3, N3);
    physicalRHS.SetValue([L1](float x, float y, float z)
    {
        return (4*(z+2)*(z+2) - 2 -4*pi*pi/L1/L1)*
               exp(-(z+2)*(z+2))*sin(2*pi*x/L1);
    }, L1, L2, L3);

    ModalField<N1, N2, N3> q(BoundaryCondition::Bounded);

    ModalField<N1, N2, N3> rhs(BoundaryCondition::Bounded);
    physicalRHS.ToModal(rhs);


    rhs.Solve(solveLaplacian, q);

    NodalField<N1, N2, N3> physicalSolution(BoundaryCondition::Bounded);
    q.ToNodal(physicalSolution);

    NodalField<N1, N2, N3> expectedSolution(BoundaryCondition::Bounded);
    expectedSolution.SetValue([L1](float x, float y, float z)
    {
        return exp(-(z+2)*(z+2))*sin(2*pi*x/L1);
    }, L1, L2, L3);


    REQUIRE(physicalSolution == expectedSolution);
}