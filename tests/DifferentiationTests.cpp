#include <catch.h>

#include "Differentiation.h"
#include "Field.h"

#include <iostream>

TEST_CASE("Fourier derivative matrices")
{
    REQUIRE(MatrixXc(FourierSecondDerivativeMatrix(5, 6, 1)).isApprox(
        MatrixXc(FourierDerivativeMatrix(5, 6, 1))*MatrixXc(FourierDerivativeMatrix(5, 6, 1))));

    REQUIRE(MatrixXc(FourierSecondDerivativeMatrix(5, 6, 2)).isApprox(
        MatrixXc(FourierDerivativeMatrix(5, 6, 2))*MatrixXc(FourierDerivativeMatrix(5, 6, 2))));
}

TEST_CASE("Simple derivatives vertical")
{
    constexpr int N1 = 2;
    constexpr int N2 = 2;
    constexpr int N3 = 64;
    stratifloat L = 5.0f;

    auto x = VerticalPoints(L, N3);

    NodalField<N1,N2,N3> f1;
    f1.SetValue([](stratifloat z){return exp(-(z-0.5)*(z-0.5));}, L);

    // convert to modal for differentiation
    NodalField<N1,N2,N3> f2;
    f2 = Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>,stratifloat,stratifloat,N1,N2,N3>(VerticalSecondDerivativeMatrix(L, N3), f1);

    NodalField<N1,N2,N3> expected;
    expected.SetValue([](stratifloat z){return (4*(z-0.5)*(z-0.5)-2)*exp(-(z-0.5)*(z-0.5));}, L);

    REQUIRE(f2 == expected);
}

TEST_CASE("Dim 1 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 2;
    constexpr int N3 = 4;
    stratifloat L = 14.0f;

    NodalField<N1,N2,N3> f1;
    f1.SetValue([L](stratifloat x, stratifloat y, stratifloat z){return cos(2*pi*x/L);}, L, 1, 1);

    ModalField<N1,N2,N3> f2;
    f1.ToModal(f2, false);

    ModalField<N1,N2,N3> f3;
    f3 = Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>,complex,complex,N1/2+1,N2,N3>(FourierDerivativeMatrix(L, N1, 1), f2);

    NodalField<N1,N2,N3> f4;
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected;
    expected.SetValue([L](stratifloat x, stratifloat y, stratifloat z){return -2*pi*sin(2*pi*x/L)/L;}, L, 1, 1);


    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5;
    f5 = Dim1MatMul<Map<const Array<complex, -1, 1>, Aligned16>,stratifloat,complex,N1/2+1,N2,N3>(FourierSecondDerivativeMatrix(L, N1, 1), f2);

    NodalField<N1,N2,N3> f6;
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2;
    expected2.SetValue([L](stratifloat x, stratifloat y, stratifloat z){return -4*pi*pi*cos(2*pi*x/L)/L/L;}, L, 1, 1);

    REQUIRE(f6 == expected2);
}

TEST_CASE("Dim 2 fourier derivatives")
{
    constexpr int N1 = 20;
    constexpr int N2 = 10;
    constexpr int N3 = 4;
    stratifloat L = 14.0f;

    NodalField<N1,N2,N3> f1;
    f1.SetValue([L](stratifloat x, stratifloat y, stratifloat z){return sin(2*pi*y/L) + x;}, 1, L, 1);


    ModalField<N1,N2,N3> f2;
    f1.ToModal(f2, false);

    ModalField<N1,N2,N3> f3;
    f3 = Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>,complex,complex,N1/2+1,N2,N3>(FourierDerivativeMatrix(L, N2, 2), f2);

    NodalField<N1,N2,N3> f4;
    f3.ToNodal(f4);

    NodalField<N1,N2,N3> expected;
    expected.SetValue([L](stratifloat x, stratifloat y, stratifloat z){return 2*pi*cos(2*pi*y/L)/L;}, 1, L, 1);

    REQUIRE(f4 == expected);

    ModalField<N1,N2,N3> f5;
    f5 = Dim2MatMul<Map<const Array<complex, -1, 1>, Aligned16>,stratifloat,complex,N1/2+1,N2,N3>(FourierSecondDerivativeMatrix(L, N2, 2), f2);

    NodalField<N1,N2,N3> f6;
    f5.ToNodal(f6);

    NodalField<N1,N2,N3> expected2;
    expected2.SetValue([L](stratifloat x, stratifloat y, stratifloat z){return -4*pi*pi*sin(2*pi*y/L)/L/L;}, 1, L, 1);

    REQUIRE(f6 == expected2);
}

// TEST_CASE("Inverse Laplacian")
// {
//     constexpr int N1 = 20;
//     constexpr int N2 = 22;
//     constexpr int N3 = 40;

//     constexpr stratifloat L1 = 14.0f;
//     constexpr stratifloat L2 = 3.5;
//     constexpr stratifloat L3 = 3.0f;

//     auto dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1, 1);
//     auto dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2, 2);

//     std::vector<ColPivHouseholderQR<MatrixXc>> solveLaplacian((N1/2 + 1)*N2);

//     // we solve each vetical line separately, so N1*N2 total solves
//     for (int j1=0; j1<N1/2 + 1; j1++)
//     {
//         for (int j2=0; j2<N2; j2++)
//         {
//             MatrixX laplacian = VerticalSecondDerivativeMatrix(BoundaryCondition::Bounded, L3, N3);

//             // add terms for horizontal derivatives
//             laplacian += dim1Derivative2.diagonal()(j1)*MatrixX::Identity(N3, N3);
//             laplacian += dim2Derivative2.diagonal()(j2)*MatrixX::Identity(N3, N3);


//             solveLaplacian[j1*N2+j2].compute(laplacian);
//         }
//     }

//     // create field in physical space
//     NodalField<N1, N2, N3> physicalRHS;
//     auto x = VerticalPoints(L3, N3);
//     physicalRHS.SetValue([L1](stratifloat x, stratifloat y, stratifloat z)
//     {
//         return (4*(z+2)*(z+2) - 2 -4*pi*pi/L1/L1)*
//                exp(-(z+2)*(z+2))*sin(2*pi*x/L1);
//     }, L1, L2, L3);

//     ModalField<N1, N2, N3> q;

//     ModalField<N1, N2, N3> rhs;
//     physicalRHS.ToModal(rhs, false);


//     rhs.Solve(solveLaplacian, q);

//     NodalField<N1, N2, N3> physicalSolution;
//     q.ToNodal(physicalSolution);

//     NodalField<N1, N2, N3> expectedSolution;
//     expectedSolution.SetValue([L1](stratifloat x, stratifloat y, stratifloat z)
//     {
//         return exp(-(z+2)*(z+2))*sin(2*pi*x/L1);
//     }, L1, L2, L3);


//     REQUIRE(physicalSolution == expectedSolution);
// }