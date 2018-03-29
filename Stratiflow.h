#pragma once
#include "Field.h"
#include "Parameters.h"
#include "Differentiation.h"

constexpr int M1 = N1/2 + 1;

using NField = NodalField<N1,N2,N3>;
using MField = ModalField<N1,N2,N3>;
using N1D = Nodal1D<N1,N2,N3>;

template<typename T>
Dim1MatMul<T, complex, complex, M1, N2, N3> ddx(const StackContainer<T, complex, M1, N2, N3>& f)
{
    static DiagonalMatrix<complex, -1> dim1Derivative = FourierDerivativeMatrix(L1, N1, 1);

    return Dim1MatMul<T, complex, complex, M1, N2, N3>(dim1Derivative, f);
}

template<typename T>
Dim2MatMul<T, complex, complex, M1, N2, N3> ddy(const StackContainer<T, complex, M1, N2, N3>& f)
{
    static DiagonalMatrix<complex, -1> dim2Derivative = FourierDerivativeMatrix(L2, N2, 2);

    return Dim2MatMul<T, complex, complex, M1, N2, N3>(dim2Derivative, f);
}

template<typename A, typename T, int K1, int K2, int K3>
Dim3MatMul<A, stratifloat, T, K1, K2, K3> ddz(const StackContainer<A, T, K1, K2, K3>& f)
{
    static MatrixX dim3Derivative = VerticalDerivativeMatrix(L3, N3);

    return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3Derivative, f);
}
