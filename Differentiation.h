#pragma once

#include "Eigen.h"
#include "Constants.h"

// Computes a finite difference matrix for use with nodal forms
MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N);
MatrixX VerticalDerivativeMatrix(stratifloat L, int N);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);

template<typename M>
void Neumannify(M& matrix, stratifloat L)
{
    int N3 = matrix.rows();

    auto D = VerticalDerivativeMatrix(L, N3);

    matrix.coeffRef(0,0) = D(0,0);
    matrix.coeffRef(0,1) = D(0,1);
    matrix.coeffRef(0,2) = D(0,2);
    matrix.coeffRef(N3-1,N3-1) = D(N3-1,N3-1);
    matrix.coeffRef(N3-1,N3-2) = D(N3-1,N3-2);
    matrix.coeffRef(N3-1,N3-3) = D(N3-1,N3-3);
}

template<typename M>
void Dirichlify(M& matrix)
{
    int N3 = matrix.rows();

    matrix.coeffRef(0,0) = 1;
    matrix.coeffRef(0,1) = 0;

    matrix.coeffRef(N3-1, N3-2) = 0;
    matrix.coeffRef(N3-1, N3-1) = 1;
}

// this makes the second and penultimate row of a matrix
// give the zero derivative at the boundary
template<typename M>
void VanishingSecondDeriv(M& matrix, stratifloat L)
{
    int N3 = matrix.rows();
    auto D2 = VerticalSecondDerivativeMatrix(L, N3);

    matrix.coeffRef(1,0) = D2(0,0);
    matrix.coeffRef(1,1) = D2(0,1);
    matrix.coeffRef(1,2) = D2(0,2);
    matrix.coeffRef(N3-2,N3-1) = D2(N3-1,N3-1);
    matrix.coeffRef(N3-2,N3-2) = D2(N3-1,N3-2);
    matrix.coeffRef(N3-2,N3-3) = D2(N3-1,N3-3);
}