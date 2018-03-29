#pragma once

#include "Eigen.h"
#include "Constants.h"

// Computes a finite difference matrix for use with nodal forms
MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N);
MatrixX VerticalDerivativeMatrix(stratifloat L, int N);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);

template<typename M>
void Neumannify(M& matrix)
{
    int N3 = matrix.rows();

    matrix.coeffRef(0,0) = 1;
    matrix.coeffRef(0,1) = -1;

    matrix.coeffRef(N3-1, N3-2) = 1;
    matrix.coeffRef(N3-1, N3-1) = -1;
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
