#pragma once

#include "Eigen.h"
#include "Constants.h"

// Computes a finite difference matrix for use with nodal forms
MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC);
MatrixX VerticalDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC);
MatrixX NeumannReinterpolationFull(stratifloat L, int N);
MatrixX NeumannReinterpolationBar(stratifloat L, int N);
MatrixX NeumannReinterpolationTilde(stratifloat L, int N);
MatrixX DirichletReinterpolation(stratifloat L, int N);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);

template<typename M>
void Neumannify(M& matrix)
{
    int N3 = matrix.rows();

    matrix.coeffRef(0,0) = 1;
    matrix.coeffRef(0,1) = 0;
    matrix.coeffRef(0,2) = 0;
    matrix.coeffRef(1,0) = 0;
    matrix.coeffRef(1,1) = -1;
    matrix.coeffRef(1,2) = 1;

    matrix.coeffRef(N3-2, N3-3) = -1;
    matrix.coeffRef(N3-2, N3-2) = 1;
    matrix.coeffRef(N3-2, N3-1) = 0;
    matrix.coeffRef(N3-1, N3-3) = 0;
    matrix.coeffRef(N3-1, N3-2) = 0;
    matrix.coeffRef(N3-1, N3-1) = 1;
}

template<typename M>
void Dirichlify(M& matrix)
{
    int N3 = matrix.rows();

    matrix.coeffRef(0,0) = 1;
    matrix.coeffRef(0,1) = 0;
    matrix.coeffRef(0,2) = 0;
    matrix.coeffRef(1,0) = 0;
    matrix.coeffRef(1,1) = 1;
    matrix.coeffRef(1,2) = 0;
    matrix.coeffRef(2,1) = 0;
    matrix.coeffRef(2,2) = 1;
    matrix.coeffRef(2,3) = 0;

    matrix.coeffRef(N3-1, N3-2) = 0;
    matrix.coeffRef(N3-1, N3-1) = 1;
    matrix.coeffRef(N3-1, N3-3) = 0;
    matrix.coeffRef(N3-2, N3-2) = 1;
    matrix.coeffRef(N3-2, N3-1) = 0;
    matrix.coeffRef(N3-2, N3-3) = 0;
    matrix.coeffRef(N3-2, N3-4) = 0;
}
