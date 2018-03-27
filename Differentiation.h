#pragma once

#include "Eigen.h"
#include "Constants.h"

// Computes a finite difference matrix for use with nodal forms
MatrixX VerticalSecondDerivativeNodalMatrix(stratifloat L, int N);
MatrixX VerticalDerivativeNodalMatrix(stratifloat L, int N);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);
