#pragma once

#include "Eigen.h"
#include "Constants.h"

// Computes a finite difference matrix for use with nodal forms
MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC);
MatrixX VerticalDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC);
MatrixX VerticalReinterpolationMatrix(stratifloat L, int N, BoundaryCondition originalBC);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);