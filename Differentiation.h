#pragma once

#include "Eigen.h"
#include "Constants.h"

using namespace Eigen;


MatrixX VerticalDerivativeMatrix(BoundaryCondition originalBC, stratifloat L, int N);
MatrixX VerticalSecondDerivativeMatrix(BoundaryCondition bc, stratifloat L, int N);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);
