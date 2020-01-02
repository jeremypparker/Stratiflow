#pragma once

#include "Eigen.h"
#include "Constants.h"

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);

