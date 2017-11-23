#pragma once

#include <Eigen/Core>

#include "Constants.h"

using namespace Eigen;


MatrixXf VerticalDerivativeMatrix(BoundaryCondition originalBC, float L, int N);
MatrixXf VerticalSecondDerivativeMatrix(BoundaryCondition bc, float L, int N);

DiagonalMatrix<float, -1> FourierSecondDerivativeMatrix(float L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(float L, int N, int dimension);
