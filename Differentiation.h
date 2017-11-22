#pragma once

#include <Eigen/Core>

#include "Constants.h"

using namespace Eigen;


MatrixXd VerticalDerivativeMatrix(BoundaryCondition originalBC, double L, int N);
MatrixXd VerticalSecondDerivativeMatrix(BoundaryCondition bc, double L, int N);

DiagonalMatrix<double, -1> FourierSecondDerivativeMatrix(double L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(double L, int N, int dimension);
