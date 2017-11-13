#pragma once

#include <Eigen/Core>

#include "Constants.h"

using namespace Eigen;


DiagonalMatrix<double, -1> VerticalDerivativeMatrix(BoundaryCondition originalBC, double L, int N);
DiagonalMatrix<double, -1> VerticalSecondDerivativeMatrix(double L, int N);

DiagonalMatrix<double, -1> FourierSecondDerivativeMatrix(double L, int N);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(double L, int N);
