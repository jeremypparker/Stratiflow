#pragma once

#include "Stratiflow.h"

ArrayXc CalculateEigenvalues(stratifloat k,
                             MatrixXc *w_eigen = nullptr,
                             MatrixXc *b_eigen = nullptr);

stratifloat LargestGrowth(stratifloat k,
                          Field1D<complex, N1, N2, N3>* w=nullptr,
                          Field1D<complex, N1, N2, N3>* b=nullptr);