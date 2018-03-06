#pragma once

#include "Stratiflow.h"

MatrixXc OrrSommerfeldLHS(stratifloat k);
MatrixXc OrrSommerfeldRHS(stratifloat k);

ArrayXc CalculateEigenvalues(stratifloat k,
                             MatrixXc *w_eigen = nullptr,
                             MatrixXc *b_eigen = nullptr);

stratifloat LargestGrowth(stratifloat k,
                          Field1D<complex, N1, N2, N3>* w=nullptr,
                          Field1D<complex, N1, N2, N3>* b=nullptr,
                          stratifloat* imag=nullptr);

void EigenModes(stratifloat k, MField& u1, MField& u2, MField& u3, MField& b);