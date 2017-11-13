#include "Differentiation.h"

#include <Eigen/Dense>

DiagonalMatrix<double, -1> VerticalDerivativeMatrix(BoundaryCondition originalBC, double L, int N)
{
    ArrayXd k = ArrayXd::LinSpaced(N, 0, N-1);
    VectorXd coeffs = pi*k/(2*L);

    //coeffs(N-1) = 0; // check this

    if (originalBC==BoundaryCondition::Neumann)
    {
        coeffs = -coeffs;
    }

    return coeffs.asDiagonal();
}


DiagonalMatrix<double, -1> VerticalSecondDerivativeMatrix(double L, int N)
{
    ArrayXd k = ArrayXd::LinSpaced(N, 0, N-1);
    VectorXd coeffs = -pi*pi*k*k/(4*L*L);

    //coeffs(N-1) = 0; // check

    return coeffs.asDiagonal();
}


ArrayXd k(int n)
{
    if (n==1)
    {
        // handle this separately for 2D
        return ArrayXd::Zero(1);
    }
    assert(n % 2 == 0); // odd case not handled
    assert(n > 0);

    ArrayXd k(n);

    // using this for k gives a result which matches the FT of the real
    // derivative
    k << ArrayXd::LinSpaced(n / 2, 0, n / 2 - 1),
         ArrayXd::LinSpaced(n / 2, -n / 2, -1);

    return k;
}

DiagonalMatrix<double, -1> FourierSecondDerivativeMatrix(double L, int N)
{
    VectorXd ret = -4*pi*pi*k(N)*k(N)/(L*L);
    return ret.asDiagonal();
}

DiagonalMatrix<complex, -1> FourierDerivativeMatrix(double L, int N)
{
    VectorXcd ret = 2.0*pi*i*k(N)/L;
    return ret.asDiagonal();
}
