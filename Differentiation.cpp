#include "Differentiation.h"

MatrixXd ChebSecondDerivativeMatrix(BoundaryCondition bc, double L, int N)
{
    assert(N%2 == 0);
    MatrixXd D(N, N);
    D.setZero();

    for (int k=0; k<N/2; k++)
    {
        for(int j=1; j<= N/2-1-k; j++)
        {
            double mult;
            if (bc == BoundaryCondition::Neumann)
            {
                mult = (k==0?4:8)*j*(2*k+j)*(k+j);
            }
            else
            {
                mult = 4*j*(2*k+j+1)*(2*k+2*j+1);
            }

            D(N/2-1-k, N/2-1-k-j) = mult;
            D(N/2+k, N/2+k+j) += mult;
        }
    }
    D /= (L*L); // because we use T(1-x/L)

    return D;
}

MatrixXd ChebDerivativeMatrix(BoundaryCondition originalBC, double L, int N)
{
    assert(N%2 == 0);
    MatrixXd D(N, N);
    D.setZero();

    for (int k=0; k<N/2; k++)
    {
        for(int j=(originalBC == BoundaryCondition::Neumann)?1:0; j<= N/2-1-k; j++)
        {
            int mult;
            if (originalBC == BoundaryCondition::Neumann)
            {
                mult = 2*(2*k+2*j);
            }
            else
            {
                mult = (k==0?1:2)*(2*k+2*j+1);
            }

            // factor of ±1/L because we use T(1±x/L)
            D(N/2-1-k, N/2-1-k-j) = mult/L;
            D(N/2+k, N/2+k+j) = -mult/L;
        }
    }

    return D;
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
