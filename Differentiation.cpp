
#include "Differentiation.h"
#include "Eigen.h"
#include "Field.h"

MatrixX VerticalSecondDerivativeNodalMatrix(stratifloat L, int N)
{
    ArrayX x = VerticalPoints(L, N);
    ArrayX d = x.head(N-1) - x.tail(N-1);

    MatrixX D = MatrixX::Zero(N,N);

    for (int j=0; j<N; j++)
    {
        if (j!=0 && j!=N-1) // ends are at infinity, so derivative is zero there
        {
            stratifloat h1 = d(j-1);
            stratifloat h2 = d(j);

            stratifloat denom = (h1+h2)*h1*h2/2.0;

            D(j,j-1) = h2/denom;
            D(j,j) = -(h1+h2)/denom;
            D(j,j+1) = h1/denom;
        }
    }

    return D;
}

MatrixX VerticalDerivativeNodalMatrix(stratifloat L, int N)
{
    ArrayX x = VerticalPoints(L, N);
    ArrayX d = x.head(N-1) - x.tail(N-1);

    MatrixX D = MatrixX::Zero(N,N);

    for (int j=0; j<N; j++)
    {
        if (j!=0 && j!=N-1) // ends are at infinity, so derivative is zero there
        {
            stratifloat h1 = d(j-1);
            stratifloat h2 = d(j);

            stratifloat denom = (h1+h2)*h1*h2;

            D(j,j-1) = h2*h2/denom;
            D(j,j) = (h1*h1-h2*h2)/denom;
            D(j,j+1) = -h1*h1/denom;
        }
    }

    return D;
}


ArrayX k(int n)
{
    if (n==1)
    {
        // handle this separately for 2D
        return ArrayX::Zero(1);
    }
    assert(n % 2 == 0); // odd case not handled
    assert(n > 0);

    ArrayX k(n);

    // using this for k gives a result which matches the FT of the real
    // derivative
    k << ArrayX::LinSpaced(n / 2, 0, n / 2 - 1),
         ArrayX::LinSpaced(n / 2, -n / 2, -1);

    return k;
}

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension)
{
    VectorX ret = -4*pi*pi*k(N)*k(N)/(L*L);

    if (dimension == 2)
    {
        return ret.asDiagonal();
    }
    else if(dimension == 1)
    {
        return ret.head(N/2 +1).asDiagonal();
    }

    assert(0);
    return DiagonalMatrix<stratifloat, -1>();
}

DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension)
{
    VectorXc ret = 2.0f*pi*i*k(N)/L;

    if (dimension == 2)
    {
        return ret.asDiagonal();
    }
    else if(dimension == 1)
    {
        return ret.head(N/2 +1).asDiagonal();
    }

    assert(0);
    return DiagonalMatrix<complex, -1>();
}
