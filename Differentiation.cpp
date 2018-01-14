
#include "Differentiation.h"
#include "Eigen.h"
#include "Field.h"

MatrixX VerticalDerivativeMatrix(BoundaryCondition originalBC, stratifloat L, int N)
{
    MatrixX D = MatrixX::Zero(N,N);

    for (int j=0; j<N; j++)
    {
        if(j==0)
        {
            D(j,j+2) = j+2;
        }
        else if(j==1)
        {
            // use (anti)symmetry of basis function to replace missing mode
            if (originalBC==BoundaryCondition::Bounded)
            {
                D(j, j) = -3*j;
            }
            else
            {
                D(j, j) = -1*j;
            }
            D(j, j+2) = j+2;
        }
        else
        {
            D(j, j-2) = j-2;

            if(j+2 < N)
            {
                D(j,j) = -2*j;
                D(j,j+2) = j+2;
            }
            else // have lost some terms, modify others to keep consistency at infinity
            {
                D(j,j) = -j;
            }
        }
    }

    D /= (4*L);
    if (originalBC==BoundaryCondition::Bounded)
    {
        D = -D;
    }

    D.row(0) *= 2;
    D.col(0) /= 2;
    D.row(N-1) *= 2;
    D.col(N-1) /= 2;

    return D;
}


MatrixX VerticalSecondDerivativeMatrix(BoundaryCondition bc, stratifloat L, int N)
{
    if (bc == BoundaryCondition::Bounded)
    {
        return VerticalDerivativeMatrix(BoundaryCondition::Decaying, L, N)
                * VerticalDerivativeMatrix(BoundaryCondition::Bounded, L, N);
    }
    else
    {
        return VerticalDerivativeMatrix(BoundaryCondition::Bounded, L, N)
                * VerticalDerivativeMatrix(BoundaryCondition::Decaying, L, N);
    }
}

MatrixX VerticalSecondDerivativeNodalMatrix(stratifloat L, int N)
{
    ArrayX x = VerticalPoints(L, N);
    ArrayX d = x.head(N-1) - x.tail(N-1);

    MatrixX D = MatrixX::Zero(N,N);

    for (int j=0; j<N; j++)
    {
        if (j==0)
        {
            stratifloat h1 = -d(0);
            stratifloat h2 = d(1)+d(0);

            stratifloat denom = (h1+h2)*h1*h2/2.0;

            D(0,0) = -(h1+h2)/denom;
            D(0,1) = h2/denom;
            D(0,2) = h1/denom;
        }
        else if (j==N-1)
        {
            stratifloat h1 = d(N-2)+d(N-3);
            stratifloat h2 = -d(N-2);

            stratifloat denom = (h1+h2)*h1*h2/2.0;

            D(N-1,N-1) = -(h1+h2)/denom;
            D(N-1,N-3) = h2/denom;
            D(N-1,N-2) = h1/denom;
        }
        else
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
