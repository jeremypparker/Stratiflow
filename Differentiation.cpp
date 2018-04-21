
#include "Differentiation.h"
#include "Eigen.h"
#include "Field.h"

MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    ArrayX d;

    if (originalBC == BoundaryCondition::Neumann)
    {
        d = dz(L, N);
    }
    else
    {
        d = dzStaggered(L, N).segment(1, N-2);
    }

    MatrixX D = MatrixX::Zero(N,N);

    for (int j=0; j<N; j++)
    {
        if (j==0)
        {
            // one sided (second order) derivatives at ends

            stratifloat h1 = d(0);
            stratifloat h2 = d(1);

            stratifloat denom = (h1+h2)*h1*h2/2.0;

            D(0,2) = h1/denom;
            D(0,1) = -(h1+h2)/denom;
            D(0,0) = h2/denom;
        }
        else if (j==N-1)
        {
            if (originalBC == BoundaryCondition::Neumann)
            {
                stratifloat h1 = d(N-2);
                stratifloat h2 = d(N-3);

                stratifloat denom = (h1+h2)*h1*h2/2.0;

                D(N-1,N-3) = h1/denom;
                D(N-1,N-2) = -(h1+h2)/denom;
                D(N-1,N-1) = h2/denom;
            }
        }
        else if (j==N-2 && originalBC == BoundaryCondition::Dirichlet)
        {
            stratifloat h1 = d(N-3);
            stratifloat h2 = d(N-4);

            stratifloat denom = (h1+h2)*h1*h2/2.0;

            D(N-2,N-4) = h1/denom;
            D(N-2,N-3) = -(h1+h2)/denom;
            D(N-2,N-2) = h2/denom;
        }
        else
        {
            // second order central finite difference everywhere else
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

MatrixX VerticalDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    MatrixX D = MatrixX::Zero(N,N);

    if (originalBC == BoundaryCondition::Neumann)
    {
        // this is second order because of symmetry
        ArrayX diff = dz(L, N);
        D.diagonal().head(N-1) = 1/diff;
        D.diagonal(1) = -1/diff;
    }
    else
    {
        // this is only first order
        ArrayX diff = dzStaggered(L,N);

        D.diagonal(-1).head(N-2) = 1/diff.segment(1,N-2);
        D.diagonal(0) = -1/diff;
    }

    return D;
}

MatrixX VerticalReinterpolationMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    // first order interpolation at one set of points of other
    MatrixX D = MatrixX::Zero(N,N);

    if (originalBC == BoundaryCondition::Neumann)
    {
        D.diagonal(1).setConstant(0.5);
        D.diagonal(0).head(N-1).setConstant(0.5);

        // zero at ends
        D.row(0).setZero();
        D.row(N-1).setZero();
        D.row(N-2).setZero();
    }
    else
    {
        ArrayX diff = dzStaggered(L,N);
        ArrayX diff2 = dz(L,N);

        D.diagonal(-1).head(N-2) = 0.5*diff2.tail(N-2)/diff.segment(1, N-2);
        D.diagonal().segment(1, N-2) = 0.5*diff2.head(N-2)/diff.segment(1, N-2);

        // no derivative at ends
        D.row(0) = D.row(1);
        D.row(N-1) = D.row(N-2);
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
