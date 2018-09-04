
#include "Differentiation.h"
#include "Eigen.h"
#include "Field.h"
#include <iomanip>

MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    MatrixX D = MatrixX::Zero(N,N);

    ArrayX DY = dz(L,N);
    ArrayX DYF = dzFractional(L,N);

    if (originalBC == BoundaryCondition::Neumann)
    {
        for (int j=1; j<=N-1; j++)
        {
            D(j,j-1) = 1/DY(j)/DYF(j);
            D(j,j) = -1/DY(j)/DYF(j) -1/DY(j+1)/DYF(j);
            D(j,j+1) = 1/DY(j+1)/DYF(j);
        }
    }
    else
    {
        for (int j=2; j<=N-1; j++)
        {
            D(j,j-1) = 1/DYF(j-1)/DY(j);
            D(j,j) = -1/DYF(j-1)/DY(j) -1/DYF(j)/DY(j);
            D(j,j+1) = 1/DYF(j)/DY(j);
        }
    }

    return D;
}

MatrixX VerticalDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    MatrixX D = MatrixX::Zero(N,N);

    ArrayX DY = dz(L,N);
    ArrayX DYF = dzFractional(L,N);

    if (originalBC == BoundaryCondition::Neumann)
    {
        for (int j=2; j<=N-1; j++)
        {
            D(j,j-1) = -1/DY(j);
            D(j,j) = 1/DY(j);
        }
    }
    else
    {
        for (int j=1; j<=N-1; j++)
        {
            D(j,j) = -1/DYF(j);
            D(j,j+1) = 1/DYF(j);
        }
    }

    return D;
}

MatrixX NeumannReinterpolationFull(stratifloat L, int N)
{
    ArrayX diff = dz(L,N);
    ArrayX diffFrac = dzFractional(L,N);

    MatrixX D = MatrixX::Zero(N,N);

    // 2nd order interpolation
    D.diagonal(-1)          = diffFrac.tail(N-1)/(2*diff.tail(N-1));
    D.diagonal(0).tail(N-1) = diffFrac.head(N-1)/(2*diff.tail(N-1));

    return D;
}

MatrixX NeumannReinterpolationBar(stratifloat L, int N)
{
    MatrixX D = MatrixX::Zero(N,N);

    // quasi 2nd order interpolation
    D.diagonal(-1).setConstant(0.5);
    D.diagonal(0).tail(N-1).setConstant(0.5);

    return D;
}

MatrixX NeumannReinterpolationTilde(stratifloat L, int N)
{
    ArrayX diff = dz(L,N);
    ArrayX diffFrac = dzFractional(L,N);

    MatrixX D = MatrixX::Zero(N,N);

    // quasi 2nd order interpolation
    D.diagonal(-1)          = diffFrac.head(N-1)/(2*diff.tail(N-1));
    D.diagonal(0).tail(N-1) = diffFrac.tail(N-1)/(2*diff.tail(N-1));

    return D;
}

MatrixX DirichletReinterpolation(stratifloat L, int N)
{
    MatrixX D = MatrixX::Zero(N,N);

    // 2nd order interpolation
    D.diagonal(0).setConstant(0.5);
    D.diagonal(1).setConstant(0.5);

    return D;
}

ArrayX k(int n)
{
    if (n==1)
    {
        // handle this separately for 2D
        return ArrayX::Zero(1);
    }

    assert(n > 0);

    ArrayX k(n);

    // using this for k gives a result which matches the FT of the real
    // derivative
    k << ArrayX::LinSpaced(n / 3 + 1, 0, n / 3),
         ArrayX::Zero(n-2*(n/3)-1),
         ArrayX::LinSpaced(n / 3, -n / 3, -1);

    return k;
}

ArrayXc KVector(stratifloat L, int N, int dimension)
{
    ArrayXc ret = i*(k(N)*(2*pi)/L);

    if (dimension == 2)
    {
        return ret;
    }
    else if(dimension == 1)
    {
        return ret.head(N/2 +1);
    }

    assert(0);
    return ArrayXc();
}

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension)
{
    return (KVector(L,N,dimension)*KVector(L,N,dimension)).real().matrix().asDiagonal();
}

DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension)
{
    return KVector(L,N,dimension).matrix().asDiagonal();

}
