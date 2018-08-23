
#include "Differentiation.h"
#include "Eigen.h"
#include "Field.h"

MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    if (originalBC == BoundaryCondition::Neumann)
    {
        return VerticalDerivativeMatrix(L,N,BoundaryCondition::Dirichlet)*
               VerticalDerivativeMatrix(L,N,BoundaryCondition::Neumann);
    }
    else
    {
        return VerticalDerivativeMatrix(L,N,BoundaryCondition::Neumann)*
               VerticalDerivativeMatrix(L,N,BoundaryCondition::Dirichlet);
    }
}

MatrixX VerticalDerivativeMatrix(stratifloat L, int N, BoundaryCondition originalBC)
{
    MatrixX D = MatrixX::Zero(N,N);

    if (originalBC == BoundaryCondition::Neumann)
    {
        ArrayX diff = dz(L,N);
        D.diagonal(0).tail(N-1) = 1/diff.tail(N-1);
        D.diagonal(-1) = -1/diff.tail(N-1);
    }
    else
    {
        ArrayX diff = dzFractional(L,N);
        D.diagonal(1).tail(N-2) = 1/diff.segment(1,N-2);
        D.diagonal(0).segment(1,N-2) = -1/diff.segment(1,N-2);
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
