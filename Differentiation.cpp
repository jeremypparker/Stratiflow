#include "Differentiation.h"

MatrixXd ChebSecondDerivativeMatrix(BoundaryCondition bc, double L, int N)
{
    MatrixXd neumann = ChebDerivativeMatrix(BoundaryCondition::Neumann, L, N);
    MatrixXd dirichlet = ChebDerivativeMatrix(BoundaryCondition::Dirichlet, L, N);

    if (bc==BoundaryCondition::Neumann)
    {
        return dirichlet*neumann;
    }
    else
    {
        return neumann*dirichlet;
    }
}

MatrixXd SymmetriseMatrix(const MatrixXd& in, BoundaryCondition bc)
{
    assert(in.rows() == in.cols());
    assert(in.rows()%2 == 1);
    int N = in.rows()-1;

    MatrixXd bigMatrix(2*N+1, 2*N+1); // this is the full spectral element system without symmetry

    bigMatrix.setZero();

    bigMatrix.block(0, 0, N+1, N+1) += in;
    bigMatrix.block(N, N, N+1, N+1) += in;

    // take into account weights
    bigMatrix.row(N) *= 0.5;

    // now use symmetry to get reduced matrix
    MatrixXd out(2*N+1, N+1);
    out.setZero();

    // these points are not imaged
    out.col(0) = bigMatrix.col(N/2);
    out.col(N) = bigMatrix.col(3*N/2);

    for (int k=1; k<=N/2; k++)
    {
        if(bc == BoundaryCondition::Neumann)
        {
            out.col(k).head(N+1) += bigMatrix.col(N/2+k).head(N+1)  // normal contribution
                                  + bigMatrix.col(N/2-k).head(N+1); // contribution from image

            out.col(N-k).tail(N+1) += bigMatrix.col(3*N/2-k).tail(N+1)
                                    + bigMatrix.col(3*N/2+k).tail(N+1);
        }
        else
        {
            out.col(k).head(N+1) += bigMatrix.col(N/2+k).head(N+1)
                                  - bigMatrix.col(N/2-k).head(N+1);

            out.col(N-k).tail(N+1) += bigMatrix.col(3*N/2-k).tail(N+1)
                                    - bigMatrix.col(3*N/2+k).tail(N+1);
        }
    }

    return out.block(N/2, 0, N+1, N+1); // discard initial and final rows as these are not used
}

ArrayXd ChebyshevGaussLobattoNodes(int N)
{
    ArrayXd x(N);

    x = -cos(ArrayXd::LinSpaced(N+1, 0, pi));

    return x;
}

ArrayXd ChebyshevBarycentricWeights(int N)
{
    // note it doesn't matter if these are scaled
    ArrayXd w(N+1);
    for (int i=0; i<=N; i++)
    {
        w(i) = pow(-1, i);
    }
    w(0) *= 0.5;
    w(N) *= 0.5;

    return w;
}

MatrixXd ChebyshevDerivativeMatrix(int N)
{
    ArrayXd x = ChebyshevGaussLobattoNodes(N);
    ArrayXd w = ChebyshevBarycentricWeights(N);


    MatrixXd D(N+1, N+1);

    for (int i=0; i<=N; i++)
    {
        D(i,i) = 0;
        for (int j=0; j<=N; j++)
        {
            if(j!=i)
            {
                D(i,j) = (w(j)/w(i))*1/((x(i)-x(j)));
                D(i,i) -= D(i,j);
            }
        }
    }

    return D;
}

MatrixXd ChebDerivativeMatrix(BoundaryCondition originalBC, double L, int N)
{
    MatrixXd D = ChebyshevDerivativeMatrix(N-1)/L;
    return SymmetriseMatrix(D, originalBC);
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
