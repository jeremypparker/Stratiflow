#include "Differentiation.h"

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

    //return bigMatrix;

    // now use symmetry to get reduced matrix
    MatrixXd out(2*N+1, N+1);
    out.setZero();

    // these points are not imaged
    out.col(0) = bigMatrix.col(N/2);
    out.col(N) = bigMatrix.col(3*N/2);

    // to avoid double counting
    bigMatrix(N, N) *= 0.5;

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

struct qAndL
{
    double q;
    double qprime;
    double L;
};
qAndL qAndLEvaluation(int N, double x)
{
    qAndL ret;

    double Lmm = 1;
    double Lm = x;
    double Lprimemm = 0;
    double Lprimem = 1;

    double Lprime;

    for (int k=2; k<=N; k++)
    {
        ret.L = (2*k-1)*x*Lm/k - (k-1)*Lmm/k;
        Lprime = Lprimemm+(2*k-1)*Lm;
        Lmm = Lm;
        Lm = ret.L;
        Lprimemm = Lprimem;
        Lprimem = Lprime;
    }

    int k = N+1;
    double Lp = (2*k-1)*x*ret.L/k - (k-1)*Lmm/k;
    double Lprimep = Lprimemm+(2*k-1)*Lm;

    ret.q = Lp - Lmm;
    ret.qprime = Lprimep - Lprimemm;

    return ret;
}

ArrayXd ChebyshevGaussLobattoNodes(int N)
{
    // ArrayXd x(N);

    // x = -cos(ArrayXd::LinSpaced(N+1, 0, pi));

    // return x;

    assert(N%2 == 0);
    ArrayXd x(N+1);
    x(0) = -1;
    x(N) = 1;
    x(N/2) = 0;

    for (int j=1; j<N/2; j++)
    {
        x(N-j) = cos((j+0.25)*pi/N - 3/(8*pi*N)/(j+0.25));

        for (int k=0; k<1000; k++)
        {
            qAndL ql = qAndLEvaluation(N, x(N-j));
            double delta = -ql.q / ql.qprime;

            x(N-j) += delta;
        }

        x(j) = -x(N-j);
    }

    return x;
}

ArrayXd GaussLobattoWeights(const ArrayXd& x)
{
    int N = x.rows() - 1;
    assert(N%2 == 0);

    ArrayXd w(N+1);

    w(0) = w(N) = 2.0/(N*(N+1));

    for (int j=0; j<N; j++)
    {
        qAndL ql = qAndLEvaluation(N, x(j));
        w(j) = 2.0/(N*(N+1)*ql.L*ql.L);
    }

    return w;
}

ArrayXd BarycentricWeights(const ArrayXd& x)
{
    // note it doesn't matter if these are scaled
    int N = x.rows() - 1;
    ArrayXd w = ArrayXd::Ones(N+1);

    for (int j=0; j<=N; j++)
    {
        for (int k=0; k<=N; k++)
        {
            if (k==j) continue;

            w(j) *= (x(j)-x(k)) * 2.0;
        }
    }

    return 1/w;

}

MatrixXd PolynomialDerivativeMatrix(const ArrayXd& x)
{
    int N = x.rows() - 1;
    ArrayXd w = BarycentricWeights(x);
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
    MatrixXd D = PolynomialDerivativeMatrix(ChebyshevGaussLobattoNodes(N-1))/L;
    return SymmetriseMatrix(D, originalBC);
}

MatrixXd ChebSecondDerivativeMatrix(BoundaryCondition bc, double L, int N)
{
    // ArrayXd x = ChebyshevGaussLobattoNodes(N-1);
    // return SymmetriseMatrix(PolynomialDerivativeMatrix(x)*PolynomialDerivativeMatrix(x)/L/L, bc);
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
