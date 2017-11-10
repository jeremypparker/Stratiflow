#include "Differentiation.h"

#include <Eigen/Dense>

MatrixXd SymmetriseMatrixDouble(const MatrixXd& in, BoundaryCondition bc)
{
    assert(in.rows() == in.cols());
    assert(in.rows()%2 == 1);
    int N = in.rows()-1;

    // matrices for each element (with imaged points)
    MatrixXd D21(N+1, N/2+1);
    MatrixXd D22(N+1, N/2+1);

    // unimaged points
    D21.col(0)   = in.col(N/2);
    D22.col(N/2) = in.col(N/2);

    for (int p=1; p<=N/2; p++)
    {
        if (bc == BoundaryCondition::Neumann)
        {
            D21.col(p)     = (in.col(N/2+p) + in.col(N/2-p));
            D22.col(N/2-p) = (in.col(N/2+p) + in.col(N/2-p));
        }
        else
        {
            D21.col(p)     = (in.col(N/2+p) - in.col(N/2-p));
            D22.col(N/2-p) = (-in.col(N/2+p) + in.col(N/2-p));
        }
    }

    MatrixXd out(N+1, N+1);
    out.setZero();

    out.row(0).head(N/2+1) += D21.row(N/2);
    out.row(N).tail(N/2+1) += D22.row(N/2);
    for (int p=1; p<=N/2; p++)
    {
        if (bc == BoundaryCondition::Neumann)
        {
            out.row(p).head(N/2+1) += D21.row(N/2+p) + D21.row(N/2-p);
            out.row(N-p).tail(N/2+1) += D22.row(N/2+p) + D22.row(N/2-p);
        }
        else
        {
            out.row(p).head(N/2+1) += D21.row(N/2+p) - D21.row(N/2-p);
            out.row(N-p).tail(N/2+1) += -D22.row(N/2+p) + D22.row(N/2-p);
        }
    }

    out.row(N/2) /= 4;

    for (int p=1; p<N/2; p++)
    {
        out.row(p) /= 2;
        out.row(N-p) /= 2;
    }

    return out;
}

MatrixXd SymmetriseMatrix(const MatrixXd& in, BoundaryCondition bc)
{
    assert(in.rows() == in.cols());
    assert(in.rows()%2 == 1);
    int N = in.rows()-1;

    // matrices for each element (with imaged points)
    MatrixXd D21(N+1, N/2+1);
    MatrixXd D22(N+1, N/2+1);

    // unimaged points
    D21.col(0)   = in.col(N/2);
    D22.col(N/2) = in.col(N/2);

    for (int p=1; p<=N/2; p++)
    {
        if (bc == BoundaryCondition::Neumann)
        {
            D21.col(p)     = (in.col(N/2+p) + in.col(N/2-p));
            D22.col(N/2-p) = (in.col(N/2+p) + in.col(N/2-p));
        }
        else
        {
            D21.col(p)     = (in.col(N/2+p) - in.col(N/2-p));
            D22.col(N/2-p) = (-in.col(N/2+p) + in.col(N/2-p));
        }
    }

    MatrixXd out(N+1, N+1);
    out.setZero();

    out.row(0).head(N/2+1) += D21.row(N/2);
    out.row(N).tail(N/2+1) += D22.row(N/2);
    for (int p=1; p<=N/2; p++)
    {
        out.row(p).head(N/2+1) += D21.row(N/2+p);
        out.row(N-p).tail(N/2+1) += D22.row(N/2-p);
    }

    out.row(N/2) /= 2;

    return out;
}

namespace
{
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
}

ArrayXd GaussLobattoNodes(int N)
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
    MatrixXd D = PolynomialDerivativeMatrix(GaussLobattoNodes(N-1))/L;
    return SymmetriseMatrix(D, originalBC);
}

MatrixXd ChebSecondDerivativeMatrix(BoundaryCondition bc, double L, int N)
{
    ArrayXd x = GaussLobattoNodes(N-1);
    MatrixXd D = PolynomialDerivativeMatrix(x);
    MatrixXd W = GaussLobattoWeights(x).matrix().asDiagonal();

    return SymmetriseMatrixDouble(-W.inverse()*D.transpose()*W*D/L/L, bc);
    // MatrixXd neumann = ChebDerivativeMatrix(BoundaryCondition::Neumann, L, N);
    // MatrixXd dirichlet = ChebDerivativeMatrix(BoundaryCondition::Dirichlet, L, N);

    // if (bc==BoundaryCondition::Neumann)
    // {
    //     return dirichlet*neumann;
    // }
    // else
    // {
    //     return neumann*dirichlet;
    // }
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
