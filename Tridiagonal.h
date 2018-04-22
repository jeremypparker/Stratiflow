#include "Eigen.h"

template<typename T, int N>
class Tridiagonal
{
public:
    Tridiagonal()
    {
        a.setZero();
        b.setZero();
        C.setZero();
    }

    void compute(const MatrixX& A)
    {
        assert(A.rows() == N && A.cols() == N);

        a.tail(N-1) = A.diagonal(-1);
        b = A.diagonal(0);
        C.head(N-1) = A.diagonal(1);

        C(0) = C(0)/b(0);
        for (int j=1; j<N-1; j++)
        {
            C(j) = C(j)/(b(j)-a(j)*C(j-1));
        }
    }

    template<typename R>
    Matrix<R, N, 1> solve(const Matrix<R, N3, 1>& d) const
    {
        Matrix<R, N, 1> x;

        Matrix<R, N, 1> D;

        // from wikipedia, Thomas algorithm

        // forward pass
        D(0) = d(0)/b(0);
        for (int j=1; j<N; j++)
        {
            D(j) = (d(j) - a(j)*D(j-1))/(b(j)-a(j)*C(j-1));
        }

        // backward pass
        x(N-1) = D(N-1);
        for (int j=N-2; j>=0; j--)
        {
            x(j) = D(j) - C(j)*x(j+1);
        }

        return x;
    }

private:
    // as per wikipedia
    Matrix<T, N, 1> a; // lower
    Matrix<T, N, 1> b; // diagonal
    Matrix<T, N, 1> C; // transformed upper
};