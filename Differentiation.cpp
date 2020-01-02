
#include "Differentiation.h"
#include "Eigen.h"
#include "Field.h"
#include <iomanip>

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

    if(dimension == 3)
    {
        return ret.head(N/2 +1);
    }
    else
    {
        return ret;
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
