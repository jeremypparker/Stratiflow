#include "Field.h"

ArrayX VerticalPoints(stratifloat L, int N)
{
    stratifloat cs = 2;

    assert(N%4 == 0); // need for alignment

    ArrayX x =  ArrayX::LinSpaced(N-1, -1, 1);

    ArrayX ret(N);
    ret.tail(N-1) = L*(cs*x*x*x + x)/(cs+1);

    stratifloat offsetPoint = (ret(1) + ret(2))/2;

    ret.tail(N-1) *= -L/offsetPoint;

    return ret;
}

ArrayX VerticalPointsFractional(stratifloat L, int N)
{
    ArrayX z = VerticalPoints(L,N);
    ArrayX ret(N);

    ret.segment(1, N-2) = 0.5*(z.segment(1,N-2)+z.tail(N-2));
    ret(0) = 2*ret(1)-ret(2);
    ret(N-1) = 2*ret(N-2)-ret(N-3);

    return ret;
}

ArrayX dz(stratifloat L, int N)
{
    ArrayX zFractional = VerticalPointsFractional(L, N);

    ArrayX ret(N);
    ret.tail(N-1) = zFractional.tail(N-1) - zFractional.head(N-1);

    return ret;
}

ArrayX dzFractional(stratifloat L, int N)
{
    ArrayX z = VerticalPoints(L, N);

    ArrayX ret(N);
    ret.segment(1,N-2) = z.tail(N-2) - z.segment(1,N-2);

    ret(N-1) = ret(N-2);
    ret(0) = ret(1);

    return ret;
}

ArrayX FourierPoints(stratifloat L, int N)
{
    return ArrayX::LinSpaced(N, 0, L - L/static_cast<stratifloat>(N));
}
