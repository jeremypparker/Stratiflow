#include "Field.h"

ArrayX VerticalPoints(stratifloat L, int N)
{
    ArrayX z = VerticalPointsFractional(L,N);
    ArrayX ret(N);

    ret.head(N-1).tail(N-2) = 0.5*(z.head(N-2)+z.tail(N-2));
    ret(0) = 2*ret(1) - ret(2);
    ret(N-1) = 2*ret(N-2) - ret(N-3);

    return ret;
}

ArrayX VerticalPointsFractional(stratifloat L, int N)
{
    assert(N%4 == 0); // need for alignment
    ArrayX x =  ArrayX::LinSpaced(N-1, -1, 1);

    return -L*tan(x*1.3)/tan(1.3);
}

ArrayX dz(stratifloat L, int N)
{
    ArrayX z = VerticalPoints(L, N);

    return z.head(N-1) - z.tail(N-1);
}

ArrayX dzFractional(stratifloat L, int N)
{
    ArrayX zFractional = VerticalPointsFractional(L, N);

    return zFractional.head(N-2) - zFractional.tail(N-2);
}

ArrayX FourierPoints(stratifloat L, int N)
{
    return ArrayX::LinSpaced(N, 0, L - L/static_cast<stratifloat>(N));
}
