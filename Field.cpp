#include "Field.h"

ArrayX VerticalPoints(stratifloat L, int N)
{
    assert(N%4 == 0); // need for alignment
    ArrayX ret =  L/(tan(ArrayX::LinSpaced(N, 0, pi)));

    // fix infinities
    ret(0) = 100000000000;
    ret(N-1) = -100000000000;

    return ret;
}

ArrayX FourierPoints(stratifloat L, int N)
{
    return ArrayX::LinSpaced(N, 0, L - L/static_cast<stratifloat>(N));
}
