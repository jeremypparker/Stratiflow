#include "Field.h"

ArrayX VerticalPoints(stratifloat L, int N)
{
    assert(N%4 == 0); // need for alignment
    ArrayX ret =  L/(tan(ArrayX::LinSpaced(N, pi/(2*N), pi-pi/(2*N))));

    return ret;
}

ArrayX FourierPoints(stratifloat L, int N)
{
    return ArrayX::LinSpaced(N, 0, L - L/static_cast<stratifloat>(N));
}
