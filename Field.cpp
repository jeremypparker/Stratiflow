#include "Field.h"

ArrayX VerticalPoints(stratifloat L, int N)
{
    assert(N%4 == 0); // need for alignment
    ArrayX x =  ArrayX::LinSpaced(N, -1, 1);

    return -L*tan(x*1.3)/tan(1.3);
}

ArrayX FourierPoints(stratifloat L, int N)
{
    return ArrayX::LinSpaced(N, 0, L - L/static_cast<stratifloat>(N));
}
