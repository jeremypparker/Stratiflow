#include "Field.h"

ArrayX FourierPoints(stratifloat L, int N)
{
    return ArrayX::LinSpaced(N, 0, L - L/static_cast<stratifloat>(N));
}


