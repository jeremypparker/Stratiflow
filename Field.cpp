#include "Field.h"

ArrayXf VerticalPoints(float L, int N)
{
    assert(N%4 == 0); // need for alignment
    ArrayXf ret =  L/(tan(ArrayXf::LinSpaced(N, 0, pi)));

    // fix infinities
    ret(0) = 100000000000;
    ret(N-1) = -100000000000;

    return ret;
}

ArrayXf FourierPoints(float L, int N)
{
    return ArrayXf::LinSpaced(N, 0, L - L/static_cast<float>(N));
}
