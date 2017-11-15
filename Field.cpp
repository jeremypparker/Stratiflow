#include "Field.h"

ArrayXd VerticalPoints(double L, int N)
{
    assert(N%2 == 1);
    ArrayXd ret =  L/(tan(ArrayXd::LinSpaced(N, 0, pi)));

    // fix infinities
    ret(0) = 100000000000;
    ret(N-1) = -100000000000;

    return ret;
}

ArrayXd FourierPoints(double L, int N)
{
    return ArrayXd::LinSpaced(N, 0, L - L/static_cast<double>(N));
}
