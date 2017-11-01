#include "Field.h"

ArrayXd ChebPoints(unsigned int N, double L)
{
    assert(N%2 == 1);
    auto points = L*(1 - cos(ArrayXd::LinSpaced(N/2+1, 0, pi/2)));
    ArrayXd ret(N);
    ret << -points.reverse(), points.tail(N/2);
    return ret;
}