#include "Field.h"

ArrayXd ChebPoints(unsigned int N, double L)
{
    assert(N%2 == 0);
    auto points = L*(1 - cos(ArrayXd::LinSpaced(N/2, pi/(2*N), pi/2-pi/(2*N))));
    ArrayXd ret(N);
    ret << -points.reverse(), points;
    return ret;
}