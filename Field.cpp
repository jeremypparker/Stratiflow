#include "Field.h"
#include "Differentiation.h"

ArrayXd ChebPoints(unsigned int N, double L)
{
    assert(N%2 == 1);
    auto points = ChebyshevGaussLobattoNodes(N-1);
    ArrayXd ret(N);
    ret << points.segment(N/2, N/2) - 1, points.head(N/2 + 1) + 1;
    return ret * L;
}