#include "Field.h"

ArrayXd VerticalPoints(double L, int N)
{
    return ArrayXd::LinSpaced(N, -L, L);
}

ArrayXd FourierPoints(double L, int N)
{
    return ArrayXd::LinSpaced(N, 0, L - L/static_cast<double>(N));
}
