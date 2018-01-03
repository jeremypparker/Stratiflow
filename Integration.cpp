#include "Integration.h"

stratifloat SolveQuadratic(stratifloat a, stratifloat b, stratifloat c, bool positiveSign)
{
    stratifloat discriminant = b*b - 4*a*c;

    if (discriminant < 0)
    {
        return 0;
    }

    if (positiveSign)
    {
        return (-b+sqrt(discriminant))/(2*a);
    }
    else
    {
        return (-b-sqrt(discriminant))/(2*a);
    }
}