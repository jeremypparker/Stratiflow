#include "FullState.h"

IMEXRK FullState::solver;

FullState operator+(const FullState& lhs, const FullState& rhs)
{
    FullState ret = lhs;
    ret += rhs;

    return ret;
}

FullState operator-(const FullState& lhs, const FullState& rhs)
{
    FullState ret = lhs;
    ret -= rhs;

    return ret;
}

FullState operator*(stratifloat scalar, const FullState& vector)
{
    FullState ret = vector;
    ret *= scalar;
    return ret;
}

FullState operator*(const FullState& vector, stratifloat scalar)
{
    return scalar*vector;
}