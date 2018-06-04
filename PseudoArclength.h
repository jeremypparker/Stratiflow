#pragma once

#include "ExtendedStateVector.h"
#include "NewtonKrylov.h"

class PseudoArclengthContinuation : public NewtonKrylov<ExtendedStateVector>
{
public:
    PseudoArclengthContinuation(ExtendedStateVector x0, ExtendedStateVector v, stratifloat deltaS)
    : x0(x0)
    , v(v)
    , deltaS(deltaS)
    {}

protected:
    virtual ExtendedStateVector EvalFunction(const ExtendedStateVector& at) override
    {
        ExtendedStateVector result;
        at.FullEvolve(T, result, false);

        result -= at;

        ExtendedStateVector displacement  = at;
        displacement -= x0;

        Ri = at.p;
        result.p = deltaS - v.Dot(displacement);

        return result;
    }

private:
    ExtendedStateVector x0;
    ExtendedStateVector v;
    stratifloat deltaS;
};