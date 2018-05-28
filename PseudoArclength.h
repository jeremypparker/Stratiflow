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

        linearAboutStart = at;
        linearAboutEnd = result;

        result -= at;

        ExtendedStateVector displacement  = at;
        displacement -= x0;
        result.p = deltaS - v.Dot(displacement);

        return result;
    }

    virtual ExtendedStateVector EvalLinearised(const ExtendedStateVector& at) override
    {
        ExtendedStateVector result;
        ExtendedStateVector Gq;
        at.LinearEvolve(T, linearAboutStart, linearAboutEnd, Gq);
        result = at;
        result -= Gq;

        result.p = v.Dot(at);

        return result;
    }

private:
    ExtendedStateVector x0;
    ExtendedStateVector v;
    stratifloat deltaS;
};