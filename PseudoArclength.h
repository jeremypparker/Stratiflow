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

    virtual void EnforceConstraints(ExtendedStateVector& at) override
    {
        at.x.RemovePhaseShift();
        RemoveAverage(at.x.u1);
        RemoveAverage(at.x.b);
    }

protected:
    virtual ExtendedStateVector EvalFunction(const ExtendedStateVector& at) override
    {
        ExtendedStateVector result;
        at.FullEvolve(T, result, false, false);

        result -= at;

        ExtendedStateVector displacement  = at;
        displacement -= x0;

        flowParams.Ri = at.p;
        result.p = deltaS - v.Dot(displacement);

        return result;
    }

private:
    ExtendedStateVector x0;
    ExtendedStateVector v;
    stratifloat deltaS;
};