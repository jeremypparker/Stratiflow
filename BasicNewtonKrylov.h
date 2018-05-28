#pragma once

#include "StateVector.h"
#include "NewtonKrylov.h"

class BasicNewtonKrylov : public NewtonKrylov<StateVector>
{
    virtual StateVector EvalFunction(const StateVector& at) override
    {
        StateVector result;
        at.FullEvolve(T, result, false);

        linearAboutStart = at;
        linearAboutEnd = result;

        result -= at;

        return result;
    }

    virtual StateVector EvalLinearised(const StateVector& at) override
    {
        StateVector result;
        StateVector Gq;
        at.LinearEvolve(T, linearAboutStart, linearAboutEnd, Gq);
        result = at;
        result -= Gq;

        return result;
    }
};