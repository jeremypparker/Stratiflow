#pragma once

#include "StateVector.h"
#include "NewtonKrylov.h"

class BasicNewtonKrylov : public NewtonKrylov<StateVector>
{
    virtual StateVector EvalFunction(const StateVector& at) override
    {
        StateVector result;
        at.FullEvolve(T, result, false);
        result -= at;

        return result;
    }
};