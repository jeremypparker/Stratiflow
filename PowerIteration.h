#pragma once
#include "StateVector.h"

template<typename VectorType>
class PowerIteration
{
public:
    PowerIteration()
    : q(K)
    , H(K, K-1)
    {
        H.setZero();
    }

    stratifloat Run(const VectorType& at, VectorType& result)
    {
        EvalFunction(at);

        VectorType real;
        VectorType imag;

        real.Randomise(0.1, true);
        imag.Randomise(0.1, true);

        stratifloat norm = sqrt(real.Norm2()+imag.Norm2());

        real *= 1/norm;
        imag *= 1/norm;

        stratifloat eigenval;

        for (int k=1; k<K; k++)
        {
            // PowerIteration Algorithm

            VectorType realNew = EvalLinearised(real);
            VectorType imagNew = EvalLinearised(imag);

            eigenval = real.Dot(realNew) + imag.Dot(imagNew);

            real = realNew;
            imag = imagNew;

            stratifloat norm = sqrt(real.Norm2()+imag.Norm2());

            real *= 1/norm;
            imag *= 1/norm;
        }

        real.PlotAll("eigReal");
        imag.PlotAll("eigImag");
        real.SaveToFile("eigReal");
        imag.SaveToFile("eigImag");

        std::cout << "Final eigenvalue: " << std::endl << eigenval << std::endl;

        return maxCoeff;
    }

protected:
    virtual VectorType EvalFunction(const VectorType& at) = 0;
    virtual VectorType EvalLinearised(const VectorType& at) = 0;

    stratifloat T = 11; // time interval for integration

    VectorType linearAboutStart;

public:
    int K = 1024; // max iterations
};

class BasicPowerIteration : public PowerIteration<StateVector>
{
    virtual StateVector EvalFunction(const StateVector& at) override
    {
        StateVector result;
        at.FullEvolve(T, result, false);

        linearAboutStart = at;

        return result;
    }

    virtual StateVector EvalLinearised(const StateVector& at) override
    {
        StateVector result;
        at.LinearEvolve(T, linearAboutStart, result);
        return result;
    }
};
