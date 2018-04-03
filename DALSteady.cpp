#include "StateVector.h"

stratifloat Residual(const StateVector& initialDirect, const StateVector& finalDirect)
{
    StateVector diff = finalDirect - initialDirect;
    return diff.Energy();
}

void Optimise(stratifloat& epsilon,
              StateVector& initialDirect,
              const StateVector& finalDirect,
              const StateVector& finalAdjoint)
{
    initialDirect += epsilon*(finalAdjoint + finalDirect - initialDirect);
}

int main()
{
    stratifloat T = 10;

    StateVector initialDirectPrevious;
    StateVector finalDirectPrevious;
    StateVector initialDirect;
    StateVector finalDirect;

    StateVector initialAdjoint;
    StateVector finalAdjoint;

    initialDirect.Randomise(0.001);

    stratifloat epsilon = 0.01; // initial gradient descent step size

    stratifloat bestResidual;
    int step = 0;
    while (true)
    {
        step++;

        // first do forward pass
        initialDirect.FullEvolve(T, finalDirect, true);

        stratifloat newResidual = Residual(initialDirect, finalDirect);

        std::cout << "Step " << step << ", residual=" << newResidual << std::endl;

        if (newResidual < bestResidual || step == 1)
        {
            bestResidual = newResidual;
            initialDirectPrevious = initialDirect;
            finalDirectPrevious = finalDirect;

            // now do adjoint
            initialAdjoint = initialDirect - finalDirect;
            initialAdjoint.AdjointEvolve(T, finalAdjoint);
        }
        else
        {
            // IC not good enough, retry with smaller epsilon
            epsilon /= 2;

            std::cout << "Epsilon: " << epsilon << std::endl;

            initialDirect = initialDirectPrevious;
            finalDirect = finalDirectPrevious;
        }

        Optimise(epsilon, initialDirect, finalDirect, finalAdjoint);
    }
}