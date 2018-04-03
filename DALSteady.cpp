#include "StateVector.h"

class DALSteady
{
public:
    DALSteady(const StateVector& IC)
    : epsilon(0.1)
    , T(10)
    , initialDirect(IC)
    {
    }

    stratifloat Residual() const
    {
        StateVector diff = finalDirect - initialDirect;
        return diff.Energy();
    }

    void Optimise()
    {
        initialDirect += epsilon*(finalAdjoint + finalDirect - initialDirect);
    }

    void AdjointIC()
    {
        initialAdjoint = initialDirect - finalDirect;
    }

    void Run()
    {
        stratifloat bestResidual;
        int step = 0;
        while (true)
        {
            step++;

            // first do forward pass
            initialDirect.FullEvolve(T, finalDirect, true);

            stratifloat newResidual = Residual();

            std::cout << "Step " << step << ", residual=" << newResidual << std::endl;

            if (newResidual < bestResidual || step == 1)
            {
                bestResidual = newResidual;
                initialDirectPrevious = initialDirect;
                finalDirectPrevious = finalDirect;

                // now do adjoint
                AdjointIC();
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

            Optimise();
        }
    }

private:
    stratifloat T; // target time
    stratifloat epsilon; // gradient descent step size

    StateVector initialDirectPrevious;
    StateVector finalDirectPrevious;
    StateVector initialDirect;
    StateVector finalDirect;

    StateVector initialAdjoint;
    StateVector finalAdjoint;
};

int main()
{
    StateVector initial;
    initial.Randomise(0.001);

    DALSteady dal(initial);

    dal.Run();
}