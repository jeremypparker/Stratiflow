#include "StateVector.h"
#include "OrrSommerfeld.h"

class DALSteady
{
public:
    DALSteady(const StateVector& IC)
    : epsilon(1)
    , T(10)
    , initialDirect(IC)
    {
    }

    void Run()
    {
        MakeCleanDir("ICs");

        stratifloat bestResidual;
        int step = 0;
        while (true)
        {
            step++;

            // store the state
            initialDirect.SaveToFile("ICs/"+std::to_string(step)+".fields");

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


    stratifloat T; // target time
    stratifloat epsilon; // gradient descent step size

    StateVector initialDirectPrevious;
    StateVector finalDirectPrevious;
    StateVector initialDirect;
    StateVector finalDirect;

    StateVector initialAdjoint;
    StateVector finalAdjoint;
};

int main(int argc, char *argv[])
{
    DumpParameters();

    StateVector initial;

    if (argc == 2)
    {
        initial.LoadFromFile(argv[1]);
    }
    else
    {
        EigenModes(2*pi/L1, initial.u1, initial.u2, initial.u3, initial.b);

        initial.Rescale(0.0001);
    }

    DALSteady dal(initial);

    dal.Run();
}