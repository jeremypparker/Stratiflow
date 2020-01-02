#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    flowParams.Ri = std::stod(argv[1]);
    flowParams.Pr = std::stod(argv[2]);
    StateVector::ResetForParams();
    DumpParameters();

    StateVector state;
    if (argc>3)
    {
        state.LoadFromFile(argv[3]);
    }

    if (argc>4)
    {
        StateVector state2;
        state2.LoadFromFile(argv[4]);

        stratifloat mult=1;

        if (argc>5)
        {
            mult = std::stod(argv[5]);
        }

        state = state2 + mult*(state2-state);
    }

    StateVector perturbation;
    perturbation.ExciteLowWavenumbers(0.5);

    if (argc == 3)
        state += perturbation;

    state.FullEvolve(100, state, true, true);

}
