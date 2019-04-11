#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    stratifloat E = std::stod(argv[1]);
    stratifloat T = std::stod(argv[2]);
    DumpParameters();

    StateVector state;
    if (argc>3)
    {
        state.LoadFromFile(argv[3]);
    }
    else
    {
         BasicArnoldi solver;

        StateVector zero;
        solver.Run(zero, state);
    }

    state.Rescale(E);

    std::cout << "Energy " << state.Energy() << std::endl;
    stratifloat mixing = state.FullEvolve(T, state, false, true, true);
    std::cout << "Mixing " << mixing << std::endl;
}
