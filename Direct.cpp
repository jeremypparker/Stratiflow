#include "StateVector.h"
#include <chrono>

int main(int argc, char* argv[])
{
    PrintParameters();

    StateVector state;

    stratifloat energy = std::stof(argv[1]);

    stratifloat T = std::stof(argv[2]);

    if (argc == 3)
    {
        state.ExciteLowWavenumbers(energy);
    }
    else
    {
        state.LoadFromFile(argv[3]);
        state.Rescale(energy);
    }

    std::cout << state.Energy() << std::endl;

    stratifloat mixing = state.FullEvolve(T, state, true, true, true);
    SaveValueToFile(mixing, "mixing");
}
