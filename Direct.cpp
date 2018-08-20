#include "StateVector.h"
#include <chrono>

int main(int argc, char* argv[])
{
    PrintParameters();

    StateVector state;

    stratifloat energy = std::stof(argv[1]);

    if (argc == 2)
    {
        state.Randomise(energy);
    }
    else
    {
        state.LoadFromFile(argv[2]);
        state.Rescale(energy);
    }

    std::cout << state.Energy() << std::endl;

    stratifloat mixing = state.FullEvolve(25, state, true, true, true);
    SaveValueToFile(mixing, "mixing");
}
