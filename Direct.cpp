#include "StateVector.h"
#include <chrono>

int main(int argc, char* argv[])
{
    PrintParameters();

    StateVector state;

    if (argc == 1)
    {
        state.Randomise(0.01);
    }
    else
    {
        state.LoadFromFile(argv[1]);
    }

    state.FullEvolve(12, state, true, true);
}
