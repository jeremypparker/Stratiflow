#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    DumpParameters();

    // load a state
    ExtendedStateVector state;
    state.LoadFromFile(argv[1]);
    Ri = state.p;

    // find eigenmode
    BasicArnoldi eigenValueSolver;
    StateVector eigenMode;
    stratifloat growth = eigenValueSolver.Run(state.x, eigenMode, false);
    std::cout << "Eigenmode growth rate: " << growth << std::endl;

    stratifloat delta = 0.001;

    // add the eigenmode
    state.x.MulAdd(delta, eigenMode);

    // follow the trajectory
    ExtendedStateVector result;
    state.FullEvolve(4000, result, true, true);

    result.SaveToFile("result+");

    // subtract the eigenmode
    state.x.MulAdd(-2*delta, eigenMode);

    // follow the trajectory
    state.FullEvolve(4000, result, true, true);

    result.SaveToFile("result-");
}
