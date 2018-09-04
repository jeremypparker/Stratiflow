#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    DumpParameters();

    // load a state
    ExtendedStateVector state;

    if (argc > 1)
    {
        state.LoadFromFile(argv[1]);
        Ri = state.p;

        // remove means
        RemoveAverage(state.x.u1, L3);
        RemoveAverage(state.x.b, L3);
    }
    else
    {
        state.p = Ri;
        state.x.ExciteLowWavenumbers(0.01);
    }

    if (argc == 3)
    {
        // load eigenmode
        StateVector eigenMode;
        eigenMode.LoadFromFile(argv[2]);

        stratifloat delta = 0.0001;

        // add the eigenmode
        state.x.MulAdd(delta, eigenMode);
    }

    // follow the trajectory
    ExtendedStateVector result;
    state.FullEvolve(10000, result, false, true);

    result.SaveToFile("result");
}
