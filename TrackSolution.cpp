#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    DumpParameters();

    // load a state
    ExtendedStateVector state;
    state.LoadFromFile(argv[1]);
    Ri = state.p;

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
    state.FullEvolve(13000, result, true, true);

    result.SaveToFile("result");
}
