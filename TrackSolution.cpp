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
    }
    else
    {
        state.p = Ri;
        state.x.Randomise(0.001);

        // remove means
        state.x.u1.stack(0,0) -= IntegrateAllSpace(state.x.u1, L1, L2, L3)/L1/L2/L3/2;
        state.x.b.stack(0,0) -= IntegrateAllSpace(state.x.b, L1, L2, L3)/L1/L2/L3/2;
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
    state.FullEvolve(13000, result, true, true);

    result.SaveToFile("result");
}
