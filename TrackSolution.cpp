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
        NeumannNodal B;
        B.SetValue([](stratifloat x, stratifloat y, stratifloat z)
        {
            return cos(x/2)*sin(y*2)/cosh(z);
        }, L1, L2, L3);

        B.ToModal(state.x.b);
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
    state.FullEvolve(200, result, false, true);

    result.SaveToFile("result");
}
