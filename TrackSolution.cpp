#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    Ri = std::stod(argv[1]);

    DumpParameters();

    StateVector state;
    state.ExciteLowWavenumbers(0.001);

    for (int n=0; n<500; n++)
    {
        state.PlotAll(std::to_string(n));

        std::cout << "Step " << n << " " << state.Energy() << " " << state.Enstrophy() << std::endl;

        state.FullEvolve(20, state, false, false);
    }
    state.SaveToFile("trackingresult");
}
