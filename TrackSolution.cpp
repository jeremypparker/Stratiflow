#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    Ri = std::stod(argv[1]);
    DumpParameters();

    StateVector state;
    if (argc>2)
    {
        state.LoadFromFile(argv[2]);
    }
    else
    {
        state.ExciteLowWavenumbers(0.001);
    }

    for (int n=0; n<3000; n++)
    {
        //state.PlotAll(std::to_string(n));

        std::cout << "Step " << n << " " << state.Energy() << " " << state.Enstrophy() << std::endl;

        state.FullEvolve(20, state, false, false);
    }
    state.SaveToFile("trackingresult");
}
