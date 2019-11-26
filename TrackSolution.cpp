#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    Ri = std::stod(argv[1]);
    Pr = std::stod(argv[2]);
    Pe = Re*Pr;
    StateVector::ResetForParams();
    DumpParameters();

    StateVector state;
    if (argc>3)
    {
        state.LoadFromFile(argv[3]);
    }

    if (argc>4)
    {
        StateVector state2;
        state2.LoadFromFile(argv[4]);

        stratifloat mult=1;

        if (argc>5)
        {
            mult = std::stod(argv[5]);
        }

        state = state2 + mult*(state2-state);
    }

    StateVector perturbation;
    perturbation.ExciteLowWavenumbers(0.5);

    if (argc == 3)
        state += perturbation;


    // state.MakeMode2();

    stratifloat timestep = 10;
    for (int n=0; n<3000; n++)
    {
        //state.PlotAll(std::to_string(n));

        std::cout << "Step " << n << " " << state.Energy() << " " << state.Enstrophy() << std::endl;

        state.FullEvolve(timestep, state, false, false);

        if(n%10 == 0)
        {
            state.AddBackground();
            state.PlotAll("state");
            state.RemoveBackground();

            state.SaveToFile("trackingresult");
        }
    }
}
