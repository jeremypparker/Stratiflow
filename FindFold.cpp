#include "ExtendedStateVector.h"
#include "TrackBifurcation.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES fold finder" << std::endl;

    if (argc>3)
    {
        LoadParameters(argv[3]);
        StateVector::ResetForParams();
    }
    DumpParameters();

    ExtendedStateVector x1, x2;
    x1.LoadFromFile(argv[1]);
    x2.LoadFromFile(argv[2]);

    Bifurcation guess;
    guess.x1 = x1.x;
    guess.x2 = x2.x;
    guess.p = (x1.p+x2.p)/2;

    stratifloat delta = (guess.x2 - guess.x1).Energy();

    TrackBifurcation solver(delta);
    solver.Run(guess);

    guess.SaveToFile("final");
}
