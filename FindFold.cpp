#include "ExtendedStateVector.h"
#include "TrackBifurcation.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES fold finder" << std::endl;

    DumpParameters();

    ExtendedStateVector initial;
    initial.LoadFromFile(argv[1]);

    Bifurcation guess;
    guess.x1 = initial.x;
    guess.x2 = initial.x;
    guess.p = initial.p;

    StateVector noise;
    noise.Randomise(0.000001);
    guess.x1 += noise;
    guess.x2 -= noise;

    stratifloat delta = (guess.x2 - guess.x1).Energy();

    TrackBifurcation solver(delta);
    solver.Run(guess);

    guess.SaveToFile("final");
}
