#include "ExtendedStateVector.h"
#include "TrackBifurcation.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES Pitchfork finder" << std::endl;

    DumpParameters();

    ExtendedStateVector initial;
    initial.LoadFromFile(argv[1]);

    Bifurcation guess;
    guess.x1 += initial.x;
    guess.x2 -= initial.x;
    guess.p = initial.p;

    stratifloat delta = (guess.x2 - guess.x1).Energy();

    TrackBifurcation solver(delta);
    solver.Run(guess);

    guess.SaveToFile("final");
}
