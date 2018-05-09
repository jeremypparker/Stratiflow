#include "ExtendedStateVector.h"
#include "TrackBifurcation.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES fold finder" << std::endl;

    DumpParameters();

    ExtendedStateVector x1, x2;
    x1.LoadFromFile(argv[1]);
    x2.LoadFromFile(argv[2]);

    Bifurcation guess;
    guess.x1 = x1.x;
    guess.x2 = x2.x;
    guess.p = x1.p;

    stratifloat delta = 1e-5;

    TrackBifurcation solver(delta);
    solver.Run(guess);

    guess.SaveToFile("final");
}
