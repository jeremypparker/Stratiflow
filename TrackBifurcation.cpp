#include "TrackBifurcation.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES bifurcation tracker" << std::endl;

    L3 = std::stof(argv[1]);
    DumpParameters();

    StateVector::ResetForParams();

    Bifurcation guess;

    if (argc == 6)
    {
        stratifloat L31 = std::stof(argv[4]);
        stratifloat L32 = std::stof(argv[5]);

        Bifurcation X1, X2;
        X1.LoadFromFile(argv[2]);
        X2.LoadFromFile(argv[3]);

        Bifurcation grad = X2;
        grad -= X1;
        grad *= 1/(L32-L31);

        guess = X2;
        guess.MulAdd(L3 - L32, grad);
    }
    else
    {
        guess.LoadFromFile(argv[2]);
    }

    stratifloat delta = (guess.x2 - guess.x1).Energy();

    TrackBifurcation solver(delta);
    solver.Run(guess);

    guess.SaveToFile("final");
}
