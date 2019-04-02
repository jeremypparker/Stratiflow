#include "BasicNewtonKrylov.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES" << std::endl;

    Ri = std::stod(argv[1]);

    if(argc>3)
    {
        Re = std::stod(argv[3]);
        Pe = Re*Pr;
        StateVector::ResetForParams();
    }
    DumpParameters();

    StateVector guess;
    guess.LoadFromFile(argv[2]);

    RemoveAverage(guess.u1, L3);
    RemoveAverage(guess.b, L3);

    BasicNewtonKrylov solver;

    solver.Run(guess);

    guess.SaveToFile("final");
}
