#include "BasicNewtonKrylov.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES" << std::endl;

    Ri = std::stod(argv[1]);
    DumpParameters();

    StateVector guess;

    if (argc == 6)
    {
        StateVector x1;
        StateVector x2;
        x1.LoadFromFile(argv[2]);
        x2.LoadFromFile(argv[3]);

        stratifloat p1 = std::stof(argv[4]);
        stratifloat p2 = std::stof(argv[5]);

        StateVector gradient = x2;
        gradient -= x1;
        gradient *= 1/(p2-p1);

        guess = x2;
        guess.MulAdd(Ri-p2, gradient);
    }
    else if (argc == 3)
    {
        if (FileExists(argv[2]))
        {
            std::cout << "Interpretting " << argv[2] << " as path to guess" << std::endl;
            // guess.LoadAndInterpolate<256,1,384>(argv[2]);
            guess.LoadFromFile(argv[2]);
        }
    }
    else
    {
        std::cerr << "Incorrect number of args" << std::endl;
    }

    RemoveAverage(guess.u1, L3);
    RemoveAverage(guess.b, L3);

    BasicNewtonKrylov solver;

    solver.Run(guess);

    guess.SaveToFile("final");
}
