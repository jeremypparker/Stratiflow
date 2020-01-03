#include "BasicNewtonKrylov.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES" << std::endl;

    flowParams.Ri = std::stod(argv[1]);

    StateVector guess;

    if (argc == 7)
    {
        flowParams.Re = 1000;
        flowParams.Pr = std::stof(argv[2]);
        StateVector::ResetForParams();

        StateVector x1;
        StateVector x2;
        x1.LoadFromFile(argv[3]);
        x2.LoadFromFile(argv[4]);

        stratifloat p1 = std::stof(argv[5]);
        stratifloat p2 = std::stof(argv[6]);

        StateVector gradient = x2;
        gradient -= x1;
        gradient *= 1/(p2-p1);

        guess = x2;
        guess.MulAdd(flowParams.Pr-p2, gradient);
    }
    else
    {
        if(argc>3)
        {
            flowParams.Re = 1000;
            flowParams.Pr = std::stof(argv[3]);
            StateVector::ResetForParams();
        }

        guess.LoadFromFile(argv[2]);
    }



    DumpParameters();


    RemoveAverage(guess.u1);
    RemoveAverage(guess.b);

    BasicNewtonKrylov solver;

    solver.Run(guess);

    guess.SaveToFile("final");
}
