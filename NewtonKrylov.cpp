#include "BasicNewtonKrylov.h"

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES" << std::endl;

    Ri = std::stod(argv[1]);

    StateVector guess;

    if (argc == 7)
    {
        Re = 1000;
        Pr = std::stof(argv[2]);
        Pe = Re*Pr;
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
        guess.MulAdd(Pr-p2, gradient);
    }
    else
    {
        if(argc>3)
        {
            Re = 1000;
            Pr = std::stof(argv[3]);
            Pe = Re*Pr;
            StateVector::ResetForParams();
        }

        guess.LoadFromFile(argv[2]);
    }



    DumpParameters();


    RemoveAverage(guess.u1, L3);
    RemoveAverage(guess.b, L3);

    BasicNewtonKrylov solver;

    solver.Run(guess);

    guess.SaveToFile("final");
}
