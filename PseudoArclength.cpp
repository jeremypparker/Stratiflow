#include "PseudoArclength.h"


int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES Pseudo-Arclength continuation" << std::endl;

    if (argc==5)
    {
        flowParams.Pr = std::stod(argv[4]);
        StateVector::ResetForParams();
    }

    DumpParameters();

    stratifloat delta = std::stof(argv[3]);

    ExtendedStateVector x1;
    ExtendedStateVector x2;
    x1.LoadFromFile(argv[1]);
    x2.LoadFromFile(argv[2]);

    flowParams.Ri = x2.p;

    x1.x.RemovePhaseShift();
    RemoveAverage(x1.x.u1, flowParams.L3);
    RemoveAverage(x1.x.b, flowParams.L3);
    x2.x.RemovePhaseShift();
    RemoveAverage(x2.x.u1, flowParams.L3);
    RemoveAverage(x2.x.b, flowParams.L3);

    // see stationarystates.pdf
    ExtendedStateVector v;

    if (x2.p == x1.p) // special case - vertical gradient
    {
        v.x = x2.x;
        v.x -= x1.x;

        v.x *= 1/v.x.Norm();

        v.p = 0;
    }
    else
    {
        v.x = x2.x;
        v.x -= x1.x;
        v.x *= 1/(x2.p - x1.p);

        v.p = 1/sqrt(1 + v.x.Norm2());
        v.x *= v.p;

        // take into account choice of sign of sqrt
        if (x2.p < x1.p)
        {
            v *= -1.0;
        }
    }

    ExtendedStateVector guess = x2;
    guess.MulAdd(delta, v);

    std::cout << "Guess averages: " << IntegrateAllSpace(guess.x.b,1,1,flowParams.L3) << " " << IntegrateAllSpace(guess.x.u1,1,1,flowParams.L3) << std::endl;

    PseudoArclengthContinuation solver(x2, v, delta);

    solver.EnforceConstraints(guess);
    solver.Run(guess);

    guess.SaveToFile("final");
}
