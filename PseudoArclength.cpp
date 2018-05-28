#include "PseudoArclength.h"


int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES Pseudo-Arclength continuation" << std::endl;

    if (argc==5)
    {
        L3 = std::stof(argv[4]);
    }

    DumpParameters();
    StateVector::ResetForParams();


    stratifloat delta = std::stof(argv[3]);

    ExtendedStateVector x1;
    ExtendedStateVector x2;
    x1.LoadFromFile(argv[1]);
    x2.LoadFromFile(argv[2]);

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

    PseudoArclengthContinuation solver(x2, v, delta);
    solver.Run(guess);

    guess.SaveToFile("final");
}
