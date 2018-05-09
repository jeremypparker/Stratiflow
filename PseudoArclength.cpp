#include "ExtendedStateVector.h"
#include "NewtonKrylov.h"

class PseudoArclengthContinuation : public NewtonKrylov<ExtendedStateVector>
{
public:
    PseudoArclengthContinuation(ExtendedStateVector x0, ExtendedStateVector v, stratifloat deltaS)
    : x0(x0)
    , v(v)
    , deltaS(deltaS)
    {}

protected:
    virtual ExtendedStateVector EvalFunction(const ExtendedStateVector& at) override
    {
        ExtendedStateVector result;
        at.FullEvolve(T, result, false);

        linearAboutStart = at;
        linearAboutEnd = result;

        result -= at;

        ExtendedStateVector displacement  = at;
        displacement -= x0;
        result.p = deltaS - v.Dot(displacement);

        return result;
    }

    virtual ExtendedStateVector EvalLinearised(const ExtendedStateVector& at) override
    {
        ExtendedStateVector result;
        ExtendedStateVector Gq;
        at.LinearEvolve(T, linearAboutStart, linearAboutEnd, Gq);
        result = at;
        result -= Gq;

        result.p = v.Dot(at);

        return result;
    }

private:
    ExtendedStateVector x0;
    ExtendedStateVector v;
    stratifloat deltaS;
};

int main(int argc, char *argv[])
{
    std::cout << "STRATIFLOW Newton-GMRES Pseudo-Arclength continuation" << std::endl;

    DumpParameters();

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
