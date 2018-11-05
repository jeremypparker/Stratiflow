#include "ExtendedStateVector.h"
#include "HopfBifurcation.h"
#include "NewtonKrylov.h"

class FindPeriodic : public NewtonKrylov<ExtendedStateVector>
{
public:
    StateVector offset;
    StateVector normal;

    virtual void EnforceConstraints(ExtendedStateVector& at)
    {
        at.x -= (normal.Dot(at.x-offset)/normal.Norm2())*normal;
    }
private:
    virtual ExtendedStateVector EvalFunction(const ExtendedStateVector& at) override
    {
        ExtendedStateVector result;

        at.x.FullEvolve(at.p, result.x, false, true);

        result.p = normal.Dot(result.x - offset);
        result.x -= at.x;

        std::cout << result.p*result.p << " " << result.x.Norm2() << std::endl;

        return result;
    }
};

int main(int argc, char *argv[])
{
    Ri = std::stof(argv[1]);
    DumpParameters();
    StateVector::ResetForParams();

    HopfBifurcation hopf;
    hopf.LoadFromFile(argv[2]);

    stratifloat amountToAdd = std::stod(argv[3]);

    ExtendedStateVector guess;

    if(argc==5)
    {
        guess.LoadFromFile(argv[4]);
    }
    else
    {
        guess.x = hopf.x + amountToAdd*hopf.v1;
        guess.p = 2*pi*11/hopf.theta;
    }

    FindPeriodic solver;

    solver.offset = hopf.x;
    solver.normal = hopf.v2;

    solver.EnforceConstraints(guess);

    solver.Run(guess);

    guess.SaveToFile("final");
}
