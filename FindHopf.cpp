#include "HopfBifurcation.h"
#include "NewtonKrylov.h"

class FindHopf : public NewtonKrylov<HopfBifurcation>
{
public:
    stratifloat weight = 1;

    virtual void EnforceConstraints(HopfBifurcation& at)
    {
        Ri = at.p;

        // ensure unit eigenvector
        stratifloat totalEnergy = at.v1.Energy() + at.v2.Energy();

        StateVector a = sqrt(weight/totalEnergy)*at.v1;
        StateVector b = sqrt(weight/totalEnergy)*at.v2;

        // now rotate so that ||v1|| = ||v2||
        stratifloat theta = 0.5*atan((a.Energy()-b.Energy())/a.Dot(b));

        at.v1 = cos(theta)*a - sin(theta)*b;
        at.v2 = sin(theta)*a + cos(theta)*b;
    }
private:
    virtual HopfBifurcation EvalFunction(const HopfBifurcation& at) override
    {
        HopfBifurcation result;

        Ri = at.p;
        at.x.FullEvolve(T, result.x, false, false);
        at.v1.LinearEvolve(T, at.x, result.v1);
        at.v2.LinearEvolve(T, at.x, result.v2);

        result.x -= at.x;
        result.v1.MulAdd(-at.lambda1, at.v1);
        result.v1.MulAdd(at.lambda2, at.v2);
        result.v2.MulAdd(-at.lambda2, at.v1);
        result.v2.MulAdd(-at.lambda1, at.v2);

        result.lambda1 = at.v1.Energy() - at.v2.Energy();
        result.lambda2 = at.lambda1*at.lambda1 + at.lambda2*at.lambda2 - 1;

        result.p = at.v1.Energy() + at.v2.Energy() - weight;

        std::cout << at.v1.Norm() << " " << at.v2.Norm() << std::endl;

        std::cout << result.x.Norm2() << " "
                   << result.v1.Norm2() << " "
                   << result.v2.Norm2() << " "
                   << result.lambda1*result.lambda1 << " "
                   << result.lambda2*result.lambda2 << " "
                   << result.p*result.p << std::endl;

        return result;
    }
};

#include "Arnoldi.h"
#include "ExtendedStateVector.h"

int main(int argc, char *argv[])
{
    Re = std::stof(argv[1]);
    Pe = Re*Pr;
    DumpParameters();
    StateVector::ResetForParams();

    HopfBifurcation guess;


    if (argc == 6)
    {
        HopfBifurcation x1;
        HopfBifurcation x2;
        x1.LoadFromFile(argv[2]);
        x2.LoadFromFile(argv[3]);

        stratifloat Re1 = std::stof(argv[4]);
        stratifloat Re2 = std::stof(argv[5]);

        HopfBifurcation gradient = x2;
        gradient -= x1;
        gradient *= 1/(Re2-Re1);

        guess = x2;
        guess.MulAdd(Re-Re2, gradient);
    }
    else
    {
        guess.LoadFromFile(argv[2]);
    }


    Ri = guess.p;

    FindHopf solver;

    solver.EnforceConstraints(guess);

    solver.Run(guess);

    guess.SaveToFile("final");
}
