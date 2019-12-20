#include "HopfBifurcation.h"
#include "NewtonKrylov.h"

class FindHopf : public NewtonKrylov<HopfBifurcation>
{
public:
    StateVector A;

    virtual void EnforceConstraints(HopfBifurcation& at)
    {
        flowParams.Ri = at.p;

        stratifloat theta = atan2(-at.v2.Dot(A),at.v1.Dot(A));
        stratifloat r = 1/(cos(theta)*at.v1.Dot(A) - sin(theta)*at.v2.Dot(A));

        StateVector newv1 = r*cos(theta)*at.v1 - r*sin(theta)*at.v2;
        at.v2 = r*sin(theta)*at.v1 + r*cos(theta)*at.v2;
        at.v1 = newv1;
    }
private:
    virtual HopfBifurcation EvalFunction(const HopfBifurcation& at) override
    {
        HopfBifurcation result;

        flowParams.Ri = at.p;
        at.x.FullEvolve(T, result.x, false, false);
        at.v1.LinearEvolve(T, at.x, result.v1);
        at.v2.LinearEvolve(T, at.x, result.v2);

        result.x -= at.x;
        result.v1.MulAdd(-cos(at.theta), at.v1);
        result.v1.MulAdd(sin(at.theta), at.v2);
        result.v2.MulAdd(-sin(at.theta), at.v1);
        result.v2.MulAdd(-cos(at.theta), at.v2);

        result.theta = at.v1.Dot(A) - 1;
        result.p = at.v2.Dot(A);

        std::cout << at.v1.Norm() << " " << at.v2.Norm() << std::endl;

        std::cout << result.x.Norm2() << " "
                   << result.v1.Norm2() << " "
                   << result.v2.Norm2() << " "
                   << result.theta*result.theta << " "
                   << result.p*result.p << std::endl;

        return result;
    }
};

#include "Arnoldi.h"
#include "ExtendedStateVector.h"

int main(int argc, char *argv[])
{
    flowParams.Re = std::stof(argv[1]);
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
        guess.MulAdd(flowParams.Re-Re2, gradient);
    }
    else
    {
        guess.LoadFromFile(argv[2]);
    }


    flowParams.Ri = guess.p;

    FindHopf solver;

    // make sure eigenvectors are a sensible size
    stratifloat rescale = guess.v1.Norm();
    guess.v1 *= 1/rescale;
    guess.v2 *= 1/rescale;

    solver.A = guess.v1;
    solver.EnforceConstraints(guess);

    solver.Run(guess);

    guess.SaveToFile("final");
}
