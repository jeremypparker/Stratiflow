#include "StateVector.h"
#include "NewtonKrylov.h"
#include <iomanip>

class HopfBifurcation
{
public:
    StateVector x;

    // complex eigenvector v1 + i v2
    StateVector v1;
    StateVector v2;

    // complex eigenvalue lambda1 + i lambda2
    stratifloat lambda1;
    stratifloat lambda2;
    stratifloat p;

    stratifloat Dot(const HopfBifurcation& other) const
    {
        return x.Dot(other.x)
             + v1.Dot(other.v1)
             + v2.Dot(other.v2)
             + lambda1*other.lambda1
             + lambda2*other.lambda2
             + p*other.p;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    void MulAdd(stratifloat b, const HopfBifurcation& A)
    {
        x.MulAdd(b,A.x);
        v1.MulAdd(b,A.v1);
        v2.MulAdd(b,A.v2);
        lambda1 += b*A.lambda1;
        lambda2 += b*A.lambda2;
        p += b*A.p;
    }

    const HopfBifurcation& operator+=(const HopfBifurcation& other)
    {
        x += other.x;
        v1 += other.v1;
        v2 += other.v2;
        lambda1 += other.lambda1;
        lambda2 += other.lambda2;
        p += other.p;
        return *this;
    }

    const HopfBifurcation& operator-=(const HopfBifurcation& other)
    {
        x -= other.x;
        v1 -= other.v1;
        v2 -= other.v2;
        lambda1 -= other.lambda1;
        lambda2 -= other.lambda2;
        p -= other.p;
        return *this;
    }

    const HopfBifurcation& operator*=(stratifloat mult)
    {
        x *= mult;
        v1 *= mult;
        v2 *= mult;
        lambda1 *= mult;
        lambda2 *= mult;
        p *= mult;
        return *this;
    }

    void Zero()
    {
        x.Zero();
        v1.Zero();
        v2.Zero();
        lambda1 = 0;
        lambda2 = 0;
        p = 0;
    }

    void SaveToFile(const std::string& filename) const
    {
        x.SaveToFile(filename+".fields");
        v1.SaveToFile(filename+"-eig1.fields");
        v2.SaveToFile(filename+"-eig2.fields");
        std::ofstream paramFile(filename+".params");
        paramFile << std::setprecision(30);
        paramFile << p;
        paramFile << std::endl << lambda1;
        paramFile << std::endl << lambda2;
    }

    void LoadFromFile(const std::string& filename)
    {
        x.LoadFromFile(filename+".fields");
        v1.LoadFromFile(filename+"-eig1.fields");
        v2.LoadFromFile(filename+"-eig2.fields");
        std::ifstream paramFile(filename+".params");
        paramFile >> p;
        paramFile >> lambda1 >> lambda2;
    }

    void EnforceBCs()
    {
        x.EnforceBCs();
        v1.EnforceBCs();
        v2.EnforceBCs();
    }

    void PlotAll(std::string directory) const
    {
        MakeCleanDir(directory);
        x.PlotAll(directory+"/x");
        v1.PlotAll(directory+"/v1");
        v2.PlotAll(directory+"/v2");
    }
};

class FindHopf : public NewtonKrylov<HopfBifurcation>
{
public:
    stratifloat weight = 1;
    stratifloat fixedLambda1;

    virtual void EnforceConstraints(HopfBifurcation& at)
    {
        Ri = at.p;

        // ensure unit eigenvalue
        at.lambda1 = fixedLambda1;
        at.lambda2 = sgn(at.lambda2)*sqrt(1-at.lambda1*at.lambda1);

        // ensure unit eigenvector
        at.v2.Rescale(weight-at.v1.Energy());
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

        result.lambda1 = at.lambda1 - fixedLambda1;
        result.lambda2 = at.lambda1*at.lambda1 + at.lambda2*at.lambda2 - 1;

        result.p = at.v1.Energy() - at.v2.Energy() - weight;

        std:: cout << result.x.Norm2() << " "
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

    solver.fixedLambda1 = guess.lambda1;
    solver.EnforceConstraints(guess);

    solver.Run(guess);

    guess.SaveToFile("final");
}
