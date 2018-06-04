#include "StateVector.h"
#include "NewtonKrylov.h"
#include <iomanip>

class CriticalPoint
{
public:
    StateVector x;
    StateVector v;
    stratifloat p;

    stratifloat Dot(const CriticalPoint& other) const
    {
        return x.Dot(other.x) + v.Dot(other.v) + p*other.p;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    void MulAdd(stratifloat b, const CriticalPoint& A)
    {
        x.MulAdd(b,A.x);
        v.MulAdd(b,A.v);
        p += b*A.p;
    }

    const CriticalPoint& operator+=(const CriticalPoint& other)
    {
        x += other.x;
        v += other.v;
        p += other.p;
        return *this;
    }

    const CriticalPoint& operator-=(const CriticalPoint& other)
    {
        x -= other.x;
        v -= other.v;
        p -= other.p;
        return *this;
    }

    const CriticalPoint& operator*=(stratifloat mult)
    {
        x *= mult;
        v *= mult;
        p *= mult;
        return *this;
    }

    void Zero()
    {
        x.Zero();
        v.Zero();
        p = 0;
    }

    void SaveToFile(const std::string& filename) const
    {
        x.SaveToFile(filename+".fields");
        v.SaveToFile(filename+"-eig.fields");
        std::ofstream paramFile(filename+".params");
        paramFile << std::setprecision(30);
        paramFile << p;
    }

    void LoadFromFile(const std::string& filename)
    {
        x.LoadFromFile(filename+".fields");
        v.LoadFromFile(filename+"-eig.fields");
        std::ifstream paramFile(filename+".params");
        paramFile >> p;
    }

    void EnforceBCs()
    {
        x.EnforceBCs();
        v.EnforceBCs();
    }

    void PlotAll(std::string directory) const
    {
        MakeCleanDir(directory);
        x.PlotAll(directory+"/x");
        v.PlotAll(directory+"/v");
    }
};

class FindCriticalPoint : public NewtonKrylov<CriticalPoint>
{
public:
    FindCriticalPoint()
    {
        phi.Randomise(0.000001, true);
    }

    StateVector phi;

private:
    virtual CriticalPoint EvalFunction(const CriticalPoint& at) override
    {
        CriticalPoint result;

        Ri = at.p;
        at.x.FullEvolve(T, result.x, false, false);
        at.v.LinearEvolve(T, at.x, result.v);

        result -= at;
        result.p = at.v.Dot(phi) - 1;

        std:: cout << result.x.Norm2() << " " << result.v.Norm2() << " " << result.p*result.p << std::endl;

        return result;
    }

    virtual void EnforceConstraints(CriticalPoint& at)
    {
        Ri = at.p;
        at.v.Rescale(weight);
    }
};

#include "Arnoldi.h"
#include "ExtendedStateVector.h"

int main(int argc, char *argv[])
{
    // L3 = std::stof(argv[2]);
    // DumpParameters();
    // StateVector::ResetForParams();

    BasicArnoldi eigenSolver;

    CriticalPoint guess;

    // if (argc==2)
    // {

    //    guess.LoadFromFile(argv[1]);
    //    Ri = guess.p;
    // }
    // else
    // {
    //     if (argc==1)
    //     {
    //         Ri = 0.245525;
    //         guess.x.Zero();
    //     }
    //     else
    //     {
    //         ExtendedStateVector loadedGuess;
    //         loadedGuess.LoadFromFile(argv[1]);

    //         guess.x = loadedGuess.x;
    //         Ri = loadedGuess.p;
    //     }
    //     eigenSolver.Run(guess.x, guess.v, false);
    //     guess.p = Ri;
    // }

    ExtendedStateVector loadedGuess;
    loadedGuess.LoadFromFile(argv[1]);

    guess.x = loadedGuess.x;
    Ri = loadedGuess.p;

    guess.p = Ri;
    guess.v = guess.x;

    FindCriticalPoint solver;

    // scale v
    stratifloat proj = guess.v.Dot(solver.phi);
    guess.v *= 1/proj;

    std::cout << "Lengths for debugging: " << solver.phi.Energy() << " " << guess.v.Energy() << " " << guess.x.Energy() << std::endl;

    std::cout << guess.v.Dot(solver.phi) << std::endl;

    solver.Run(guess);
}
