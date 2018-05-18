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
    virtual CriticalPoint EvalFunction(const CriticalPoint& at) override
    {
        CriticalPoint result;

        Ri = at.p;
        at.x.FullEvolve(T, result.x, false);
        at.v.LinearEvolve(T, at.x, result.x, result.v);

        // need this stuff for later
        linearAboutStart = at;
        linearAboutEnd.x = result.x;
        linearAboutEnd.p = at.p;
        StateVector temp = at.x + eps*at.v;
        temp.FullEvolve(T, linearAboutEnd.v, false, false);

        result -= at;
        result.p = at.v.Energy() - 1;

        std:: cout << result.x.Norm2() << " " << result.v.Norm2() << " " << result.p*result.p << std::endl;

        return result;
    }

    virtual CriticalPoint EvalLinearised(const CriticalPoint& at) override
    {

        StateVector x = linearAboutStart.x;

        StateVector dx = at.x;
        StateVector dv = at.v;

        StateVector v = linearAboutStart.v;
        StateVector f = linearAboutEnd.x;

        StateVector fdv;
        StateVector fdx, fvdx;

        StateVector fv = linearAboutEnd.v;

        StateVector temp;

        Ri = linearAboutStart.p;

        temp = x + eps*dv;
        temp.FullEvolve(T, fdv, false, false);


        Ri = linearAboutStart.p + eps*at.p;

        temp = x;
        temp.MulAdd(eps,dx);
        temp.FullEvolve(T, fdx, false, false);

        temp.MulAdd(eps,v);
        temp.FullEvolve(T, fvdx, false, false);


        CriticalPoint result;
        //          I.dx      - J.dx
        result.x = dx - (1/eps)*(fdx-f);
        //         I.dv  -     J.dv                  - dx.H.v
        result.v = dv - (1/eps)*(fdv-f) - (1/eps/eps)*(fvdx - fdx - fv + f);
        result.p = -dv.Dot(v);

        return result;
    }

    stratifloat eps = 0.0000001;
};

#include "Arnoldi.h"
#include "ExtendedStateVector.h"

int main(int argc, char *argv[])
{
    BasicArnoldi eigenSolver;

    CriticalPoint guess;

    if (argc==2)
    {
        guess.LoadFromFile(argv[1]);
    }
    else
    {
        if (argc==1)
        {
            Ri = 0.245525;
            guess.x.Zero();
        }
        else
        {
            ExtendedStateVector loadedGuess;
            loadedGuess.LoadFromFile(argv[1]);

            guess.x = loadedGuess.x;
            Ri = loadedGuess.p;
        }
        eigenSolver.Run(guess.x, guess.v, true);
        guess.v.Rescale(1);
        guess.p = Ri;
    }

    FindCriticalPoint solver;
    solver.Run(guess);
}