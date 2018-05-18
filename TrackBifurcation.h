#pragma once
#include "StateVector.h"
#include "NewtonKrylov.h"

class Bifurcation
{
public:
    StateVector x1;
    StateVector x2;
    stratifloat p;

    stratifloat Dot(const Bifurcation& other) const
    {
        return x1.Dot(other.x1) + x2.Dot(other.x2) + p*other.p;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    void MulAdd(stratifloat b, const Bifurcation& A)
    {
        x1.MulAdd(b,A.x1);
        x2.MulAdd(b,A.x2);
        p += b*A.p;
    }

    const Bifurcation& operator+=(const Bifurcation& other)
    {
        x1 += other.x1;
        x2 += other.x2;
        p += other.p;
        return *this;
    }

    const Bifurcation& operator-=(const Bifurcation& other)
    {
        x1 -= other.x1;
        x2 -= other.x2;
        p -= other.p;
        return *this;
    }

    const Bifurcation& operator*=(stratifloat mult)
    {
        x1 *= mult;
        x2 *= mult;
        p *= mult;
        return *this;
    }

    void Zero()
    {
        x1.Zero();
        x2.Zero();
        p = 0;
    }

    void LinearEvolve(stratifloat T,
                      const Bifurcation& about,
                      const Bifurcation& aboutResult,
                      Bifurcation& result) const
    {
        assert(about.p == aboutResult.p);

        stratifloat eps = 0.0000001;

        result = about;
        result.MulAdd(eps, *this);

        result.FullEvolve(T, result, false, false);

        result -= aboutResult;
        result *= 1/eps;
    }

    void FullEvolve(stratifloat T,
                    Bifurcation& result,
                    bool snapshot = false,
                    bool screenshot = true) const
    {
        stratifloat RiOld = Ri;
        Ri = p;

        x1.FullEvolve(T, result.x1, snapshot, screenshot);
        x2.FullEvolve(T, result.x2, snapshot, screenshot);
        result.p = p;

        Ri = RiOld;
    }

    void SaveToFile(const std::string& filename) const
    {
        x1.SaveToFile(filename+"-1.fields");
        x2.SaveToFile(filename+"-2.fields");
        std::ofstream paramFile(filename+".params");
        paramFile << p;
    }

    void LoadFromFile(const std::string& filename)
    {
        x1.LoadFromFile(filename+"-1.fields");
        x2.LoadFromFile(filename+"-2.fields");
        std::ifstream paramFile(filename+".params");
        paramFile >> p;
    }

    void EnforceBCs()
    {
        x1.EnforceBCs();
        x2.EnforceBCs();
    }

    void PlotAll(std::string directory) const
    {
        MakeCleanDir(directory);
        x1.PlotAll(directory+"/x1");
        x2.PlotAll(directory+"/x2");
    }
};


class TrackBifurcation : public NewtonKrylov<Bifurcation>
{
public:
    TrackBifurcation(stratifloat delta)
    : delta(delta)
    {}
protected:
    virtual Bifurcation EvalFunction(const Bifurcation& at) override
    {
        Bifurcation result;
        at.FullEvolve(T, result, false);

        linearAboutStart = at;
        linearAboutEnd = result;

        result -= at;

        result.p = delta - (at.x2-at.x1).Energy();

        std::cout << result.x1.Norm2() << " " << result.x2.Norm2() << " " << result.p*result.p << std::endl;
        std::cout << delta << " " << (at.x2-at.x1).Energy() << std::endl;

        return result;
    }

    virtual Bifurcation EvalLinearised(const Bifurcation& at) override
    {
        Bifurcation result;
        Bifurcation Gq;
        at.LinearEvolve(T, linearAboutStart, linearAboutEnd, Gq);
        result = at;
        result -= Gq;

        result.p = (linearAboutStart.x2-linearAboutStart.x1).Dot(at.x2-at.x1);

        return result;
    }

private:
    stratifloat delta;
};
