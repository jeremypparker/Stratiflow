#pragma once

#include "StateVector.h"

#include <iomanip>

// A vector in the extended state space including one real parameter (Ri)
class ExtendedStateVector
{
public:
    StateVector x;
    stratifloat p;

    stratifloat Dot(const ExtendedStateVector& other) const
    {
        Ri = p;
        return x.Dot(other.x) + p*other.p;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    void MulAdd(stratifloat b, const ExtendedStateVector& A)
    {
        x.MulAdd(b,A.x);
        p += b*A.p;
    }

    const ExtendedStateVector& operator+=(const ExtendedStateVector& other)
    {
        x += other.x;
        p += other.p;
        return *this;
    }

    const ExtendedStateVector& operator-=(const ExtendedStateVector& other)
    {
        x -= other.x;
        p -= other.p;
        return *this;
    }

    const ExtendedStateVector& operator*=(stratifloat mult)
    {
        x *= mult;
        p *= mult;
        return *this;
    }

    void Zero()
    {
        x.Zero();
        p = 0;
    }

    void LinearEvolve(stratifloat T,
                      const ExtendedStateVector& about,
                      const ExtendedStateVector& aboutResult,
                      ExtendedStateVector& result) const
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
                    ExtendedStateVector& result,
                    bool snapshot = false,
                    bool screenshot = true) const
    {
        stratifloat RiOld = Ri;
        Ri = p;

        x.FullEvolve(T, result.x, snapshot, screenshot);
        result.p = p;

        Ri = RiOld;
    }

    void SaveToFile(const std::string& filename) const
    {
        x.SaveToFile(filename+".fields");
        std::ofstream paramFile(filename+".params");
        paramFile << std::setprecision(30);
        paramFile << p;
    }

    void LoadFromFile(const std::string& filename)
    {
        x.LoadFromFile(filename+".fields");
        std::ifstream paramFile(filename+".params");
        paramFile >> p;
    }

    void EnforceBCs()
    {
        x.EnforceBCs();
    }

    void PlotAll(std::string directory) const
    {
        x.PlotAll(directory);
    }
};
