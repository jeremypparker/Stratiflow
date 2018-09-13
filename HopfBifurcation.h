#include "StateVector.h"
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
