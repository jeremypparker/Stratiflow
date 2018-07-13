#pragma once

#include "IMEXRK.h"
#include "OrrSommerfeld.h"

// This class contains a full state's information
// its operations are not particularly efficient
// so it should only be used for high level algorithms
class StateVector
{
public:
    NeumannModal u1;
    NeumannModal u2;
    DirichletModal u3;
    NeumannModal b;
    NeumannModal p;

    void FullEvolve(stratifloat T, StateVector& result, bool snapshot = false, bool screenshot = true) const;

    void FixedEvolve(stratifloat deltaT, int steps, std::vector<StateVector>& result) const;

    void LinearEvolve(stratifloat T, const StateVector& about, StateVector& result) const;

    void AdjointEvolve(stratifloat deltaT, int steps, const std::vector<StateVector>& intermediate, StateVector& result) const;


    const StateVector& operator+=(const StateVector& other)
    {
        u1 += other.u1;
        if (ThreeDimensional)
        {
            u2 += other.u2;
        }
        u3 += other.u3;
        b += other.b;

        return *this;
    }

    const StateVector& operator-=(const StateVector& other)
    {
        u1 -= other.u1;
        if (ThreeDimensional)
        {
            u2 -= other.u2;
        }
        u3 -= other.u3;
        b  -= other.b;
        return *this;
    }

    const StateVector& MulAdd(stratifloat a, const StateVector& B)
    {
        u1 += a*B.u1;
        if (ThreeDimensional)
        {
            u2 += a*B.u2;
        }
        u3 += a*B.u3;
        b  += a*B.b;
        return *this;
    }

    const StateVector& operator*=(stratifloat other)
    {
        u1 *= other;
        if (ThreeDimensional)
        {
            u2 *= other;
        }
        u3 *= other;
        b  *= other;
        return *this;
    }

    stratifloat Dot(const StateVector& other) const
    {
        stratifloat prod = InnerProd(u1, other.u1, L3)
                         + InnerProd(u3, other.u3, L3)
                         + Ri*InnerProd(b, other.b, L3); // TODO: is this correct PE?

        if (ThreeDimensional)
        {
            prod += InnerProd(u2, other.u2, L3);
        }

        return prod;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Energy() const
    {
        return 0.5*Norm2();
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    void Zero()
    {
        u1.Zero();
        if (ThreeDimensional)
        {
            u2.Zero();
        }
        u3.Zero();
        b.Zero();
        p.Zero();
    }

    void Rescale(stratifloat energy);

    void Randomise(stratifloat energy, bool restrictToMiddle = false)
    {
        u1.RandomizeCoefficients(0.3);
        if (ThreeDimensional)
        {
            u2.RandomizeCoefficients(0.3);
        }
        u3.RandomizeCoefficients(0.3);
        b.RandomizeCoefficients(0.3);

        if (restrictToMiddle)
        {
            for (int j=0; j<N3/4; j++)
            {
                u1.slice(j).setZero();
                u1.slice(N3-j-1).setZero();

                u2.slice(j).setZero();
                u2.slice(N3-j-1).setZero();

                u3.slice(j).setZero();
                u3.slice(N3-j-1).setZero();

                b.slice(j).setZero();
                b.slice(N3-j-1).setZero();
            }
        }

        Rescale(energy);
    }

    void LoadFromFile(const std::string& filename)
    {
        if (EndsWith(filename, ".fields"))
        {
            solver.LoadFlow(filename);
        }
        else
        {
            solver.LoadFlow(filename+".fields");
        }
        CopyFromSolver();
    }

    void SaveToFile(const std::string& filename) const
    {
        CopyToSolver();
        solver.PopulateNodalVariables();

        if (EndsWith(filename, ".fields"))
        {
            solver.SaveFlow(filename);
        }
        else
        {
            solver.SaveFlow(filename+".fields");
        }
    }

    void EnforceBCs()
    {
        u3.ZeroEnds();
        u1.NeumannEnds();
        u2.NeumannEnds();
        b.NeumannEnds();
    }

    void AddBackground()
    {
        NeumannNodal U1;
        NeumannNodal B;

        Neumann1D U_;
        Neumann1D B_;

        U_.SetValue(InitialU, L3);
        B_.SetValue(InitialB, L3);



        u1.ToNodal(U1);
        b.ToNodal(B);

        U1 += U_;
        B += B_;

        U1.ToModal(u1);
        B.ToModal(b);
    }

    void RemoveBackground()
    {
        NeumannNodal U1;
        NeumannNodal B;

        Neumann1D U_;
        Neumann1D B_;

        U_.SetValue(InitialU, L3);
        B_.SetValue(InitialB, L3);



        u1.ToNodal(U1);
        b.ToNodal(B);

        U1 -= U_;
        B -= B_;

        U1.ToModal(u1);
        B.ToModal(b);

    }

    void PlotAll(std::string directory) const
    {
        MakeCleanDir(directory);

        HeatPlot(u1, L1, L3, 0, directory+"/u1.png");
        if (ThreeDimensional)
        {
            HeatPlot(u2, L1, L3, 0, directory+"/u2.png");
        }
        HeatPlot(u3, L1, L3, 0, directory+"/u3.png");
        HeatPlot(b, L1, L3, 0, directory+"/b.png");

        DirichletModal StaggeredTemp;
        StaggeredTemp = ddz(u1)+-1.0*ddx(u3);
        HeatPlot(StaggeredTemp, L1, L3, 0, directory+"/vorticity.png");
    }

    stratifloat ToEigenMode(stratifloat energy, int mode=1)
    {
        stratifloat growth = EigenModes(2*pi*mode/L1, u1, u2, u3, b);
        p.Zero();
        Rescale(energy);

        return growth;
    }

    static void ResetForParams()
    {
        solver = IMEXRK();
    }

private:
    void CopyToSolver() const
    {
        solver.u1 = u1;
        if (ThreeDimensional)
        {
            solver.u2 = u2;
        }
        solver.u3 = u3;
        solver.b = b;
        solver.u2.Zero();
        solver.p = p;
    }

    void CopyFromSolver()
    {
        CopyFromSolver(*this);
    }

    void CopyFromSolver(StateVector& into) const
    {
        into.u1 = solver.u1;
        if (ThreeDimensional)
        {
            into.u2 = solver.u2;
        }
        into.u3 = solver.u3;
        into.b = solver.b;
        into.p = solver.p;
    }

    static IMEXRK solver;
};

StateVector operator+(const StateVector& lhs, const StateVector& rhs);
StateVector operator-(const StateVector& lhs, const StateVector& rhs);
StateVector operator*(stratifloat scalar, const StateVector& vector);
StateVector operator*(const StateVector& vector, stratifloat scalar);
