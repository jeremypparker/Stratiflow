#include "IMEXRK.h"

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

    void FullEvolve(stratifloat T, StateVector& result, bool snapshot = false) const;

    void LinearEvolve(stratifloat T, StateVector& result) const;

    void LinearEvolveFixed(stratifloat T, const StateVector& about, StateVector& result) const;

    void AdjointEvolve(stratifloat T, StateVector& result) const;

    void CalcPressure()
    {
        CopyToSolver();
        solver.SolveForPressure();
        CopyFromSolver();
    }

    const StateVector& operator+=(const StateVector& other)
    {
        u1 += other.u1;
        if (ThreeDimensional)
        {
            u2 += other.u2;
        }
        u3 += other.u3;
        b += other.b;
        CalcPressure();

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
        CalcPressure();

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
        CalcPressure();

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
        CalcPressure();

        return *this;
    }

    stratifloat Dot(const StateVector& other) const
    {
        stratifloat prod = 0.5f*(InnerProd(u1, other.u1, L3)
                               + InnerProd(u3, other.u3, L3)
                               + Ri*InnerProd(b, other.b, L3)); // TODO: is this correct PE?

        if (ThreeDimensional)
        {
            prod += 0.5f*InnerProd(u2, other.u2, L3);
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
        solver.LoadFlow(filename);
        CopyFromSolver();
    }

    void SaveToFile(const std::string& filename) const
    {
        CopyToSolver();
        solver.PopulateNodalVariables();
        solver.SaveFlow(filename);
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
