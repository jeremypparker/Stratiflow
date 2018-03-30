#include "IMEXRK.h"

// This class contains a full state's information
// its operations are not particularly efficient
// so it should only be used for high level algorithms
class FullState
{
public:
    FullState()
    : solver()
    {}

    FullState(const FullState& other)
    : u1(other.u1)
    , u3(other.u3)
    , b(other.b)
    , p(other.p)
    , solver()
    {
    }

    const FullState& operator=(const FullState& other)
    {
        u1 = other.u1;
        u3 = other.u3;
        b = other.b;
        p = other.p;

        return *this;
    }

    NeumannModal u1;
    DirichletModal u3;
    NeumannModal b;
    NeumannModal p;

    void FullEvolve(stratifloat T, FullState& result, bool snapshot = false) const
    {
        CopyToSolver();

        solver.SetBackground(InitialU, InitialB);

        solver.FilterAll();
        solver.PopulateNodalVariables();
        solver.RemoveDivergence(0.0f);

        stratifloat t = 0.0f;

        int step = 0;

        bool done = false;

        static int runnum = 0;
        runnum++;
        solver.PrepareRun(std::string("images-")+std::to_string(runnum)+"/");
        solver.PlotAll(std::to_string(t)+".png", true);
        while (t < T)
        {
            // on last step, arrive exactly
            if (t + solver.deltaT > T)
            {
                solver.deltaT = T - t;
                solver.UpdateForTimestep();
                done = true;
            }

            solver.TimeStep();
            t += solver.deltaT;

            if(step%50==0)
            {
                stratifloat cfl = solver.CFL();
            }

            if (snapshot)
            {
                solver.StoreSnapshot(t);
            }

            step++;
            std::cout << step << " " << t << std::endl;

            if (done)
            {
                break;
            }
        }

        solver.PlotAll(std::to_string(t)+".png", true);

        CopyFromSolver(result);
    }

    void LinearEvolve(stratifloat T, FullState& result) const
    {
        CopyToSolver();

        solver.FilterAll();
        solver.PopulateNodalVariables();
        solver.RemoveDivergence(0.0f);

        stratifloat t = 0.0f;

        int step = 0;

        bool done = false;

        static int runnum = 0;
        runnum++;
        solver.PrepareRunLinear(std::string("images-linear-")+std::to_string(runnum)+"/");
        solver.PlotAll(std::to_string(t)+".png", false);
        while (t < T)
        {
            // on last step, arrive exactly
            if (t + solver.deltaT > T)
            {
                solver.deltaT = T - t;
                solver.UpdateForTimestep();
                done = true;
            }

            solver.TimeStepLinear(t);
            t += solver.deltaT;

            if(step%50==0)
            {
                stratifloat cfl = solver.CFLadjoint();
            }

            step++;
            std::cout << step << " " << t << std::endl;

            if (done)
            {
                break;
            }
        }

        solver.PlotAll(std::to_string(t)+".png", false);
        CopyFromSolver(result);
    }

    void CalcPressure()
    {
        CopyToSolver();
        solver.SolveForPressure();
        CopyFromSolver();
    }

    const FullState& operator+=(const FullState& other)
    {
        u1 += other.u1;
        u3 += other.u3;
        b += other.b;
        CalcPressure();

        return *this;
    }

    const FullState& operator-=(const FullState& other)
    {
        u1 -= other.u1;
        u3 -= other.u3;
        b  -= other.b;
        CalcPressure();

        return *this;
    }

    const FullState& MulAdd(stratifloat a, const FullState& B)
    {
        u1 += a*B.u1;
        u3 += a*B.u3;
        b  += a*B.b;
        CalcPressure();

        return *this;
    }

    const FullState& operator*=(stratifloat other)
    {
        u1 *= other;
        u3 *= other;
        b  *= other;
        CalcPressure();

        return *this;
    }

    stratifloat Dot(const FullState& other) const
    {
        stratifloat prod = 0.5f*(InnerProd(u1, other.u1, L3)
                               + InnerProd(u3, other.u3, L3)
                               + Ri*InnerProd(b, other.b, L3)); // TODO: is this correct PE?
        return prod;
    }

    stratifloat Norm() const
    {
        return sqrt(Dot(*this));
    }

    void Zero()
    {
        u1.Zero();
        u3.Zero();
        b.Zero();
        p.Zero();
    }

    void LoadFromFile(const std::string& filename)
    {
        solver.LoadFlow(filename);
        CopyFromSolver();
    }

private:
    void CopyToSolver() const
    {
        solver.u1 = u1;
        solver.u3 = u3;
        solver.b = b;
        solver.u2.Zero();
        solver.p = p;
    }

    void CopyFromSolver()
    {
        CopyFromSolver(*this);
    }

    void CopyFromSolver(FullState& into) const
    {
        into.u1 = solver.u1;
        into.u3 = solver.u3;
        into.b = solver.b;
        into.p = solver.p;
    }

    mutable IMEXRK solver;
};