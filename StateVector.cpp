#include "StateVector.h"

void StateVector::FullEvolve(stratifloat T, StateVector& result, bool snapshot, bool screenshot) const
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
    solver.PrepareRun(std::string("images-")+std::to_string(runnum)+"/", screenshot);

    if (screenshot)
    {
        solver.PlotAll(std::to_string(t)+".png", true);
    }

    solver.deltaT = 0.01;
    solver.UpdateForTimestep();

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
            std::cout << step << " " << t << std::endl;

            if (screenshot)
            {
                solver.PlotAll(std::to_string(t)+".png", true);
            }
        }

        if (snapshot)
        {
            solver.StoreSnapshot(t);
        }

        step++;

        if (done)
        {
            break;
        }
    }
    if (screenshot)
    {
        solver.PlotAll(std::to_string(t)+".png", true);
    }
    CopyFromSolver(result);
}

void StateVector::LinearEvolve(stratifloat T, const StateVector& about, const StateVector& aboutResult, StateVector& result) const
{
    stratifloat eps = 0.0000001;

    result = about;
    result.MulAdd(eps, *this);

    result.FullEvolve(T, result, false, false);

    result -= aboutResult;
    result *= 1/eps;
}

void StateVector::LinearEvolve(stratifloat T, StateVector& result) const
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

    solver.deltaT = 0.01;
    solver.UpdateForTimestep();

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
            std::cout << step << " " << t << std::endl;
        }

        step++;

        if (done)
        {
            break;
        }
    }

    solver.PlotAll(std::to_string(t)+".png", false);
    CopyFromSolver(result);
}

void StateVector::AdjointEvolve(stratifloat T, StateVector& result) const
{
    CopyToSolver();

    solver.FilterAll();
    solver.PopulateNodalVariablesAdjoint();
    solver.RemoveDivergence(0.0f);

    stratifloat t = T;

    static int runnum = 0;
    runnum++;
    solver.PrepareRunAdjoint(std::string("images-adjoint-")+std::to_string(runnum)+"/");
    solver.PlotAll(std::to_string(t)+".png", false);


    bool done = false;
    int step = 0;
    while (t > 0)
    {
        // on last step, arrive exactly
        if (t - solver.deltaT < 0)
        {
            solver.deltaT = t;
            solver.UpdateForTimestep();
            done = true;
        }

        solver.TimeStepAdjoint(t, true);
        t -= solver.deltaT;

        if(step%50==0)
        {
            stratifloat cfl = solver.CFLadjoint();
            std::cout << step << " " << t << std::endl;
            solver.PlotAll(std::to_string(t)+".png", false);
        }

        step++;

        if (done)
        {
            break;
        }
    }

    solver.PlotAll(std::to_string(t)+".png", false);
    CopyFromSolver(result);
}

void StateVector::Rescale(stratifloat energy)
{
    stratifloat scale;

    // energies are entirely quadratic
    // which makes this easy

    stratifloat energyBefore = Energy();

    if (energyBefore!=0.0f)
    {
        scale = sqrt(energy/energyBefore);
    }
    else
    {
        scale = 0.0f;
    }

    u1 *= scale;
    if (ThreeDimensional)
    {
        u2 *= scale;
    }
    u3 *= scale;
    b *= scale;
}

IMEXRK StateVector::solver;

StateVector operator+(const StateVector& lhs, const StateVector& rhs)
{
    StateVector ret = lhs;
    ret += rhs;

    return ret;
}

StateVector operator-(const StateVector& lhs, const StateVector& rhs)
{
    StateVector ret = lhs;
    ret -= rhs;

    return ret;
}

StateVector operator*(stratifloat scalar, const StateVector& vector)
{
    StateVector ret = vector;
    ret *= scalar;
    return ret;
}

StateVector operator*(const StateVector& vector, stratifloat scalar)
{
    return scalar*vector;
}
