#include "StateVector.h"

void StateVector::FullEvolve(stratifloat T, StateVector& result, bool snapshot, bool screenshot) const
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
    solver.PrepareRun(std::string("images-")+std::to_string(runnum)+"/", screenshot);
    MakeCleanDir("snapshots");

    const int stepinterval = 100;

    stratifloat mixing = 0;

    while (t+0.0001 < T)
    {
        if(step%stepinterval==0)
        {
            stratifloat cfl = solver.CFL();
            std::cout << step << " " << t << " " << (solver.KE() + solver.PE()) << std::endl;

            // finish exactly for last step
            stratifloat remaining = T-t;
            int remainingSteps = (remaining / solver.deltaT)+1;
            if (remainingSteps < stepinterval)
            {
                // make timestep slightly shorter
                solver.deltaT = remaining/remainingSteps;
                solver.UpdateForTimestep();
            }

            if (screenshot)
            {
                solver.PlotAll(std::to_string(step)+"-"+std::to_string(t)+".png", true);
            }

            if (snapshot)
            {
                solver.SaveFlow("snapshots/"+std::to_string(step)+"-"+std::to_string(t)+".fields");
            }
        }

        solver.TimeStep();

        t += solver.deltaT;
        step++;

        if (t+0.0001>=T)
        {
            if (screenshot)
            {
                solver.PlotAll(std::to_string(step)+"-"+std::to_string(t)+".png", true);
            }

            if (snapshot)
            {
                solver.SaveFlow("snapshots/"+std::to_string(step)+"-"+std::to_string(t)+".fields");
            }
        }
    }

    CopyFromSolver(result);
}

void StateVector::FixedEvolve(stratifloat deltaT, int steps, std::vector<StateVector>& result) const
{
    result.resize(steps);

    CopyToSolver();

    solver.FilterAll();
    solver.PopulateNodalVariables();
    solver.RemoveDivergence(0.0f);

    solver.deltaT = deltaT;
    solver.UpdateForTimestep();

    solver.PrepareRun(std::string("blah"), false);

    for (int step=0; step<steps; step++)
    {
        CopyFromSolver(result[step]);
        solver.TimeStep();
    }
}

void StateVector::LinearEvolve(stratifloat T, const StateVector& about, StateVector& result) const
{
    CopyToSolver();

    solver.SetBackground(about.u1, about.u2, about.u3, about.b);
    solver.FilterAll();
    solver.PopulateNodalVariables();
    solver.RemoveDivergence(0.0f);

    stratifloat t = 0.0f;

    int step = 0;

    bool done = false;

    static int runnum = 0;
    runnum++;
    solver.PrepareRunLinear(std::string("images-linear-")+std::to_string(runnum)+"/", false);

    solver.deltaT = 0.01;
    solver.UpdateForTimestep();

    const int stepinterval = 100;

    while (t+0.0001 < T)
    {
        if(step%stepinterval==0)
        {
            stratifloat cfl = solver.CFLlinear();
            std::cout << step << " " << t << std::endl;

            // finish exactly for last step
            stratifloat remaining = T-t;
            int remainingSteps = (remaining / solver.deltaT)+1;
            if (remainingSteps < stepinterval)
            {
                // make timestep slightly shorter
                solver.deltaT = remaining/remainingSteps;
                solver.UpdateForTimestep();
            }
        }

        solver.TimeStepLinear();
        t += solver.deltaT;
        step++;
    }

    CopyFromSolver(result);
}

void StateVector::LinearEvolve(stratifloat deltaT, int steps, const std::vector<StateVector>& intermediate, StateVector& result) const
{
    CopyToSolver();

    solver.FilterAll();
    solver.PopulateNodalVariables();
    solver.RemoveDivergence(0.0f);

    static int runnum = 0;
    runnum++;
    solver.PrepareRunLinear("blah", false);

    solver.deltaT = deltaT;
    solver.UpdateForTimestep();

    for (int step=0; step<steps; step++)
    {
        solver.TimeStepLinear(intermediate[step].u1,
                              intermediate[step].u2,
                              intermediate[step].u3,
                              intermediate[step].b,
                              intermediate[step+1].u1,
                              intermediate[step+1].u2,
                              intermediate[step+1].u3,
                              intermediate[step+1].b);
    }

    CopyFromSolver(result);
}

void StateVector::AdjointEvolve(stratifloat deltaT, int steps, const std::vector<StateVector>& intermediate, StateVector& result) const
{
    CopyToSolver();

    solver.FilterAll();
    solver.PopulateNodalVariables();
    solver.RemoveDivergence(0.0f);

    static int runnum = 0;
    runnum++;
    solver.PrepareRunAdjoint(std::string("images-adjoint-")+std::to_string(runnum)+"/");

    solver.deltaT = deltaT;
    solver.UpdateForTimestep();

    for (int step=0; step<steps; step++)
    {
        solver.TimeStepAdjoint(intermediate[steps-1-step].u1,
                               intermediate[steps-1-step].u2,
                               intermediate[steps-1-step].u3,
                               intermediate[steps-1-step].b,
                               intermediate[steps-step].u1,
                               intermediate[steps-step].u2,
                               intermediate[steps-step].u3,
                               intermediate[steps-step].b);
    }

    CopyFromSolver(result);
}

void StateVector::Rescale(stratifloat energy)
{
    CopyToSolver();
    solver.FilterAll();
    solver.PopulateNodalVariables();
    solver.RemoveDivergence(0.0f);
    CopyFromSolver();

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
    if (gridParams.ThreeDimensional)
    {
        u2 *= scale;
    }
    u3 *= scale;
    b *= scale;

    CopyToSolver();
    solver.FilterAll();
    solver.PopulateNodalVariables();
    solver.RemoveDivergence(0.0f);
    CopyFromSolver();
}

void StateVector::ExciteLowWavenumbers(stratifloat energy)
{
    u1.ExciteLowWavenumbers();
    u2.ExciteLowWavenumbers();
    u3.ExciteLowWavenumbers();
    b.ExciteLowWavenumbers();

    Rescale(energy);
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
