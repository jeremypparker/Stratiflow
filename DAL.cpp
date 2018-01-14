#include "IMEXRK.h"

int main(int argc, char *argv[])
{
    const stratifloat targetTime = 40.0;
    const stratifloat energy = 0.001;
    const stratifloat residualTarget = 0.001;
    const stratifloat minimumEpsilon = 0.000000001;

    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    // these are the initial conditions for the current step
    IMEXRK::MField u10(BoundaryCondition::Bounded);
    IMEXRK::MField u20(BoundaryCondition::Bounded);
    IMEXRK::MField u30(BoundaryCondition::Decaying);
    IMEXRK::MField b0(BoundaryCondition::Bounded);

    // these are the initial conditions for the previous step - so we can reset if necessary
    IMEXRK::MField previousu10(BoundaryCondition::Bounded);
    IMEXRK::MField previousu20(BoundaryCondition::Bounded);
    IMEXRK::MField previousu30(BoundaryCondition::Decaying);
    IMEXRK::MField previousb0(BoundaryCondition::Bounded);

    IMEXRK::M1D backgroundB(BoundaryCondition::Bounded);

    IMEXRK::MField previousv1(BoundaryCondition::Bounded);
    IMEXRK::MField previousv2(BoundaryCondition::Bounded);
    IMEXRK::MField previousv3(BoundaryCondition::Decaying);
    IMEXRK::MField previousvb(BoundaryCondition::Bounded);

    stratifloat previousIntegral = -1000;


    // first optional parameter is the maximum number of DAL loops
    int maxiterations = 50;
    if (argc > 1)
    {
        maxiterations = std::stoi(argv[1]);
    }

    // second optional parameter is which old initial condition to load
    int p; // which DAL step we are on
    if (argc > 2)
    {
        std::cout << "Loading ICs..." << std::endl;

        p = std::stoi(argv[2]);
        solver.LoadFlow("ICs/"+std::to_string(p)+".fields");
    }
    else
    {
        // if none supplied, set ICs manually
        std::cout << "Setting ICs..." << std::endl;

        MakeCleanDir("ICs");

        p = 0;

        IMEXRK::MField initialU1(BoundaryCondition::Bounded);
        IMEXRK::MField initialU2(BoundaryCondition::Bounded);
        IMEXRK::MField initialU3(BoundaryCondition::Decaying);
        IMEXRK::MField initialB(BoundaryCondition::Bounded);

        // put energy in the lowest third of the spatial modes
        initialU1.RandomizeCoefficients(0.3);
        initialU2.RandomizeCoefficients(0.3);
        initialU3.RandomizeCoefficients(0.3);
        initialB.RandomizeCoefficients(0.3);

        solver.SetInitial(initialU1, initialU2, initialU3, initialB);
    }

    solver.RemoveDivergence(0.0f);

    // rescale energy
    {
        stratifloat energyBefore = solver.KE() + solver.PE();
        stratifloat scale = sqrt(energy/energyBefore);

        solver.u1 *= scale;
        solver.u2 *= scale;
        solver.u3 *= scale;
        solver.b *= scale;
    }

    u10 = solver.u1;
    u20 = solver.u2;
    u30 = solver.u3;
    b0 = solver.b;

    stratifloat E0 = -1;

    stratifloat epsilon = 0.01;

    std::ofstream energyFile("energy.dat");
    for (; p<maxiterations; p++) // Direct-adjoint loop
    {
        // add background flow
        std::cout << "Setting background..." << std::endl;
        {
            stratifloat R = 2;

            IMEXRK::N1D Ubar(BoundaryCondition::Bounded);
            IMEXRK::N1D Bbar(BoundaryCondition::Bounded);
            Ubar.SetValue([](stratifloat z){return tanh(z);}, IMEXRK::L3);
            Bbar.SetValue([R](stratifloat z){return -tanh(R*z);}, IMEXRK::L3);

            solver.SetBackground(Ubar, Bbar);

            Bbar.ToModal(backgroundB);
        }

        E0 = solver.KE() + solver.PE();
        std::cout << "E0: " << E0 << std::endl;

        stratifloat totalTime = 0.0f;


        stratifloat saveEvery = 1.0f;
        int lastFrame = -1;
        int step = 0;
        bool done = false;

        stratifloat JoverKintegrated = 0;

        solver.PrepareRun("images/");

        // save initial condition
        solver.StoreSnapshot(totalTime);
        solver.SaveFlow("ICs/"+std::to_string(p)+".fields");

        // also save images for animation
        solver.PlotAll(std::to_string(totalTime)+".png", true);
        while (totalTime < targetTime)
        {
            // on last step, arrive exactly
            if (totalTime + solver.deltaT > targetTime)
            {
                solver.deltaT = targetTime - totalTime;
                solver.UpdateForTimestep();
                done = true;
            }

            solver.TimeStep();

            JoverKintegrated += solver.JoverK() * solver.deltaT;

            totalTime += solver.deltaT;

            solver.StoreSnapshot(totalTime);

            if(step%50==0)
            {
                stratifloat cfl = solver.CFL();
                std::cout << "  Step " << step << ", time " << totalTime
                        << ", CFL number: " << cfl << std::endl;

                std::cout << "  Average timings: " << solver.totalExplicit / (step+1)
                        << ", " << solver.totalImplicit / (step+1)
                        << ", " << solver.totalDivergence / (step+1)
                        << std::endl;
            }

            int frame = static_cast<int>(totalTime / saveEvery);

            if (frame>lastFrame)
            {
                lastFrame=frame;

                solver.PlotAll(std::to_string(totalTime)+".png", true);

                energyFile << totalTime
                        << " " << solver.KE()
                        << " " << solver.PE()
                        << " " << solver.JoverK()
                        << std::endl;
            }

            step++;

            if (done)
            {
                break;
            }
        }

        std::cout << "Integral of J/K: " << JoverKintegrated << std::endl;

        if (JoverKintegrated > previousIntegral)
        {
            previousIntegral = JoverKintegrated;

            previousu10 = u10;
            previousu20 = u20;
            previousu30 = u30;
            previousb0 = b0;

            // it's going well, slowly increase step size
            epsilon *= 1.1;
        }
        else
        {
            // we have overshot, try with a smaller step
            epsilon /= 2;

            std::cout << "Epsilon: " << epsilon << std::endl;

            if (epsilon < minimumEpsilon)
            {
                std::cout << "Epsilon too small, cannot converge" << std::endl;
                break;
            }

            solver.u1 = previousv1;
            solver.u2 = previousv2;
            solver.u3 = previousv3;
            solver.b = previousvb;

            u10 = previousu10;
            u20 = previousu20;
            u30 = previousu30;
            b0 = previousb0;

            energyFile << "STEP " << p << " and residual= "
                   << solver.Optimise(epsilon, E0, u10, u20, u30, b0, backgroundB)
                   << std::endl;

            continue;
        }

        {
            IMEXRK::N1D Ubar(BoundaryCondition::Bounded);
            IMEXRK::N1D Bbar(BoundaryCondition::Bounded);
            solver.SetBackground(Ubar, Bbar);
        }

        totalTime = targetTime;
        lastFrame = 10000;

        step = 0;
        done = false;

        solver.PrepareRunAdjoint("imagesadj/");
        while (totalTime > 0)
        {
            // on last step, arrive exactly
            if (totalTime + solver.deltaT < 0)
            {
                solver.deltaT = totalTime;
                solver.UpdateForTimestep();
                done = true;
            }

            solver.TimeStepAdjoint(totalTime);
            totalTime -= solver.deltaT;

            if(step%50==0)
            {
                stratifloat cfl = solver.CFLadjoint();
                std::cout << "  Step " << step << ", time " << totalTime
                        << ", CFL number: " << cfl << std::endl;

                std::cout << "  Average timings: " << solver.totalForcing / (step+1)
                        << ", " << solver.totalExplicit / (step+1)
                        << ", " << solver.totalImplicit / (step+1)
                        << ", " << solver.totalDivergence / (step+1)
                        << std::endl;
            }

            int frame = static_cast<int>(totalTime / saveEvery);

            if (frame<lastFrame)
            {
                lastFrame=frame;

                solver.PlotAll(std::to_string(totalTime)+".png", false);
            }

            step++;

            if (done)
            {
                break;
            }
        }

        previousv1 = solver.u1;
        previousv2 = solver.u2;
        previousv3 = solver.u3;
        previousvb = solver.b;

        stratifloat residual = solver.Optimise(epsilon, E0, u10, u20, u30, b0, backgroundB);

        energyFile << "STEP " << p << " and residual= " << residual << std::endl;

        if (residual < residualTarget)
        {
            std::cout << "Successfully converged" << std::endl;
            break;
        }
    }

    f3_cleanup_threads();

    return 0;
}
