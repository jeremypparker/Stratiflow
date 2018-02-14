#include "IMEXRK.h"

int main(int argc, char *argv[])
{
    const stratifloat targetTime = 47.0;
    const stratifloat energy = 0.001;
    const stratifloat residualTarget = 0.001;
    const stratifloat minimumEpsilon = 0.0000001;

    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    // these are the initial conditions for the current step
    MField u10(BoundaryCondition::Bounded);
    MField u20(BoundaryCondition::Bounded);
    MField u30(BoundaryCondition::Decaying);
    MField b0(BoundaryCondition::Bounded);

    // these are the initial conditions for the previous step - so we can reset if necessary
    MField previousu10(BoundaryCondition::Bounded);
    MField previousu20(BoundaryCondition::Bounded);
    MField previousu30(BoundaryCondition::Decaying);
    MField previousb0(BoundaryCondition::Bounded);


    MField previousv1(BoundaryCondition::Bounded);
    MField previousv2(BoundaryCondition::Bounded);
    MField previousv3(BoundaryCondition::Decaying);
    MField previousvb(BoundaryCondition::Bounded);

    stratifloat previousIntegral = -1000;

    M1D backgroundB(BoundaryCondition::Bounded);
    M1D backgroundU(BoundaryCondition::Bounded);

    {
        N1D Bbar(BoundaryCondition::Bounded);
        Bbar.SetValue(InitialB, L3);
        Bbar.ToModal(backgroundB);

        N1D Ubar(BoundaryCondition::Bounded);
        Ubar.SetValue(InitialU, L3);
        Ubar.ToModal(backgroundU);
    }


    // first optional parameter is the maximum number of DAL loops
    int maxiterations = 50;
    if (argc > 1)
    {
        maxiterations = std::stoi(argv[1]);
    }

    // second optional parameter is which old initial condition to load
    int p; // which DAL step we are on
    std::ofstream energyFile; // file to which to output energy and residuals
    if (argc > 2)
    {
        std::cout << "Loading ICs..." << std::endl;

        p = std::stoi(argv[2]);
        solver.LoadFlow("ICs/"+std::to_string(p)+".fields");

        // in this case, only append to the existing energy file
        energyFile.open("energy.dat", std::fstream::out | std::fstream::app);
    }
    else
    {
        // if none supplied, set ICs manually
        std::cout << "Setting ICs..." << std::endl;

        // fresh run, so clean up old stuff
        MakeCleanDir("ICs");

        p = 0;

        MField initialU1(BoundaryCondition::Bounded);
        MField initialU2(BoundaryCondition::Bounded);
        MField initialU3(BoundaryCondition::Decaying);
        MField initialB(BoundaryCondition::Bounded);

        // put energy in the lowest third of the spatial modes
        initialU1.RandomizeCoefficients(0.3);
        initialU2.RandomizeCoefficients(0.3);
        initialU3.RandomizeCoefficients(0.3);
        initialB.RandomizeCoefficients(0.3);

        solver.SetInitial(initialU1, initialU2, initialU3, initialB);

        solver.RemoveDivergence(0.0f);

        // in this case, overwrite any old file
        energyFile.open("energy.dat", std::fstream::out);
    }

    stratifloat epsilon = 0.05; // gradient ascent step size

    for (; p<maxiterations; p++) // Direct-adjoint loop
    {
        // add background flow
        solver.SetBackground(InitialU, InitialB);

        stratifloat totalTime = 0.0f;


        stratifloat saveEvery = 1.0f;
        int lastFrame = 0;
        int step = 0;
        bool done = false;

        stratifloat JoverKintegrated = 0;

        if (EnforceSymmetry)
        {
            // slight hack: we expect all optimals to have symmetry, so enforce this
            solver.u1.Antisymmetrise();
            solver.u2.Antisymmetrise();
            solver.u3.Antisymmetrise();
            solver.b.Antisymmetrise();
        }

        std::cout << "E0: " << solver.KE() + solver.PE() << std::endl;
        solver.RescaleForEnergy(energy); // rescale to ensure we don't drift
        std::cout << "E0 (after rescale): " << solver.KE() + solver.PE() << std::endl;

        solver.PrepareRun("images/");

        // save initial condition
        solver.StoreSnapshot(totalTime);
        solver.SaveFlow("ICs/"+std::to_string(p)+".fields");
        energyFile << totalTime
                   << " " << solver.KE()
                   << " " << solver.PE()
                   << " " << solver.JoverK()
                   << std::endl;

        u10 = solver.u1;
        u20 = solver.u2;
        u30 = solver.u3;
        b0 = solver.b;

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
                   << solver.Optimise(epsilon, energy, u10, u20, u30, b0, backgroundB, backgroundU)
                   << std::endl;

            continue;
        }

        {
            N1D Ubar(BoundaryCondition::Bounded);
            N1D Bbar(BoundaryCondition::Bounded);
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
            if (totalTime - solver.deltaT < 0)
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

        stratifloat residual = solver.Optimise(epsilon, energy, u10, u20, u30, b0, backgroundB, backgroundU);

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
