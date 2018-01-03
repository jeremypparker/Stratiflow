#include "Stratiflow.h"

int main(int argc, char *argv[])
{
    stratifloat targetTime = 10.0;

    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    IMEXRK::MField oldu1(BoundaryCondition::Bounded);
    IMEXRK::MField oldu2(BoundaryCondition::Bounded);
    IMEXRK::MField oldu3(BoundaryCondition::Decaying);
    IMEXRK::MField oldb(BoundaryCondition::Bounded);

    IMEXRK::MField oldoldu1(BoundaryCondition::Bounded);
    IMEXRK::MField oldoldu2(BoundaryCondition::Bounded);
    IMEXRK::MField oldoldu3(BoundaryCondition::Decaying);
    IMEXRK::MField oldoldb(BoundaryCondition::Bounded);

    IMEXRK::M1D backgroundB(BoundaryCondition::Bounded);

    IMEXRK::MField previousv1(BoundaryCondition::Bounded);
    IMEXRK::MField previousv2(BoundaryCondition::Bounded);
    IMEXRK::MField previousv3(BoundaryCondition::Decaying);
    IMEXRK::MField previousvb(BoundaryCondition::Bounded);

    stratifloat previousIntegral = -1000;

    int p; // which DAL step we are on

    if (argc == 2)
    {
        std::cout << "Loading ICs..." << std::endl;

        p = std::stoi(argv[1]);
        solver.LoadFlow("ICs/"+std::to_string(p)+".fields");
    }
    else
    {
        std::cout << "Setting ICs..." << std::endl;

        exec("rm -rf ICs");
        exec("mkdir -p ICs");

        p = 0;

        IMEXRK::NField initialU1(BoundaryCondition::Bounded);
        IMEXRK::NField initialU2(BoundaryCondition::Bounded);
        IMEXRK::NField initialU3(BoundaryCondition::Decaying);
        IMEXRK::NField initialB(BoundaryCondition::Bounded);
        auto x3 = VerticalPoints(IMEXRK::L3, IMEXRK::N3);

        // add a perturbation to allow instabilities to develop

        stratifloat bandmax = 4;
        for (int j=0; j<IMEXRK::N3; j++)
        {
            if (x3(j) > -bandmax && x3(j) < bandmax)
            {
                initialU1.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
                initialU2.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
                initialU3.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
            }
        }
        solver.SetInitial(initialU1, initialU2, initialU3, initialB);
    }

    solver.RemoveDivergence(0.0f);

    oldu1 = solver.u1;
    oldu2 = solver.u2;
    oldu3 = solver.u3;
    oldb = solver.b;

    stratifloat E0 = -1;

    stratifloat epsilon = 0.01;

    std::ofstream energyFile("energy.dat");
    for (; p<100; p++) // Direct-adjoint loop
    {
        exec("rm -rf images/u1 images/u2 images/u3 images/buoyancy images/pressure");
        exec("rm -rf imagesadj/u1 imagesadj/u2 imagesadj/u3 imagesadj/buoyancy imagesadj/pressure");
        exec("rm -rf /local/scratch/public/jpp39/snapshots/*");
        exec("mkdir -p images/u1 images/u2 images/u3 images/buoyancy images/pressure");
        exec("mkdir -p imagesadj/u1 imagesadj/u2 imagesadj/u3 imagesadj/buoyancy imagesadj/pressure");

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

        //if (E0 == -1)
        //{
            E0 = solver.KE() + solver.PE();
        //}
        std::cout << "E0: " << (solver.KE() + solver.PE()) << std::endl;

        stratifloat totalTime = 0.0f;
        solver.SaveFlow("/local/scratch/public/jpp39/snapshots/"+std::to_string(totalTime)+".fields");

        // save initial condition
        solver.SaveFlow("ICs/"+std::to_string(p)+".fields", false);

        // also save images for animation
        solver.PlotPressure("images/pressure/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotBuoyancy("images/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotVerticalVelocity("images/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotSpanwiseVelocity("images/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);

        stratifloat saveEvery = 1.0f;
        int lastFrame = -1;
        int step = 0;
        solver.totalExplicit = 0;
        solver.totalImplicit = 0;
        solver.totalDivergence = 0;
        solver.totalForcing = 0;
        bool done = false;

        stratifloat JoverKintegrated = 0;

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

            solver.SaveFlow("/local/scratch/public/jpp39/snapshots/"+std::to_string(totalTime)+".fields");

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

                solver.PlotPressure("images/pressure/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotBuoyancy("images/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotVerticalVelocity("images/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotSpanwiseVelocity("images/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);

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

            oldoldu1 = oldu1;
            oldoldu2 = oldu2;
            oldoldu3 = oldu3;
            oldoldb = oldb;
        }
        else
        {
            // we have overshot, try with a smaller step
            epsilon /= 2;

            std::cout << "Epsilon: " << epsilon << std::endl;

            solver.u1 = previousv1;
            solver.u2 = previousv2;
            solver.u3 = previousv3;
            solver.b = previousvb;

            oldu1 = oldoldu1;
            oldu2 = oldoldu2;
            oldu3 = oldoldu3;
            oldb = oldoldb;

            energyFile << "STEP " << p << " and residual= "
                   << solver.Optimise(epsilon, E0, oldu1, oldu2, oldu3, oldb, backgroundB)
                   << std::endl;

            continue;
        }

        // clear everything for adjoint loop
        {
            solver.u1.Zero();
            solver.u2.Zero();
            solver.u3.Zero();
            solver.b.Zero();
            solver.p.Zero();
        }
        {
            IMEXRK::N1D Ubar(BoundaryCondition::Bounded);
            IMEXRK::N1D Bbar(BoundaryCondition::Bounded);
            solver.SetBackground(Ubar, Bbar);
        }

        totalTime = targetTime;
        lastFrame = 10000;

        solver.BuildFilenameMap();

        step = 0;
        solver.totalExplicit = 0;
        solver.totalImplicit = 0;
        solver.totalDivergence = 0;
        solver.totalForcing = 0;

        done = false;
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

                solver.PlotPressure("imagesadj/pressure/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotBuoyancy("imagesadj/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotVerticalVelocity("imagesadj/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotSpanwiseVelocity("imagesadj/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotStreamwiseVelocity("imagesadj/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
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

        energyFile << "STEP " << p << " and residual= "
                   << solver.Optimise(epsilon, E0, oldu1, oldu2, oldu3, oldb, backgroundB)
                   << std::endl;
    }

    f3_cleanup_threads();

    return 0;
}
