#include "Stratiflow.h"

int main(int argc, char *argv[])
{
    stratifloat targetTime = 10.0;
    stratifloat energy = 0.001;

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

    if (argc > 2)
    {
        std::cout << "Loading ICs..." << std::endl;

        p = std::stoi(argv[2]);
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

        initialU1.SetValue([](stratifloat x, stratifloat y, stratifloat z)
        {
            return 0.1*cos(2*pi*x/IMEXRK::L1)*exp(-z*z);
        }, IMEXRK::L1, IMEXRK::L2, IMEXRK::L3);

        // stratifloat bandmax = 4;
        // for (int j=0; j<IMEXRK::N3; j++)
        // {
        //     if (x3(j) > -bandmax && x3(j) < bandmax)
        //     {
        //         initialU1.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
        //             * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
        //         initialU2.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
        //             * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
        //         initialU3.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
        //             * Array<stratifloat, IMEXRK::N1, IMEXRK::N2>::Random(IMEXRK::N1, IMEXRK::N2);
        //     }
        // }
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

    oldu1 = solver.u1;
    oldu2 = solver.u2;
    oldu3 = solver.u3;
    oldb = solver.b;

    stratifloat E0 = -1;

    stratifloat epsilon = 0.01;

    int maxiterations = 50;
    if (argc > 2)
    {
        maxiterations = std::stoi(argv[1]);
    }

    std::ofstream energyFile("energy.dat");
    for (; p<maxiterations; p++) // Direct-adjoint loop
    {
        exec("rm -rf images/u1 images/u2 images/u3 images/buoyancy images/pressure images/vorticity images/perturbvorticity");
        exec("rm -rf imagesadj/u1 imagesadj/u2 imagesadj/u3 imagesadj/buoyancy imagesadj/pressure");
        exec("rm -rf snapshots");
        exec("mkdir -p images/u1 images/u2 images/u3 images/buoyancy images/pressure images/vorticity images/perturbvorticity");
        exec("mkdir -p imagesadj/u1 imagesadj/u2 imagesadj/u3 imagesadj/buoyancy imagesadj/pressure");
        exec("mkdir -p snapshots");

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
        solver.StoreSnapshot(totalTime);

        // save initial condition
        solver.SaveFlow("ICs/"+std::to_string(p)+".fields");

        // also save images for animation
        solver.PlotPressure("images/pressure/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotBuoyancy("images/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotVerticalVelocity("images/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotSpanwiseVelocity("images/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
        solver.PlotSpanwiseVorticity("images/vorticity/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);

        stratifloat saveEvery = 1.0f;
        int lastFrame = -1;
        int step = 0;
        bool done = false;

        stratifloat JoverKintegrated = 0;

        solver.PrepareRun();
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

                solver.PlotPressure("images/pressure/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotBuoyancy("images/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotVerticalVelocity("images/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotSpanwiseVelocity("images/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotSpanwiseVorticity("images/vorticity/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotPerturbationVorticity("images/perturbvorticity/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);

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

            // it's going well, slowly increase step size
            epsilon *= 1.1;
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

        {
            IMEXRK::N1D Ubar(BoundaryCondition::Bounded);
            IMEXRK::N1D Bbar(BoundaryCondition::Bounded);
            solver.SetBackground(Ubar, Bbar);
        }

        totalTime = targetTime;
        lastFrame = 10000;

        step = 0;
        done = false;

        solver.PrepareRunAdjoint();
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
                solver.PlotBuoyancy("imagesadj/buoyancy/"+std::to_string(totalTime)+".png", IMEXRK::N2/2, false);
                solver.PlotVerticalVelocity("imagesadj/u3/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotSpanwiseVelocity("imagesadj/u2/"+std::to_string(totalTime)+".png", IMEXRK::N2/2);
                solver.PlotStreamwiseVelocity("imagesadj/u1/"+std::to_string(totalTime)+".png", IMEXRK::N2/2, false);
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
