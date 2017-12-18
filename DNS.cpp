#include "Stratiflow.h"

int main()
{
    stratifloat targetTime = 150.0;

    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    IMEXRK::MField oldu1(BoundaryCondition::Bounded);
    IMEXRK::MField oldu2(BoundaryCondition::Bounded);
    IMEXRK::MField oldu3(BoundaryCondition::Decaying);
    IMEXRK::MField oldb(BoundaryCondition::Decaying);

    IMEXRK::M1D backgroundB(BoundaryCondition::Bounded);

    std::cout << "Setting ICs..." << std::endl;
    {
        IMEXRK::NField initialU1(BoundaryCondition::Bounded);
        IMEXRK::NField initialU2(BoundaryCondition::Bounded);
        IMEXRK::NField initialU3(BoundaryCondition::Decaying);
        IMEXRK::NField initialB(BoundaryCondition::Decaying);
        auto x3 = VerticalPoints(IMEXRK::L3, IMEXRK::N3);

        // nudge with something like the eigenmode
        initialU3.SetValue([](stratifloat x, stratifloat y, stratifloat z){return 0.1*cos(2*pi*x/16.0f)/cosh(z)/cosh(z);}, IMEXRK::L1, IMEXRK::L2, IMEXRK::L3);

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
        solver.RemoveDivergence(0.0f);

        oldu1 = solver.u1;
        oldu2 = solver.u2;
        oldu3 = solver.u3;
        oldb = solver.b;
    }

    std::ofstream energyFile("energy.dat");

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
        Bbar.SetValue([R](stratifloat z){return tanh(R*z);}, IMEXRK::L3);

        solver.SetBackground(Ubar, Bbar);

        Bbar.ToModal(backgroundB);
    }

    stratifloat E0 = solver.KE() + solver.PE();

    std::cout << "E0: " << E0 << std::endl;

    stratifloat totalTime = 0.0f;
    solver.SaveFlow("/local/scratch/public/jpp39/snapshots/"+std::to_string(totalTime)+".fields");

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
    while (totalTime < targetTime)
    {
        solver.TimeStep();
        totalTime += solver.deltaT;

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

    }


    f3_cleanup_threads();

    return 0;
}
