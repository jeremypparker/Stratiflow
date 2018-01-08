#include "Stratiflow.h"
#include "OSUtils.cpp"

int main(int argc, char *argv[])
{
    stratifloat targetTime = 200.0;

    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    if (argc == 2)
    {
        std::cout << "Loading ICs..." << std::endl;
        solver.LoadFlow(argv[1]);
    }
    else
    {
        std::cout << "Setting ICs..." << std::endl;
        IMEXRK::NField initialU1(BoundaryCondition::Bounded);
        IMEXRK::NField initialU2(BoundaryCondition::Bounded);
        IMEXRK::NField initialU3(BoundaryCondition::Decaying);
        IMEXRK::NField initialB(BoundaryCondition::Bounded);
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
    }

    solver.RemoveDivergence(0.0f);

    std::ofstream energyFile("energy.dat");

    MakeCleanDir("images/u1");
    MakeCleanDir("images/u2");
    MakeCleanDir("images/u3");
    MakeCleanDir("images/buoyancy");
    MakeCleanDir("images/vorticity");
    MakeCleanDir("images/perturbvorticity");

    // add background flow
    std::cout << "Setting background..." << std::endl;
    {
        stratifloat R = 2;

        IMEXRK::N1D Ubar(BoundaryCondition::Bounded);
        IMEXRK::N1D Bbar(BoundaryCondition::Bounded);
        Ubar.SetValue([](stratifloat z){return tanh(z);}, IMEXRK::L3);
        Bbar.SetValue([R](stratifloat z){return -tanh(R*z);}, IMEXRK::L3);

        solver.SetBackground(Ubar, Bbar);
    }

    stratifloat totalTime = 0.0f;

    stratifloat saveEvery = 1.0f;
    int lastFrame = -1;
    int step = 0;

    solver.PrepareRun();
    solver.PlotAll("images", std::to_string(totalTime)+".png", true);
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

            solver.PlotAll("images", std::to_string(totalTime)+".png", true);

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
