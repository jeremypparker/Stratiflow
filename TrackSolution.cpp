#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    Ri = std::stod(argv[1]);
    DumpParameters();

    StateVector state;
    if (argc>2)
    {
        state.LoadFromFile(argv[2]);
    }

    if (argc>3)
    {
        StateVector state2;
        state2.LoadFromFile(argv[3]);

        stratifloat mult=1;

        if (argc>4)
        {
            mult = std::stod(argv[4]);
        }

        state = state2 + mult*(state2-state);
    }

    StateVector perturbation;
    perturbation.ExciteLowWavenumbers(0.0001);

    if (argc == 2)
        state += perturbation;

    // things for Poincare section
    bool aboveSection = true;
    StateVector lastIntersection = state;
    stratifloat timeLastIntersection = 0;

    StateVector sectionOrigin = state;
    StateVector sectionNormal;
    bool haveSectionNormal = false;

    stratifloat timestep = 10;
    for (int n=0; n<3000; n++)
    {
        //state.PlotAll(std::to_string(n));

        std::cout << "Step " << n << " " << state.Energy() << " " << state.Enstrophy() << std::endl;

        state.FullEvolve(timestep, state, false, false);
       
        state.AddBackground(); 
        state.PlotAll(std::to_string(n));
        state.RemoveBackground();
//        if (!haveSectionNormal)
//        {
//            sectionNormal = state-sectionOrigin;
//            sectionNormal *= 1/sectionNormal.Norm();
//            haveSectionNormal = true;
//        }
//
//        // check if intersection Poincare section
//        bool nowAboveSection = (state-sectionOrigin).Dot(sectionNormal) > 0;
//
//       if (!aboveSection && nowAboveSection)
//        {
//            std::cout << "POINCARE INTERSECTION. Time since last="
//                      << (n*timestep - timeLastIntersection) << ", distance="
//                      << (state - lastIntersection).Norm() << std::endl;
//
//            lastIntersection = state;
//            timeLastIntersection = n*timestep;
//        }
//
//        aboveSection = nowAboveSection;
        state.SaveToFile("trackingresult");
    }
}
