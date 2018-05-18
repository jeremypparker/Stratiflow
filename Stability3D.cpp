#include "Arnoldi.h"
#include "ExtendedStateVector.h"

int main(int argc, char* argv[])
{
    PrintParameters();

    ExtendedStateVector testState;
    testState.LoadFromFile(argv[1]);

    Ri = testState.p;

    BasicArnoldi solver;

    StateVector eigenMode;
    stratifloat growth = solver.Run(testState.x, eigenMode);

    if (growth > 1)
    {
        std::cout << "STATE IS UNSTABLE" << std::endl;
    }
    else
    {
        std::cout << "STATE IS STABLE" << std::endl;
    }
}

// stratifloat growth(stratifloat p)
// {
//     Ri = p;
//     StateVector background;
//     StateVector result;

//     BasicArnoldi solver;
//     return solver.Run(background, result);
// }

// int main(int argc, char *argv[])
// {
//     stratifloat RiLower = 0.245;
//     stratifloat RiHigher = 0.245625;

//     stratifloat GrowthLower = growth(RiLower);
//     stratifloat GrowthHigher = growth(RiHigher);

//     while (true)
//     {
//         stratifloat RiMiddle = 0.5*(RiLower+RiHigher);

//         stratifloat GrowthMiddle = growth(RiMiddle);

//         std::cout << "Ri: " << RiLower << " " << RiMiddle << " " << RiHigher << std::endl;
//         std::cout << "Growth: " << GrowthLower << " " << GrowthMiddle << " " << GrowthHigher << std::endl;

//         if (GrowthMiddle > 1)
//         {
//             RiLower = RiMiddle;
//             GrowthLower = GrowthMiddle;
//         }
//         else
//         {
//             RiHigher = RiMiddle;
//             GrowthHigher = GrowthMiddle;
//         }
//     }
// }
