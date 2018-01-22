#include "IMEXRK.h"

int main(int argc, char *argv[])
{
    const stratifloat energy = 0.001;

    IMEXRK solver;

    for (int p=0; p<1000; p++)
    {
        MField initialU1(BoundaryCondition::Bounded);
        MField initialU2(BoundaryCondition::Bounded);
        MField initialU3(BoundaryCondition::Decaying);
        MField initialB(BoundaryCondition::Bounded);

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<stratifloat> rng(-1.0,1.0);

        initialU1.RandomizeCoefficients((rng(generator)+1)/3);
        initialU2.RandomizeCoefficients((rng(generator)+1)/3);
        initialU3.RandomizeCoefficients((rng(generator)+1)/3);
        initialB.RandomizeCoefficients((rng(generator)+1)/3);

        initialU1 *= rng(generator);
        initialU2 *= rng(generator);
        initialU3 *= rng(generator);
        initialB *= rng(generator);

        solver.SetInitial(initialU1, initialU2, initialU3, initialB);
        solver.PopulateNodalVariables();

        solver.RemoveDivergence(0.0f);

        solver.SetBackground(InitialU, InitialB);
        solver.RescaleForEnergy(energy);

        stratifloat velocityNorm = sqrt(InnerProd(solver.u1, solver.u1, L3) + InnerProd(solver.u3, solver.u3, L3));
        stratifloat buoyancyNorm = sqrt(InnerProd(solver.b, solver.b, L3));

        std::cout << velocityNorm << " " << buoyancyNorm << std::endl;
    }

    return 0;
}
