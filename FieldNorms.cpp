#include "IMEXRK.h"
#include "OSUtils.cpp"
#include "OrrSommerfeld.h"

int main(int argc, char *argv[])
{
    IMEXRK solver;
    solver.SetBackground(InitialU, InitialB);
    solver.PopulateNodalVariables();

    for (int p=0; p<50; p++)
    {
        solver.LoadFlow("ICs/"+std::to_string(p)+".fields");

        stratifloat velocityNorm = sqrt(InnerProd(solver.u1, solver.u1, L3) + InnerProd(solver.u3, solver.u3, L3));
        stratifloat buoyancyNorm = sqrt(InnerProd(solver.b, solver.b, L3));

        std::cout << velocityNorm << " " << buoyancyNorm << std::endl;
    }

    return 0;
}