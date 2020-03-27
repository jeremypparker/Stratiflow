#include "StateVector.h"

int main(int argc, char* argv[])
{
    StateVector adjointState;
    adjointState.LoadFromFile(argv[1]);

    StateVector directState;
    directState.LoadFromFile(argv[2]);

    StateVector result;

    result.u1 = -adjointState.u1;
    result.u2 = -adjointState.u2;
    result.u3 = -adjointState.u3;
    result.b = -adjointState.b;

    result.Rescale(0.01);

    std::cout << "Energy of final thing " << result.Energy() << std::endl;

    stratifloat residual = (directState-result).Norm2()/directState.Norm2();

    SaveValueToFile(residual, "residual");

    result.SaveToFile("final");

    //stratifloat delta = (directState-result).Norm();
    //std::cout << "Magnitude of change " << delta << std::endl;

    // if (delta < 0.00001)
    // {
    //     return 5;
    // }

    return 0;
}
