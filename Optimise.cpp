#include "StateVector.h"

int main(int argc, char* argv[])
{
    StateVector directState;
    directState.LoadFromFile(argv[1]);
    directState.RemoveBackground();

    StateVector adjointState;
    adjointState.LoadFromFile(argv[2]);

    // now in directState and adjointState we should have both at t=0

    // perform optimisation
    stratifloat epsilon = std::stof(argv[3]);

    // scale for more uniform updating
    adjointState.b *= 1/Ri;

    stratifloat udotu = directState.Dot(directState);
    stratifloat udotv = directState.Dot(adjointState);
    stratifloat vdotv = adjointState.Dot(adjointState);

    stratifloat lambda = 0;
    while(lambda==0)
    {
        lambda = SolveQuadratic(epsilon*udotu,
                                -2*epsilon*udotv + 2*udotu,
                                epsilon*vdotv - 2*udotv,
                                true);
        if (lambda==0)
        {
            std::cout << "Reducing step size" << std::endl;
            epsilon /= 2;
        }
    }

    StateVector deriv = lambda*directState-adjointState;
    StateVector result = directState + epsilon*deriv;

    stratifloat residual = deriv.Norm2();

    SaveValueToFile(residual, "residual");

    result.Rescale(directState.Energy());

    result.SaveToFile("final");

    std::cout << "Current epsilon " << epsilon << std::endl;
    std::cout << "Bound epsilon " << -2*udotu/udotv << std::endl;

    stratifloat delta = (directState-result).Norm();
    std::cout << "Magnitude of change " << delta << std::endl;

    if (delta < 0.00001)
    {
        return 5;
    }

    return 0;
}