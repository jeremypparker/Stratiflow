#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    StateVector::ResetForParams();
    DumpParameters();

    stratifloat T = std::stof(argv[1]);

    StateVector state;

    stratifloat k = 0.25;
    stratifloat m = 3;
    stratifloat omega = sqrt(flowParams.Ri*k*k/(k*k+m*m));

    std::cout << "Horizontal phase speed " << omega/k << std::endl;


    stratifloat s = 0.75; 

    Nodal U1;
    U1.SetValue([=](float x, float y, float z){return (s*omega/k)*sin(k*x+m*z)+sin(z);}, flowParams.L1, flowParams.L2, flowParams.L3);
    U1.ToModal(state.u1);

    Nodal U3;
    U3.SetValue([=](float x, float y, float z){return -(s*omega/m)*sin(k*x+m*z);}, flowParams.L1, flowParams.L2, flowParams.L3);
    U3.ToModal(state.u3);

    Nodal B;
    B.SetValue([=](float x, float y, float z){return (s/m)*cos(k*x+m*z);}, flowParams.L1, flowParams.L2, flowParams.L3);
    B.ToModal(state.b);

    state.FullEvolve(T, state, true, true);
}
