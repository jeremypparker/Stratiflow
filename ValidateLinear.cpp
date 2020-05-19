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

    Nodal U1;
    U1.SetValue([=](float x, float y, float z){return 0.5*omega*(m/k)*sin(k*x+m*z)+sin(z);}, flowParams.L1, flowParams.L2, flowParams.L3);
    U1.ToModal(state.u1);

    Nodal U3;
    U3.SetValue([=](float x, float y, float z){return -0.5*omega*sin(k*x+m*z);}, flowParams.L1, flowParams.L2, flowParams.L3);
    U3.ToModal(state.u3);

    Nodal B;
    B.SetValue([=](float x, float y, float z){return 0.5*cos(k*x+m*z);}, flowParams.L1, flowParams.L2, flowParams.L3);
    B.ToModal(state.b);

    StateVector endState;

    state.FullEvolve(T, endState, true, true);
    
    StateVector linearState;
    linearState.LoadFromFile(argv[2]);

    linearState.PlotAll("linearInitial");

    state = state + 0.0000001*linearState;

    state.FullEvolve(T, state, false, false);

    linearState = (1.0/0.0000001)*(state - endState);
    linearState.PlotAll("linearFinal");
}
