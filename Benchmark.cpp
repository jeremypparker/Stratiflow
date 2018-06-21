#include "StateVector.h"
#include <chrono>

int main(int argc, char* argv[])
{
    PrintParameters();

    StateVector result;
    StateVector state;

    DirichletNodal U3;
    U3.SetValue([](stratifloat x, stratifloat y, stratifloat z){
        return cos(2*pi*x/L1)*cos(2*pi*y/L2)*exp(-z*z);
    }, L1, L2, L3);

    U3.ToModal(state.u3);


    auto tBefore = std::chrono::high_resolution_clock::now();
    state.FullEvolve(5, result, false, true);
    auto tAfter = std::chrono::high_resolution_clock::now();

    long totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(tAfter-tBefore).count();

    std::cout << "Total time: " << totalTime << std::endl;
}
