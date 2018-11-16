#include "StateVector.h"

int main(int argc, char* argv[])
{
    // backwards integrates the adjoint equation between two given direct snapshots

    // Load state at tmax of interval
    std::string filenameabove(argv[1]);
    int extension = filenameabove.find(".fields");
    int hyphen = filenameabove.find_last_of("-");
    int slash = filenameabove.find_last_of("/");

    int stepabove = std::stoi(filenameabove.substr(slash+1, hyphen-slash-1));
    stratifloat timeabove = std::stof(filenameabove.substr(hyphen+1, extension-hyphen-1));

    // Load state at tmin
    std::string filenamebelow(argv[2]);
    extension = filenamebelow.find(".fields");
    hyphen = filenamebelow.find_last_of("-");
    slash = filenamebelow.find_last_of("/");

    int stepbelow = std::stoi(filenamebelow.substr(slash+1, hyphen-slash-1));
    stratifloat timebelow = std::stof(filenamebelow.substr(hyphen+1, extension-hyphen-1));

    std::cout << "Between " << timebelow << " and " << timeabove << std::endl;
    stratifloat steps = stepabove - stepbelow;
    stratifloat deltaT = (timeabove - timebelow)/steps;
    std::cout << steps << " steps, deltaT=" << deltaT << std::endl;

    // Fill in the gaps by doing extra forward integration
    StateVector directState;
    directState.LoadFromFile(filenamebelow);

    std::vector<StateVector> intermediateStates;
    directState.FixedEvolve(deltaT, steps, intermediateStates);

    directState.LoadFromFile(filenameabove);
    intermediateStates.push_back(directState);

    // Now do adjoint integration
    StateVector adjointState;
    adjointState.LoadFromFile("adjoint-"+std::to_string(stepabove)+".fields");
    adjointState.PlotAll(std::to_string(timeabove));
    adjointState.AdjointEvolve(deltaT, steps, intermediateStates, adjointState);
    adjointState.SaveToFile("adjoint-"+std::to_string(stepbelow)+".fields");
    adjointState.PlotAll(std::to_string(timebelow));
}