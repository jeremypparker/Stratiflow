#include "StateVector.h"
#include <chrono>

int main(int argc, char* argv[])
{
    PrintParameters();


    // Load state at tmax of interval
    std::string filenameabove(argv[2]);
    int extension = filenameabove.find(".fields");
    int hyphen = filenameabove.find_last_of("-");
    int slash = filenameabove.find_last_of("/");

    int stepabove = std::stoi(filenameabove.substr(slash+1, hyphen-slash-1));
    stratifloat timeabove = std::stof(filenameabove.substr(hyphen+1, extension-hyphen-1));

    // Load state at tmin
    std::string filenamebelow(argv[1]);
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


    // Now do linear integration
    StateVector linearState;
    std::string linearFile = "direct-"+std::to_string(stepbelow)+".fields";

    if (FileExists(linearFile))
    {
        linearState.LoadFromFile(linearFile);
    }
    else
    {
        if(argc>3)
        {
            linearState.LoadFromFile(argv[3]);
        }
        else
        {
            linearState.ExciteLowWavenumbers(0.1);
        }

        stratifloat energy = 0.01;

        std::cout << "Before rescale, energy = " << linearState.Energy() << std::endl;

        linearState.Rescale(energy);
    }

    stratifloat initialEnergy = linearState.Energy();
    std::cout << initialEnergy << std::endl;

    linearState.PlotAll(std::to_string(timebelow));
    linearState.LinearEvolve(deltaT, steps, intermediateStates, linearState);
    linearState.SaveToFile("direct-"+std::to_string(stepabove)+".fields");
    linearState.SaveToFile("final.fields");
    linearState.PlotAll(std::to_string(timeabove));

    SaveValuesToFile({timeabove, linearState.Energy(), linearState.KE(), linearState.PE()}, "energies");
}
