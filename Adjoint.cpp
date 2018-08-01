#include "StateVector.h"
#include <algorithm>

struct Snapshot
{
    int step;
    stratifloat time;
    std::string filename;
};

int main(int argc, char* argv[])
{
    std::vector<Snapshot> snapshots;

    auto dir = opendir(argv[1]);
    struct dirent* file = nullptr;
    while((file=readdir(dir)))
    {
        std::string filename(file->d_name);
        int extension = filename.find(".fields");
        int hyphen = filename.find("-");

        if (extension!=-1 && hyphen!=-1)
        {
            Snapshot snapshot;
            snapshot.step = std::stoi(filename.substr(0, hyphen));
            snapshot.time = std::stof(filename.substr(hyphen+1, extension-hyphen-1));
            snapshot.filename = argv[1]+std::string("/")+filename;

            snapshots.push_back(snapshot);
        }
    }
    closedir(dir);

    std::sort(snapshots.begin(), snapshots.end(), [](Snapshot& a, Snapshot& b){return a.step<b.step;});

    std::vector<StateVector> intermediateStates;
    StateVector directState;
    StateVector adjointState;

    for (auto snapshot = snapshots.end(); std::prev(snapshot)!=snapshots.begin(); snapshot--)
    {
        const Snapshot& shotabove = *std::prev(snapshot);
        const Snapshot& shotbelow = *std::prev(std::prev(snapshot));

        std::cout << "Between " << shotbelow.time << " and " << shotabove.time << std::endl;
        stratifloat steps = shotabove.step - shotbelow.step;
        stratifloat deltaT = (shotabove.time - shotbelow.time)/steps;

        std::cout << "Timestep " << deltaT << std::endl;

        // Fill in the gaps by doing extra forward integration
        directState.LoadFromFile(shotbelow.filename);
        directState.FixedEvolve(deltaT, steps, intermediateStates);

        directState.LoadFromFile(shotabove.filename);
        intermediateStates.push_back(directState);

        // Now do adjoint integration
        adjointState.PlotAll(std::to_string(shotabove.time));
        adjointState.AdjointEvolve(deltaT, steps, intermediateStates, adjointState);
    }

    directState.LoadFromFile(snapshots[0].filename);

    // now in directState and adjointState we should have both at t=0

    // perform optimisation
    stratifloat epsilon = std::stof(argv[2]);

    // scale for more uniform updating
    adjointState.b *= 1/Ri;

    stratifloat udotu = directState.Dot(directState);
    stratifloat udotv = directState.Dot(adjointState);
    stratifloat vdotv = adjointState.Dot(adjointState);

    stratifloat lambda = 0;
    while(lambda==0)
    {
        lambda = SolveQuadratic(epsilon*udotu,
                                2*epsilon*udotv - 2*udotu,
                                epsilon*vdotv - 2*udotv);
        if (lambda==0)
        {
            std::cout << "Reducing step size" << std::endl;
            epsilon /= 2;
        }
    }

    StateVector deriv = adjointState + lambda*directState;
    StateVector result = directState - epsilon*deriv;

    stratifloat residual = deriv.Norm2();

    SaveValueToFile(residual, "residual");

    result.Rescale(directState.Energy());

    result.SaveToFile("final");
}