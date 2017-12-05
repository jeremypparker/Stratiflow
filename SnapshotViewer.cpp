#include "Field.h"
#include "Graph.h"

#include <string>
#include <fstream>
#include <dirent.h>
#include <iostream>


template<int N1, int N2, int N3>
void LoadBuoyancy(std::string filename, NodalField<N1,N2,N3>& buoyancy)
{
    std::cout << "Loading " << filename << std::endl;

    std::ifstream filestream(filename, std::ios::in | std::ios::binary);

    filestream.seekg(N1*N2*N3*3*sizeof(float));

    buoyancy.Load(filestream);
}

int main(int argc, char *argv[])
{
    constexpr int N1 = 256;
    constexpr int N2 = 1;
    constexpr int N3 = 256;

    const float L1 = 32;
    const float L2 = 4;
    const float L3 = 6;

    float time = strtof(argv[1], nullptr);

    std::string filenameabove;
    std::string filenamebelow;
    float timeabove = 10000000000.0f;
    float timebelow = -1.0f;

    auto dir = opendir("snapshots");
    struct dirent* file = nullptr;
    while((file=readdir(dir)))
    {
        std::string foundfilename(file->d_name);
        
        int end = foundfilename.find(".fields");
        float foundtime = strtof(foundfilename.substr(0, end).c_str(), nullptr);

        if(foundtime >= time && foundtime < timeabove)
        {
            timeabove = foundtime;
            filenameabove = "snapshots/"+foundfilename;
        }
        if(foundtime < time && foundtime > timebelow)
        {
            timebelow = foundtime;
            filenamebelow = "snapshots/"+foundfilename;
        }
    }
    closedir(dir);

    NodalField<N1,N2,N3> bAbove(BoundaryCondition::Neumann);
    NodalField<N1,N2,N3> bBelow(BoundaryCondition::Neumann);

    LoadBuoyancy(filenameabove, bAbove);
    LoadBuoyancy(filenamebelow, bBelow);

    
    ModalField<N1,N2,N3> bModal(BoundaryCondition::Neumann);
    NodalField<N1,N2,N3> bNodal(BoundaryCondition::Neumann);

    bNodal = ((time-timebelow)/(timeabove-timebelow))*bAbove + ((timeabove-time)/(timeabove-timebelow))*bBelow;

    bNodal.ToModal(bModal);

    HeatPlot(bModal, L1, L3, 0, "output.png");

    return 0;
}