#include "Field.h"
#include "Graph.h"

#include <string>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <map>


template<int N1, int N2, int N3>
void LoadBuoyancy(std::string filename, NodalField<N1,N2,N3>& buoyancy)
{
    std::cout << "Loading " << filename << std::endl;

    std::ifstream filestream(filename, std::ios::in | std::ios::binary);

    filestream.seekg(N1*N2*N3*3*sizeof(float));

    buoyancy.Load(filestream);
}

std::map<float, std::string> BuildFilenameMap()
{
    
    std::map<float, std::string> ret;
    auto dir = opendir("snapshots");
    struct dirent* file = nullptr;
    while((file=readdir(dir)))
    {
        std::string foundfilename(file->d_name);
        int end = foundfilename.find(".fields");
        float foundtime = strtof(foundfilename.substr(0, end).c_str(), nullptr);

        ret.insert(std::pair<float, std::string>(foundtime, "snapshots/"+foundfilename));
    }
    closedir(dir);

    return ret;
}



int main(int argc, char *argv[])
{
    constexpr int N1 = 256;
    constexpr int N2 = 1;
    constexpr int N3 = 400;

    const float L1 = 32;
    const float L2 = 4;
    const float L3 = 8;

    float time = strtof(argv[1], nullptr);

    std::string filenameabove;
    std::string filenamebelow;
    float timeabove;
    float timebelow;

    auto filenamemap = BuildFilenameMap();

    for (auto entry = filenamemap.begin(); entry != filenamemap.end(); entry++)
    {
        if(entry->first>=time)
        {
            timeabove = entry->first;
            filenameabove = entry->second;

            entry--;
            timebelow = entry->first;
            filenamebelow = entry->second;

            break;
        }
    }

    NodalField<N1,N2,N3> bAbove(BoundaryCondition::Bounded);
    NodalField<N1,N2,N3> bBelow(BoundaryCondition::Bounded);

    LoadBuoyancy(filenameabove, bAbove);
    LoadBuoyancy(filenamebelow, bBelow);


    ModalField<N1,N2,N3> bModal(BoundaryCondition::Bounded);
    NodalField<N1,N2,N3> bNodal(BoundaryCondition::Bounded);

    bNodal = ((time-timebelow)/(timeabove-timebelow))*bAbove + ((timeabove-time)/(timeabove-timebelow))*bBelow;

    bNodal.ToModal(bModal);

    HeatPlot(bModal, L1, L3, 0, "output.png");

    return 0;
}