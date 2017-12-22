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

    filestream.seekg(N1*N2*N3*3*sizeof(stratifloat));

    buoyancy.Load(filestream);
}

std::map<stratifloat, std::string> BuildFilenameMap()
{
    
    std::map<stratifloat, std::string> ret;
    auto dir = opendir("snapshots");
    struct dirent* file = nullptr;
    while((file=readdir(dir)))
    {
        std::string foundfilename(file->d_name);
        int end = foundfilename.find(".fields");
        stratifloat foundtime = strtof(foundfilename.substr(0, end).c_str(), nullptr);

        ret.insert(std::pair<stratifloat, std::string>(foundtime, "snapshots/"+foundfilename));
    }
    closedir(dir);

    return ret;
}

template<int N1, int N2, int N3>
NodalField<N1,N2,N3> GetBuoyancy(stratifloat time)
{
    static auto filenamemap = BuildFilenameMap();
    static auto entry = filenamemap.end();
    entry--;

    while(entry->first > time && entry != filenamemap.begin())
    {
        entry--;
    }

    static stratifloat timeabove = -1;
    static stratifloat timebelow = -1;

    static NodalField<N1,N2,N3> bAbove(BoundaryCondition::Bounded);
    static NodalField<N1,N2,N3> bBelow(BoundaryCondition::Bounded);

    if (timebelow != entry->first)
    {
        if (timebelow == std::next(entry)->first)
        {
            bAbove = bBelow;
        }   
        else
        {
            LoadBuoyancy(entry->second, bAbove);
        }
        LoadBuoyancy(std::next(entry)->second, bBelow);

        timeabove = entry->first;
        timebelow = std::next(entry)->first;
    }

    NodalField<N1,N2,N3> bNodal(BoundaryCondition::Bounded);

    bNodal = ((time-timebelow)/(timeabove-timebelow))*bAbove + ((timeabove-time)/(timeabove-timebelow))*bBelow;

    return bNodal;
}

int main(int argc, char *argv[])
{
    constexpr int N1 = 256;
    constexpr int N2 = 1;
    constexpr int N3 = 400;

    const stratifloat L1 = 32;
    const stratifloat L2 = 4;
    const stratifloat L3 = 8;

    //stratifloat time = strtof(argv[1], nullptr);

    ModalField<N1,N2,N3> bModal(BoundaryCondition::Bounded);

    matplotlibcpp::figure();

    for (stratifloat time = 20.0f; time>0.5f; time -= 0.5f)
    {    
        GetBuoyancy<N1,N2,N3>(time).ToModal(bModal);

        HeatPlot(bModal, L1, L3, 0, "output.png");
    }

    return 0;
}