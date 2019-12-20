#include "Parameters.h"
#include <fstream>
#include <iostream>
#include <iomanip>

FlowParams flowParams
    = {8.885765876, 1.570795, 10, 1000, 0.16, 0.7, false};

#define PrintParam(parameter) paramFile << #parameter << " " << parameter << std::endl

void LoadParamInto(std::ifstream& paramFile, const std::string& name, stratifloat& into)
{
    paramFile.seekg(0);
    for (std::string line; std::getline(paramFile, line);)
    {
        if (line.substr(0, name.length()) == name)
        {
            into = std::stof(line.substr(name.length()+1));
            return;
        }
    }

    std::cerr << "Could not load parameter " << name << std::endl;
}

void LoadParamInto(std::ifstream& paramFile, const std::string& name, int& into)
{
    paramFile.seekg(0);
    for (std::string line; std::getline(paramFile, line);)
    {
        if (line.substr(0, name.length()) == name)
        {
            into = std::stoi(line.substr(name.length()+1));
            return;
        }
    }

    std::cerr << "Could not load parameter " << name;
}

void CheckParameter(std::ifstream& paramFile, const std::string& name, int check)
{
    int val;
    LoadParamInto(paramFile, name, val);

    if (val != check)
    {
        throw std::string("Error: parameter ") + name + " not correct";
    }
}

#define CheckParam(parameter) CheckParameter(paramFile, #parameter, parameter)
#define LoadParam(parameter) LoadParamInto(paramFile, #parameter, parameter)

void DumpParameters()
{
    std::ofstream paramFile;
    paramFile.open("params.dat", std::fstream::out);

    paramFile << std::setprecision(30);

    PrintParam(gridParams.N1);
    PrintParam(gridParams.N2);
    PrintParam(gridParams.N3);
    PrintParam(gridParams.ThreeDimensional);

    PrintParam(flowParams.L1);
    PrintParam(flowParams.L2);
    PrintParam(flowParams.L3);
    PrintParam(flowParams.Re);
    PrintParam(flowParams.Ri);
    PrintParam(flowParams.Pr);
    PrintParam(flowParams.EvolveBackground);
}

void LoadParameters(const std::string& file)
{
    std::ifstream paramFile;
    paramFile.open(file, std::fstream::in);

    CheckParam(gridParams.N1);
    CheckParam(gridParams.N2);
    CheckParam(gridParams.N3);
    CheckParam(gridParams.ThreeDimensional);

    // LoadParam(flowParams.L1);
    // LoadParam(flowParams.L2);
    LoadParam(flowParams.L3);
    // LoadParam(flowParams.Re);
    // LoadParam(flowParams.Ri);
    CheckParam(flowParams.EvolveBackground);
}

void PrintParameters()
{
    auto& paramFile = std::cout;

    PrintParam(gridParams.N1);
    PrintParam(gridParams.N2);
    PrintParam(gridParams.N3);
    PrintParam(gridParams.ThreeDimensional);

    PrintParam(flowParams.L1);
    PrintParam(flowParams.L2);
    PrintParam(flowParams.L3);
    PrintParam(flowParams.Re);
    PrintParam(flowParams.Ri);
    PrintParam(flowParams.Pr);
    PrintParam(flowParams.EvolveBackground);
}
