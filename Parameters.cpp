#include "Parameters.h"
#include <fstream>
#include <iostream>
#include <iomanip>

// These are runtime parameters - could load from a file in future

stratifloat L1 = 12.56637;// size of domain streamwise
stratifloat L2 = 4.0f;  // size of domain spanwise
stratifloat L3 = 15.0f; // half-size of domain vertically
stratifloat Re = 500;
stratifloat Ri = 0.24;
stratifloat R = 1;
stratifloat Pr = R*R;
stratifloat Pe = Re*Pr;
bool EnforceSymmetry = false;

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

    PrintParam(N1);
    PrintParam(N2);
    PrintParam(N3);
    PrintParam(ThreeDimensional);
    PrintParam(EvolveBackground);

    PrintParam(L1);
    PrintParam(L2);
    PrintParam(L3);
    PrintParam(Re);
    PrintParam(Ri);
    PrintParam(R);
}

void LoadParameters(const std::string& file)
{
    std::ifstream paramFile;
    paramFile.open(file, std::fstream::in);

    CheckParam(N1);
    CheckParam(N2);
    CheckParam(N3);
    CheckParam(ThreeDimensional);
    CheckParam(EvolveBackground);

    // LoadParam(L1);
    // LoadParam(L2);
    LoadParam(L3);
    // LoadParam(Re);
    // LoadParam(Ri);
    // LoadParam(R);
}

void PrintParameters()
{
    auto& paramFile = std::cout;

    PrintParam(N1);
    PrintParam(N2);
    PrintParam(N3);
    PrintParam(ThreeDimensional);
    PrintParam(EvolveBackground);

    PrintParam(L1);
    PrintParam(L2);
    PrintParam(L3);
    PrintParam(Re);
    PrintParam(Ri);
    PrintParam(R);
}
