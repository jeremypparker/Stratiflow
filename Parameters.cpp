#include "Parameters.h"
#include <fstream>

// These are runtime parameters - could load from a file in future

stratifloat L1 = 12;// size of domain streamwise
stratifloat L2 = 4.0f;  // size of domain spanwise
stratifloat L3 = 8.0f; // half-size of domain vertically
stratifloat Re = 500;
stratifloat Ri = 0.16;
stratifloat R = 1;
stratifloat Pr = R*R;
stratifloat Pe = Re*Pr;
bool EnforceSymmetry = false;

#define PrintParam(parameter) paramFile << #parameter << " " << parameter << std::endl

void DumpParameters()
{
    std::ofstream paramFile;
    paramFile.open("params.dat", std::fstream::out);

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
