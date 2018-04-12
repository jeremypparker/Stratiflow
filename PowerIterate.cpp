#include "StateVector.h"

int main(int argc, char* argv[])
{
    Ri = std::stof(argv[2]);

    StateVector stationaryPoint;
    stationaryPoint.LoadFromFile(argv[1]);

    // saved field does not include background shear/stratification about which we want to linearise
    stationaryPoint.AddBackground();

    StateVector perturbation;
    perturbation.Randomise(0.0001, true);

    perturbation.LinearEvolveFixed(1000, stationaryPoint, perturbation);
}