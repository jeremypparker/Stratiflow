#pragma once
#include "Constants.h"
#include <string>

// These must be defined at compile time
struct GridParams
{
    int N1; // Number of streamwise gridpoints
    int N2; // Number of spanwise gridpoints
    int N3; // Number of vertical gridpoints
    Dimensionality dimensionality;

    bool ThirdDimension() const // whether to resolve spanwise direction
    {
        return dimensionality == Dimensionality::ThreeDimensional
            || dimensionality == Dimensionality::TwoAndAHalf;
    }
};

// In principle these can be changed mid-run
struct FlowParams
{
    stratifloat L1; // streamwise (periodic) domain size
    stratifloat L2; // spanwise (periodic domain size)
    stratifloat L3; // vertical domain half-height
    stratifloat Re; // Reynolds number
    stratifloat Ri; // bulk Richardson number
    stratifloat Pr; // Prandtl number
    bool EvolveBackground;
};

// background shear
inline stratifloat InitialU(stratifloat z)
{
    return tanh(z);
}

void DumpParameters();
void PrintParameters();
void LoadParameters(const std::string& file);

constexpr GridParams gridParams
    = {256, 2, 768, Dimensionality::TwoAndAHalf};

extern FlowParams flowParams;
