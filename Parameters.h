#pragma once
#include "Constants.h"
#include <string>

// These are the used-modifiable parameters for Stratiflow

// These are runtime parameters - values in Parameters.cpp
extern stratifloat L1;
extern stratifloat L2;
extern stratifloat L3;
extern stratifloat Re;
extern stratifloat Ri;
extern stratifloat R;
extern stratifloat Pr;
extern stratifloat Pe;
extern bool EnforceSymmetry;

// These must be defined at compile time

// SOLVER PARAMETERS //
constexpr int N1 = 128; // Number of streamwise gridpoints
constexpr int N2 = 1;   // Number of spanwise gridpoints
constexpr int N3 = 768; // Number of vertical gridpoints

constexpr bool ThreeDimensional = false; // whether to resolve spanwise direction
constexpr bool EvolveBackground = false;

// background shear
inline stratifloat InitialU(stratifloat z)
{
    return tanh(z);
}

void DumpParameters();
void PrintParameters();
void LoadParameters(const std::string& file);
