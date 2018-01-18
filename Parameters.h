#pragma once
#include "Constants.h"

// These are the used-modifiable parameters for Stratiflow

// SOLVER PARAMETERS //
constexpr int N1 = 384; // Number of streamwise gridpoints
constexpr int N2 = 1;   // Number of spanwise gridpoints
constexpr int N3 = 440; // Number of vertical gridpoints

constexpr stratifloat L1 = 14.228; // size of domain streamwise
constexpr stratifloat L2 = 4.0f;  // size of domain spanwise
constexpr stratifloat L3 = 5.0f; // vertical scaling factor

constexpr bool ThreeDimensional = false; // whether to resolve spanwise direction

constexpr bool SnapshotToMemory = false;

constexpr EnergyType EnergyConstraint = EnergyType::MadeUp;

// FLOW PARAMETERS //
constexpr stratifloat Re = 1000;
constexpr stratifloat Ri = 0.1;
constexpr stratifloat Pe = Re; // Pe != Re not yet supported!
constexpr stratifloat R = 1;

// background shear
inline stratifloat InitialU(stratifloat z)
{
    return tanh(z);
}

// background stratification
inline stratifloat InitialB(stratifloat z)
{
    return -tanh(R*z);
}
