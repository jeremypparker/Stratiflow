#pragma once
#include "Constants.h"

// These are the used-modifiable parameters for Stratiflow

// SOLVER PARAMETERS //
constexpr int N1 = 384; // Number of streamwise gridpoints
constexpr int N2 = 1;   // Number of spanwise gridpoints
constexpr int N3 = 440; // Number of vertical gridpoints

constexpr stratifloat L1 = 13.649; // size of domain streamwise
constexpr stratifloat L2 = 4.0f;  // size of domain spanwise
constexpr stratifloat L3 = 5.0f; // vertical scaling factor

constexpr bool ThreeDimensional = false; // whether to resolve spanwise direction

constexpr bool SnapshotToMemory = false;

constexpr EnergyType EnergyConstraint = EnergyType::Correct;

// FLOW PARAMETERS //
constexpr stratifloat Re = 1000;
constexpr stratifloat Ri = 1.0/6.0;
constexpr stratifloat R = 2.8284;
constexpr stratifloat Pr = R*R;
constexpr stratifloat Pe = Re*Pr;

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

constexpr stratifloat zlim = 8;

inline stratifloat zFunc(stratifloat z)
{
    if (z>zlim || z<-zlim)
    {
        return 0;
    }
    else
    {
        return z;
    }
}

inline stratifloat zFilter(stratifloat z)
{
    if (z>zlim || z<-zlim)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}
