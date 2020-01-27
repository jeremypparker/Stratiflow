#pragma once
#include "Constants.h"
#include <string>

// These must be defined at compile time
struct GridParams
{
    int N1; // Number of streamwise gridpoints
    int N2; // Number of spanwise gridpoints
    int N3; // Number of vertical gridpoints
    bool ThreeDimensional; // whether to resolve spanwise direction
};

// In principle these can be changed mid-run
struct FlowParams
{
    stratifloat L1; // streamwise (periodic) domain size
    stratifloat L2; // spanwise (periodic) domain size
    stratifloat L3; // vertical (periodic) domain size
    stratifloat Re; // Reynolds number
    stratifloat Ri; // bulk Richardson number
    stratifloat Pr; // Prandtl number
};


void DumpParameters();
void PrintParameters();
void LoadParameters(const std::string& file);

constexpr GridParams gridParams
    = {384, 24, 96, true};

extern FlowParams flowParams;
