#pragma once

#include <complex>

constexpr float pi = 3.14159265358979;
constexpr std::complex<float> i(0, 1);

enum class BoundaryCondition
{
    Dirichlet,
    Neumann
};

using complex = std::complex<float>;
