#pragma once

#include <complex>

constexpr double pi = 3.14159265358979;
constexpr std::complex<double> i(0, 1);

enum class BoundaryCondition
{
    Dirichlet,
    Neumann
};

using complex = std::complex<double>;
