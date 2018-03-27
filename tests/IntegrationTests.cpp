#include <catch.h>

#include "Integration.h"

TEST_CASE("Integration of gaussian")
{
    stratifloat L3 = 2.0f;
    NodalField<1, 1, 32> gaussian;
    gaussian.SetValue([](stratifloat z){return exp(-z*z);}, L3);

    stratifloat integral = IntegrateAllSpace(gaussian, 1, 1, L3);

    REQUIRE(integral == Approx(sqrt(pi)));
}

TEST_CASE("Horizontal average")
{
    stratifloat L1 = 5.0f;
    stratifloat L2 = 4.0f;
    stratifloat L3 = 3.0f;

    constexpr int N1 = 40;
    constexpr int N2 = 12;
    constexpr int N3 = 28;

    NodalField<N1,N2,N3> f;
    f.SetValue([L1, L2](stratifloat x, stratifloat y, stratifloat z){
        return 5*exp(-z*z)+2*sin(2*pi*x/L1)+sin(2*pi*y/L2)*sin(2*pi*y/L2);
    }, L1, L2, L3);

    Nodal1D<N1,N2,N3> expected;
    expected.SetValue([L2](stratifloat z){return 5*exp(-z*z)+0.5;}, L3);

    ModalField<N1,N2,N3> fM;
    f.ToModal(fM);

    Nodal1D<N1,N2,N3> average;
    HorizontalAverage(fM, average);

    REQUIRE(average == expected);
}

TEST_CASE("Integrate vertically")
{
    stratifloat L3 = 2.0f;

    constexpr int N1 = 40;
    constexpr int N2 = 12;
    constexpr int N3 = 28;

    Nodal1D<N1,N2,N3> f;
    f.SetValue([](stratifloat z){return exp(-(z-1)*(z-1));}, L3);

    stratifloat integral = IntegrateVertically(f, L3);

    REQUIRE(integral == Approx(sqrt(pi)));
}

TEST_CASE("I test")
{
    constexpr int N1 = 320;
    constexpr int N2 = 1;
    constexpr int N3 = 440;

    constexpr stratifloat L1 = 16.0f;
    constexpr stratifloat L2 = 4.0f;
    constexpr stratifloat L3 = 5.0f;

    Nodal1D<N1,N2,N3> U_;
    NodalField<N1,N2,N3> U1;

    U_.SetValue([](stratifloat z){return tanh(z);}, L3);

    U1.SetValue([L1](stratifloat x, stratifloat y, stratifloat z){
        return 10*cos(2*pi*x/L1);
    }, L1, L2, L3);

    NodalField<N1,N2,N3> nnTemp;
    nnTemp = U1+U_;

    ModalField<N1,N2,N3> boundedTemp;
    nnTemp.ToModal(boundedTemp);

    Nodal1D<N1,N2,N3> ave;
    HorizontalAverage(boundedTemp, ave);

    Nodal1D<N1,N2,N3> one;
    one.SetValue([](stratifloat z){return 1;}, L3);

    Nodal1D<N1,N2,N3> integrand;
    integrand = one + (-1)*ave*ave;

    stratifloat result = IntegrateVertically(integrand, L3);

    REQUIRE(result == Approx(2.0).epsilon(0.001));
}

TEST_CASE("Quadratic")
{
    REQUIRE(SolveQuadratic(1,-1,-6,true) == Approx(3.0));
    REQUIRE(SolveQuadratic(1,-1,-6,false) == Approx(-2.0));
}
