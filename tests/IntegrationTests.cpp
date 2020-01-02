#include <catch.h>

#include "Integration.h"

TEST_CASE("Integration of gaussian")
{
    stratifloat L3 = 2.0f;
    NodalField<1, 1, 32> gaussian;
    gaussian.SetValue([](stratifloat x, stratifloat y, stratifloat z){return exp(-z*z);}, L3, 1, 1);

    stratifloat integral = IntegrateAllSpace(gaussian, 1, 1, L3);

    REQUIRE(integral == Approx(sqrt(pi)));
}

TEST_CASE("Quadratic")
{
    REQUIRE(SolveQuadratic(1,-1,-6,true) == Approx(3.0));
    REQUIRE(SolveQuadratic(1,-1,-6,false) == Approx(-2.0));
}
