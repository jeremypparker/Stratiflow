#include <catch.h>

#include "Integration.h"

TEST_CASE("Integration of gaussian")
{
    stratifloat L3 = 2.0f;
    NodalField<1, 1, 32> gaussian(BoundaryCondition::Decaying);
    gaussian.SetValue([](stratifloat z){return exp(-z*z);}, L3);

    stratifloat integral = IntegrateAllSpace(gaussian, 1, 1, L3);

    REQUIRE(integral == Approx(sqrt(pi)));
}