#include <catch.h>

#include "Integration.h"

TEST_CASE("Integration of gaussian")
{
    float L3 = 2.0f;
    NodalField<1, 1, 31> gaussian(BoundaryCondition::Decaying);
    gaussian.SetValue([](float z){return exp(-z*z);}, L3);

    float integral = IntegrateAllSpace(gaussian, 1, 1, L3);

    REQUIRE(integral == Approx(sqrt(pi)));
}