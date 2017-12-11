#include <catch.h>

#include "Field.h"

TEST_CASE("Basic Field")
{
    Field<stratifloat, 3, 4, 5> f1(BoundaryCondition::Decaying); // constructor

    f1.slice(1).setRandom();

    auto f2 = f1; // copy constructor
    REQUIRE(f2 == f1);

    Field<stratifloat, 3, 4, 5> f3(BoundaryCondition::Decaying);
    f3.slice(2).setRandom();

    f2 = f3; // assignment operator
    REQUIRE(f2 == f3);

    REQUIRE(f2 != f1);
}

TEST_CASE("Zero")
{
    NodalField<2, 8, 4> f1(BoundaryCondition::Bounded);
    for (int j=0; j<4; j++)
    {
        f1.slice(j).setRandom();
    }

    NodalField<2, 8, 4> f2(BoundaryCondition::Bounded);

    REQUIRE(f1 != f2);
    f1.Zero();
    REQUIRE(f1 == f2);
}

TEST_CASE("Slice and stack")
{
    Field<stratifloat, 2, 2, 2> f1(BoundaryCondition::Decaying);
    f1.slice(0) << 1, 2,
                   3, 4;
    f1.slice(1) << 2, 3,
                   4, 5;

    ArrayX expected(2);
    expected << 2, 3;

    ArrayX actual = f1.stack(0, 1);
    REQUIRE(actual.isApprox(expected));

    const auto& f2 = f1;
    actual = f2.stack(0,1); // this time returns ConstStack
    REQUIRE(actual.isApprox(expected));

    REQUIRE(f1(1,0,1) == 4);
    REQUIRE(f1(0,1,1) == 3);
}

TEST_CASE("Stackwise Matmul")
{
    NodalField<5, 6, 8> f1(BoundaryCondition::Decaying);
    NodalField<5, 6, 8> f2(BoundaryCondition::Decaying);

    DiagonalMatrix<stratifloat,-1> mat = VectorX::Constant(8, 5.0f).asDiagonal();

    f2 = Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>,stratifloat,stratifloat,5,6,8>(mat, f1);

    f1 *= 5.0f;
    REQUIRE(f2 == f1);
}

TEST_CASE("Multiply Add")
{
    ModalField<5,6,8> f1(BoundaryCondition::Bounded);
    ModalField<5,6,8> f2(BoundaryCondition::Bounded);
    for (int j=0; j<8; j++)
    {
        f1.slice(j).setConstant(5);
        f2.slice(j).setConstant(1);
    }

    f1 += (-3.0f)*f2;

    f2 *= 2;

    REQUIRE(f1 == f2);
}

TEST_CASE("Decaying Modal/Nodal")
{
    constexpr int N1 = 4;
    constexpr int N2 = 1;
    constexpr int N3 = 32;
    NodalField<N1, N2, N3> f1(BoundaryCondition::Decaying);
    for (int j1=0; j1<N1; j1++)
    {
        for (int j2=0; j2<N2; j2++)
        {
            f1.stack(j1, j2).setConstant(j1);
        }
    }

    // because it's decaying
    f1.slice(0).setZero();
    f1.slice(N3-1).setZero();

    ModalField<N1, N2, N3> f2(BoundaryCondition::Decaying);
    f1.ToModal(f2);

    NodalField<N1, N2, N3> f3(BoundaryCondition::Decaying);
    f2.ToNodal(f3);

    REQUIRE(f1 == f3);
}

TEST_CASE("Bounded Modal/Nodal")
{
    NodalField<2, 8, 4> f1(BoundaryCondition::Bounded);
    for (int j=0; j<4; j++)
    {
        f1.slice(j).setConstant(j);
    }

    ModalField<2, 8, 4> f2(BoundaryCondition::Bounded);
    f1.ToModal(f2);

    NodalField<2, 8, 4> f3(BoundaryCondition::Bounded);
    f2.ToNodal(f3);

    REQUIRE(f1 == f3);
}

TEST_CASE("Max")
{
    NodalField<3,4,2> f1(BoundaryCondition::Bounded);

    f1.slice(0) << 3, 5, -1, 8,
                   2, 4, 2, 0,
                   2, 5, 3, 2;
    f1.slice(1) << 0, 5, -1, 8,
                   2, 4, 0, 0,
                   2, 5, 3, -10;

    REQUIRE(f1.Max() == 10);
}