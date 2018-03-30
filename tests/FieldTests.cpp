#include <catch.h>

#include "Field.h"

TEST_CASE("Basic Field")
{
    Field<stratifloat, 3, 4, 5> f1(BoundaryCondition::Dirichlet); // constructor

    f1.slice(1).setRandom();

    auto f2 = f1; // copy constructor
    REQUIRE(f2 == f1);

    Field<stratifloat, 3, 4, 5> f3(BoundaryCondition::Dirichlet);
    f3.slice(2).setRandom();

    f2 = f3; // assignment operator
    REQUIRE(f2 == f3);

    REQUIRE(f2 != f1);
}

TEST_CASE("Zero")
{
    NodalField<2, 8, 4> f1(BoundaryCondition::Dirichlet);
    for (int j=0; j<4; j++)
    {
        f1.slice(j).setRandom();
    }

    NodalField<2, 8, 4> f2(BoundaryCondition::Dirichlet);

    REQUIRE(f1 != f2);
    f1.Zero();
    REQUIRE(f1 == f2);
}

TEST_CASE("Slice and stack")
{
    Field<stratifloat, 2, 2, 2> f1(BoundaryCondition::Dirichlet);
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
    NodalField<5, 6, 8> f1(BoundaryCondition::Dirichlet);
    NodalField<5, 6, 8> f2(BoundaryCondition::Dirichlet);

    DiagonalMatrix<stratifloat,-1> mat = VectorX::Constant(8, 5.0f).asDiagonal();

    f2 = Dim3MatMul<Map<const Array<stratifloat, -1, 1>, Aligned16>,stratifloat,stratifloat,5,6,8>(mat, f1);

    f1 *= 5.0f;
    REQUIRE(f2 == f1);
}

TEST_CASE("Multiply Add")
{
    ModalField<5,6,8> f1(BoundaryCondition::Dirichlet);
    ModalField<5,6,8> f2(BoundaryCondition::Dirichlet);
    for (int j=0; j<8; j++)
    {
        f1.slice(j).setConstant(5);
        f2.slice(j).setConstant(1);
    }

    f1 += (-3.0f)*f2;

    f2 *= 2;

    REQUIRE(f1 == f2);
}

TEST_CASE("Modal/Nodal")
{
    NodalField<4, 8, 4> f1(BoundaryCondition::Dirichlet);
    for (int j1=0; j1<4; j1++)
    {
        for (int j2=0; j2<8; j2++)
        {
            f1.stack(j1, j2).setConstant(j1*j2);
        }
    }

    ModalField<4, 8, 4> f2(BoundaryCondition::Dirichlet);
    f1.ToModal(f2, false);

    NodalField<4, 8, 4> f3(BoundaryCondition::Dirichlet);
    f2.ToNodal(f3);

    std::cout << f1.slice(0) << std::endl << std::endl;
    std::cout << f3.slice(0) << std::endl << std::endl << std::endl;

    std::cout << f1.slice(2) << std::endl << std::endl;
    std::cout << f3.slice(2) << std::endl << std::endl << std::endl;

    REQUIRE(f1 == f3);
}

TEST_CASE("Max")
{
    NodalField<3,4,2> f1(BoundaryCondition::Dirichlet);

    f1.slice(0) << 3, 5, -1, 8,
                   2, 4, 2, 0,
                   2, 5, 3, 2;
    f1.slice(1) << 0, 5, -1, 8,
                   2, 4, 0, 0,
                   2, 5, 3, -10;

    REQUIRE(f1.Max() == 10);
}