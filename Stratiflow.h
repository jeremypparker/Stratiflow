#pragma once
#include "Field.h"
#include "Parameters.h"
#include "Differentiation.h"

constexpr int M1 = N1/2 + 1;

class NeumannNodal : public NodalField<N1,N2,N3>
{
public:
    NeumannNodal() : NodalField(BoundaryCondition::Neumann) {}
    using NodalField::operator=;
};

class NeumannModal : public ModalField<N1,N2,N3>
{
public:
    NeumannModal() : ModalField(BoundaryCondition::Neumann) {}
    using ModalField::operator=;
};

class DirichletNodal : public NodalField<N1,N2,N3>
{
public:
    DirichletNodal() : NodalField(BoundaryCondition::Dirichlet) {}
    using NodalField::operator=;

};

class DirichletModal : public ModalField<N1,N2,N3>
{
public:
    DirichletModal() : ModalField(BoundaryCondition::Dirichlet) {}
    using ModalField::operator=;
};

class Neumann1D : public Nodal1D<N1,N2,N3>
{
public:
    Neumann1D() : Nodal1D(BoundaryCondition::Neumann) {}
    using Nodal1D::operator=;
};

class Dirichlet1D : public Nodal1D<N1,N2,N3>
{
public:
    Dirichlet1D() : Nodal1D(BoundaryCondition::Dirichlet) {}
    using Nodal1D::operator=;
};

template<typename T>
Dim1MatMul<T, complex, complex, M1, N2, N3> ddx(const StackContainer<T, complex, M1, N2, N3>& f)
{
    static DiagonalMatrix<complex, -1> dim1Derivative = FourierDerivativeMatrix(L1, N1, 1);

    return Dim1MatMul<T, complex, complex, M1, N2, N3>(dim1Derivative, f);
}

template<typename T>
Dim2MatMul<T, complex, complex, M1, N2, N3> ddy(const StackContainer<T, complex, M1, N2, N3>& f)
{
    static DiagonalMatrix<complex, -1> dim2Derivative = FourierDerivativeMatrix(L2, N2, 2);

    return Dim2MatMul<T, complex, complex, M1, N2, N3>(dim2Derivative, f);
}

template<typename A, typename T, int K1, int K2, int K3>
Dim3MatMul<A, stratifloat, T, K1, K2, K3> ddz(const StackContainer<A, T, K1, K2, K3>& f)
{
    if (f.BC() == BoundaryCondition::Neumann)
    {
        static MatrixX dim3Derivative = VerticalDerivativeMatrix(L3, N3, f.BC());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3Derivative, f, BoundaryCondition::Dirichlet);
    }
    else
    {
        static MatrixX dim3Derivative = VerticalDerivativeMatrix(L3, N3, f.BC());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3Derivative, f, BoundaryCondition::Neumann);
    }
}

template<typename A, typename T, int K1, int K2, int K3>
Dim3MatMul<A, stratifloat, T, K1, K2, K3> Reinterpolate(const StackContainer<A, T, K1, K2, K3>& f)
{
    if (f.BC() == BoundaryCondition::Neumann)
    {
        static MatrixX reint = VerticalReinterpolationMatrix(L3, N3, f.BC());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(reint, f, BoundaryCondition::Dirichlet);
    }
    else
    {
        static MatrixX reint = VerticalReinterpolationMatrix(L3, N3, f.BC());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(reint, f, BoundaryCondition::Neumann);
    }
}

namespace
{
void InterpolateProduct(const NeumannNodal& A, const NeumannNodal& B, NeumannModal& to)
{
    static NeumannNodal prod;
    prod = A*B;
    prod.ToModal(to);
}
void InterpolateProduct(const NeumannNodal& A, const NeumannNodal& B, DirichletModal& to)
{
    static DirichletNodal prod;
    prod = Reinterpolate(A)*Reinterpolate(B);
    prod.ToModal(to);
}
void InterpolateProduct(const NeumannNodal& A, const DirichletNodal& B, DirichletModal& to)
{
    static DirichletNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const DirichletNodal& B, const NeumannNodal& A, DirichletModal& to)
{
    static DirichletNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const NeumannNodal& B, const DirichletNodal& A, NeumannModal& to)
{
    static NeumannNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const DirichletNodal& A, const NeumannNodal& B, NeumannModal& to)
{
    static NeumannNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const DirichletNodal& A, const DirichletNodal& B, NeumannModal& to)
{
    static NeumannNodal prod;
    prod = Reinterpolate(A)*Reinterpolate(B);
    prod.ToModal(to);
}
void InterpolateProduct(const DirichletNodal& A, const DirichletNodal& B, DirichletModal& to)
{
    static DirichletNodal prod;
    prod = A*B;
    prod.ToModal(to);
}
void InterpolateProduct(const NeumannNodal& A1, const NeumannNodal& A2,
                        const NeumannNodal& B1, const NeumannNodal& B2,
                        NeumannModal& to)
{
    static NeumannNodal prod;
    prod = A1*B1 + A2*B2;
    prod.ToModal(to);
}
void InterpolateProduct(const NeumannNodal& A1, const NeumannNodal& A2,
                        const DirichletNodal& B1, const DirichletNodal& B2,
                        DirichletModal& to)
{
    static DirichletNodal prod;
    prod = Reinterpolate(A1)*B1 + Reinterpolate(A2)*B2;
    prod.ToModal(to);
}
}
