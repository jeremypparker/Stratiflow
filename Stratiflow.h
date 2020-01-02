#pragma once
#include "Field.h"
#include "Parameters.h"
#include "Differentiation.h"

constexpr int M1 = gridParams.N1/2 + 1;

class Modal : public ModalField<gridParams.N1,gridParams.N2,gridParams.N3>
{
public:
    Modal() : ModalField() {}
    using ModalField::operator=;
};

class Nodal : public NodalField<gridParams.N1,gridParams.N2,gridParams.N3>
{
public:
    Nodal() : NodalField() {}
    using NodalField::operator=;

};


template<typename T>
Dim1MatMul<T, complex, complex, M1, gridParams.N2, gridParams.N3> ddx(const StackContainer<T, complex, M1, gridParams.N2, gridParams.N3>& f)
{
    static DiagonalMatrix<complex, -1> dim1Derivative = FourierDerivativeMatrix(flowParams.L1, gridParams.N1, 1);

    return Dim1MatMul<T, complex, complex, M1, gridParams.N2, gridParams.N3>(dim1Derivative, f);
}

template<typename T>
Dim2MatMul<T, complex, complex, M1, gridParams.N2, gridParams.N3> ddy(const StackContainer<T, complex, M1, gridParams.N2, gridParams.N3>& f)
{
    static DiagonalMatrix<complex, -1> dim2Derivative = FourierDerivativeMatrix(flowParams.L2, gridParams.N2, 2);

    return Dim2MatMul<T, complex, complex, M1, gridParams.N2, gridParams.N3>(dim2Derivative, f);
}

template<typename T, int K1, int K2, int K3>
Dim3MatMul<T, complex, complex, K1, K2, K3> ddz(const StackContainer<T, complex, K1, K2, K3>& f)
{
    static MatrixXc dim3Derivative = FourierDerivativeMatrix(flowParams.L3, gridParams.N3, 3);
    return Dim3MatMul<T, complex, complex, K1, K2, K3>(dim3Derivative, f);
}


namespace
{
void InterpolateProduct(const Nodal& A, const Nodal& B, Modal& to)
{
    static Nodal prod;
    prod = A*B;
    prod.ToModal(to);
}

void InterpolateProduct(const Nodal& A1, const Nodal& A2,
                        const Nodal& B1, const Nodal& B2,
                        Modal& to)
{
    static Nodal prod;
    prod = A1*B1 + A2*B2;
    prod.ToModal(to);
}
}
