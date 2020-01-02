#pragma once

#include "Field.h"

template<int N1, int N2, int N3>
void RemoveHorizontalAverage(ModalField<N1,N2,N3>& field)
{
    field.stack(0,0).setZero();
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const ModalField<N1,N2,N3>& u, stratifloat L1, stratifloat L2, stratifloat L3)
{
    return real(u(0,0,0));
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const NodalField<N1,N2,N3>& U, stratifloat L1, stratifloat L2, stratifloat L3)
{
    static ModalField<N1,N2,N3> u;
    U.ToModal(u);
    return IntegrateAllSpace(u, L1, L2, L3);
}

template<int N1, int N2, int N3>
void RemoveAverage(ModalField<N1,N2,N3>& field, stratifloat L3)
{
    field(0,0,0) = 0;
}

template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, stratifloat L3, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> U;
    U.Reset();

    U = A*B*weight;

    return IntegrateAllSpace(U, 1, 1, L3);
}

template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> A;
    static NodalField<N1,N2,N3> B;

    A.Reset();
    B.Reset();


    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A, B, L3, weight);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, stratifloat L3)
{
    static NodalField<N1,N2,N3> U;
    U.Reset();

    U = A*B;

    return IntegrateAllSpace(U, 1, 1, L3);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3)
{
    static NodalField<N1,N2,N3> A;
    static NodalField<N1,N2,N3> B;

    A.Reset();
    B.Reset();

    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A,B,L3);
}

stratifloat SolveQuadratic(stratifloat a, stratifloat b, stratifloat c, bool positiveSign=false);
