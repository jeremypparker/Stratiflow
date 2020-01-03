#pragma once

#include "Field.h"

template<int N1, int N2, int N3>
void RemoveHorizontalAverage(ModalField<N1,N2,N3>& field)
{
    field.stack(0,0).setZero();
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const ModalField<N1,N2,N3>& u)
{
    return real(u(0,0,0));
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const NodalField<N1,N2,N3>& U)
{
    static ModalField<N1,N2,N3> u;
    U.ToModal(u);
    return IntegrateAllSpace(u);
}

template<int N1, int N2, int N3>
void RemoveAverage(ModalField<N1,N2,N3>& field)
{
    field(0,0,0) = 0;
}

template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> U;
    U.Reset();

    U = A*B*weight;

    return IntegrateAllSpace(U);
}

template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> A;
    static NodalField<N1,N2,N3> B;

    A.Reset();
    B.Reset();


    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A, B, weight);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B)
{
    static NodalField<N1,N2,N3> U;
    U.Reset();

    U = A*B;

    return IntegrateAllSpace(U);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b)
{
    static NodalField<N1,N2,N3> A;
    static NodalField<N1,N2,N3> B;

    A.Reset();
    B.Reset();

    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A,B);
}

stratifloat SolveQuadratic(stratifloat a, stratifloat b, stratifloat c, bool positiveSign=false);
