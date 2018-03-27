#pragma once

#include "Field.h"


template<int N1, int N2, int N3>
stratifloat IntegrateVertically(const Nodal1D<N1,N2,N3>& U, stratifloat L3)
{
    static ArrayX z = VerticalPoints(L3,N3);

    stratifloat result = 0;

    for (int k=1; k<N3-2; k++) // ignore endpoints, should be zero
    {
        result += (z(k)-z(k+1))*(U.Get()(k+1)+U.Get()(k))*0.5;
    }

    return result;
}

template<int N1, int N2, int N3>
void HorizontalAverage(const ModalField<N1,N2,N3>& integrand, Nodal1D<N1,N2,N3>& into)
{
    // the zero mode gives the horizontal average (should always be real)
    into.Get() = real(integrand.stack(0,0));
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const NodalField<N1,N2,N3>& U, stratifloat L1, stratifloat L2, stratifloat L3)
{
    static ModalField<N1,N2,N3> u;
    U.ToModal(u);
    static Nodal1D<N1,N2,N3> horzAve;
    HorizontalAverage(u,horzAve);
    return IntegrateVertically(horzAve,L3)*L1*L2;
}


template<int N1, int N2, int N3>
void RemoveHorizontalAverage(ModalField<N1,N2,N3>& integrand)
{
    // the zero mode gives the horizontal average
    integrand.stack(0,0).setZero();
}

template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, stratifloat L3, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> U;

    U = A*B*weight;

    return IntegrateAllSpace(U, 1, 1, L3);
}


template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> A;
    static NodalField<N1,N2,N3> B;

    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A, B, L3, weight);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3)
{
    static Nodal1D<N1,N2,N3> one;
    one.SetValue([](stratifloat z){return 1;}, L3);
    return InnerProd(a, b, L3, one);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, stratifloat L3)
{
    static Nodal1D<N1,N2,N3> one;
    one.SetValue([](stratifloat z){return 1;}, L3);
    return InnerProd(A, B, L3, one);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3, stratifloat weight)
{
    static Nodal1D<N1,N2,N3> w;
    w.SetValue([weight](stratifloat z){return weight;}, L3);
    return InnerProd(a, b, L3, w);
}

stratifloat SolveQuadratic(stratifloat a, stratifloat b, stratifloat c, bool positiveSign=false);