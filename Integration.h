#pragma once

#include "Field.h"

template<int N1, int N2, int N3>
stratifloat IntegrateVertically(const Nodal1D<N1,N2,N3>& U, stratifloat L3)
{
    if (U.BC() == BoundaryCondition::Neumann)
    {
        static ArrayX z = VerticalPoints(L3,N3);

        stratifloat result = 0;

        result += (z(0)-z(1))*(3*U.Get()(1)+U.Get()(0))*0.125;

        for (int k=1; k<N3-2; k++)
        {
            result += (z(k)-z(k+1))*(U.Get()(k+1)+U.Get()(k))*0.5;
        }

        result += (z(N3-2)-z(N3-1))*(3*U.Get()(N3-2)+U.Get()(N3-1))*0.125;

        return result;
    }
    else
    {
        static ArrayX z = VerticalPointsStaggered(L3,N3);

        stratifloat result = 0;

        for (int k=0; k<N3-2; k++)
        {
            result += (z(k)-z(k+1))*(U.Get()(k+1)+U.Get()(k))*0.5;
        }

        return result;
    }
}

template<int N1, int N2, int N3>
void HorizontalAverage(const ModalField<N1,N2,N3>& integrand, Nodal1D<N1,N2,N3>& into)
{
    // the zero mode gives the horizontal average (should always be real)
    into.Get() = real(integrand.stack(0,0));
}

template<int N1, int N2, int N3>
void RemoveHorizontalAverage(ModalField<N1,N2,N3>& field)
{
    field.stack(0,0).setZero();
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const ModalField<N1,N2,N3>& u, stratifloat L1, stratifloat L2, stratifloat L3)
{
    static Nodal1D<N1,N2,N3> horzAve(u.BC());
    horzAve.Reset(u.BC());
    HorizontalAverage(u,horzAve);
    return IntegrateVertically(horzAve,L3)*L1*L2;
}

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const NodalField<N1,N2,N3>& U, stratifloat L1, stratifloat L2, stratifloat L3)
{
    static ModalField<N1,N2,N3> u(U.BC());
    u.Reset(U.BC());
    U.ToModal(u);
    static Nodal1D<N1,N2,N3> horzAve(U.BC());
    horzAve.Reset(U.BC());
    HorizontalAverage(u,horzAve);
    return IntegrateVertically(horzAve,L3)*L1*L2;
}


template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, stratifloat L3, const StackContainer<C,T,N1,N2,N3>& weight)
{
    assert(A.BC() == B.BC());
    static NodalField<N1,N2,N3> U(A.BC());
    U.Reset(A.BC());

    U = A*B*weight;

    return IntegrateAllSpace(U, 1, 1, L3);
}

template<typename C, typename T, int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3, const StackContainer<C,T,N1,N2,N3>& weight)
{
    static NodalField<N1,N2,N3> A(a.BC());
    static NodalField<N1,N2,N3> B(b.BC());

    A.Reset(a.BC());
    B.Reset(b.BC());


    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A, B, L3, weight);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const NodalField<N1,N2,N3>& A, const NodalField<N1,N2,N3>& B, stratifloat L3)
{
    assert(A.BC() == B.BC());
    static NodalField<N1,N2,N3> U(A.BC());
    U.Reset(A.BC());

    U = A*B;

    return IntegrateAllSpace(U, 1, 1, L3);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3)
{
    static NodalField<N1,N2,N3> A(a.BC());
    static NodalField<N1,N2,N3> B(b.BC());

    A.Reset(a.BC());
    B.Reset(b.BC());

    a.ToNodal(A);
    b.ToNodal(B);

    return InnerProd(A, B, L3);
}

template<int N1, int N2, int N3>
stratifloat InnerProd(const ModalField<N1,N2,N3>& a, const ModalField<N1,N2,N3>& b, stratifloat L3, stratifloat weight)
{
    static Nodal1D<N1,N2,N3> w(BoundaryCondition::Neumann);
    w.SetValue([weight](stratifloat z){return weight;}, L3);
    return InnerProd(a, b, L3, w);
}

stratifloat SolveQuadratic(stratifloat a, stratifloat b, stratifloat c, bool positiveSign=false);