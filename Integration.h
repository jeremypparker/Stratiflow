#pragma once

#include "Field.h"

template<int N1, int N2, int N3>
stratifloat IntegrateAllSpace(const NodalField<N1,N2,N3>& U, stratifloat L1, stratifloat L2, stratifloat L3)
{
    // it better decay for us to integrate it
    assert(U.BC() == BoundaryCondition::Decaying);

    // the inverse of the weight function
    static Nodal1D<N1,N2,N3> w(BoundaryCondition::Bounded);
    w.SetValue([L3](stratifloat z){return 1+z*z/L3/L3;}, L3);

    // we hope this product doesn't get too large, otherwise we're stuck
    static NodalField<N1,N2,N3> I(BoundaryCondition::Bounded);
    I = U*w;

    // because these ones won't play nicely with the regularised positions
    I.slice(0).setZero();
    I.slice(N3-1).setZero();

    static ModalField<N1,N2,N3> J(BoundaryCondition::Bounded);
    I.ToModal(J);

    // the zero mode gives the contribution that remains after integration
    return real(J(0,0,0)) * pi * L1 * L2 * L3;
}

template<int N1, int N2, int N3>
stratifloat IntegrateVertically(const Nodal1D<N1,N2,N3>& U, stratifloat L3)
{
    assert(U.BC() == BoundaryCondition::Decaying);

    // the inverse of the weight function
    static Nodal1D<N1,N2,N3> w(BoundaryCondition::Bounded);
    w.SetValue([L3](stratifloat z){return 1+z*z/L3/L3;}, L3);

    // we hope this product doesn't get too large, otherwise we're stuck
    static Nodal1D<N1,N2,N3> I(BoundaryCondition::Bounded);
    I = U*w;

    // because these ones won't play nicely with the regularised positions
    I.Get()(0) = 0;
    I.Get()(N3-1) = 0;

    static Modal1D<N1,N2,N3> J(BoundaryCondition::Bounded);
    I.ToModal(J);

    // the zero mode gives the contribution that remains after integration
    return J.Get()(0) * pi * L3;
}