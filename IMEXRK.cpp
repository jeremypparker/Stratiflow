#include "IMEXRK.h"

constexpr int IMEXRK::s;
constexpr stratifloat IMEXRK::beta[];
constexpr stratifloat IMEXRK::zeta[];

void IMEXRK::TimeStep()
{
    // see Numerical Renaissance
    for (int k=0; k<s; k++)
    {
        ExplicitCN(k, EvolveBackground);
        BuildRHS();
        FinishRHS(k);

        ImplicitUpdate(k, EvolveBackground);

        RemoveDivergence(1/h[k]);

        //if (k==s-1)
        //{
            FilterAll();
        //}

        PopulateNodalVariables();
    }
}

void IMEXRK::RemoveDivergence(stratifloat pressureMultiplier)
{
    // construct the diverence of u
    if(ThreeDimensional)
    {
        divergence = ddx(u1) + ddy(u2) + ddz(u3);
    }
    else
    {
        divergence = ddx(u1) + ddz(u3);
    }

    // set value at boundary to zero
    divergence.ZeroEnds();

    // solve Δq = ∇·u as linear system Aq = divergence
    divergence.Solve(solveLaplacian, q);

    // subtract the gradient of this from the velocity
    u1 -= ddx(q);
    if(ThreeDimensional)
    {
        u2 -= ddy(q);
    }
    u3 -= ddz(q);

    // also add it on to p for the next step
    // this is scaled to match the p that was added before
    // effectively we have forward euler
    p += pressureMultiplier*q;
}

void IMEXRK::ImplicitUpdate(int k, bool evolveBackground)
{
    CNSolve(R1, u1, k);
    if(ThreeDimensional)
    {
        CNSolve(R2, u2, k);
    }
    CNSolve(R3, u3, k);
    CNSolveBuoyancy(RB, b, k);

    if (EvolveBackground)
    {
        CNSolve1D(RU_, U_, k);
        CNSolveBuoyancy1D(RB_, B_, k);
    }
}

void IMEXRK::FinishRHS(int k)
{
    // now add on explicit terms to RHS
    R1 += (h[k]*beta[k])*r1;
    if(ThreeDimensional)
    {
        R2 += (h[k]*beta[k])*r2;
    }
    R3 += (h[k]*beta[k])*r3;
    RB += (h[k]*beta[k])*rB;
}

void IMEXRK::ExplicitCN(int k, bool evolveBackground)
{
    //   old      last rk step         pressure         explicit CN
    R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u1)+MatMulDim2(dim2Derivative2, u1)+MatMulDim3(dim3Derivative2Neumann, u1));
    if(ThreeDimensional)
    {
    R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u2)+MatMulDim2(dim2Derivative2, u2)+MatMulDim3(dim3Derivative2Neumann, u2));
    }
    R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) + (0.5f*h[k]/Re)*(MatMulDim1(dim1Derivative2, u3)+MatMulDim2(dim2Derivative2, u3)+MatMulDim3(dim3Derivative2Dirichlet, u3));
    RB = b  + (h[k]*zeta[k])*rB                  + (0.5f*h[k]/Pe)*(MatMulDim1(dim1Derivative2, b)+MatMulDim2(dim2Derivative2, b)+MatMulDim3(dim3Derivative2Neumann, b));

    if (evolveBackground)
    {
        // for the 1D variables u_ and b_ (background flow) we only use vertical derivative matrix
        RU_ = U_ + (0.5f*h[k]/Re)*MatMul1D(dim3Derivative2Neumann, U_);
        RB_ = B_ + (0.5f*h[k]/Pe)*MatMul1D(dim3Derivative2Neumann, B_);
    }
}

void IMEXRK::BuildRHS()
{
    // build up right hand sides for the implicit solve in R

    // buoyancy force without hydrostatic part
    neumannTemp = b;
    RemoveHorizontalAverage(neumannTemp);
    r3 = -Ri*Reinterpolate(neumannTemp); // buoyancy force

    //////// NONLINEAR TERMS ////////

    // calculate products at nodes in physical space

    // take into account background shear for nonlinear terms
    U1_tot = U1 + U_;

    InterpolateProduct(U1_tot, U1_tot, neumannTemp);
    InterpolateProduct(U1_tot, U3, dirichletTemp);
    r1 = -(ddx(neumannTemp)+ddz(dirichletTemp));

    InterpolateProduct(U3, U3, neumannTemp);
    r3 -= ddx(dirichletTemp)+ddz(neumannTemp);

    if(ThreeDimensional)
    {
        InterpolateProduct(U2, U2, neumannTemp);
        InterpolateProduct(U2, U3, dirichletTemp);
        r2 = -(ddy(neumannTemp)+ddz(dirichletTemp));
        r3 -= ddy(dirichletTemp);

        InterpolateProduct(U1_tot, U2, neumannTemp);
        r1 -= ddy(neumannTemp);
        r2 -= ddx(neumannTemp);
    }

    // buoyancy nonlinear terms
    InterpolateProduct(B, U3, dirichletTemp);

    if(ThreeDimensional)
    {
        InterpolateProduct(U2, B, neumannTemp);
        rB = -(ddz(dirichletTemp)+ddy(neumannTemp));
    }
    else
    {
        rB = -ddz(dirichletTemp);
    }

    InterpolateProduct(U1_tot, B, neumannTemp);

    // advection term from background buoyancy
    ndTemp = U3*dB_dz;
    ndTemp.ToModal(dirichletTemp);
    rB -= ddx(neumannTemp)+Reinterpolate(dirichletTemp);
}

void IMEXRK::BuildRHSLinear()
{
    // build up right hand sides for the implicit solve in R

    // buoyancy force without hydrostatic part
    neumannTemp = b;
    RemoveHorizontalAverage(neumannTemp);
    r3 = -Ri*Reinterpolate(neumannTemp); // buoyancy force

    nnTemp = U1_tot + U_;

    //////// NONLINEAR TERMS ////////
    InterpolateProduct(U1, nnTemp, neumannTemp);
    InterpolateProduct(nnTemp, U1, U3, U3_tot, dirichletTemp);
    r1 = -(2.0*ddx(neumannTemp) +  ddz(dirichletTemp));

    InterpolateProduct(U3, U3_tot, neumannTemp);
    r3 -= ddx(dirichletTemp) + 2.0*ddz(neumannTemp);

    if(ThreeDimensional)
    {
        InterpolateProduct(U2, U2_tot, neumannTemp);
        r2 = -2.0*ddy(neumannTemp);

        InterpolateProduct(U2_tot, U2, U3, U3_tot, dirichletTemp);
        r3 -= ddy(dirichletTemp);
        r2 -= ddz(dirichletTemp);

        InterpolateProduct(U2_tot, U2, U1, nnTemp, neumannTemp);
        r1 -= ddy(neumannTemp);
        r2 -= ddx(neumannTemp);
    }

    // buoyancy nonlinear terms
    InterpolateProduct(nnTemp, U1, B, B_tot, neumannTemp);
    InterpolateProduct(B, B_tot, U3_tot, U3, dirichletTemp);
    rB = -(ddx(neumannTemp) + ddz(dirichletTemp));

    if(ThreeDimensional)
    {
        InterpolateProduct(U2_tot, U2, B, B_tot, neumannTemp);
        rB -= ddy(neumannTemp);
    }

    // advection term from background buoyancy
    ndTemp = U3*dB_dz;
    ndTemp.ToModal(dirichletTemp);
    rB -= Reinterpolate(dirichletTemp);
}

void IMEXRK::BuildRHSAdjoint()
{
    // build up right hand sides for the implicit solve in R

    // adjoint buoyancy
    bForcing -= Ri*Reinterpolate(U3);

    //////// NONLINEAR TERMS ////////
    // advection of adjoint quantities by the direct flow
    InterpolateProduct(U1, U1_tot, neumannTemp);
    InterpolateProduct(U1, U3_tot, dirichletTemp);
    r1 = ddx(neumannTemp) + ddz(dirichletTemp);

    if(ThreeDimensional)
    {
        InterpolateProduct(U1, U2_tot, neumannTemp);
        r1 += ddy(neumannTemp);
    }

    if(ThreeDimensional)
    {
        InterpolateProduct(U2, U1_tot, neumannTemp);
        r2 = ddx(neumannTemp);
        InterpolateProduct(U2, U2_tot, neumannTemp);
        InterpolateProduct(U2, U3_tot, dirichletTemp);
        r2 += ddy(neumannTemp) + ddz(dirichletTemp);
    }

    InterpolateProduct(U1_tot, U3, dirichletTemp);
    InterpolateProduct(U3, U3_tot, neumannTemp);
    r3 = ddx(dirichletTemp) + ddz(neumannTemp);

    if(ThreeDimensional)
    {
        InterpolateProduct(U2_tot, U3, dirichletTemp);
        r3 += ddy(dirichletTemp);
    }

    InterpolateProduct(B, U1_tot, neumannTemp);
    InterpolateProduct(B, U3_tot, dirichletTemp);
    rB = ddx(neumannTemp) + ddz(dirichletTemp);
    if(ThreeDimensional)
    {
        InterpolateProduct(B, U2_tot, neumannTemp);
        rB += ddy(neumannTemp);
    }

    // extra adjoint nonlinear terms
    neumannTemp = ddx(u1_tot);
    neumannTemp.ToNodal(nnTemp);
    nnTemp2 = nnTemp*U1;
    if(ThreeDimensional)
    {
        neumannTemp = ddx(u2_tot);
        neumannTemp.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
    }
    dirichletTemp = ddx(u3_tot);
    dirichletTemp.ToNodal(ndTemp);
    u1Forcing -= nnTemp2 + Reinterpolate(ndTemp*U3);

    if(ThreeDimensional)
    {
        neumannTemp = ddy(u1_tot);
        neumannTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*U1;
        neumannTemp = ddy(u2_tot);
        neumannTemp.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
        dirichletTemp = ddy(u3_tot);
        dirichletTemp.ToNodal(ndTemp);
        u2Forcing -= nnTemp2 + Reinterpolate(ndTemp*U3);
    }

    dirichletTemp = ddz(u1_tot);
    dirichletTemp.ToNodal(ndTemp);
    ndTemp2 = ndTemp*Reinterpolate(U1);
    if(ThreeDimensional)
    {
        dirichletTemp = ddz(u2_tot);
        dirichletTemp.ToNodal(ndTemp);
        ndTemp2 += ndTemp*Reinterpolate(U2);
    }
    neumannTemp = ddz(u3_tot);
    neumannTemp.ToNodal(nnTemp);
    u3Forcing -= ndTemp2 + Reinterpolate(nnTemp)*U3;


    neumannTemp = ddx(b_tot);
    neumannTemp.ToNodal(nnTemp);
    u1Forcing -= nnTemp*B;

    if(ThreeDimensional)
    {
        neumannTemp = ddy(b_tot);
        neumannTemp.ToNodal(nnTemp);
        u2Forcing -= nnTemp*B;
    }

    dirichletTemp = ddz(b_tot);
    dirichletTemp.ToNodal(ndTemp);
    u3Forcing -= ndTemp*Reinterpolate(B);

    // Now include all the forcing terms
    u1Forcing.ToModal(neumannTemp);
    r1 += neumannTemp;
    if (ThreeDimensional)
    {
        u2Forcing.ToModal(neumannTemp);
        r2 += neumannTemp;
    }
    u3Forcing.ToModal(dirichletTemp);
    r3 += dirichletTemp;
    bForcing.ToModal(neumannTemp);
    rB += neumannTemp;
}
