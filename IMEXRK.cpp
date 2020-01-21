#include "IMEXRK.h"
#include <iomanip>

constexpr int IMEXRK::s;
constexpr stratifloat IMEXRK::beta[];
constexpr stratifloat IMEXRK::zeta[];

void IMEXRK::TimeStep()
{
    // see Numerical Renaissance
    for (int k=0; k<s; k++)
    {
        ExplicitRK(k);
        BuildRHS();
        FinishRHS(k);

        CrankNicolson(k);

        RemoveDivergence(1/h[k]);

        FilterAll();
        PopulateNodalVariables();
    }
}

void IMEXRK::TimeStepLinear()
{
    // see Numerical Renaissance
    for (int k=0; k<s; k++)
    {
        ExplicitRK(k);
        BuildRHSLinear();
        FinishRHS(k);

        CrankNicolson(k);
        RemoveDivergence(1/h[k]);

        FilterAll();
        PopulateNodalVariables();
    }
}

void IMEXRK::RemoveDivergence(stratifloat pressureMultiplier)
{
    // construct the diverence of u
    if(gridParams.ThreeDimensional)
    {
        divergence = ddx(u1) + ddy(u2) + ddz(u3);
    }
    else
    {
        divergence = ddx(u1) + ddz(u3);
    }

    // solve Δq = ∇·u as linear system Aq = divergence
    divergence.Solve(solveLaplacian, q);

    // subtract the gradient of this from the velocity
    u1 -= ddx(q);
    if(gridParams.ThreeDimensional)
    {
        u2 -= ddy(q);
    }
    u3 -= ddz(q);

    // also add it on to p for the next step
    // this is scaled to match the p that was added before
    // effectively we have forward euler
    p += pressureMultiplier*q;
}

void IMEXRK::CrankNicolson(int k)
{
    R1 += (0.5f*h[k]/flowParams.Re)*(MatMulDim1(dim1Derivative2, u1)
                         +MatMulDim2(dim2Derivative2, u1)
                         +MatMulDim3(dim3Derivative2, u1));
    CNSolve(R1, u1, k);

    if(gridParams.ThreeDimensional)
    {
        R2 += (0.5f*h[k]/flowParams.Re)*(MatMulDim1(dim1Derivative2, u2)
                             +MatMulDim2(dim2Derivative2, u2)
                             +MatMulDim3(dim3Derivative2, u2));
        CNSolve(R2, u2, k);
    }

    R3 += (0.5f*h[k]/flowParams.Re)*(MatMulDim1(dim1Derivative2, u3)
                         +MatMulDim2(dim2Derivative2, u3)
                         +MatMulDim3(dim3Derivative2, u3));
    CNSolve(R3, u3, k);

    RB += (0.5f*h[k]/flowParams.Re/flowParams.Pr)*(MatMulDim1(dim1Derivative2, b)
                         +MatMulDim2(dim2Derivative2, b)
                         +MatMulDim3(dim3Derivative2, b));
    CNSolveBuoyancy(RB, b, k);

}

void IMEXRK::FinishRHS(int k)
{
    // now add on explicit terms to RHS
    R1 += (h[k]*beta[k])*r1;
    if(gridParams.ThreeDimensional)
    {
        R2 += (h[k]*beta[k])*r2;
    }
    R3 += (h[k]*beta[k])*r3;
    RB += (h[k]*beta[k])*rB;
}

void IMEXRK::ExplicitRK(int k)
{
    //   old      last rk step         pressure
    R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) ;
    if(gridParams.ThreeDimensional)
    {
    R2 = u2 + (h[k]*zeta[k])*r2 + (-h[k])*ddy(p) ;
    }
    R3 = u3 + (h[k]*zeta[k])*r3 + (-h[k])*ddz(p) ;
    RB = b  + (h[k]*zeta[k])*rB                  ;

    r1.Zero();
    r2.Zero();
    r3.Zero();
    rB.Zero();
}

void IMEXRK::BuildRHS()
{
    // build up right hand sides for the implicit solve in R

    // buoyancy force without hydrostatic part
    modalTemp1 = b;
    RemoveHorizontalAverage(modalTemp1);
    r3 += flowParams.Ri*modalTemp1; // buoyancy force

    // background stratification term
    rB -= u3;

    //////// NONLINEAR TERMS ////////
    // calculate products at nodes in physical space

    InterpolateProduct(U1, U1, modalTemp1);
    InterpolateProduct(U1, U3, modalTemp2);
    r1 -= ddx(modalTemp1) + ddz(modalTemp2);
    InterpolateProduct(U3, U3, modalTemp1);
    r3 -= ddx(modalTemp2) + ddz(modalTemp1);

    if(gridParams.ThreeDimensional)
    {
        InterpolateProduct(U2, U2, modalTemp1);
        r2 -= ddy(modalTemp1);

        InterpolateProduct(U2, U3, modalTemp1);
        r2 -= ddz(modalTemp1);
        r3 -= ddy(modalTemp1);

        InterpolateProduct(U1, U2, modalTemp1);
        r1 -= ddy(modalTemp1);
        r2 -= ddx(modalTemp1);
    }

    // buoyancy nonlinear terms
    if(gridParams.ThreeDimensional)
    {
        InterpolateProduct(U2, B, modalTemp1);
        rB -= ddy(modalTemp1);
    }

    InterpolateProduct(B, U3, modalTemp2);
    InterpolateProduct(U1, B, modalTemp1);
    rB -= ddx(modalTemp1)+ddz(modalTemp2);
}

void IMEXRK::BuildRHSLinear()
{
    // build up right hand sides for the implicit solve in R

    // buoyancy force without hydrostatic part
    modalTemp1 = b;
    RemoveHorizontalAverage(modalTemp1);
    r3 += flowParams.Ri*modalTemp1; // buoyancy force

    // background stratification term
    rB -= u3;

    //////// NONLINEAR TERMS ////////
    // calculate products at nodes in physical space

    InterpolateProduct(U1, U1_tot, modalTemp1);
    InterpolateProduct(U1, U1_tot, U3_tot, U3, modalTemp2);
    r1 -= 2.0*ddx(modalTemp1) + ddz(modalTemp2);

    InterpolateProduct(U3, U3_tot, modalTemp1);
    r3 -= ddx(modalTemp2) + 2.0*ddz(modalTemp1);

    if(gridParams.ThreeDimensional)
    {
        InterpolateProduct(U2, U2_tot, modalTemp2);
        InterpolateProduct(U2, U2_tot, U3_tot, U3, modalTemp1);
        r2 -= ddz(modalTemp1) + 2.0*ddy(modalTemp2);
        r3 -= ddy(modalTemp1);

        InterpolateProduct(U1, U1_tot, U2_tot, U2, modalTemp1);
        r1 -= ddy(modalTemp1);
        r2 -= ddx(modalTemp1);
    }

    // buoyancy nonlinear terms

    if(gridParams.ThreeDimensional)
    {
        InterpolateProduct(B, B_tot, U2_tot, U2, modalTemp1);
        rB -= ddy(modalTemp1);
    }

    InterpolateProduct(B, B_tot, U3_tot, U3, modalTemp2);
    InterpolateProduct(B, B_tot, U1_tot, U1, modalTemp1);
    rB -= ddx(modalTemp1) + ddz(modalTemp2);
}

void IMEXRK::BuildRHSAdjoint()
{
    // build up right hand sides for the implicit solve in R

    // adjoint buoyancy
    bForcing += flowParams.Ri*U3;

    // background stratification term
    r3 -= b;

    //////// NONLINEAR TERMS ////////
    // advection of adjoint quantities by the direct flow
    InterpolateProduct(U1, U1_tot, modalTemp1);
    InterpolateProduct(U1, U3_tot, modalTemp2);
    r1 += ddx(modalTemp1) + ddz(modalTemp2);

    InterpolateProduct(U1_tot, U3, modalTemp2);
    InterpolateProduct(U3, U3_tot, modalTemp1);
    r3 += ddx(modalTemp2) + ddz(modalTemp1);

    if(gridParams.ThreeDimensional)
    {
        InterpolateProduct(U2, U2_tot, modalTemp2);
        InterpolateProduct(U2, U3_tot, modalTemp1);
        r2 += ddy(modalTemp2) + ddz(modalTemp1);

        InterpolateProduct(U2_tot, U3, modalTemp2);
        r3 += ddy(modalTemp2);

        InterpolateProduct(U1, U2_tot, modalTemp1);
        r1 += ddy(modalTemp1);

        InterpolateProduct(U2, U1_tot, modalTemp1);
        r2 += ddx(modalTemp1);
    }

    // buoyancy nonlinear terms
    InterpolateProduct(B, U3_tot, modalTemp1);
    rB += ddz(modalTemp1);

    if(gridParams.ThreeDimensional)
    {
        InterpolateProduct(B, U2_tot, modalTemp1);
        rB += ddy(modalTemp1);
    }

    InterpolateProduct(B, U1_tot, modalTemp1);
    rB += ddx(modalTemp1);


    // extra adjoint nonlinear terms
    modalTemp1 = ddx(u1_tot);
    modalTemp1.ToNodal(nnTemp);
    nnTemp2 = nnTemp*U1;
    if(gridParams.ThreeDimensional)
    {
        modalTemp1 = ddx(u2_tot);
        modalTemp1.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
    }
    modalTemp2 = ddx(u3_tot);
    modalTemp2.ToNodal(ndTemp);
    u1Forcing -= nnTemp2 + ndTemp*U3;

    if(gridParams.ThreeDimensional)
    {
        modalTemp1 = ddy(u1_tot);
        modalTemp1.ToNodal(nnTemp);
        nnTemp2 = nnTemp*U1;
        modalTemp1 = ddy(u2_tot);
        modalTemp1.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
        modalTemp2 = ddy(u3_tot);
        modalTemp2.ToNodal(ndTemp);
        u2Forcing -= nnTemp2 + ndTemp*U3;
    }

    modalTemp2 = ddz(u1_tot);
    modalTemp2.ToNodal(ndTemp);
    ndTemp2 = ndTemp*U1;
    if(gridParams.ThreeDimensional)
    {
        modalTemp2 = ddz(u2_tot);
        modalTemp2.ToNodal(ndTemp);
        ndTemp2 += ndTemp*U2;
    }
    modalTemp1 = ddz(u3_tot);
    modalTemp1.ToNodal(nnTemp);
    u3Forcing -= ndTemp2 + nnTemp*U3;


    modalTemp1 = ddx(b_tot);
    modalTemp1.ToNodal(nnTemp);
    u1Forcing -= nnTemp*B;

    if(gridParams.ThreeDimensional)
    {
        modalTemp1 = ddy(b_tot);
        modalTemp1.ToNodal(nnTemp);
        u2Forcing -= nnTemp*B;
    }

    modalTemp2 = ddz(b_tot);
    modalTemp2.ToNodal(ndTemp);
    u3Forcing -= ndTemp*B;

    // Now include all the forcing terms
    u1Forcing.ToModal(modalTemp1);
    r1 += modalTemp1;
    if (gridParams.ThreeDimensional)
    {
        u2Forcing.ToModal(modalTemp1);
        r2 += modalTemp1;
    }
    u3Forcing.ToModal(modalTemp2);
    r3 += modalTemp2;
    bForcing.ToModal(modalTemp1);
    rB += modalTemp1;
}
