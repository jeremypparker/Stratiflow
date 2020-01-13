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
        ExplicitRK(k, flowParams.EvolveBackground);
        BuildRHS();
        FinishRHS(k);

        CrankNicolson(k, flowParams.EvolveBackground);

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
    if(gridParams.ThirdDimension())
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
    if(gridParams.ThirdDimension())
    {
        u2 -= ddy(q);
    }
    u3 -= ddz(q);

    // also add it on to p for the next step
    // this is scaled to match the p that was added before
    // effectively we have forward euler
    p += pressureMultiplier*q;
}

void IMEXRK::CrankNicolson(int k, bool evolveBackground)
{
    R1 += (0.5f*h[k]/flowParams.Re)*(MatMulDim1(dim1Derivative2, u1)
                         +MatMulDim2(dim2Derivative2, u1)
                         +MatMulDim3(dim3Derivative2Neumann, u1));
    CNSolve(R1, u1, k);

    if(gridParams.ThirdDimension())
    {
        R2 += (0.5f*h[k]/flowParams.Re)*(MatMulDim1(dim1Derivative2, u2)
                             +MatMulDim2(dim2Derivative2, u2)
                             +MatMulDim3(dim3Derivative2Neumann, u2));
        CNSolve(R2, u2, k);
    }

    R3 += (0.5f*h[k]/flowParams.Re)*(MatMulDim1(dim1Derivative2, u3)
                         +MatMulDim2(dim2Derivative2, u3)
                         +MatMulDim3(dim3Derivative2Dirichlet, u3));
    CNSolve(R3, u3, k);

    RB += (0.5f*h[k]/flowParams.Re/flowParams.Pr)*(MatMulDim1(dim1Derivative2, b)
                         +MatMulDim2(dim2Derivative2, b)
                         +MatMulDim3(dim3Derivative2Neumann, b));
    CNSolveBuoyancy(RB, b, k);

    if (flowParams.EvolveBackground)
    {
        RU_ = U_ + (0.5f*h[k]/flowParams.Re)*MatMul1D(dim3Derivative2Neumann, U_);
        CNSolve1D(RU_, U_, k);
    }
}

void IMEXRK::FinishRHS(int k)
{
    // now add on explicit terms to RHS
    R1 += (h[k]*beta[k])*r1;
    if(gridParams.ThirdDimension())
    {
        R2 += (h[k]*beta[k])*r2;
    }
    R3 += (h[k]*beta[k])*r3;
    RB += (h[k]*beta[k])*rB;
}

void IMEXRK::ExplicitRK(int k, bool evolveBackground)
{
    //   old      last rk step         pressure
    R1 = u1 + (h[k]*zeta[k])*r1 + (-h[k])*ddx(p) ;
    if(gridParams.ThirdDimension())
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
    neumannTemp = b;
    RemoveHorizontalAverage(neumannTemp);
    r3 += flowParams.Ri*ReinterpolateFull(neumannTemp); // buoyancy force

    // background stratification term
    dirichletTemp = u3;
    rB -= ReinterpolateDirichlet(dirichletTemp);

    //////// NONLINEAR TERMS ////////
    // calculate products at nodes in physical space

    // take into account background shear for nonlinear terms
    U1_tot = U1 + U_;

    InterpolateProduct(U1_tot, U1_tot, neumannTemp);
    r1 -= ddx(neumannTemp);

    DifferentiateProductBar(U1_tot, U3, neumannTemp);
    r1 -= neumannTemp;

    InterpolateProductTilde(U1_tot, U3, dirichletTemp);
    InterpolateProduct(U3, U3, neumannTemp);
    r3 -= ddx(dirichletTemp)+ddz(neumannTemp);

    if(gridParams.ThirdDimension())
    {
        InterpolateProduct(U2, U2, neumannTemp);
        r2 -= ddy(neumannTemp);

        DifferentiateProductBar(U2, U3, neumannTemp);
        r2 -= neumannTemp;

        InterpolateProductTilde(U2, U3, dirichletTemp);
        r3 -= ddy(dirichletTemp);

        InterpolateProduct(U1_tot, U2, neumannTemp);
        r1 -= ddy(neumannTemp);
        r2 -= ddx(neumannTemp);
    }

    // buoyancy nonlinear terms
    DifferentiateProductBar(B, U3, neumannTemp);
    rB -= neumannTemp;

    if(gridParams.ThirdDimension())
    {
        InterpolateProduct(U2, B, neumannTemp);
        rB -= ddy(neumannTemp);
    }

    InterpolateProduct(U1_tot, B, neumannTemp);
    rB -= ddx(neumannTemp);
}

void IMEXRK::BuildRHSLinear()
{
    // build up right hand sides for the implicit solve in R

    // buoyancy force without hydrostatic part
    neumannTemp = b;
    RemoveHorizontalAverage(neumannTemp);
    r3 += flowParams.Ri*ReinterpolateFull(neumannTemp); // buoyancy force

    // background stratification term
    dirichletTemp = u3;
    rB -= ReinterpolateDirichlet(dirichletTemp);

    //////// NONLINEAR TERMS ////////
    // calculate products at nodes in physical space

    // take into account background shear for nonlinear terms
    nnTemp = U1_tot + U_;

    InterpolateProduct(U1, nnTemp, neumannTemp);
    r1 -= 2.0*ddx(neumannTemp);

    DifferentiateProductBar(U1, nnTemp, U3_tot, U3, neumannTemp);
    r1 -= neumannTemp;

    InterpolateProductTilde(U1, nnTemp, U3_tot, U3, dirichletTemp);
    InterpolateProduct(U3, U3_tot, neumannTemp);
    r3 -= ddx(dirichletTemp)+2.0*ddz(neumannTemp);

    if(gridParams.ThirdDimension())
    {
        DifferentiateProductBar(U2, U2_tot, U3_tot, U3, neumannTemp);
        r2 -= neumannTemp;

        if (gridParams.dimensionality == Dimensionality::ThreeDimensional)
        {
            InterpolateProduct(U2, U2_tot, neumannTemp);
            r2 -= 2.0*ddy(neumannTemp);

            InterpolateProductTilde(U2, U2_tot, U3_tot, U3,  dirichletTemp);
            r3 -= ddy(dirichletTemp);
        }

        InterpolateProduct(U1, nnTemp, U2_tot, U2, neumannTemp);
        if (gridParams.dimensionality == Dimensionality::ThreeDimensional)
        {
            r1 -= ddy(neumannTemp);
        }
        r2 -= ddx(neumannTemp);
    }

    // buoyancy nonlinear terms
    DifferentiateProductBar(B, B_tot, U3_tot, U3, neumannTemp);
    rB -= neumannTemp;

    if(gridParams.ThirdDimension())
    {
        InterpolateProduct(B, B_tot, U2_tot, U2, neumannTemp);
        rB -= ddy(neumannTemp);
    }

    InterpolateProduct(B, B_tot, nnTemp, U1, neumannTemp);
    rB -= ddx(neumannTemp);
}

void IMEXRK::BuildRHSAdjoint()
{
    // build up right hand sides for the implicit solve in R

    // adjoint buoyancy
    bForcing += flowParams.Ri*ReinterpolateDirichlet(U3);

    //////// NONLINEAR TERMS ////////
    // advection of adjoint quantities by the direct flow
    InterpolateProduct(U1, U1_tot, neumannTemp);
    r1 += ddx(neumannTemp);

    DifferentiateProductBar(U1, U3_tot, neumannTemp);
    r1 += neumannTemp;

    InterpolateProductTilde(U1_tot, U3, dirichletTemp);
    InterpolateProduct(U3, U3_tot, neumannTemp);
    r3 += ddx(dirichletTemp)+ddz(neumannTemp);

    if(gridParams.ThirdDimension())
    {
        InterpolateProduct(U2, U2_tot, neumannTemp);
        r2 += ddy(neumannTemp);

        DifferentiateProductBar(U2, U3_tot, neumannTemp);
        r2 += neumannTemp;

        InterpolateProductTilde(U2_tot, U3, dirichletTemp);
        r3 += ddy(dirichletTemp);

        InterpolateProduct(U1, U2_tot, neumannTemp);
        r1 += ddy(neumannTemp);

        InterpolateProduct(U2, U1_tot, neumannTemp);
        r2 += ddx(neumannTemp);
    }

    // buoyancy nonlinear terms
    DifferentiateProductBar(B, U3_tot, neumannTemp);
    rB += neumannTemp;

    if(gridParams.ThirdDimension())
    {
        InterpolateProduct(B, U2_tot, neumannTemp);
        rB += ddy(neumannTemp);
    }

    InterpolateProduct(B, U1_tot, neumannTemp);
    rB += ddx(neumannTemp);


    // extra adjoint nonlinear terms
    neumannTemp = ddx(u1_tot);
    neumannTemp.ToNodal(nnTemp);
    nnTemp2 = nnTemp*U1;
    if(gridParams.ThirdDimension())
    {
        neumannTemp = ddx(u2_tot);
        neumannTemp.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
    }
    dirichletTemp = ddx(u3_tot);
    dirichletTemp.ToNodal(ndTemp);
    u1Forcing -= nnTemp2 + ReinterpolateDirichlet(ndTemp*U3);

    if(gridParams.ThirdDimension())
    {
        neumannTemp = ddy(u1_tot);
        neumannTemp.ToNodal(nnTemp);
        nnTemp2 = nnTemp*U1;
        neumannTemp = ddy(u2_tot);
        neumannTemp.ToNodal(nnTemp);
        nnTemp2 += nnTemp*U2;
        dirichletTemp = ddy(u3_tot);
        dirichletTemp.ToNodal(ndTemp);
        u2Forcing -= nnTemp2 + ReinterpolateDirichlet(ndTemp*U3);
    }

    dirichletTemp = ddz(u1_tot);
    dirichletTemp.ToNodal(ndTemp);
    ndTemp2 = ndTemp*ReinterpolateFull(U1);
    if(gridParams.ThirdDimension())
    {
        dirichletTemp = ddz(u2_tot);
        dirichletTemp.ToNodal(ndTemp);
        ndTemp2 += ndTemp*ReinterpolateFull(U2);
    }
    neumannTemp = ddz(u3_tot);
    neumannTemp.ToNodal(nnTemp);
    u3Forcing -= ndTemp2 + ReinterpolateFull(nnTemp)*U3;


    neumannTemp = ddx(b_tot);
    neumannTemp.ToNodal(nnTemp);
    u1Forcing -= nnTemp*B;

    if(gridParams.ThirdDimension())
    {
        neumannTemp = ddy(b_tot);
        neumannTemp.ToNodal(nnTemp);
        u2Forcing -= nnTemp*B;
    }

    dirichletTemp = ddz(b_tot);
    dirichletTemp.ToNodal(ndTemp);
    u3Forcing -= ndTemp*ReinterpolateFull(B);

    // Now include all the forcing terms
    u1Forcing.ToModal(neumannTemp);
    r1 += neumannTemp;
    if (gridParams.ThirdDimension())
    {
        u2Forcing.ToModal(neumannTemp);
        r2 += neumannTemp;
    }
    u3Forcing.ToModal(dirichletTemp);
    r3 += dirichletTemp;
    bForcing.ToModal(neumannTemp);
    rB += neumannTemp;
}
