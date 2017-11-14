#include "Field.h"
#include "Differentiation.h"
#include "Graph.h"

#include <iostream>

class IMEXRK
{
public:
    static constexpr int N1 = 128;
    static constexpr int N2 = 1;
    static constexpr int N3 = 201;

    static constexpr double L1 = 9.44; // size of domain streamwise
    static constexpr double L2 = 15.0;  // size of domain spanwise
    static constexpr double L3 = 15.0; // half size of domain vertically

    const double deltaT = 0.001;
    const double Re = 1000;
    const double Pe = 1000;

    using NField = NodalField<N1,N2,N3>;
    using MField = ModalField<N1,N2,N3>;

public:
    IMEXRK()
    : u1(BoundaryCondition::Neumann)
    , u2(BoundaryCondition::Neumann)
    , u3(BoundaryCondition::Dirichlet)
    , p(BoundaryCondition::Neumann)
    , b(BoundaryCondition::Neumann)

    , R1(u1), R2(u2), R3(u3), RB(b)
    , r1(u1), r2(u2), r3(u3), rB(b)
    , U1(BoundaryCondition::Neumann)
    , U2(BoundaryCondition::Neumann)
    , U3(BoundaryCondition::Dirichlet)
    , B(BoundaryCondition::Neumann)
    , dirichletTemp(BoundaryCondition::Dirichlet)
    , neumannTemp(BoundaryCondition::Neumann)
    , ndTemp(BoundaryCondition::Dirichlet)
    , nnTemp(BoundaryCondition::Neumann)
    , mdProduct(BoundaryCondition::Dirichlet)
    , mnProduct(BoundaryCondition::Neumann)
    , divergence(mnProduct)
    , q(p)
    {
        dim1Derivative = FourierDerivativeMatrix(L1, N1);
        dim2Derivative = FourierDerivativeMatrix(L2, N2);
        dim3DerivativeNeumann = VerticalDerivativeMatrix(BoundaryCondition::Neumann, L3, N3);
        dim3DerivativeDirichlet = VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L3, N3);

        dim1Derivative2 = FourierSecondDerivativeMatrix(L1, N1);
        dim2Derivative2 = FourierSecondDerivativeMatrix(L2, N2);
        dim3Derivative2 = VerticalSecondDerivativeMatrix(L3, N3);

        // we solve each vetical line separately, so N1*N2 total solves
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                MatrixXd laplacian = dim3Derivative2;

                // add terms for horizontal derivatives
                laplacian += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                laplacian += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);

                if (j1==0 && j2==0)
                {
                    // // despite the fact we want neumann boundary conditions,
                    // // we need to impose a boundary value for non-singularity
                    // laplacian.row(0).setConstant(2);
                    // laplacian(0,0) = 1; // the form of DCT we are using has end coefficients different
                    // laplacian(0,N3-1) = 1;

                    // stop matrix being singular
                    laplacian(0,0) = 0.0001;
                }


                solveLaplacian[j1*N2+j2].compute(laplacian);


                // for viscous terms
                explicitSolveVelocity[j1*N2+j2] = dim3Derivative2;
                explicitSolveVelocity[j1*N2+j2] += dim1Derivative2.diagonal()(j1)*MatrixXd::Identity(N3, N3);
                explicitSolveVelocity[j1*N2+j2] += dim2Derivative2.diagonal()(j2)*MatrixXd::Identity(N3, N3);
                explicitSolveBuoyancy[j1*N2+j2] = explicitSolveVelocity[j1*N2+j2];
                explicitSolveVelocity[j1*N2+j2] /= Re;
                explicitSolveBuoyancy[j1*N2+j2] /= Pe;

                for (int k=0; k<s; k++)
                {
                    implicitSolveVelocity[s*(j1*N2+j2) + k].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveVelocity[j1*N2+j2]);
                    implicitSolveBuoyancy[s*(j1*N2+j2) + k].compute(
                        MatrixXd::Identity(N3, N3)-0.5*h[k]*explicitSolveBuoyancy[j1*N2+j2]);
                }
            }
        }
    }


    void TimeStep()
    {
        // see Numerical Renaissance
        for (int k=0; k<s; k++)
        {
            //PlotVerticalVelocity("images/u3initial_"+std::to_string(k)+".png", 0);

            ExplicitUpdate(k);
            ImplicitUpdate(k);

            //PlotVerticalVelocity("images/u3before_"+std::to_string(k)+".png", 0);
            //PlotPressure("images/pressurebefore_"+std::to_string(k)+".png", 0);

            RemoveDivergence(1/h[k]);

            //SolveForPressure();

            //PlotVerticalVelocity("images/u3after_"+std::to_string(k)+".png", 0);
            //PlotPressure("images/pressureafter_"+std::to_string(k)+".png", 0);
        }
    }

    void Quiver(std::string filename, int j2)
    {
        u1.ToNodal(U1);
        u3.ToNodal(U3);

        QuiverPlot(U1, U3, L1, L3, j2, filename);
    }

    void Profile(std::string filename, int j1, int j2) const
    {
        u1.ToNodal(U1);
        Interpolate(U1.stack(j1, j2), L3, BoundaryCondition::Neumann, filename);
    }

    void PlotBuoyancy(std::string filename, int j2) const
    {
        b.ToNodal(B);
        HeatPlot(B, L1, L3, j2, filename);
    }

    void PlotPressure(std::string filename, int j2) const
    {
        p.ToNodal(B);
        HeatPlot(B, L1, L3, j2, filename);
    }

    void PlotVerticalVelocity(std::string filename, int j2) const
    {
        u3.ToNodal(U3);
        HeatPlot(U3, L1, L3, j2, filename);
    }

    void PlotStreamwiseVelocity(std::string filename, int j2) const
    {
        u1.ToNodal(U1);
        HeatPlot(U1, L1, L3, j2, filename);
    }

    void AddVariables(NField velocity1, NField velocity3, NField buoyancy)
    {
        u1.ToNodal(U1);
        u3.ToNodal(U3);
        b.ToNodal(B);

        U1 += velocity1;
        U3 += velocity3;
        B += buoyancy;

        U1.ToModal(u1);
        U3.ToModal(u3);
        B.ToModal(b);
    }

    // gives an upper bound on cfl number
    double CFL()
    {
        u1.ToNodal(U1);
        //u2.ToNodal(U2);
        u3.ToNodal(U3);

        double delta1 = L1/N1;
        double delta2 = L2/N2;
        double delta3 = 2*L3/(N3-1);

        double cfl = U1.Max()/delta1 /*+ U2.Max()/delta2*/ + U3.Max()/delta3;
        cfl *= deltaT;

        return cfl;
    }

    void RemoveDivergence(double pressureMultiplier=1.0)
    {
        // construct the diverence of u
        divergence.Zero();

        u1.Dim1MatMul(dim1Derivative, neumannTemp);
        divergence += neumannTemp;
        //u2.Dim2MatMul(dim2Derivative, neumannTemp);
        //divergence += neumannTemp;
        u3.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        divergence += neumannTemp;

        //divergence.ToNodal(B);
        //HeatPlot(B, L1, L3, 0, "images/divergence.png");

        // boundary value
        //divergence(0,0,0) = 0;

        // solve Δq = ∇·u as linear system Aq = divergence
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                divergence.Dim3Solve(solveLaplacian[j1*N2+j2], j1, j2, q);
            }
        }

        //q.ToNodal(B);
        //HeatPlot(B, L1, L3, 0, "images/pressureupdate.png");

        //std::cout << q.stack(5, 0) << std::endl << std::endl;

        // subtract the gradient of this from the velocity
        q.Dim1MatMul(dim1Derivative, neumannTemp);
        u1 -= neumannTemp;
        //q.Dim2MatMul(dim2Derivative, neumannTemp);
        //u2 -= neumannTemp;
        q.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        u3 -= dirichletTemp;
        //std::cout << dirichletTemp.stack(5, 0) << std::endl << std::endl;

        // also add it on to p for the next step
        // this is scaled to match the p that was added before
        // effectively we have forward euler
        p += pressureMultiplier*q;
    }

    void FilterVariables()
    {
        u1.Filter();
        //u2.Filter();
        u3.Filter();
        b.Filter();
    }

    void SolveForPressure()
    {
        // solve Δp = -∇u∇u
        // (store rhs in divergence)
        divergence.Zero();


        // todo: cleanup use of temp variables
        u1.Dim1MatMul(dim1Derivative, neumannTemp);
        neumannTemp.ToNodal(U1);
        NodalProduct(U1, U1, nnTemp);
        nnTemp.ToModal(mnProduct);
        divergence -= mnProduct;

        u3.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        neumannTemp.ToNodal(U2);
        NodalProduct(U2, U2, nnTemp);
        nnTemp.ToModal(mnProduct);
        divergence -= mnProduct;

        u1.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        dirichletTemp.ToNodal(ndTemp);
        u3.Dim1MatMul(dim1Derivative, dirichletTemp);
        dirichletTemp.ToNodal(U3);
        NodalProduct(ndTemp, U3, nnTemp);
        nnTemp.ToModal(mnProduct);
        divergence += (-2)*mnProduct;


        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                divergence.Dim3Solve(solveLaplacian[j1*N2+j2], j1, j2, p);
            }
        }
    }


private:
    void ImplicitUpdate(int k)
    {
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                R1.Dim3Solve(implicitSolveVelocity[s*(j1*N2+j2) + k], j1, j2, u1);
                //R2.Dim3Solve(implicitSolveVelocity[s*(j1*N2+j2) + k], j1, j2, u2);
                R3.Dim3Solve(implicitSolveVelocity[s*(j1*N2+j2) + k], j1, j2, u3);
                RB.Dim3Solve(implicitSolveBuoyancy[s*(j1*N2+j2) + k], j1, j2, b);
            }
        }
    }

    void ExplicitUpdate(int k)
    {
        // calculate rhs terms and accumulate in y
        R1 = u1;
        //R2 = u2;
        R3 = u3;
        RB = b;

        // add term from last rk step
        R1 += (h[k]*zeta[k])*r1;
        //R2 += (h[k]*zeta[k])*r2;
        R3 += (h[k]*zeta[k])*r3;
        RB += (h[k]*zeta[k])*rB;

        // explicit part of CN
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                u1.Dim3MatMul(explicitSolveVelocity[j1*N2+j2], j1, j2, neumannTemp);
            }
        }
        R1 += 0.5*h[k]*neumannTemp;

        // for (int j1=0; j1<N1; j1++)
        // {
        //     for (int j2=0; j2<N2; j2++)
        //     {
        //         u2.Dim3MatMul(explicitSolveVelocity[j1*N2+j2], j1, j2, neumannTemp);
        //     }
        // }
        // R2 += 0.5*h[k]*neumannTemp;

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                u3.Dim3MatMul(explicitSolveVelocity[j1*N2+j2], j1, j2, dirichletTemp);
            }
        }
        R3 += 0.5*h[k]*dirichletTemp;

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                b.Dim3MatMul(explicitSolveBuoyancy[j1*N2+j2], j1, j2, neumannTemp);
            }
        }
        RB += 0.5*h[k]*neumannTemp;

        // now construct explicit terms
        r1.Zero();
        //r2.Zero();
        r3.Zero();
        rB.Zero();

        //////// NONLINEAR TERMS ////////

        // calculate products at nodes in physical space
        u1.ToNodal(U1);
        //u2.ToNodal(U2);
        u3.ToNodal(U3);
        b.ToNodal(B);

        NodalProduct(U1, U1, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        r1 -= neumannTemp;

        // NodalProduct(U1, U2, nnTemp);
        // nnTemp.ToModal(mnProduct);
        // mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        // r2 -= neumannTemp;
        // mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        // r1 -= neumannTemp;

        NodalProduct(U1, U3, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim1MatMul(dim1Derivative, dirichletTemp);
        r3 -= dirichletTemp;
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        r1 -= neumannTemp;

        // NodalProduct(U2, U2, nnTemp);
        // nnTemp.ToModal(mnProduct);
        // mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        // r2 -= neumannTemp;

        // NodalProduct(U2, U3, ndTemp);
        // ndTemp.ToModal(mdProduct);
        // mdProduct.Dim2MatMul(dim2Derivative, dirichletTemp);
        // r3 -= dirichletTemp;
        // mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        // r2 -= neumannTemp;

        NodalProduct(U3, U3, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        r3 -= dirichletTemp;

        // buoyancy nonlinear terms
        NodalProduct(U1, B, nnTemp);
        nnTemp.ToModal(mnProduct);
        mnProduct.Dim1MatMul(dim1Derivative, neumannTemp);
        rB -= neumannTemp;

        // NodalProduct(U2, B, nnTemp);
        // nnTemp.ToModal(mnProduct);
        // mnProduct.Dim2MatMul(dim2Derivative, neumannTemp);
        // rB -= neumannTemp;

        NodalProduct(U3, B, ndTemp);
        ndTemp.ToModal(mdProduct);
        mdProduct.Dim3MatMul(dim3DerivativeDirichlet, neumannTemp);
        rB -= neumannTemp;

        // now add on explicit terms
        R1 += (h[k]*beta[k])*r1;
        // R2 += (h[k]*beta[k])*r2;
        R3 += (h[k]*beta[k])*r3;
        RB += (h[k]*beta[k])*rB;

        // now add on pressure term
        p.Dim1MatMul(dim1Derivative, neumannTemp);
        R1 += (-h[k])*neumannTemp;

        // p.Dim2MatMul(dim2Derivative, neumannTemp);
        // R2 += (-h[k])*neumannTemp;

        p.Dim3MatMul(dim3DerivativeNeumann, dirichletTemp);
        R3 += (-h[k])*dirichletTemp;
    }

private:
    // these are the actual variables we care about
    MField u1, u2, u3, b;
    MField p;

    // parameters for the scheme
    const int s = 3;
    const double h[3] = {deltaT*8.0/15.0, deltaT*2.0/15.0, deltaT*5.0/15.0};
    const double beta[3] = {1.0, 25.0/8.0, 9.0/4.0};
    const double zeta[3] = {0, -17.0/8.0, -5.0/4.0};

    // these are intermediate variables used in the computation, preallocated for efficiency
    MField R1, R2, R3, RB;
    MField r1, r2, r3, rB;
    mutable NField U1, U2, U3, B;
    mutable NField ndTemp, nnTemp;
    mutable MField mdProduct, mnProduct;
    mutable MField dirichletTemp, neumannTemp;
    MField& divergence; // reference to share memory
    MField q;

    // these are precomputed matrices for performing and solving derivatives
    DiagonalMatrix<complex, -1> dim1Derivative;
    DiagonalMatrix<complex, -1> dim2Derivative;
    MatrixXd dim3DerivativeNeumann;
    MatrixXd dim3DerivativeDirichlet;

    DiagonalMatrix<double, -1> dim1Derivative2;
    DiagonalMatrix<double, -1> dim2Derivative2;
    DiagonalMatrix<double, -1> dim3Derivative2;

    std::array<MatrixXcd, N1*N2> explicitSolveVelocity;
    std::array<MatrixXcd, N1*N2> explicitSolveBuoyancy;
    std::array<ColPivHouseholderQR<MatrixXcd>, 3*N1*N2> implicitSolveVelocity;
    std::array<ColPivHouseholderQR<MatrixXcd>, 3*N1*N2> implicitSolveBuoyancy;
    std::array<ColPivHouseholderQR<MatrixXcd>, N1*N2> solveLaplacian;
};

int main()
{
    IMEXRK solver;

    IMEXRK::NField initialU1(BoundaryCondition::Neumann);
    IMEXRK::NField initialU3(BoundaryCondition::Dirichlet);
    IMEXRK::NField initialB(BoundaryCondition::Neumann);
    auto x3 = VerticalPoints(IMEXRK::L3, IMEXRK::N3);

    double interfaceoffset = 1.0;

    for (int j=0; j<IMEXRK::N3; j++)
    {
        if (j!=0 && j!=IMEXRK::N3-1)// && j!= IMEXRK::N3/2)// (j==5 || j == IMEXRK::N3-5)//
        {
            initialU1.slice(j) += 0.1*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);
            initialU3.slice(j) += 0.1*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2);

            //initialU1.slice(j) += 0.1*x3(j)*x3(j)*ArrayXd::Random(IMEXRK::N1, IMEXRK::N2)/(IMEXRK::L3*IMEXRK::L3);
        }
    }
    solver.AddVariables(initialU1, initialU3, initialB);
    //solver.FilterVariables();

    // add background flow
    initialU1.Zero();
    initialU3.Zero();
    initialB.Zero();
    for (int j=0; j<IMEXRK::N3; j++)
    {
        initialU1.slice(j).setConstant(tanh(x3(j)-interfaceoffset));
        initialB.slice(j).setConstant(tanh(x3(j)-interfaceoffset));

        // if (x3(j)-interfaceoffset>0)
        // {
        //     initialB.slice(j).setConstant(-1);
        // }
        // if (x3(j)-interfaceoffset<0)
        // {
        //     initialB.slice(j).setConstant(1);
        // }

        // band to see fluid velocity
        initialB.slice(j)(0, IMEXRK::N2/2) = 0;
        initialB.slice(j)(1, IMEXRK::N2/2) = 0;
        initialB.slice(j)(2, IMEXRK::N2/2) = 0;
        initialB.slice(j)(3, IMEXRK::N2/2) = 0;
        initialB.slice(j)(4, IMEXRK::N2/2) = 0;
    }

    solver.AddVariables(initialU1, initialU3, initialB);

    solver.RemoveDivergence(0.0);
    solver.SolveForPressure();

    solver.PlotPressure("images/pressure/initial.png", IMEXRK::N2/2);

    for (int step=0; step<10000; step++)
    {
        //solver.FilterVariables();
        solver.TimeStep();

        if(step%200==0)
        {
            //solver.Quiver("images/"+std::to_string(step)+".png", IMEXRK::N2/2);
            //solver.Profile("images/"+std::to_string(step)+"profile.png", 0, 0);
            solver.PlotPressure("images/pressure/"+std::to_string(step)+".png", IMEXRK::N2/2);
            solver.PlotBuoyancy("images/buoyancy/"+std::to_string(step)+".png", IMEXRK::N2/2);
            solver.PlotVerticalVelocity("images/u3/"+std::to_string(step)+".png", IMEXRK::N2/2);
            solver.PlotStreamwiseVelocity("images/u1/"+std::to_string(step)+".png", IMEXRK::N2/2);

            double cfl = solver.CFL();
            std::cout << "Step " << step << ", time " << step*solver.deltaT
                      << ", CFL number: " << cfl << std::endl;
        }
    }

    return 0;
}