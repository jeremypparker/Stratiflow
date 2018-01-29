#include "OrrSommerfeld.h"

#include "Differentiation.h"

ArrayXc CalculateEigenvalues(stratifloat k, MatrixXc *w_eigen, MatrixXc *b_eigen)
{
    // The result of this is the vertical profile of the vertical velocity and buoyancy


    N1D U(BoundaryCondition::Bounded);
    N1D B(BoundaryCondition::Bounded);
    U.SetValue(InitialU, L3);
    B.SetValue(InitialB, L3);

    N1D Upp(BoundaryCondition::Bounded);
    N1D Bp(BoundaryCondition::Decaying);

    {
        M1D u(BoundaryCondition::Bounded);
        M1D b(BoundaryCondition::Bounded);
        U.ToModal(u);
        B.ToModal(b);

        M1D bp(BoundaryCondition::Decaying);
        M1D upp(BoundaryCondition::Bounded);
        bp = ddz(b);
        upp = ddz(ddz(u));

        bp.ToNodal(Bp);
        upp.ToNodal(Upp);
    }

    auto D2 = VerticalSecondDerivativeNodalMatrix(L3, N3);
    auto I = MatrixX::Identity(N3, N3);

    MatrixX Um = U.Get().matrix().asDiagonal();
    MatrixX Uppm = Upp.Get().matrix().asDiagonal();
    MatrixX Bpm = Bp.Get().matrix().asDiagonal();

    MatrixXc A11 = i*k*Um*(D2-k*k*I)
                   -i*k*Uppm*I
                   -(1/Re)*(D2-k*k*I)*(D2-k*k*I);

    MatrixXc A12 = k*k*Ri*I;

    MatrixXc A21 = -Bpm;

    MatrixXc A22 = i*k*Um
                   -(1/Pe)*(D2-k*k*I);

    MatrixXc C11 = -(D2-k*k*I);

    MatrixXc C12 = MatrixXc::Zero(N3, N3);

    MatrixXc C21 = MatrixXc::Zero(N3, N3);

    MatrixXc C22 = -I;

    // values at infinity must be zero, so ignore those bits
    MatrixXc A(2*(N3-2), 2*(N3-2));
    MatrixXc C(2*(N3-2), 2*(N3-2));

    A << A11.block(1, 1, N3-2, N3-2), A12.block(1, 1, N3-2, N3-2),
         A21.block(1, 1, N3-2, N3-2), A22.block(1, 1, N3-2, N3-2);
    C << C11.block(1, 1, N3-2, N3-2), C12.block(1, 1, N3-2, N3-2),
         C21.block(1, 1, N3-2, N3-2), C22.block(1, 1, N3-2, N3-2);

    // eigen's generalised eigenvalue solver currently supports only real matrices
    // so this is a hack around that

    MatrixXc CinvA = C.inverse() * A;

    ComplexEigenSolver<MatrixXc> solver;

    solver.compute(CinvA);
    if (solver.info() != Success)
    {
        std::cout << "Failed to converge!" << std::endl;
    }

    if (w_eigen!=nullptr)
    {
        w_eigen->block(1,0,N3-2,N3-2) = solver.eigenvectors().topRows(N3-2);
    }

    if (b_eigen!=nullptr)
    {
        b_eigen->block(1,0,N3-2,N3-2) = solver.eigenvectors().bottomRows(N3-2);
    }

    return solver.eigenvalues();
}

stratifloat LargestGrowth(stratifloat k,
                          Field1D<complex, N1, N2, N3>* w,
                          Field1D<complex, N1, N2, N3>* b)
{
    MatrixXc w_eigen(N3,2*(N3-2));
    MatrixXc b_eigen(N3,2*(N3-2));

    auto eigenvalues = CalculateEigenvalues(k, &w_eigen, &b_eigen);

    // find maximum growth
    stratifloat largest = -100000;
    stratifloat jmax = -1;
    for (int j=0; j<eigenvalues.size(); j++)
    {
        auto eigenvalue = eigenvalues[j];

        if (real(eigenvalue) > largest)
        {
            largest = real(eigenvalue);
            jmax = j;
        }
    }

    std::cout << k << " : " << largest << std::endl;

    if (w!=nullptr)
    {
        w->Get() = w_eigen.col(jmax);
    }
    if (b!=nullptr)
    {
        b->Get() = b_eigen.col(jmax);
    }

    return largest;
}

void EigenModes(stratifloat k, MField& u1, MField& u2, MField& u3, MField& b)
{
    // find the vertical profile of eigenmodes
    Field1D<complex, N1, N2, N3> w_hat(BoundaryCondition::Decaying);
    Field1D<complex, N1, N2, N3> b_hat(BoundaryCondition::Decaying);

    LargestGrowth(k, &w_hat, &b_hat);

    // multiply out the modes
    NField W(BoundaryCondition::Decaying);
    NField B(BoundaryCondition::Bounded);

    auto x = FourierPoints(L1, N1);

    for3D(N1,N2,N3)
    {
        W(j1,j2,j3) = real(w_hat.Get()(j3) * exp(i*k*x(j1)));

        // todo: make consistent the definition of b
        B(j1,j2,j3) = -real(b_hat.Get()(j3) * exp(i*k*x(j1)));
    } endfor3D

    W.ToModal(u3);
    B.ToModal(b);

    // incompressibility gives us streamwise velocity
    u1 = (i/k)*ddz(u3); // only one fourier mode so can just divide to integrate

    // squire's theorem tells us spanwise velocity is zero
    u2.Zero();

}