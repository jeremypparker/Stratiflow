#include "OrrSommerfeld.h"

#include "Differentiation.h"

#include <unsupported/Eigen/MatrixFunctions>

MatrixXc OrrSommerfeldLHS(stratifloat k)
{
    N1D U;
    N1D B;
    U.SetValue(InitialU, L3);
    B.SetValue(InitialB, L3);

    N1D Upp;
    Upp = ddz(ddz(U));
    N1D Bp;
    Bp = ddz(B);

    auto D2 = VerticalSecondDerivativeNodalMatrix(L3, N3);
    auto I = MatrixX::Identity(N3, N3);

    MatrixX Um = U.Get().matrix().asDiagonal();
    MatrixX Uppm = Upp.Get().matrix().asDiagonal();
    MatrixX Bpm = Bp.Get().matrix().asDiagonal();

    MatrixXc A11 = i*Um*(D2-k*k*I)
                   -i*Uppm*I
                   -(1/Re/k)*(D2-k*k*I)*(D2-k*k*I);

    MatrixXc A12 = k*Ri*I;

    MatrixXc A21 = (-1/k)*Bpm;

    MatrixXc A22 = i*Um
                   -(1/Pe/k)*(D2-k*k*I);

    // values at infinity must be zero, so ignore those bits
    MatrixXc A(2*(N3-2), 2*(N3-2));

    A << A11.block(1, 1, N3-2, N3-2), A12.block(1, 1, N3-2, N3-2),
         A21.block(1, 1, N3-2, N3-2), A22.block(1, 1, N3-2, N3-2);

    return A;
}

MatrixXc OrrSommerfeldRHS(stratifloat k)
{
    N1D U;
    N1D B;
    U.SetValue(InitialU, L3);
    B.SetValue(InitialB, L3);

    N1D Upp;
    Upp = ddz(ddz(U));
    N1D Bp;
    Bp = ddz(B);

    auto D2 = VerticalSecondDerivativeNodalMatrix(L3, N3);
    auto I = MatrixX::Identity(N3, N3);

    MatrixX Um = U.Get().matrix().asDiagonal();
    MatrixX Uppm = Upp.Get().matrix().asDiagonal();
    MatrixX Bpm = Bp.Get().matrix().asDiagonal();

    MatrixXc C11 = -(D2-k*k*I);

    MatrixXc C12 = MatrixXc::Zero(N3, N3);

    MatrixXc C21 = MatrixXc::Zero(N3, N3);

    MatrixXc C22 = -I;

    // values at infinity must be zero, so ignore those bits
    MatrixXc C(2*(N3-2), 2*(N3-2));

    C << C11.block(1, 1, N3-2, N3-2), C12.block(1, 1, N3-2, N3-2),
         C21.block(1, 1, N3-2, N3-2), C22.block(1, 1, N3-2, N3-2);

    return C;
}

ArrayXc CalculateEigenvalues(stratifloat k, MatrixXc *w_eigen, MatrixXc *b_eigen)
{
    // The result of this is the vertical profile of the vertical velocity and buoyancy

    MatrixXc A = OrrSommerfeldLHS(k);
    MatrixXc C = OrrSommerfeldRHS(k);

    // eigen's generalised eigenvalue solver currently supports only real matrices
    // so this is a hack around that
    MatrixXc CinvA = C.inverse() * A;

    ComplexEigenSolver<MatrixXc> solver;

    solver.compute(CinvA, w_eigen!=nullptr || b_eigen!=nullptr);
    if (solver.info() != Success)
    {
        std::cout << "Failed to converge!" << std::endl;
    }

    if (w_eigen!=nullptr)
    {
        w_eigen->block(1,0,N3-2,N3-2) = solver.eigenvectors().block(0,0,N3-2,N3-2);
    }

    if (b_eigen!=nullptr)
    {
        b_eigen->block(1,0,N3-2,N3-2) = solver.eigenvectors().block(N3-2,0,N3-2,N3-2);
    }

    return solver.eigenvalues();
}

stratifloat LargestGrowth(stratifloat k,
                          Field1D<complex, N1, N2, N3>* w,
                          Field1D<complex, N1, N2, N3>* b,
                          stratifloat* imagpart)
{
    if (w==nullptr && b==nullptr && imagpart!=nullptr)
    {
        MatrixXc A = OrrSommerfeldLHS(k);
        MatrixXc C = OrrSommerfeldRHS(k);

        MatrixXc CinvA = C.inverse() * A;

        stratifloat T = 2000;

        // matrix exponential means real part of eigenvalue becomes modulus of eigenvalue
        MatrixXc M = (T*CinvA).exp();

        // do power iteration on a random initial vector
        VectorXc x = VectorXc::Random(CinvA.rows());
        stratifloat lambda;
        for (int j=0; j<400; j++)
        {
            x.normalize();

            VectorXc Mx = M*x;

            complex eigApprox = x.dot(Mx)/x.dot(x);

            lambda = std::abs(eigApprox);

            x = Mx; // iterate
        }

        return std::log(lambda)/T;
    }
    MatrixXc w_eigen(N3,2*(N3-2));
    MatrixXc b_eigen(N3,2*(N3-2));

    ArrayXc eigenvalues;
    if (w!=nullptr|| b!=nullptr)
    {
        eigenvalues = CalculateEigenvalues(k, &w_eigen, &b_eigen);
    }
    else
    {
        eigenvalues = CalculateEigenvalues(k);
    }

    // find maximum growth
    int jmax;
    stratifloat largest = eigenvalues.real().maxCoeff(&jmax);

    //std::cout << k << " : " << largest << std::endl;

    if (w!=nullptr)
    {
        w->Get() = w_eigen.col(jmax);
    }
    if (b!=nullptr)
    {
        b->Get() = b_eigen.col(jmax);
    }
    if (imagpart!=nullptr)
    {
        // assume they come in conjugate pairs, so take abs
        *imagpart = std::abs(eigenvalues.imag()(jmax));
    }

    return largest;
}

void EigenModes(stratifloat k, MField& u1, MField& u2, MField& u3, MField& b)
{
    // find the vertical profile of eigenmodes
    Field1D<complex, N1, N2, N3> w_hat;
    Field1D<complex, N1, N2, N3> b_hat;

    LargestGrowth(k, &w_hat, &b_hat);

    // multiply out the modes
    NField W;
    NField B;

    auto x = FourierPoints(L1, N1);

    for3D(N1,N2,N3)
    {
        complex w_hat_j;
        complex b_hat_j;
        if (EnforceSymmetry)
        {
            // eigenvalues come in complex conjugate pairs, with hermitian symmetry of eigenfunctions
            // make sure we include both parts equally
            w_hat_j = w_hat.Get()(j3) + std::conj(w_hat.Get()(N3-j3-1));
            b_hat_j = b_hat.Get()(j3) + std::conj(b_hat.Get()(N3-j3-1));
        }
        else
        {
            w_hat_j = w_hat.Get()(j3);
            b_hat_j = b_hat.Get()(j3);
        }

        W(j1,j2,j3) = real(w_hat_j * exp(i*k*x(j1)));

        // todo: make consistent the definition of b
        B(j1,j2,j3) = -real(b_hat_j * exp(i*k*x(j1)));
    } endfor3D

    W.ToModal(u3);
    B.ToModal(b);

    // incompressibility gives us streamwise velocity
    u1 = (i/k)*ddz(u3); // only one fourier mode so can just divide to integrate

    // squire's theorem tells us spanwise velocity is zero
    u2.Zero();

}
