#include "Stratiflow.h"
#include "Differentiation.h"

ArrayXc CalculateEigenvalues(stratifloat k, MatrixXc *w_eigen = nullptr, MatrixXc *b_eigen = nullptr)
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

    MatrixXc A(2*N3, 2*N3);
    MatrixXc C(2*N3, 2*N3);

    A << A11, A12, A21, A22;
    C << C11, C12, C21, C22;

    // eigen's generalised eigenvalue solver currently supports only real matrices
    // so this is a hack around that

    MatrixXc CinvA = C.inverse() * A;

    ComplexEigenSolver<MatrixXc> solver;

    solver.compute(CinvA);

    if (w_eigen!=nullptr)
    {
        *w_eigen = solver.eigenvectors().topRows(N3);
    }

    if (b_eigen!=nullptr)
    {
        *b_eigen = solver.eigenvectors().bottomRows(N3);
    }

    return solver.eigenvalues();
}

stratifloat LargestGrowth(stratifloat k,
                          Field1D<complex, N1, N2, N3>* w=nullptr,
                          Field1D<complex, N1, N2, N3>* b=nullptr)
{
    MatrixXc w_eigen(N3,2*N3);
    MatrixXc b_eigen(N3,2*N3);

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

int main()
{

    stratifloat kmax;
    stratifloat growthmax = -10000;

    stratifloat k_lower = 0.00001;
    stratifloat k_upper = 2.0;

    for (int n=0; n<5; n++)
    {
        stratifloat deltak = (k_upper-k_lower)/10;

        for (stratifloat k=k_lower; k<=k_upper; k+=deltak)
        {
            auto largest = LargestGrowth(k);

            if (largest>growthmax)
            {
                growthmax = largest;
                kmax = k;
            }
        }

        k_lower = kmax - deltak;
        k_upper = kmax + deltak;
    }

    std::cout << "Maximum growth rate " << growthmax << " at " << kmax << std::endl;
    std::cout << "Wavelength of fastest growing mode is " << 2*pi/kmax << std::endl;

    return 0;
}