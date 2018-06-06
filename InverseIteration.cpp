#include "ExtendedStateVector.h"

int main(int argc, const char* argv[])
{
    ExtendedStateVector background;
    background.LoadFromFile(argv[1]);

    Ri = background.p;

    StateVector b;
    //b.Randomise(0.1);

    b = background.x;

    b *= 1/b.Norm();

    stratifloat mu = 1; // eigenvalue we want to find
    stratifloat T = 5;
    stratifloat epsilon = 0.05;

    int iter = 0;
    while(true)
    {
        b.PlotAll("eig"+std::to_string(iter));

        // find a residual
        StateVector bT;
        b.LinearEvolve(T, background.x, bT);

        stratifloat muApprox = bT.Norm() / b.Norm();

        bT.MulAdd(-muApprox, b);

        std::cout << "Inverse iteration " << iter << ", RESIUDAL " << bT.Norm() << std::endl;
        std::cout << "Approx eigenvalue: " << muApprox << std::endl;

        iter++;


        int K = 512; // max iterations
        std::vector<StateVector> q(K);
        MatrixX H(K, K-1); // upper Hessenberg matrix



        VectorX y; // result in new basis

        q[0] = b;
        q[0].EnforceBCs();

        stratifloat beta = q[0].Norm();
        std::cout << beta << std::endl;
        q[0] *= 1/beta;

        K = q.size();
        for (int k=1; k<K; k++)
        {
            // Arnoldi Algorithm
            // find orthogonal basis q1,...,qn
            // from x, A x, A^2 x, ...

            // q_k = A q_k-1
            q[k-1].LinearEvolve(T, background.x, q[k]);
            q[k].MulAdd(-mu, q[k-1]);

            // remove component in direction of preceding vectors
            for (int j=0; j<k; j++)
            {
                H(j,k-1) = q[j].Dot(q[k]);
                q[k].MulAdd(-H(j,k-1), q[j]);
            }

            // normalise
            H(k,k-1) = q[k].Norm();
            q[k] *= 1/H(k,k-1);

            // enforce BCs
            q[k].EnforceBCs();

            // Construct least squares problem in this basis
            VectorX Beta(k+1);
            Beta.setZero();
            Beta[0] = beta;

            MatrixX subH = H.block(0,0,k+1,k);

            // Now we solve Hy = Beta

            // follows notation of Chandler & Kerswell 2013

            // first H = UDV*
            JacobiSVD<MatrixX> svd(subH, ComputeFullU | ComputeFullV);
            MatrixX U = svd.matrixU();
            MatrixX V = svd.matrixV();
            ArrayX d = svd.singularValues();

            // solve problem in space of singular vectors
            VectorX p = U.transpose() * Beta; // p = U* Beta
            VectorX z = p.array()/d; // D z = p

            // z = V* y
            y = V*z;

            stratifloat residual = (subH*y - Beta).norm()/beta;

            std::cout << "GMRES STEP " << k << ", RESIDUAL: " << residual << std::endl;

            if (residual < epsilon)
            {
                K = k+1;
                break;
            }
        }

        // Now compute the solution using the basis vectors
        b.Zero();
        for (int k=0; k<K-1; k++)
        {
            b.MulAdd(y[k], q[k]);
        }

        // normalise
        b *= 1/b.Norm();
    }
}