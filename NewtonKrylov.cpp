#include "FullState.h"

class NewtonKrylov
{
public:
    // using the GMRES routine, perform Newton-Raphson iteration
    // x : an initial guess, also the result when finished
    void NewtonRaphson(FullState& x)
    {
        FullState dx; // update, solve into here to save reallocating memory
        dx = x; // initial guess for update
        for (int n=0; n<N; n++)
        {
            // first nonlinearly evolve current state
            FullState rhs; // = G-x
            x.FullEvolve(T, rhs, true);
            rhs -= x;

            stratifloat residual = rhs.Norm();

            std::cout << "NEWTON STEP " << n << ", RESIDUAL: " << residual << std::endl;

            // solve matrix system
            GMRES(rhs, dx, 0.001);

            // update
            x += dx;
        }
    }

private:
    // solves A x = G-x0 for x
    // where A = I-G_x
    // GMRES is a Krylov-subspace method, hence Newton-Krylov
    void GMRES(const FullState& rhs, FullState& x, stratifloat epsilon) const
    {
        int K = 128; // max iterations

        std::vector<FullState> q(K);

        MatrixX H(K, K-1); // upper Hessenberg matrix representing A
        H.setZero();

        VectorX y; // result in new basis

        FullState r; // r = rhs - A x = rhs + G_x x - x
        x.LinearEvolve(T, r);
        r += rhs;
        r -= x;

        stratifloat beta = r.Norm();

        std::cout << beta << std::endl;

        q[0] = r;
        q[0] *= 1/beta;
        for (int k=1; k<K; k++)
        {
            // Arnoldi Algorithm
            // find orthogonal basis q1,...,qn
            // from x, A x, A^2 x, ...

            // q_k = A q_k-1
            FullState Gq;
            q[k-1].LinearEvolve(T, Gq);
            q[k] = q[k-1];
            q[k] -= Gq;

            // remove component in direction of preceding vectors
            for (int j=0; j<k; j++)
            {
                H(j,k-1) = q[j].Dot(q[k]);
                q[k].MulAdd(-H(j,k-1), q[j]);
            }

            // normalise
            H(k,k-1) = q[k].Norm();
            q[k] *= 1/H(k,k-1);

            VectorX Beta(k+1);
            Beta.setZero();
            Beta[0] = beta;

            MatrixX subH = H.block(0,0,k+1,k);
            y = subH.colPivHouseholderQr().solve(Beta);

            stratifloat residual = (subH*y - Beta).norm()/beta;

            std::cout << "GMRES STEP " << k << ", RESIDUAL: " << residual << std::endl;

            if (residual < epsilon)
            {
                K = k+1;
                break;
            }
        }

        // Now compute the solution using the basis vectors
        for (int k=0; k<K; k++)
        {
            x.MulAdd(y[k], q[k]);
        }
    }

    int N = 10; // max iterations
    stratifloat T = 5; // time interval for integration
};

int main(int argc, char *argv[])
{
    NewtonKrylov solver;

    FullState stationaryPoint;

    if (argc == 2)
    {
        stationaryPoint.LoadFromFile(argv[1]);
    }
    else
    {
        // check it converges to steady state
        stationaryPoint.u1.RandomizeCoefficients(0.3);
        stationaryPoint.u3.RandomizeCoefficients(0.3);
        stationaryPoint.b.RandomizeCoefficients(0.3);

        stationaryPoint.u1 *= 0.0001;
        stationaryPoint.u3 *= 0.0001;
        stationaryPoint.b *= 0.0001;
    }

    solver.NewtonRaphson(stationaryPoint);
}