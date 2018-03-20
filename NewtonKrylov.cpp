#include "FullState.h"

class NewtonKrylov
{
public:
    // using the GMRES routine, perform Newton-Raphson iteration
    // x : an initial guess, also the result when finished
    void NewtonRaphson(FullState& x)
    {
        FullState dx; // update, solve into here to save reallocating memory
        for (int n=0; n<N; n++)
        {
            // first nonlinearly evolve current state
            FullState rhs; // = G-x
            x.FullEvolve(T, rhs, true);
            rhs -= x;

            stratifloat residual = rhs.Norm();

            std::cout << "STEP " << n << ", RESIDUAL: " << residual << std::endl;

            // solve matrix system
            GMRES(rhs, dx);

            // update
            x += dx;
        }
    }

private:
    // solves (I-Gx) dx = G-x for dx
    // GMRES is a Krylov-subspace method, hence Newton-Krylov
    void GMRES(const FullState& rhs, FullState& dx) const
    {
        // TODO

        int K = 10;

        // Arnoldi Algorithm
        // find orthogonal basis q1,...,qn
        // from dx, A dx, A^2 dx, ...
        // where A = I-Gx

        std::vector<FullState> q(K);
        std::vector<stratifloat> H(K*K);

        q[0] = dx;
        for (int k=1; k<K; k++)
        {
            // q_k = A q_k-1
            FullState Gq;
            q[k-1].LinearEvolve(T, Gq);
            q[k] = q[k-1];
            q[k] -= Gq;

            // remove component in direction of proceeding vectors
            for (int j=0; j<k; j++)
            {
                H[k*K+j] = q[j].Dot(q[k]);
                q[k].MulAdd(-H[k*K+j], q[j]);
            }

            // normalise
            H[k*K+k] = q[k].Norm();
            q[k] *= 1/H[k*K+k];
        }
    }

    int N = 100; // max iterations
    stratifloat T = 10; // time interval for integration
};

int main()
{
    // check it converges to steady state

    NewtonKrylov solver;

    FullState stationaryPoint;

    NField initialU1(BoundaryCondition::Bounded);
    NField initialU3(BoundaryCondition::Decaying);
    auto x3 = VerticalPoints(L3, N3);

    stratifloat bandmax = 4;
    for (int j=0; j<N3; j++)
    {
        if (x3(j) > -bandmax && x3(j) < bandmax)
        {
            initialU1.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                * Array<stratifloat, N1, N2>::Random(N1, N2);
            initialU3.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                * Array<stratifloat, N1, N2>::Random(N1, N2);
        }
    }

    initialU1.ToModal(stationaryPoint.u1);
    initialU3.ToModal(stationaryPoint.u3);

    solver.NewtonRaphson(stationaryPoint);
}