#include "StateVector.h"
#include "ExtendedStateVector.h"

template<typename VectorType>
class Arnoldi
{
public:
    Arnoldi()
    : q(K)
    , H(K, K-1)
    {
        H.setZero();
    }

    void Run(const VectorType& at)
    {
        EvalFunction(at);

        q[0].Randomise(0.1, true);

        q[0].EnforceBCs();

        q[0] *= 1/q[0].Norm();

        K = q.size();
        for (int k=1; k<K; k++)
        {
            // Arnoldi Algorithm
            // find orthogonal basis q1,...,qn
            // from x, A x, A^2 x, ...

            // q_k = A q_k-1
            q[k] = EvalLinearised(q[k-1]);

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

            MatrixX subH = H.block(0,0,k,k);

            EigenSolver<MatrixX> ces(subH, false);
            ArrayXc complexEigenvalues = ces.eigenvalues();
            ArrayX eigenvalues = sqrt(complexEigenvalues.real()*complexEigenvalues.real()
                                    + complexEigenvalues.imag()*complexEigenvalues.imag());
            int maxIndex;
            stratifloat maxCoeff = eigenvalues.maxCoeff(&maxIndex);

            std::cout << "At step " << k << ", maximum growth rate: " << maxCoeff << std::endl;
        }

        MatrixX subH = H.block(0,0,K-1,K-1);

        EigenSolver<MatrixX> ces(subH, true);
        ArrayXc complexEigenvalues = ces.eigenvalues();
        ArrayX eigenvalues = sqrt(complexEigenvalues.real()*complexEigenvalues.real()
                                + complexEigenvalues.imag()*complexEigenvalues.imag());
        int maxIndex = -1;
        stratifloat maxEigenvalue=-10000;
        stratifloat maxCoeff = eigenvalues.maxCoeff(&maxIndex);
        VectorXc eigenvector = ces.eigenvectors().col(maxIndex);

        StateVector eigenfunction;
        // StateVector eigenfunctionI;
        for (int k=0; k<K-1; k++)
        {
            eigenfunction += eigenvector(k).real() * q[k];
            // eigenfunctionI += eigenvector(k).imag() * q[k];
        }

        eigenfunction.PlotAll("eigReal");
        // eigenfunctionI.PlotAll("eigImag");

        std::cout << "Final eigenvalue: " << complexEigenvalues(maxIndex) << std::endl;
        std::cout << "Final eigenvector: " << std::endl << eigenvector << std::endl;

        std::cout << "Full matrix: " << std::endl << subH << std::endl;

        std::cout << "Full eigenvalues: " << std::endl << complexEigenvalues << std::endl;
    }

protected:
    virtual VectorType EvalFunction(const VectorType& at) = 0;
    virtual VectorType EvalLinearised(const VectorType& at) = 0;

    stratifloat T = 4.5; // time interval for integration

    VectorType linearAboutStart;
    VectorType linearAboutEnd;

public:
    int K = 256; // max iterations
    std::vector<VectorType> q;
    MatrixX H; // upper Hessenberg matrix
};

class BasicArnoldi : public Arnoldi<StateVector>
{
    virtual StateVector EvalFunction(const StateVector& at) override
    {
        StateVector result;
        at.FullEvolve(T, result, false);

        linearAboutStart = at;
        linearAboutEnd = result;

        return result;
    }

    virtual StateVector EvalLinearised(const StateVector& at) override
    {
        StateVector result;
        at.LinearEvolve(T, linearAboutStart, linearAboutEnd, result);
        return result;
    }
};

int main(int argc, char *argv[])
{
    StateVector::ResetForParams();

    ExtendedStateVector input;
    input.LoadFromFile(argv[1]);

    StateVector background;
    background = input.x;

    Ri = input.p;

    BasicArnoldi solver;
    solver.Run(background);
}
