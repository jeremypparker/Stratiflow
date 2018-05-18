#pragma once
#include "StateVector.h"

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

    stratifloat Run(const VectorType& at, VectorType& result, bool getSecond=false)
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
            eigenvalues(maxIndex) = 0;

            stratifloat maxCoeff2 = eigenvalues.maxCoeff(&maxIndex);

            std::cout << "At step " << k << ", maximum growth rates: " << maxCoeff << " & " << maxCoeff2 << std::endl;
        }

        MatrixX subH = H.block(0,0,K-1,K-1);

        EigenSolver<MatrixX> ces(subH, true);
        ArrayXc complexEigenvalues = ces.eigenvalues();
        ArrayX eigenvalues = sqrt(complexEigenvalues.real()*complexEigenvalues.real()
                                + complexEigenvalues.imag()*complexEigenvalues.imag());
        int maxIndex;
        stratifloat maxCoeff = eigenvalues.maxCoeff(&maxIndex);
        eigenvalues(maxIndex) = 0;
        int maxIndex2;
        stratifloat maxCoeff2 = eigenvalues.maxCoeff(&maxIndex2);
        VectorXc eigenvector = ces.eigenvectors().col(maxIndex);
        VectorXc eigenvector2 = ces.eigenvectors().col(maxIndex2);

        VectorType result2;
        for (int k=0; k<K-1; k++)
        {
            result += eigenvector(k).real() * q[k];
            result2 += eigenvector2(k).real() * q[k];
        }

        result.PlotAll("eigReal");
        result2.PlotAll("eig2Real");

        std::cout << "Final eigenvalue: " << complexEigenvalues(maxIndex) << std::endl;
        std::cout << "Final eigenvector: " << std::endl << eigenvector << std::endl;

        std::cout << "Final eigenvalue: " << complexEigenvalues(maxIndex2) << std::endl;
        std::cout << "Final eigenvector: " << std::endl << eigenvector2 << std::endl;

        std::cout << "Full matrix: " << std::endl << subH << std::endl;

        std::cout << "Full eigenvalues: " << std::endl << complexEigenvalues << std::endl;

        if (getSecond)
        {
            result = result2;
            return maxCoeff2;
        }
        else
        {
            return maxCoeff;
        }
    }

protected:
    virtual VectorType EvalFunction(const VectorType& at) = 0;
    virtual VectorType EvalLinearised(const VectorType& at) = 0;

    stratifloat T = 11; // time interval for integration

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
