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

        VectorType phaseShift;
        phaseShift.u1 = ddx(at.u1);
        phaseShift.u2 = ddx(at.u2);
        phaseShift.u3 = ddx(at.u3);
        phaseShift.b = ddx(at.b);

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


            // transform phase shift into arnoldi space
            VectorX phaseShiftTransformed = VectorX::Zero(k);
            for (int j=0; j<k; j++)
            {
                phaseShiftTransformed[j] = phaseShift.Dot(q[j]);
            }

            // exclude things that look like a phase shift
            for (int j=0; j<k; j++)
            {
                complex proj = ces.eigenvectors().col(j).dot(phaseShiftTransformed);
                stratifloat prod = phaseShiftTransformed.norm()*ces.eigenvectors().col(j).norm();
                if (proj.real() > 0.7*prod || proj.real() < -0.7*prod)
                {
                    eigenvalues(j) = 0;
                }
            }

            int maxIndex;
            stratifloat maxCoeff = eigenvalues.maxCoeff(&maxIndex);
            eigenvalues(maxIndex) = 0;

            stratifloat maxCoeff2 = eigenvalues.maxCoeff(&maxIndex);
            eigenvalues(maxIndex) = 0;

            stratifloat maxCoeff3 = eigenvalues.maxCoeff(&maxIndex);

            std::cout << "At step " << k << ", maximum growth rates: "
                      << maxCoeff << " & "
                      << maxCoeff2 << " & "
                      << maxCoeff3 << std::endl;
        }

        MatrixX subH = H.block(0,0,K-1,K-1);

        EigenSolver<MatrixX> ces(subH, true);
        ArrayXc complexEigenvalues = ces.eigenvalues();
        ArrayX eigenvalues = sqrt(complexEigenvalues.real()*complexEigenvalues.real()
                                + complexEigenvalues.imag()*complexEigenvalues.imag());


        // transform phase shift into arnoldi space
        VectorX phaseShiftTransformed = VectorX::Zero(K-1);
        for (int j=0; j<K-1; j++)
        {
            phaseShiftTransformed[j] = phaseShift.Dot(q[j]);
        }

        // exclude things that look like a phase shift
        for (int j=0; j<K-1; j++)
        {
                complex proj = ces.eigenvectors().col(j).dot(phaseShiftTransformed);
                stratifloat prod = phaseShiftTransformed.norm()*ces.eigenvectors().col(j).norm();
                if (proj.real() > 0.7*prod || proj.real() < -0.7*prod)
                {
                    eigenvalues(j) = 0;
                }
        }



        int maxIndex;
        stratifloat maxCoeff = eigenvalues.maxCoeff(&maxIndex);

        eigenvalues(maxIndex) = 0;
        int maxIndex2;
        stratifloat maxCoeff2 = eigenvalues.maxCoeff(&maxIndex2);

        eigenvalues(maxIndex2) = 0;
        int maxIndex3;
        stratifloat maxCoeff3 = eigenvalues.maxCoeff(&maxIndex3);

        VectorXc eigenvector = ces.eigenvectors().col(maxIndex);
        VectorXc eigenvector2 = ces.eigenvectors().col(maxIndex2);
        VectorXc eigenvector3 = ces.eigenvectors().col(maxIndex3);

        VectorType result2;
        VectorType result3;

        VectorType imag1;
        VectorType imag2;
        VectorType imag3;
        for (int k=0; k<K-1; k++)
        {
            result += eigenvector(k).real() * q[k];
            imag1 += eigenvector(k).imag() * q[k];
            result2 += eigenvector2(k).real() * q[k];
            imag2 += eigenvector2(k).imag() * q[k];
            result3 += eigenvector3(k).real() * q[k];
            imag3 += eigenvector3(k).imag() * q[k];
        }

        result.PlotAll("eigReal");
        result2.PlotAll("eig2Real");
        result3.PlotAll("eig3Real");

        result.SaveToFile("eigReal");
        imag1.SaveToFile("eigImag");
        result2.SaveToFile("eig2Real");
        imag2.SaveToFile("eig2Imag");
        result3.SaveToFile("eig3Real");
        imag3.SaveToFile("eig3Imag");

        std::cout << "Final eigenvalue: " << complexEigenvalues(maxIndex) << std::endl;
        std::cout << "Final eigenvector: " << std::endl << eigenvector << std::endl;

        std::cout << "Final eigenvalue: " << complexEigenvalues(maxIndex2) << std::endl;
        std::cout << "Final eigenvector: " << std::endl << eigenvector2 << std::endl;

        std::cout << "Final eigenvalue: " << complexEigenvalues(maxIndex3) << std::endl;
        std::cout << "Final eigenvector: " << std::endl << eigenvector3 << std::endl;

        //std::cout << "Full matrix: " << std::endl << subH << std::endl;

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

public:
    int K = 512; // max iterations
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

        return result;
    }

    virtual StateVector EvalLinearised(const StateVector& at) override
    {
        StateVector result;
        at.LinearEvolve(T, linearAboutStart, result);
        return result;
    }
};
