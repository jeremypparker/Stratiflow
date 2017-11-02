#pragma once

#include "Constants.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <fftw3.h>
#include <vector>
#include <utility>

using namespace Eigen;

template<typename T, int N1, int N2, int N3>
class Field
{
public:
    Field(BoundaryCondition bc)
    : _data(N1*N2*N3, 0)
    , _bc(bc)
    {
    }

    Field(const Field<T, N1, N2, N3>& other)
    : _data(other._data)
    , _bc(other._bc)
    {
    }

    const Field<T, N1, N2, N3>& operator=(const Field<T, N1, N2, N3>& other)
    {
        assert(other.BC() == BC());
        _data = other._data;
        return *this;
    }

    bool operator==(const Field<T, N1, N2, N3>& other) const
    {
        if (other.BC() != BC())
        {
            return false;
        }

        // because of the way isApprox works, need to fail on both slice and stack to count
        bool failedSlice = false;
        for (int j=0; j<N3; j++)
        {
            if (!(slice(j)+ 0.001).isApprox(other.slice(j) + 0.001, 0.05))
            {
                failedSlice = true;
            }
        }
        if(failedSlice)
        {
            for (int j1=0; j1<N1; j1++)
            {
                for (int j2=0; j2<N2; j2++)
                {
                    if (!(stack(j1, j2)+0.001).isApprox(other.stack(j1, j2) + 0.001, 0.05))
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    bool operator!=(const Field<T, N1, N2, N3>& other) const
    {
        return !operator==(other);
    }

    const Field<T, N1, N2, N3>& operator*=(T mult)
    {
        for (int j=0; j<N3; j++)
        {
            slice(j) *= mult;
        }

        return *this;
    }

    const Field<T, N1, N2, N3>& operator+=(std::pair<const Field<T, N1, N2, N3>*, T> pair)
    {
        assert(pair.first->BC()==BC());
        for (int j=0; j<N3; j++)
        {
            slice(j) += pair.second * pair.first->slice(j);
        }

        return *this;
    }
    const Field<T, N1, N2, N3>& operator+=(const Field<T, N1, N2, N3>& other)
    {
        assert(other.BC()==BC());
        for (int j=0; j<N3; j++)
        {
            slice(j) += other.slice(j);
        }

        return *this;
    }

    using Slice = Map<Array<T, -1, -1>, Unaligned>;
    using Stack = Map<Array<T, N3, 1>, Unaligned, Stride<1, N2*N1>>;
    using ConstSlice = Map<const Array<T, -1, -1>, Unaligned>;
    using ConstStack = Map<const Array<T, N3, 1>, Unaligned, Stride<1, N1*N2>>;

    Slice slice(int n3)
    {
        return Slice(&Raw()[N1*N2*n3], N1, N2);
    }
    ConstSlice slice(int n3) const
    {
        return ConstSlice(&Raw()[N1*N2*n3], N1, N2);
    }

    Stack stack(int n1, int n2)
    {
        return Stack(&Raw()[N1*n2 + n1]);
    }
    ConstStack stack(int n1, int n2) const
    {
        return ConstStack(&Raw()[N1*n2 + n1]);
    }

    T* Raw()
    {
        return _data.data();
    }
    const T* Raw() const
    {
        return _data.data();
    }

    BoundaryCondition BC() const
    {
        return _bc;
    }

    void Zero()
    {
        for(T& datum : _data)
        {
            datum = 0;
        }
    }

    void Dim3MatMul(const MatrixXcd& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N3 && matrix.cols() == N3);
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                Dim3MatMul(matrix, j1, j2, result);
            }
        }
    }

    void Dim3MatMul(const MatrixXcd& matrix, int j1, int j2, Field<T, N1, N2, N3>& result) const
    {
        result.stack(j1, j2) = matrix * stack(j1, j2).matrix();
    }

    void Dim1MatMul(const DiagonalMatrix<T, -1>& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N1);
        for (int j=0; j<N3; j++)
        {
            result.slice(j) = matrix*slice(j).matrix();
        }
    }

    void Dim2MatMul(const DiagonalMatrix<T, -1>& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N2);
        for (int j=0; j<N3; j++)
        {
            result.slice(j) = slice(j).matrix() * matrix;
        }
    }

    template<typename Solver>
    void Dim3Solve(Solver& solver, Field<T, N1, N2, N3>& result) const
    {
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                Dim3Solve(solver, j1, j2, result);
            }
        }
    }

    template<typename Solver>
    void Dim3Solve(Solver& solver, int j1, int j2, Field<T, N1, N2, N3>& result) const
    {
        result.stack(j1, j2) = solver.solve(stack(j1, j2).matrix());
    }

private:
    // stored in column-major ordering of size (N1, N2, N3)
    std::vector<T> _data;

    BoundaryCondition _bc;
};

template<int N1, int N2, int N3>
class ModalField;

template<int N1, int N2, int N3>
class NodalField : public Field<double, N1, N2, N3>
{
    using Field<double, N1, N2, N3>::Field;
public:
    void ToModal(ModalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // copy the input data into complex numbers
        std::vector<complex> inputData(N1*N2*N3);
        for (unsigned int j=0; j<N1*N2*N3; j++)
        {
            inputData[j] = this->Raw()[j];
        }

        // do FFT in 1st and 2nd dimensions
        int dims[] = {N2, N1};
        auto plan = fftw_plan_many_dft(2,
                                       dims,
                                       N3,
                                       reinterpret_cast<fftw_complex*>(inputData.data()),
                                       nullptr,
                                       1,
                                       N1*N2,
                                       reinterpret_cast<fftw_complex*>(other.Raw()),
                                       nullptr,
                                       1,
                                       N1*N2,
                                       FFTW_FORWARD,
                                       FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        other *= 1/static_cast<double>(N1*N2);
    }
};

template<int N1, int N2, int N3>
class ModalField : public Field<complex, N1, N2, N3>
{
    using Field<complex, N1, N2, N3>::Field;
public:
    void ToNodal(NodalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // make a copy of the input data as it is modified by the transform
        std::vector<complex> inputData(N1*N2*N3);
        for (unsigned int j=0; j<N1*N2*N3; j++)
        {
            inputData[j] = this->Raw()[j];
        }
        std::vector<complex> outputData(N1*N2*N3);

        // do IFT in 1st and 2nd dimensions
        int dims[] = {N2, N1};
        auto plan = fftw_plan_many_dft(2,
                                       dims,
                                       N3,
                                       reinterpret_cast<fftw_complex*>(inputData.data()),
                                       nullptr,
                                       1,
                                       N1*N2,
                                       reinterpret_cast<fftw_complex*>(outputData.data()),
                                       nullptr,
                                       1,
                                       N1*N2,
                                       FFTW_BACKWARD,
                                       FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // copy back complex output into real buffer
        for (unsigned int j=0; j<N1*N2*N3; j++)
        {
            other.Raw()[j] = outputData[j].real();
        }
    }
};

ArrayXd ChebPoints(unsigned int N, double L);

template<int N1, int N2, int N3>
std::pair<const Field<complex, N1, N2, N3>*, complex> operator*(complex scalar,
                                                                const ModalField<N1, N2, N3>& field)
{
    return std::pair<const Field<complex, N1, N2, N3>*, complex>(&field, scalar);
}

template<int N1, int N2, int N3>
void NodalProduct(const NodalField<N1, N2, N3>& f1,
             const NodalField<N1, N2, N3>& f2,
             NodalField<N1, N2, N3>& result)
{
    for (int j=0; j<N3; j++)
    {
        result.slice(j) = f1.slice(j)*f2.slice(j);
    }
}