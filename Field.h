#pragma once

#include "Constants.h"
#include "ThreadPool.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <fftw3.h>
#include <vector>
#include <utility>
#include <functional>

using namespace Eigen;

ArrayXd VerticalPoints(double L, int N);
ArrayXd FourierPoints(double L, int N);

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
    const Field<T, N1, N2, N3>& operator-=(const Field<T, N1, N2, N3>& other)
    {
        assert(other.BC()==BC());
        for (int j=0; j<N3; j++)
        {
            slice(j) -= other.slice(j);
        }

        return *this;
    }

    using Slice = Map<Array<T, -1, -1>, Unaligned>;
    using Stack = Map<Array<T, N3, 1>, Unaligned, Stride<1, N2*N1>>;
    using ConstSlice = Map<const Array<T, -1, -1>, Unaligned>;
    using ConstStack = Map<const Array<T, N3, 1>, Unaligned, Stride<1, N1*N2>>;

    Slice slice(int n3)
    {
        assert(n3>=0 && n3<N3);
        return Slice(&Raw()[N1*N2*n3], N1, N2);
    }
    ConstSlice slice(int n3) const
    {
        assert(n3>=0 && n3<N3);
        return ConstSlice(&Raw()[N1*N2*n3], N1, N2);
    }

    Stack stack(int n1, int n2)
    {
        assert(n1>=0 && n1<N1);
        assert(n2>=0 && n2<N2);
        return Stack(&Raw()[N1*n2 + n1]);
    }
    ConstStack stack(int n1, int n2) const
    {
        assert(n1>=0 && n1<N1);
        assert(n2>=0 && n2<N2);
        return ConstStack(&Raw()[N1*n2 + n1]);
    }

    T& operator()(int n1, int n2, int n3)
    {
        return Raw()[N1*N2*n3 + N1*n2 + n1];
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

    template<typename M>
    void Dim3MatMul(const M& matrix, Field<T, N1, N2, N3>& result) const
    {
        int each = N1/maxthreads + 1;
        for (int first=0; first<N1; first+=each)
        {
            int last = first+each;
            if(last>N1)
            {
                last = N1;
            }

            ThreadPool::Get().ExecuteAsync(
                [&matrix,&result,first,last,this]()
                {
                    for (int j1=first; j1<last; j1++)
                    {
                        for (int j2=0; j2<N2; j2++)
                        {
                            Dim3MatMul(matrix, j1, j2, result);
                        }
                    }
                });
        }

        ThreadPool::Get().WaitAll();
    }
    template<typename M>
    void Dim3MatMul(const std::array<M, N1*N2>& matrices, Field<T, N1, N2, N3>& result) const
    {
        assert(matrices.size() == N1*N2);

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                ThreadPool::Get().ExecuteAsync(
                    [&matrices,&result,j1,j2,this]()
                    {
                        Dim3MatMul(matrices[j1*N2+j2], j1, j2, result);
                    });
            }
        }

        ThreadPool::Get().WaitAll();
    }

    void Dim1MatMul(const DiagonalMatrix<T, -1>& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N1);

        int each = N3/maxthreads + 1;
        for (int first=0; first<N3; first+=each)
        {
            int last = first+each;
            if(last>N3)
            {
                last = N3;
            }

            ThreadPool::Get().ExecuteAsync(
                [&matrix,&result,first,last,this]()
                {
                    for (int j3=first; j3<last; j3++)
                    {
                        result.slice(j3) = matrix * slice(j3).matrix();
                    }
                });
        }

        ThreadPool::Get().WaitAll();
    }

    void Dim2MatMul(const DiagonalMatrix<T, -1>& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N2);

        int each = N3/maxthreads + 1;
        for (int first=0; first<N3; first+=each)
        {
            int last = first+each;
            if(last>N3)
            {
                last = N3;
            }

            ThreadPool::Get().ExecuteAsync(
                [&matrix,&result,first,last,this]()
                {
                    for (int j3=first; j3<last; j3++)
                    {
                        result.slice(j3) = slice(j3).matrix() * matrix;
                    }
                });
        }

        ThreadPool::Get().WaitAll();
    }

    // template<typename Solver>
    // void Dim3Solve(Solver& solver, Field<T, N1, N2, N3>& result) const
    // {
    //     for (int j1=0; j1<N1; j1++)
    //     {
    //         for (int j2=0; j2<N2; j2++)
    //         {
    //             Dim3Solve(solver, j1, j2, result);
    //         }
    //     }
    // }

    template<typename Solver>
    void Dim3Solve(std::array<Solver, N1*N2>& solvers, Field<T, N1, N2, N3>& result) const
    {
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                ThreadPool::Get().ExecuteAsync(
                    [&solvers,&result,j1,j2,this]()
                    {
                        Dim3Solve(solvers[j1*N2+j2], j1, j2, result);
                    });
            }
        }

        ThreadPool::Get().WaitAll();
    }

private:
    template<typename M>
    void Dim3MatMul(const M& matrix, int j1, int j2, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == matrix.cols());
        assert(matrix.rows() == N3);

        result.stack(j1, j2) = matrix * stack(j1, j2).matrix();
    }

    template<typename Solver>
    void Dim3Solve(Solver& solver, int j1, int j2, Field<T, N1, N2, N3>& result) const
    {
        assert(solver.rows() == N3);

        result.stack(j1, j2) = solver.solve(stack(j1, j2).matrix());
    }


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

        std::vector<complex> intermediateData(N1*N2*N3, 0); // todo: don't allocate

        // first do (co)sine transform in 3rd dimension
        {
            int dims;
            fftw_r2r_kind kind;

            double* in = const_cast<double*>(this->Raw());
            double* out = reinterpret_cast<double*>(intermediateData.data());

            if (this->BC() == BoundaryCondition::Neumann)
            {
                kind = FFTW_REDFT00;
                dims = N3;
            }
            else
            {
                kind = FFTW_RODFT00;
                in += N1*N2;
                out += 2*N1*N2;
                dims = N3-2;
            }
            auto plan = fftw_plan_many_r2r(1,
                                           &dims,
                                           N1*N2,
                                           in,
                                           nullptr,
                                           N1*N2,
                                           1,
                                           out,
                                           nullptr,
                                           N1*N2*2,
                                           2,
                                           &kind,
                                           FFTW_ESTIMATE);

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }

        // then do FFT in 1st and 2nd dimensions
        int dims[] = {N2, N1};
        auto plan = fftw_plan_many_dft(2,
                                       dims,
                                       N3,
                                       reinterpret_cast<fftw_complex*>(intermediateData.data()),
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

        other *= 1/static_cast<double>(N1*N2*2*(N3-1));

        if (this->BC() == BoundaryCondition::Dirichlet)
        {
            other.slice(0).setZero();
            other.slice(N3-1).setZero();
        }
    }

    double Max() const
    {
        double max = 0;

        for (int j3=0; j3<N3; j3++)
        {
            double norm = this->slice(j3).matrix().template lpNorm<Infinity>();
            if (norm > max)
            {
                max = norm;
            }
        }

        return max;
    }

    void SetValue(std::function<double(double)> f, double L3)
    {
        ArrayXd z = VerticalPoints(L3, N3);
        for (int j3=0; j3<N3; j3++)
        {
            this->slice(j3).setConstant(f(z(j3)));
        }
    }

    void SetValue(std::function<double(double,double,double)> f, double L1, double L2, double L3)
    {
        ArrayXd x = FourierPoints(L1, N1);
        ArrayXd y = FourierPoints(L2, N2);
        ArrayXd z = VerticalPoints(L3, N3);

        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                for (int j3=0; j3<N3; j3++)
                {
                    (*this)(j1,j2,j3) = f(x(j1), y(j2), z(j3));
                }
            }
        }
    }
};

template<int N1, int N2, int N3>
class ModalField : public Field<complex, N1, N2, N3>
{
    using Field<complex, N1, N2, N3>::Field;
public:
    void ToNodalHorizontal(NodalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // make a copy of the input data as it is modified by the transform
        std::vector<complex> inputData(N1*N2*N3); // todo: don't allocate
        for (unsigned int j=0; j<N1*N2*N3; j++)
        {
            inputData[j] = this->Raw()[j];
        }
        std::vector<complex> outputData(N1*N2*N3); // todo: don't allocate

        // do IFT in 1st and 2nd dimensions
        {
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
        }

        // todo: remove
        // copy back complex output into real buffer
        for (unsigned int j=0; j<N1*N2*N3; j++)
        {
            other.Raw()[j] = outputData[j].real();
        }
    }

    void ToNodalNoFilter(NodalField<N1, N2, N3>& other) const
    {
        ToNodalHorizontal(other);

        // then do (co)sine transform in 3rd dimension
        {
            int dims;
            fftw_r2r_kind kind;
            double* in = other.Raw();
            double* out = other.Raw();

            if (this->BC() == BoundaryCondition::Neumann)
            {
                kind = FFTW_REDFT00;
                dims = N3;
            }
            else
            {
                kind = FFTW_RODFT00;
                in += N1*N2;
                out += N1*N2;
                dims = N3-2;
            }
            auto plan = fftw_plan_many_r2r(1,
                                           &dims,
                                           N1*N2,
                                           in,
                                           nullptr,
                                           N1*N2,
                                           1,
                                           out,
                                           nullptr,
                                           N1*N2,
                                           1,
                                           &kind,
                                           FFTW_ESTIMATE);

            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }

        if (this->BC() == BoundaryCondition::Dirichlet)
        {
            other.slice(0).setZero();
            other.slice(N3-1).setZero();
        }

    }

    void ToNodal(NodalField<N1, N2, N3>& other)
    {
        Filter();

        ToNodalNoFilter(other);
    }

    void Filter()
    {
        if (N3>2)
        {
            for (int j3=2*N3/3; j3<N3; j3++)
            {
                this->slice(j3).setZero();
            }
        }

        if (N1>2)
        {
            for (int j1=N1/3; j1<=2*N1/3; j1++)
            {
                for (int j2=0; j2<N2; j2++)
                {
                    this->stack(j1, j2).setZero();
                }
            }
        }

        if (N2>2)
        {
            for (int j2=N2/3; j2<=2*N2/3; j2++)
            {
                for (int j1=0; j1<N1; j1++)
                {
                    this->stack(j1, j2).setZero();
                }
            }
        }
    }

};

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
    assert(result.BC() == BoundaryCondition::Neumann
           || f1.BC() == BoundaryCondition::Dirichlet
           || f2.BC() == BoundaryCondition::Dirichlet);

    int each = N3/maxthreads + 1;
    for (int first=0; first<N3; first+=each)
    {
        int last = first+each;
        if(last>N3)
        {
            last = N3;
        }

        ThreadPool::Get().ExecuteAsync(
            [&f1,&f2,&result,first,last]()
            {
                for (int j3=first; j3<last; j3++)
                {
                    result.slice(j3) = f1.slice(j3)*f2.slice(j3);
                }
            });
    }

    ThreadPool::Get().WaitAll();
}

template<int N1, int N2, int N3>
void NodalSum(const NodalField<N1, N2, N3>& f1,
             const NodalField<N1, N2, N3>& f2,
             NodalField<N1, N2, N3>& result)
{
    assert(result.BC() == BoundaryCondition::Neumann
           || (f1.BC() == BoundaryCondition::Dirichlet
           && f2.BC() == BoundaryCondition::Dirichlet));

    int each = N3/maxthreads + 1;
    for (int first=0; first<N3; first+=each)
    {
        int last = first+each;
        if(last>N3)
        {
            last = N3;
        }

        ThreadPool::Get().ExecuteAsync(
            [&f1,&f2,&result,first,last]()
            {
                for (int j3=first; j3<last; j3++)
                {
                    result.slice(j3) = f1.slice(j3) + f2.slice(j3);
                }
            });
    }

    ThreadPool::Get().WaitAll();
}