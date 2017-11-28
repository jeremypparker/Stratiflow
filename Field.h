#pragma once

#include "Constants.h"
#include "ThreadPool.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <cassert>

#include <fftw3.h>

#ifdef USE_CUDA
#include <cufft.h>
#include <cuda_runtime.h>
#endif

#include <vector>
#include <utility>
#include <functional>

using namespace Eigen;

ArrayXf VerticalPoints(float L, int N);
ArrayXf FourierPoints(float L, int N);

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
        ParallelPerStack([mult,this](int j1, int j2){
            stack(j1, j2) *= mult;
        });

        return *this;
    }

    const Field<T, N1, N2, N3>& operator+=(std::pair<const Field<T, N1, N2, N3>*, T> pair)
    {
        assert(pair.first->BC()==BC());

        ParallelPerStack([&pair,this](int j1, int j2){
            stack(j1, j2) += pair.second * pair.first->stack(j1, j2);
        });

        return *this;
    }
    const Field<T, N1, N2, N3>& operator+=(const Field<T, N1, N2, N3>& other)
    {
        assert(other.BC()==BC());

        ParallelPerStack([&other,this](int j1, int j2){
            stack(j1, j2) += other.stack(j1, j2);
        });

        return *this;
    }
    const Field<T, N1, N2, N3>& operator-=(const Field<T, N1, N2, N3>& other)
    {
        assert(other.BC()==BC());

        ParallelPerStack([&other,this](int j1, int j2){
            stack(j1, j2) -= other.stack(j1, j2);
        });

        return *this;
    }

    using Slice = Map<Array<T, -1, -1>, Unaligned, Stride<N3*N1, N3>>;
    using Stack = Map<Array<T, N3, 1>, Aligned16>;
    using ConstSlice = Map<const Array<T, -1, -1>, Unaligned, Stride<N3*N1, N3>>;
    using ConstStack = Map<const Array<T, N3, 1>, Aligned16>;

    Slice slice(int n3)
    {
        assert(n3>=0 && n3<N3);
        return Slice(&Raw()[n3], N1, N2);
    }
    ConstSlice slice(int n3) const
    {
        assert(n3>=0 && n3<N3);
        return ConstSlice(&Raw()[n3], N1, N2);
    }

    Stack stack(int n1, int n2)
    {
        assert(n1>=0 && n1<N1);
        assert(n2>=0 && n2<N2);
        return Stack(&Raw()[(N1*n2 + n1)*N3]);
    }
    ConstStack stack(int n1, int n2) const
    {
        assert(n1>=0 && n1<N1);
        assert(n2>=0 && n2<N2);
        return ConstStack(&Raw()[(N1*n2 + n1)*N3]);
    }

    T& operator()(int n1, int n2, int n3)
    {
        return Raw()[(N1*n2 + n1)*N3 + n3];
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
        ParallelPerStack(
            [&matrix,&result,this](int j1, int j2)
            {
                Dim3MatMul(matrix, j1, j2, result);
            }
        );
    }
    template<typename M>
    void MatMul(const std::array<M, N1*N2>& matrices, Field<T, N1, N2, N3>& result) const
    {
        ParallelPerStack(
            [&matrices,&result,this](int j1, int j2)
            {
                Dim3MatMul(matrices[j1*N2+j2], j1, j2, result);
            }
        );
    }

    void Dim1MatMul(const DiagonalMatrix<T, -1>& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N1);

        ParallelPerStack([&result,&matrix,this](int j1, int j2){
            result.stack(j1, j2) = matrix.diagonal()(j1) * stack(j1,j2);
        });
    }

    void Dim2MatMul(const DiagonalMatrix<T, -1>& matrix, Field<T, N1, N2, N3>& result) const
    {
        assert(matrix.rows() == N2);

        ParallelPerStack([&result,&matrix,this](int j1, int j2){
            result.stack(j1, j2) = matrix.diagonal()(j2) * stack(j1,j2);
        });
    }

    template<typename Solver>
    void Solve(std::array<Solver, N1*N2>& solvers, Field<T, N1, N2, N3>& result) const
    {
        ParallelPerStack(
            [&solvers,&result,this](int j1, int j2)
            {
                Dim3Solve(solvers[j1*N2+j2], j1, j2, result);
            }
        );
    }

    void ParallelPerStack(std::function<void(int j1, int j2)> f) const
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
                [&f,first,last]()
                {
                    for (int j1=first; j1<last; j1++)
                    {
                        for (int j2=0; j2<N2; j2++)
                        {
                            f(j1, j2);
                        }
                    }
                });
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

        Matrix<T, N3, 1> col = stack(j1, j2);
        Matrix<T, N3, 1> res = solver.solve(col);
        result.stack(j1, j2) = res;
    }


    // stored in column-major ordering of size (N1, N2, N3)
    std::vector<T, aligned_allocator<T>> _data;

    BoundaryCondition _bc;
};

template<int N1, int N2, int N3>
class ModalField;

template<int N1, int N2, int N3>
class NodalField : public Field<float, N1, N2, N3>
{
public:
    NodalField(BoundaryCondition bc)
    : Field<float, N1, N2, N3>(bc)
    , intermediateData(N1*N2*N3, 0)
    {
        // do some ffts to find optimal method
        std::vector<float> data0(N1*N2*N3);
        std::vector<float> data1(N1*N2*N3);
        std::vector<complex> data2((N1/2+1)*N2*N3);

        {
            int dims;
            fftwf_r2r_kind kind;

            float* in = data0.data();
            float* out = data1.data();

            if (this->BC() == BoundaryCondition::Neumann)
            {
                kind = FFTW_REDFT00;
                dims = N3;
            }
            else
            {
                kind = FFTW_RODFT00;
                in += 1;
                out += 1;
                dims = N3-2;
            }
            auto plan = fftwf_plan_many_r2r(1,
                                           &dims,
                                           N1*N2,
                                           in,
                                           &dims,
                                           1,
                                           N3,
                                           out,
                                           &dims,
                                           1,
                                           N3,
                                           &kind,
                                           FFTW_PATIENT);

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }

        {
            int dims[] = {N2, N1};
            int odims[] = {N2, (N1/2+1)};
            auto plan = fftwf_plan_many_dft_r2c(2,
                                        dims,
                                        N3,
                                        data1.data(),
                                        dims,
                                        N3,
                                        1,
                                        reinterpret_cast<fftwf_complex*>(data2.data()),
                                        odims,
                                        N3,
                                        1,
                                        FFTW_PATIENT | FFTW_DESTROY_INPUT);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    }

    void ToModal(ModalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // first do (co)sine transform in 3rd dimension
        {
            int dims;
            fftwf_r2r_kind kind;

            float* in = const_cast<float*>(this->Raw());
            float* out = intermediateData.data();

            if (this->BC() == BoundaryCondition::Neumann)
            {
                kind = FFTW_REDFT00;
                dims = N3;
            }
            else
            {
                kind = FFTW_RODFT00;
                in += 1;
                out += 1;
                dims = N3-2;
            }
            auto plan = fftwf_plan_many_r2r(1,
                                           &dims,
                                           N1*N2,
                                           in,
                                           &dims,
                                           1,
                                           N3,
                                           out,
                                           &dims,
                                           1,
                                           N3,
                                           &kind,
                                           FFTW_ESTIMATE);

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }

        // then do FFT in 1st and 2nd dimensions
        {
#ifdef USE_CUDA
            static cufftComplex *data = nullptr;
            static cufftReal *realdata = nullptr;

            size_t datasize = sizeof(cufftComplex)*(N1/2+1)*N2*N3;
            size_t smalldatasize = sizeof(cufftReal)*N1*N2*N3;
            static cufftHandle plan;


            if (!data)
            {
                cudaMalloc((void**)&data, datasize);
                cudaMalloc((void**)&realdata, smalldatasize);

                int dims[] = {N2, N1};
                int odims[] = {N2, (N1/2+1)};
                cufftPlanMany(&plan, // todo: do beforehand
                            2,
                            dims,
                            dims,
                            N3,
                            1,
                            odims,
                            N3,
                            1,
                            CUFFT_R2C,
                            N3);
            }

            cudaMemcpy(realdata, intermediateData.data(), smalldatasize, cudaMemcpyHostToDevice);
            cufftExecR2C(plan, realdata, data);
            cudaMemcpy(other.Raw(), data, datasize, cudaMemcpyDeviceToHost);
#else
            int dims[] = {N2, N1};
            int odims[] = {N2, (N1/2+1)};
            auto plan = fftwf_plan_many_dft_r2c(2,
                                        dims,
                                        N3,
                                        intermediateData.data(),
                                        dims,
                                        N3,
                                        1,
                                        reinterpret_cast<fftwf_complex*>(other.Raw()),
                                        odims,
                                        N3,
                                        1,
                                        FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
#endif
        }

        other *= 1/static_cast<float>(N1*N2*2*(N3-1));

        if (this->BC() == BoundaryCondition::Dirichlet)
        {
            other.slice(0).setZero();
            other.slice(N3-1).setZero();
        }

        other.Filter();
    }

    float Max() const
    {
        float max = 0;

        for (int j3=0; j3<N3; j3++)
        {
            float norm = this->slice(j3).matrix().template lpNorm<Infinity>();
            if (norm > max)
            {
                max = norm;
            }
        }

        return max;
    }

    void SetValue(std::function<float(float)> f, float L3)
    {
        ArrayXf z = VerticalPoints(L3, N3);
        for (int j3=0; j3<N3; j3++)
        {
            this->slice(j3).setConstant(f(z(j3)));
        }
    }

    void SetValue(std::function<float(float,float,float)> f, float L1, float L2, float L3)
    {
        ArrayXf x = FourierPoints(L1, N1);
        ArrayXf y = FourierPoints(L2, N2);
        ArrayXf z = VerticalPoints(L3, N3);

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

private:
    mutable std::vector<float, aligned_allocator<float>> intermediateData;
};

template<int N1, int N2, int N3>
class ModalField : public Field<complex, N1/2+1, N2, N3>
{
    static constexpr int actualN1 = N1/2 + 1;
public:
    ModalField(BoundaryCondition bc)
    : Field<complex, N1/2+1, N2, N3>(bc)
    {
        // do fft to measure time
        std::vector<float> data0(N1*N2*N3);
        {
            int dims;
            fftwf_r2r_kind kind;
            float* in = data0.data();
            float* out = data0.data();

            if (this->BC() == BoundaryCondition::Neumann)
            {
                kind = FFTW_REDFT00;
                dims = N3;
            }
            else
            {
                kind = FFTW_RODFT00;
                in += 1;
                out += 1;
                dims = N3-2;
            }
            auto plan = fftwf_plan_many_r2r(1,
                                           &dims,
                                           N1*N2,
                                           in,
                                           &dims,
                                           1,
                                           N3,
                                           out,
                                           &dims,
                                           1,
                                           N3,
                                           &kind,
                                           FFTW_PATIENT);

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }
    }


    void ToNodalHorizontal(NodalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // do IFT in 1st and 2nd dimensions

#ifdef USE_CUDA
        static cufftComplex *data = nullptr;
        static cufftReal *realdata = nullptr;

        size_t datasize = sizeof(cufftComplex)*actualN1*N2*N3;
        size_t smalldatasize = sizeof(cufftReal)*N1*N2*N3;
        static cufftHandle plan;


        if (!data)
        {
            cudaMalloc((void**)&data, datasize);
            cudaMalloc((void**)&realdata, smalldatasize);

            int dims[] = {N2, N1};
            int idims[] = {N2, actualN1};
            cufftPlanMany(&plan, // todo: do beforehand
                        2,
                        dims,
                        idims,
                        N3,
                        1,
                        dims,
                        N3,
                        1,
                        CUFFT_C2R,
                        N3);
        }

        cudaMemcpy(data, this->Raw(), datasize, cudaMemcpyHostToDevice);
        cufftExecC2R(plan, data, realdata);
        cudaMemcpy(other.Raw(), realdata, smalldatasize, cudaMemcpyDeviceToHost);
#else

        // make a copy of the input data as it is modified by the transform
        if(inputData.size() == 0)
        {
            inputData.resize(actualN1*N2*N3);
        }
        for (unsigned int j=0; j<actualN1*N2*N3; j++)
        {
            inputData[j] = this->Raw()[j];
        }

        int dims[] = {N2, N1};
        int idims[] = {N2, actualN1};
        auto plan = fftwf_plan_many_dft_c2r(2,
                                        dims,
                                        N3,
                                        reinterpret_cast<fftwf_complex*>(inputData.data()),
                                        idims,
                                        N3,
                                        1,
                                        other.Raw(),
                                        dims,
                                        N3,
                                        1,
                                        FFTW_PATIENT);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
#endif
    }

    void ToNodal(NodalField<N1, N2, N3>& other) const
    {
        ToNodalHorizontal(other);

        // then do (co)sine transform in 3rd dimension
        {
            int dims;
            fftwf_r2r_kind kind;
            float* in = other.Raw();
            float* out = other.Raw();

            if (this->BC() == BoundaryCondition::Neumann)
            {
                kind = FFTW_REDFT00;
                dims = N3;
            }
            else
            {
                kind = FFTW_RODFT00;
                in += 1;
                out += 1;
                dims = N3-2;
            }
            auto plan = fftwf_plan_many_r2r(1,
                                           &dims,
                                           N1*N2,
                                           in,
                                           &dims,
                                           1,
                                           N3,
                                           out,
                                           &dims,
                                           1,
                                           N3,
                                           &kind,
                                           FFTW_ESTIMATE);

            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        }

        if (this->BC() == BoundaryCondition::Dirichlet)
        {
            other.slice(0).setZero();
            other.slice(N3-1).setZero();
        }

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
            for (int j1=N1/3; j1<actualN1; j1++)
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
                for (int j1=0; j1<actualN1; j1++)
                {
                    this->stack(j1, j2).setZero();
                }
            }
        }
    }

private:
    mutable std::vector<complex, aligned_allocator<complex>> inputData;
};

template<int N1, int N2, int N3>
std::pair<const ModalField<N1, N2, N3>*, complex> operator*(complex scalar,
                                                                const ModalField<N1, N2, N3>& field)
{
    return std::pair<const ModalField<N1, N2, N3>*, complex>(&field, scalar);
}

template<int N1, int N2, int N3>
void NodalProduct(const NodalField<N1, N2, N3>& f1,
             const NodalField<N1, N2, N3>& f2,
             NodalField<N1, N2, N3>& result)
{
    assert(result.BC() == BoundaryCondition::Neumann
           || f1.BC() == BoundaryCondition::Dirichlet
           || f2.BC() == BoundaryCondition::Dirichlet);

    result.ParallelPerStack(
        [&result,&f1,&f2](int j1, int j2)
        {
            result.stack(j1, j2) = f1.stack(j1, j2)*f2.stack(j1, j2);
        }
    );
}

template<int N1, int N2, int N3>
void NodalSum(const NodalField<N1, N2, N3>& f1,
             const NodalField<N1, N2, N3>& f2,
             NodalField<N1, N2, N3>& result)
{
    assert(result.BC() == BoundaryCondition::Neumann
           || (f1.BC() == BoundaryCondition::Dirichlet
           && f2.BC() == BoundaryCondition::Dirichlet));

    result.ParallelPerStack(
        [&result,&f1,&f2](int j1, int j2)
        {
            result.stack(j1, j2) = f1.stack(j1, j2)+f2.stack(j1, j2);
        }
    );
}