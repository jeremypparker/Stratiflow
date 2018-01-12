#pragma once

#include "Constants.h"
#include "ThreadPool.h"
#include "Eigen.h"

#include <cassert>

#include <fftw3.h>

#include <vector>
#include <utility>
#include <functional>
#include <iostream>
#include <iterator>
#include <fstream>
#include <random>


using namespace Eigen;

ArrayX VerticalPoints(stratifloat L, int N);
ArrayX FourierPoints(stratifloat L, int N);

template<typename A, typename T, int N1, int N2, int N3>
class StackContainer
{
public:
    virtual A stack(int n1, int n2) const = 0;
    virtual BoundaryCondition BC() const = 0;
};

template<typename A, typename T, int N1, int N2, int N3>
class ScalarProduct : public StackContainer<CwiseBinaryOp<internal::scalar_product_op<T, T>, const CwiseNullaryOp<internal::scalar_constant_op<T>,const Array<T, -1, 1>>, const A>, T, N1, N2, N3>
{
public:
    ScalarProduct(T scalar, const StackContainer<A, T, N1, N2, N3>* other)
    : scalar(scalar)
    , rhs(other)
    {}

    virtual
        CwiseBinaryOp<internal::scalar_product_op<T, T>, const CwiseNullaryOp<internal::scalar_constant_op<T>,const Array<T, -1, 1>>, const A>
        stack(int n1, int n2) const override
    {
        return scalar*rhs->stack(n1, n2);
    }
    virtual BoundaryCondition BC() const override
    {
        return rhs->BC();
    }
private:
    const StackContainer<A, T, N1, N2, N3>* rhs;
    T scalar;
};

template<typename A, typename B, typename T, int N1, int N2, int N3>
class ComponentwiseSum : public StackContainer<CwiseBinaryOp<internal::scalar_sum_op<T, T>, const A, const B>, T, N1, N2, N3>
{
public:
    ComponentwiseSum(const StackContainer<A, T, N1, N2, N3>* lhs, const StackContainer<B, T, N1, N2, N3>* rhs)
    : lhs(lhs)
    , rhs(rhs)
    {}
    virtual
        CwiseBinaryOp<internal::scalar_sum_op<T, T>, const A, const B>
        stack(int n1, int n2) const override
    {
        return lhs->stack(n1, n2) + rhs->stack(n1, n2);
    }
    virtual BoundaryCondition BC() const override
    {
        if(lhs->BC() == BoundaryCondition::Decaying && rhs->BC() == BoundaryCondition::Decaying)
        {
            return BoundaryCondition::Decaying;
        }
        else
        {
            return BoundaryCondition::Bounded;
        }
    }
private:
    const StackContainer<A, T, N1, N2, N3>* lhs;
    const StackContainer<B, T, N1, N2, N3>* rhs;
};

template<typename A, typename B, typename T, int N1, int N2, int N3>
class ComponentwiseProduct : public StackContainer<CwiseBinaryOp<internal::scalar_product_op<T, T>, const A, const B>, T, N1, N2, N3>
{
public:
    ComponentwiseProduct(const StackContainer<A, T, N1, N2, N3>* lhs, const StackContainer<B, T, N1, N2, N3>* rhs)
    : lhs(lhs)
    , rhs(rhs)
    {}
    virtual
        CwiseBinaryOp<internal::scalar_product_op<T, T>, const A, const B>
        stack(int n1, int n2) const override
    {
        return lhs->stack(n1, n2) * rhs->stack(n1, n2);
    }
    virtual BoundaryCondition BC() const override
    {
        if(lhs->BC() == BoundaryCondition::Decaying || rhs->BC() == BoundaryCondition::Decaying)
        {
            return BoundaryCondition::Decaying;
        }
        else
        {
            return BoundaryCondition::Bounded;
        }
    }
private:
    const StackContainer<A, T, N1, N2, N3>* lhs;
    const StackContainer<B, T, N1, N2, N3>* rhs;
};

template<typename A, typename T1, typename T2, int N1, int N2, int N3>
class Dim1MatMul : public StackContainer<CwiseBinaryOp<internal::scalar_product_op<T1, T2>, const CwiseNullaryOp<internal::scalar_constant_op<T1>,const Array<T1, -1, 1>>, const A>, T2, N1, N2, N3>
{
public:
    Dim1MatMul(const DiagonalMatrix<T1, -1>& matrix, const StackContainer<A, T2, N1, N2, N3>& field)
    : matrix(matrix)
    , field(field)
    {}

    virtual
        CwiseBinaryOp<internal::scalar_product_op<T1, T2>, const CwiseNullaryOp<internal::scalar_constant_op<T1>,const Array<T1, -1, 1>>, const A>
        stack(int n1, int n2) const override
    {
        return matrix.diagonal()(n1)*field.stack(n1, n2);
    }
    virtual BoundaryCondition BC() const override
    {
        return field.BC();
    }
private:
    const StackContainer<A, T2, N1, N2, N3>& field;
    const DiagonalMatrix<T1, -1>& matrix;
};

template<typename A, typename T1, typename T2, int N1, int N2, int N3>
class Dim2MatMul : public StackContainer<CwiseBinaryOp<internal::scalar_product_op<T1, T2>, const CwiseNullaryOp<internal::scalar_constant_op<T1>,const Array<T1, -1, 1>>, const A>, T2, N1, N2, N3>
{
public:
    Dim2MatMul(const DiagonalMatrix<T1, -1>& matrix, const StackContainer<A, T2, N1, N2, N3>& field)
    : matrix(matrix)
    , field(field)
    {}

    virtual
        CwiseBinaryOp<internal::scalar_product_op<T1, T2>, const CwiseNullaryOp<internal::scalar_constant_op<T1>,const Array<T1, -1, 1>>, const A>
        stack(int n1, int n2) const override
    {
        return matrix.diagonal()(n2)*field.stack(n1, n2);
    }
    virtual BoundaryCondition BC() const override
    {
        return field.BC();
    }
private:
    const StackContainer<A, T2, N1, N2, N3>& field;
    const DiagonalMatrix<T1, -1>& matrix;
};

template<typename A, typename T1, typename T2, int N1, int N2, int N3>
class Dim3MatMul : public StackContainer<ArrayWrapper<const Product<Matrix<T1,-1,-1>, MatrixWrapper<A>>>, T2, N1, N2, N3>
{
public:
    Dim3MatMul(const Matrix<T1, -1, -1>& matrix, const StackContainer<A, T2, N1, N2, N3>& field, BoundaryCondition resultingBC)
    : matrix(matrix)
    , field(field)
    , resultingBC(resultingBC)
    {}

    Dim3MatMul(const Matrix<T1, -1, -1>& matrix, const StackContainer<A, T2, N1, N2, N3>& field)
    : matrix(matrix)
    , field(field)
    , resultingBC(field.BC())
    {}

    virtual
        ArrayWrapper<const Product<Matrix<T1,-1,-1>, MatrixWrapper<A>>>
        stack(int n1, int n2) const override
    {
        return (matrix*field.stack(n1, n2).matrix()).array();
    }
    virtual BoundaryCondition BC() const override
    {
        return resultingBC;
    }
private:
    const StackContainer<A, T2, N1, N2, N3>& field;
    const Matrix<T1, -1, -1>& matrix;
    BoundaryCondition resultingBC;
};

template<typename A, typename T1, typename T2, int N1, int N2, int N3>
class MatMul : public StackContainer<ArrayWrapper<const Product<Matrix<T1,-1,-1>, MatrixWrapper<A>>>, T2, N1, N2, N3>
{
public:
    MatMul(const std::vector<Matrix<T1, -1, -1>>& matrices, const StackContainer<A, T2, N1, N2, N3>& field)
    : matrices(matrices)
    , field(field)
    , resultingBC(field.BC())
    {}


    virtual
        ArrayWrapper<const Product<Matrix<T1,-1,-1>, MatrixWrapper<A>>>
        stack(int n1, int n2) const override
    {
        return (matrices[n1*N2+n2]*field.stack(n1, n2).matrix()).array();
    }
    virtual BoundaryCondition BC() const override
    {
        return resultingBC;
    }
private:
    const StackContainer<A, T2, N1, N2, N3>& field;
    const std::vector<Matrix<T1, -1, -1>>& matrices;
    BoundaryCondition resultingBC;
};

template<typename T, int N1, int N2, int N3>
class Field : public StackContainer<Map<const Array<T, -1, 1>, Aligned16>,T,N1,N2,N3>
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

    template<typename A>
    const Field<T, N1, N2, N3>& operator=(const StackContainer<A,T,N1,N2,N3>& other)
    {
        //assert(BC() == BoundaryCondition::Bounded || other.BC() == BoundaryCondition::Decaying);

        ParallelPerStack([&other,this](int j1, int j2){
            stack(j1, j2) = other.stack(j1,j2);
        });

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

    template<typename A>
    const Field<T, N1, N2, N3>& operator+=(const StackContainer<A, T, N1, N2, N3>& other)
    {
        assert(other.BC()==BC());

        ParallelPerStack([&other,this](int j1, int j2){
            stack(j1, j2) += other.stack(j1, j2);
        });

        return *this;
    }
    template<typename A>
    const Field<T, N1, N2, N3>& operator-=(const StackContainer<A, T, N1, N2, N3>& other)
    {
        assert(other.BC()==BC());

        ParallelPerStack([&other,this](int j1, int j2){
            stack(j1, j2) -= other.stack(j1, j2);
        });

        return *this;
    }

    using Slice = Map<Array<T, -1, -1>, Unaligned, Stride<N3*N1, N3>>;
    using Stack = Map<Array<T, -1, 1>, Aligned16>;
    using ConstSlice = Map<const Array<T, -1, -1>, Unaligned, Stride<N3*N1, N3>>;
    using ConstStack = Map<const Array<T, -1, 1>, Aligned16>;

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
        return Stack(&Raw()[(N1*n2 + n1)*N3], N3);
    }
    virtual ConstStack stack(int n1, int n2) const override
    {
        assert(n1>=0 && n1<N1);
        assert(n2>=0 && n2<N2);
        return ConstStack(&Raw()[(N1*n2 + n1)*N3], N3);
    }

    T& operator()(int n1, int n2, int n3)
    {
        return Raw()[(N1*n2 + n1)*N3 + n3];
    }
    T operator()(int n1, int n2, int n3) const
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

    virtual BoundaryCondition BC() const override
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


    template<typename Solver>
    void Solve(std::vector<Solver>& solvers, Field<T, N1, N2, N3>& result) const
    {
        assert(solvers.size() == N1*N2);
        ParallelPerStack(
            [&solvers,&result,this](int j1, int j2)
            {
                Dim3Solve(solvers[j1*N2+j2], j1, j2, result);
            }
        );
    }

    virtual void ParallelPerStack(std::function<void(int j1, int j2)> f) const
    {
        #pragma omp parallel for
        for (int j1=0; j1<N1; j1++)
        {
            for (int j2=0; j2<N2; j2++)
            {
                f(j1, j2);
            }
        }
    }

    void Save(std::ofstream& filestream)
    {
        filestream.write(reinterpret_cast<char*>(Raw()), sizeof(T)*N1*N2*N3);
    }

    void Load(std::ifstream& filestream)
    {
        filestream.read(reinterpret_cast<char*>(Raw()), sizeof(T)*N1*N2*N3);
    }

private:

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

template<typename T, int N1, int N2, int N3>
class Field1D : public StackContainer<Map<const Array<T, -1, 1>, Aligned16>, T, N1, N2, N3>
{
public:
    Field1D(BoundaryCondition bc)
    : _data(N3)
    , _bc(bc)
    {
        _data.setZero();
    }

    template<typename A>
    const Field1D<T, N1, N2, N3>& operator=(const StackContainer<A,T,N1,N2,N3>& other)
    {
        //assert(other.BC() == BC());

        Get() = other.stack(0, 0);

        return *this;
    }

    bool operator==(const Field1D<T, N1, N2, N3>& other) const
    {
        if (other.BC() != BC())
        {
            return false;
        }

        return Get().isApprox(other.Get(), 0.05);
    }

    const Field1D<T, N1, N2, N3>& operator*=(T mult)
    {
        Get() *= mult;

        return *this;
    }

    virtual Map<const Array<T, -1, 1>, Aligned16> stack(int n1, int n2) const override
    {
        return Map<const Array<T, -1, 1>, Aligned16>(Raw(), N3);
    }


    template<typename Solver>
    void Solve(Solver& solver, Field1D<T, N1, N2, N3>& result) const
    {
        assert(solver.rows() == N3);

        Matrix<T, N3, 1> in = Get();
        Matrix<T, N3, 1> out(N3);
        out = solver.solve(in);
        result.Get() = out;
    }

    virtual BoundaryCondition BC() const override
    {
        return _bc;
    }

    T* Raw()
    {
        return _data.data();
    }
    const T* Raw() const
    {
        return _data.data();
    }

    Array<T, -1, 1>& Get()
    {
        return _data;
    }

    const Array<T, -1, 1>& Get() const
    {
        return _data;
    }
private:
    BoundaryCondition _bc;
    Array<T, -1, 1> _data;
};

template<int N1, int N2, int N3>
class Nodal1D;

template<int N1, int N2, int N3>
class Modal1D : public Field1D<stratifloat, 1, 1, N3>
{
public:
    using Field1D<stratifloat, 1, 1, N3>::Field1D;
    using Field1D<stratifloat, 1, 1, N3>::operator=;

    void ToNodal(Nodal1D<N1,N2,N3>& other) const
    {
        int size;
        f3_r2r_kind kind;

        stratifloat* in = const_cast<stratifloat*>(this->Raw());
        stratifloat* out = other.Raw();

        if (this->BC() == BoundaryCondition::Bounded)
        {
            kind = FFTW_REDFT00;
            size = N3;
        }
        else
        {
            kind = FFTW_RODFT00;
            in += 1;
            out += 1;
            size = N3-2;
        }

        auto plan = f3_plan_r2r_1d(size, in, out, kind, FFTW_ESTIMATE);

        f3_execute(plan);
        f3_destroy_plan(plan);

        if (this->BC() == BoundaryCondition::Decaying)
        {
            other.Raw()[0] = 0.0f;
            other.Raw()[N3-1] = 0.0f;
        }
    }

    void Filter()
    {
        if (N3>2)
        {
            for (int j3=2*N3/3; j3<N3; j3++)
            {
                this->Raw()[j3] = 0;
            }
        }
    }

};

template<int N1, int N2, int N3>
class Nodal1D : public Field1D<stratifloat, N1, N2, N3>
{
public:
    using Field1D<stratifloat, N1, N2, N3>::Field1D;
    using Field1D<stratifloat, N1, N2, N3>::operator=;

    void ToModal(Modal1D<N1,N2,N3>& other) const
    {
        int size;
        f3_r2r_kind kind;

        stratifloat* in = const_cast<stratifloat*>(this->Raw());
        stratifloat* out = other.Raw();

        if (this->BC() == BoundaryCondition::Bounded)
        {
            kind = FFTW_REDFT00;
            size = N3;
        }
        else
        {
            kind = FFTW_RODFT00;
            in += 1;
            out += 1;
            size = N3-2;
        }

        auto plan = f3_plan_r2r_1d(size, in, out, kind, FFTW_ESTIMATE);

        f3_execute(plan);
        f3_destroy_plan(plan);

        other *= 1/static_cast<stratifloat>(2*(N3-1));

        if (this->BC() == BoundaryCondition::Decaying)
        {
            other.Raw()[0] = 0.0f;
            other.Raw()[N3-1] = 0.0f;
        }
    }

    void SetValue(std::function<stratifloat(stratifloat)> f, stratifloat L3)
    {
        ArrayX z = VerticalPoints(L3, N3);
        for (int j3=0; j3<N3; j3++)
        {
            this->Get()(j3) = f(z(j3));
        }
    }
};


template<int N1, int N2, int N3>
class ModalField;

template<int N1, int N2, int N3>
class NodalField : public Field<stratifloat, N1, N2, N3>
{
public:
    using Field<stratifloat, N1, N2, N3>::operator=;

    NodalField(BoundaryCondition bc)
    : Field<stratifloat, N1, N2, N3>(bc)
    {
        if (!plan)
        {
            plan = f3_plan_dft_r2c_3d(2*(N3-1),
                                      N2,
                                      N1,
                                      intermediateData.data(),
                                      reinterpret_cast<f3_complex*>(intermediateData.data()),
                                      FFTW_MEASURE);

            assert(plan);
            f3_execute(plan);
        }
    }

    void ToModal(ModalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // first transpose to format required for fft (and use symmetry to pad out)

        int n1 = 2*(N1/2+1);
        int n2 = N2;
        int n3 = 2*(N3-1);

        // if statement outside the loops, though harder to read and gives code duplication
        // is much more efficient (compiler doesn't seem clever enough to break it out in this case)
        if (this->BC() == BoundaryCondition::Decaying)
        {
            // loop in blocks for cache efficiency
            #pragma omp parallel for
            for3D(N1,n2,n3)
            {
                stratifloat& outVal = intermediateData[j3*n1*n2 + j2*n1 + j1];

                if (j3 == 0 || j3 == N3-1)
                {
                    outVal = 0;
                }
                else if (j3<N3)
                {
                    outVal = this->operator()(j1,j2,j3);
                }
                else
                {
                    // odd symmetry gives a sine transform in 3rd direction
                    outVal = -this->operator()(j1,j2,n3-j3);
                }
            } endfor3D
        }
        else
        {
            // loop in blocks for cache efficiency
            #pragma omp parallel for
            for3D(N1,n2,n3)
            {
                stratifloat& outVal = intermediateData[j3*n1*n2 + j2*n1 + j1];

                if (j3<N3)
                {
                    outVal = this->operator()(j1,j2,j3);
                }
                else
                {
                    // even symmetry gives a cosine transform in 3rd dim
                    outVal = this->operator()(j1,j2,n3-j3);
                }
            } endfor3D
        }


        // then actually perform fft
        f3_execute(plan);

        // we use an inplace transform, so the complex result is stored in a real array
        complex *fftResult = reinterpret_cast<complex*>(intermediateData.data());

        // then transpose back

        n1 = N1/2 + 1;
        n2 = N2;
        n3 = N3; // actually the output has 2*(N3-1) entries, but with symmetry

        if (this->BC() == BoundaryCondition::Bounded)
        {
            #pragma omp parallel for
            for3D(n1,n2,n3)
            {
                // this is a transpose in disguise, as the access order of operator()
                // is 3, 1, 2
                other(j1,j2,j3) = fftResult[j3*n2*n1 + j2*n1 + j1];
            } endfor3D
        }
        else
        {
            #pragma omp parallel for
            for3D(n1,n2,n3)
            {
                other(j1,j2,j3) = i*fftResult[j3*n2*n1 + j2*n1 + j1];
            } endfor3D
        }

        other *= 1/static_cast<stratifloat>(N1*N2*2*(N3-1));

        if (this->BC() == BoundaryCondition::Decaying)
        {
            other.slice(0).setZero();
            other.slice(N3-1).setZero();
        }

        other.Filter();
    }

    stratifloat Max() const
    {
        stratifloat max = 0;

        for (int j3=0; j3<N3; j3++)
        {
            stratifloat norm = this->slice(j3).matrix().template lpNorm<Infinity>();
            if (norm > max)
            {
                max = norm;
            }
        }

        return max;
    }

    void SetValue(std::function<stratifloat(stratifloat)> f, stratifloat L3)
    {
        ArrayX z = VerticalPoints(L3, N3);
        for (int j3=0; j3<N3; j3++)
        {
            this->slice(j3).setConstant(f(z(j3)));
        }
    }

    void SetValue(std::function<stratifloat(stratifloat,stratifloat,stratifloat)> f, stratifloat L1, stratifloat L2, stratifloat L3)
    {
        ArrayX x = FourierPoints(L1, N1);
        ArrayX y = FourierPoints(L2, N2);
        ArrayX z = VerticalPoints(L3, N3);

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

    NodalField& operator-=(stratifloat other)
    {
        this->ParallelPerStack([other,this](int j1, int j2){
            this->stack(j1, j2) -= other;
        });

        return *this;
    }

    using Field<stratifloat, N1, N2, N3>::operator-=;

private:
    static std::vector<stratifloat, aligned_allocator<stratifloat>> intermediateData;
    static f3_plan plan;
};

template<int N1, int N2, int N3>
std::vector<stratifloat, aligned_allocator<stratifloat>> NodalField<N1,N2,N3>::intermediateData(2*(N1/2+1)*N2*2*(N3-1), 0);
template<int N1, int N2, int N3>
f3_plan NodalField<N1,N2,N3>::plan(0);

template<int N1, int N2, int N3>
class ModalField : public Field<complex, N1/2+1, N2, N3>
{
    static constexpr int actualN1 = N1/2 + 1;
public:
    using Field<complex, actualN1, N2, N3>::operator=;

    ModalField(BoundaryCondition bc)
    : Field<complex, N1/2+1, N2, N3>(bc)
    {
        if (!plan)
        {
            plan = f3_plan_dft_c2r_3d(2*(N3-1),
                                      N2,
                                      N1,
                                      reinterpret_cast<f3_complex*>(intermediateData.data()),
                                      intermediateData.data(),
                                      FFTW_MEASURE);

            assert(plan);
            f3_execute(plan);
        }
    }


    void ToNodalHorizontal(NodalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // do IFT in 1st and 2nd dimensions

        // make a copy of the input data as it is modified by the transform
        std::vector<complex, aligned_allocator<complex>> inputData(actualN1*N2*N3);
        for (unsigned int j=0; j<actualN1*N2*N3; j++)
        {
            inputData[j] = this->Raw()[j];
        }

        int dims[] = {N2, N1};
        int idims[] = {N2, actualN1};
        auto plan = f3_plan_many_dft_c2r(2,
                                        dims,
                                        N3,
                                        reinterpret_cast<f3_complex*>(inputData.data()),
                                        idims,
                                        N3,
                                        1,
                                        other.Raw(),
                                        dims,
                                        N3,
                                        1,
                                        FFTW_ESTIMATE);
        f3_execute(plan);
        f3_destroy_plan(plan);
    }

    void ToNodal(NodalField<N1, N2, N3>& other) const
    {
        assert(other.BC() == this->BC());

        // we use an inplace transform, so the complex input is stored in a real array
        complex *fftInput = reinterpret_cast<complex*>(intermediateData.data());

        // transpose

        int n1 = N1/2+1;
        int n2 = N2;
        int n3 = 2*(N3-1);

        if (this->BC() == BoundaryCondition::Bounded)
        {
            #pragma omp parallel for
            for3D(n1,n2,n3)
            {
                complex val;
                if (j3<N3)
                {
                    val = this->operator()(j1,j2,j3);
                }
                else
                {
                    val = this->operator()(j1,j2,n3-j3);
                }

                fftInput[j3*n2*n1 + j2*n1 + j1] = val;
            } endfor3D
        }
        else
        {
            #pragma omp parallel for
            for3D(n1,n2,n3)
            {
                complex val;
                if (j3 == 0 || j3 == N3-1)
                {
                    val = 0;
                }
                else if (j3<N3)
                {
                    val = this->operator()(j1,j2,j3);
                }
                else
                {
                    val = -this->operator()(j1,j2,n3-j3);
                }

                fftInput[j3*n2*n1 + j2*n1 + j1] = -i*val;
            } endfor3D
        }

        // transform
        f3_execute(plan);

        // transpose back

        #pragma omp parallel for
        for3D(N1,N2,N3)
        {
            other(j1,j2,j3) = intermediateData[j3*2*n1*n2 + j2*2*n1 + j1];
        } endfor3D

        if (this->BC() == BoundaryCondition::Decaying)
        {
            // now enforce zeros
            other.slice(0).setZero();
            other.slice(N3-1).setZero();
        }

    }

    void Filter()
    {
        if (N3>2)
        {
            #pragma omp parallel for
            for (int j3=2*N3/3; j3<N3; j3++)
            {
                this->slice(j3).setZero();
            }
        }

        if (N1>2)
        {
            #pragma omp parallel for collapse(2)
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
            #pragma omp parallel for collapse(2)
            for (int j2=N2/3; j2<=2*N2/3; j2++)
            {
                for (int j1=0; j1<actualN1; j1++)
                {
                    this->stack(j1, j2).setZero();
                }
            }
        }
    }

    void RandomizeCoefficients(stratifloat cutoff)
    {
        assert(cutoff < 2.0/3.0);

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<stratifloat> rng(-1.0,1.0);

        for (int j1=0; j1<0.5*cutoff*N1; j1++)
        {
            for (int j3=0; j3<cutoff*N3; j3++)
            {
                for (int j2=0; j2<0.5*cutoff*N2; j2++)
                {
                    this->operator()(j1,j2,j3) = rng(generator);
                }
                for (int j2=N2-1; j2>(1-0.5*cutoff)*N2; j2--)
                {
                    this->operator()(j1,j2,j3) = rng(generator);
                }
            }
        }
    }

    virtual void ParallelPerStack(std::function<void(int j1, int j2)> f) const override
    {
        int maxN1 = std::min(N1/2 + 1, 2*(N1/2 + 1)/3+1);
        int halfMaxN2 = N2/3+1;

        if(N2>1)
        {
            #pragma omp parallel for collapse(2)
            for (int j1=0; j1<maxN1; j1++)
            {
                for (int j2=0; j2<halfMaxN2; j2++)
                {
                    f(j1, j2);
                    f(j1, N2-1-j2);
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int j1=0; j1<maxN1; j1++)
            {
                f(j1, 0);
            }
        }
    }

private:
    static std::vector<stratifloat, aligned_allocator<stratifloat>> intermediateData;
    static f3_plan plan;
};

template<int N1, int N2, int N3>
std::vector<stratifloat, aligned_allocator<stratifloat>> ModalField<N1,N2,N3>::intermediateData(2*(N1/2+1)*N2*2*(N3-1), 0);
template<int N1, int N2, int N3>
f3_plan ModalField<N1,N2,N3>::plan(0);

template<typename A, typename T, int N1, int N2, int N3>
ScalarProduct<A, T, N1, N2, N3> operator*(T scalar,
                                       const StackContainer<A, T, N1, N2, N3>& field)
{
    return ScalarProduct<A, T, N1, N2, N3>(scalar, &field);
}

template<typename A, int N1, int N2, int N3>
ScalarProduct<A, complex, N1, N2, N3> operator*(stratifloat scalar,
                                       const StackContainer<A, complex, N1, N2, N3>& field)
{
    return ScalarProduct<A, complex, N1, N2, N3>(scalar, &field);
}

template<typename A, typename T, int N1, int N2, int N3>
ScalarProduct<A, T, N1, N2, N3> operator*(notstratifloat scalar,
                                       const StackContainer<A, T, N1, N2, N3>& field)
{
    return ScalarProduct<A, T, N1, N2, N3>((stratifloat)scalar, &field);
}


template<typename A, typename B, typename T, int N1, int N2, int N3>
ComponentwiseSum<A, B, T, N1, N2, N3> operator+(const StackContainer<A, T, N1, N2, N3>& lhs, const StackContainer<B, T, N1, N2, N3>& rhs)
{
    return ComponentwiseSum<A, B, T, N1, N2, N3>(&lhs, &rhs);
}

template<typename A, typename B, typename T, int N1, int N2, int N3>
ComponentwiseProduct<A, B, T, N1, N2, N3> operator*(const StackContainer<A, T, N1, N2, N3>& lhs, const StackContainer<B, T, N1, N2, N3>& rhs)
{
    return ComponentwiseProduct<A, B, T, N1, N2, N3>(&lhs, &rhs);
}
