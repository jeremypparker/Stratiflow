#pragma once

#include <complex>

#ifdef USE_DOUBLE
    #define stratifloat double
    #define notstratifloat float

    #define ArrayX    ArrayXd
    #define ArrayXc   ArrayXcd
    #define ArrayXX   ArrayXXd
    #define ArrayXXc  ArrayXXcd
    #define VectorX   VectorXd
    #define VectorXc  VectorXcd
    #define MatrixX   MatrixXd
    #define MatrixXc  MatrixXcd

    #define f3_init_threads         fftw_init_threads
    #define f3_plan_with_nthreads   fftw_plan_with_nthreads
    #define f3_cleanup_threads      fftw_cleanup_threads
    #define f3_r2r_kind             fftw_r2r_kind
    #define f3_execute              fftw_execute
    #define f3_destroy_plan         fftw_destroy_plan
    #define f3_plan_many_r2r        fftw_plan_many_r2r
    #define f3_plan_many_dft_r2c    fftw_plan_many_dft_r2c
    #define f3_plan_many_dft_c2r    fftw_plan_many_dft_c2r
    #define f3_complex              fftw_complex 
#else
    #define stratifloat float
    #define notstratifloat double

    #define ArrayX    ArrayXf
    #define ArrayXc   ArrayXcf
    #define ArrayXX   ArrayXXf
    #define ArrayXXc  ArrayXXcf
    #define VectorX   VectorXf
    #define VectorXc  VectorXcf
    #define MatrixX   MatrixXf
    #define MatrixXc  MatrixXcf

    #define f3_init_threads         fftwf_init_threads
    #define f3_plan_with_nthreads   fftwf_plan_with_nthreads
    #define f3_cleanup_threads      fftwf_cleanup_threads
    #define f3_r2r_kind             fftwf_r2r_kind
    #define f3_execute              fftwf_execute
    #define f3_destroy_plan         fftwf_destroy_plan
    #define f3_plan_many_r2r        fftwf_plan_many_r2r
    #define f3_plan_many_dft_r2c    fftwf_plan_many_dft_r2c
    #define f3_plan_many_dft_c2r    fftwf_plan_many_dft_c2r
    #define f3_complex              fftwf_complex 
#endif

constexpr stratifloat pi = 3.14159265358979;
constexpr std::complex<stratifloat> i(0, 1);

enum class BoundaryCondition
{
    Decaying,
    Bounded
};

using complex = std::complex<stratifloat>;
