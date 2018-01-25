#include "FFT.h"

#include <cassert>
#include <vector>

#ifdef USE_CUDA

int fftw_init_threads() 
{ 
    return 0; 
}

void fftw_plan_with_nthreads(int nthreads) 
{
}

void fftw_cleanup_threads()
{
}

void Perform1DR2R(int size, const stratifloat* in, stratifloat* out, f3_r2r_kind kind)
{
    int fftSize;

    if (kind == FFTW_REDFT00)
    {
        fftSize = 2*(size-1);
    }
    else if (kind == FFTW_RODFT00)
    {
        fftSize = 2*(size+1);
    }
    else
    {
        assert(0);
    }

    std::vector<complex> fftData(fftSize);

    if (kind == FFTW_RODFT00)
    {
        for (int j=0; j<fftSize; j++)
        {
            complex &outVal = fftData[j];

            if (j == 0 || j == size+1)
            {
                outVal = 0;
            }
            else if (j<size+1)
            {
                outVal = in[j+1];
            }
            else
            {
                // odd symmetry gives a sine transform in 3rd direction
                outVal = -in[fftSize-j-1];
            }
        }
    }
    else
    {
        for (int j=0; j<fftSize; j++)
        {
            complex &outVal = fftData[j];

            if (j<size)
            {
                outVal = in[j];
            }
            else
            {
                // even symmetry gives a cosine transform in 3rd direction
                outVal = in[fftSize-j];
            }
        }
    }

    auto plan = f3_plan_dft_1d(fftSize, 
                               reinterpret_cast<f3_complex*>(fftData.data()), 
                               reinterpret_cast<f3_complex*>(fftData.data()), 
                               FFTW_FORWARD, 
                               FFTW_ESTIMATE);
    assert(plan);
    f3_execute(plan);
    f3_destroy_plan(plan);

    if (kind == FFTW_RODFT00)
    {
        for (int j=0; j<size; j++)
        {
            out[j] = -imag(fftData[j+1]);
        }
    }
    else
    {
        for (int j=0; j<size; j++)
        {
            out[j] = real(fftData[j]);
        }
    }
}

#else

void Perform1DR2R(int size, const stratifloat* in, stratifloat* out, f3_r2r_kind kind)
{
    // the const_cast is legit when using FFTW_ESTIMATE
    auto plan = f3_plan_r2r_1d(size, const_cast<stratifloat*>(in), out, kind, FFTW_ESTIMATE);
    assert(plan);
    f3_execute(plan);
    f3_destroy_plan(plan);
}

#endif