#pragma once

#include "Constants.h"

#ifdef USE_CUDA
#include <cufftw.h>

// cuda's fft wrapper does not define certain functions, so we have to work around this

int fftw_init_threads();
void fftw_plan_with_nthreads(int nthreads);
void fftw_cleanup_threads();

enum fftw_r2r_kind
{
FFTW_RODFT00,
FFTW_REDFT00
};

#define fftwf_r2r_kind fftw_r2r_kind
#define fftwf_init_threads fftw_init_threads
#define fftwf_plan_with_nthreads fftw_plan_with_nthreads
#define fftwf_cleanup_threads fftw_cleanup_threads

#else
#include <fftw3.h>
#endif

void Setup();
void Cleanup();

// hackish solution to ensure mpi is initialised when we need it
class InitialiserClass
{
public:
    InitialiserClass()
    {
        if (counter++ == 0)
        {
            Setup();
        }
    }

    ~InitialiserClass()
    {
        counter--;
        if (counter==0)
        {
            Cleanup();
        }
    }
private:
    static int counter;
};
static InitialiserClass initialiser;
