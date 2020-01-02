#include "FFT.h"

#include <cassert>
#include <vector>
#include <iostream>
#include <omp.h>

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

#else

#endif

void Setup()
{
    // We use printf here because of weird std bugs when using cout
    printf("Setting up Stratiflow\n");

    if (f3_init_threads() == 0)
    {
        fprintf(stderr, "Failed to initialise fftw threads.\n");
    }

    f3_plan_with_nthreads(omp_get_max_threads());

    printf("Using %d threads\n", omp_get_max_threads());
}

void Cleanup()
{
    f3_cleanup_threads();
}

int InitialiserClass::counter;