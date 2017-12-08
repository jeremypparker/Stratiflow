#define CATCH_CONFIG_RUNNER
#include <catch.h>
#include <omp.h>
#include <fftw3.h>


int main(int argc, char* argv[])
{
    fftwf_init_threads();
    fftwf_plan_with_nthreads(omp_get_max_threads());

    int result = Catch::Session().run(argc, argv);

    fftwf_cleanup_threads();

    return result;
}