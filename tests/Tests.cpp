#define CATCH_CONFIG_RUNNER
#include <catch.h>
#include <omp.h>

#include "FFT.h"
#include "Constants.h"


int main(int argc, char* argv[])
{
    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    int result = Catch::Session().run(argc, argv);

    f3_cleanup_threads();

    return result;
}