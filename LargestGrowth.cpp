#include "OrrSommerfeld.h"

int main()
{

    stratifloat kmax;
    stratifloat growthmax = -10000;

    stratifloat k_lower = 0.00001;
    stratifloat k_upper = 2.0;

    for (int n=0; n<5; n++)
    {
        stratifloat deltak = (k_upper-k_lower)/10;

        for (stratifloat k=k_lower; k<=k_upper; k+=deltak)
        {
            auto largest = LargestGrowth(k);

            if (largest>growthmax)
            {
                growthmax = largest;
                kmax = k;
            }
        }

        k_lower = kmax - deltak;
        k_upper = kmax + deltak;
    }

    std::cout << "Maximum growth rate " << growthmax << " at " << kmax << std::endl;
    std::cout << "Wavelength of fastest growing mode is " << 2*pi/kmax << std::endl;

    return 0;
}