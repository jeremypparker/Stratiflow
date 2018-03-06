#include "OrrSommerfeld.h"
#include "Graph.h"
#include "Differentiation.h"
#include <Eigen/SVD>

int main()
{
    for (R=0.5; R<3; R+=0.02)
    {
        Pr = R*R;
        Pe = Pr*Re;

        for (int j=1; j<=10; j++)
        {
            stratifloat k=j/50.0;
            stratifloat growth = LargestGrowth(k);

            std::cout << k << " " << R << " " << growth << " " << std::endl;
        }
    }

    return 0;
}
