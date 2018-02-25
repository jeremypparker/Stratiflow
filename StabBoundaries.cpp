#include "OrrSommerfeld.h"
#include "Graph.h"
#include "Differentiation.h"
#include <Eigen/SVD>

int main()
{
    for (R=2.5f; R<3.5; R+=0.1)
    {
        Pr = R*R;
        Pe = Pr*Re;

        for (int j=1; j<=10; j++)
        {
            stratifloat k=j/10.0;
            MatrixXc A = OrrSommerfeldLHS(k);
            JacobiSVD<MatrixXc> svd(A);
            stratifloat cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

            std::cout << k << " " << R << " " << cond << std::endl;
        }
    }

    return 0;
}