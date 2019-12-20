#pragma once

#include "Field.h"

#include <matplotlib-cpp.h>

#ifdef DEBUG_PLOT

template<int N1, int N2, int N3>
inline void HeatPlot(const NodalField<N1, N2, N3> &U, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    matplotlibcpp::figure();

    std::vector<stratifloat> imdata(N1*N3);

    for (int n1=0; n1<N1; n1++)
    {
        for (int n3=0; n3<N3; n3++)
        {
            imdata[n3*N1 + n1] = U(n1,j2,n3);
        }
    }

    matplotlibcpp::imshow(imdata, N3, N1, 0, N1, 0, N3);

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}

#else

template<int K1, int K2, int K3>
inline void HeatPlot(const NodalField<K1, K2, K3> &U, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    const stratifloat zcutoff = 6;

    const int N1 = L1*50;
    const int N3 = 2*zcutoff*50;

    matplotlibcpp::figure(L1, zcutoff);

    ArrayX oldNeumannPoints = VerticalPointsFractional(L3, K3);
    ArrayX oldDirichletPoints = VerticalPoints(L3, K3);

    std::vector<stratifloat> imdata(N1*2*N3);

    for (int j1=0; j1<N1*2; j1++)
    {
        stratifloat x = j1*L1/N1;

        int k1_left = static_cast<int>(x/(L1/K1));
        int k1_right = k1_left+1;

        stratifloat x_left = k1_left*L1/K1;
        stratifloat x_right = k1_right*L1/K1;

        stratifloat weight_left = (x_right-x)/(x_right-x_left);
        stratifloat weight_right = (x-x_left)/(x_right-x_left);

        while (k1_left<0) k1_left += K1;
        while (k1_right<0) k1_right += K1;
        while (k1_left>=K1) k1_left -= K1;
        while (k1_right>=K1) k1_right -= K1;

        for (int j3=0; j3<N3; j3++)
        {

            stratifloat z = j3*2*zcutoff/N3 - zcutoff;


            int k3_below;
            int k3_above = 0;
            stratifloat z_below;
            stratifloat z_above;

            do
            {
                k3_above++;
                k3_below = k3_above-1;

                if (U.BC() == BoundaryCondition::Neumann)
                {
                    z_below = oldNeumannPoints(k3_below);
                    z_above = oldNeumannPoints(k3_above);
                }
                else
                {
                    z_below = oldDirichletPoints(k3_below);
                    z_above = oldDirichletPoints(k3_above);
                }
            } while(z_above<z);

            stratifloat weight_above = (z-z_below)/(z_above-z_below);
            stratifloat weight_below = (z_above-z)/(z_above-z_below);

            imdata[(N3-1-j3)*N1*2 + j1] = weight_left*weight_below*U(k1_left,j2,k3_below)
                                      + weight_left*weight_above*U(k1_left,j2,k3_above)
                                      + weight_right*weight_below*U(k1_right,j2,k3_below)
                                      + weight_right*weight_above*U(k1_right,j2,k3_above);
        }
    }

    matplotlibcpp::imshow(imdata, N3, 2*N1, 0, 2*L1, -zcutoff, zcutoff);

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}

#endif

template<int N1, int N2, int N3>
inline void HeatPlot(const ModalField<N1, N2, N3> &u, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    NodalField<N1, N2, N3> U(u.BC());

    u.ToNodal(U);

    HeatPlot<N1,N2,N3>(U, L1, L3, j2, filename);
}
