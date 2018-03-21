#pragma once

#include "Field.h"

#include <matplotlib-cpp.h>

ArrayX Evaluate(const ArrayX& a, const ArrayX& x, stratifloat L, BoundaryCondition bc)
{
    ArrayX y = ArrayX::Zero(x.size());

    ArrayX theta = atan(L/x);

    for (int j=0; j<theta.size(); j++)
    {
        if (theta(j)<0)
        {
            theta(j) += pi;
        }
    }


    for (int k=0; k<a.size(); k++)
    {
        stratifloat c = 2;
        if (k==0 || k==a.size()-1)
        {
            if (bc==BoundaryCondition::Bounded)
            {
                c = 1;
            }
            else
            {
                c = 0;
            }
        }

        if (bc==BoundaryCondition::Bounded)
        {
            y += c*a(k)*cos(k*theta);
        }
        else
        {
            y += c*a(k)*sin(k*theta);
        }
    }

    return y;
}

template<int N1, int N2, int N3>
inline void HeatPlot1D(const Nodal1D<N1, N2, N3> &U, std::string filename)
{
    matplotlibcpp::figure();

    int cols = N3;
    int rows = N3;

    std::vector<stratifloat> imdata(rows*cols);

    for (int col=0; col<cols; col++)
    {
        for (int row=0; row<rows; row++)
        {
            imdata[row*cols + col] = U.Get()(row);
        }
    }

    matplotlibcpp::imshow(imdata, rows, cols);

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}

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

    matplotlibcpp::imshow(imdata, N3, N1);

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}

template<int N1, int N2, int N3>
inline void HeatPlot(const ModalField<N1, N2, N3> &u, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    NodalField<N1, N2, N3> U(u.BC());

    u.ToNodal(U);

    HeatPlot(U, L1, L3, j2, filename);
}

#else

template<int N1, int N2, int N3>
inline void HeatPlot(const NodalField<N1, N2, N3> &U, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    matplotlibcpp::figure();


    int scale = 1;

    int cols = N1*scale;
    int rows = N1*scale;

    ArrayX x = ArrayX::LinSpaced(rows, 0.5*L1*scale, -0.5*L1*scale);

    std::vector<stratifloat> imdata(rows*cols);

    for (int col=0; col<N1; col++)
    {
        ArrayX y = Evaluate(U.stack(col, j2), x, L3, U.BC());

        for (int row=0; row<rows; row++)
        {
            for (int repeat = 0; repeat < scale; repeat++)
            {
                imdata[row*cols + repeat*N1 + col] = y(row);
            }
        }
    }

    matplotlibcpp::imshow(imdata, rows, cols);

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}

template<int N1, int N2, int N3>
inline void HeatPlot(const ModalField<N1, N2, N3> &u, stratifloat L1, stratifloat L3, int j2, std::string filename)
{
    NodalField<N1, N2, N3> U(u.BC());
    u.ToNodalHorizontal(U);

    HeatPlot(U, L1, L3, j2, filename);
}

#endif