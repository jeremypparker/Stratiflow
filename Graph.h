#pragma once

#include "Field.h"

#include <matplotlib-cpp.h>

ArrayXf Evaluate(const ArrayXf& a, const ArrayXf& x, float L, BoundaryCondition bc)
{
    ArrayXf y = ArrayXf::Zero(x.size());

    ArrayXf theta = atan(L/x);

    for (int j=0; j<theta.size(); j++)
    {
        if (theta(j)<0)
        {
            theta(j) += pi;
        }
    }


    for (int k=0; k<a.size(); k++)
    {
        float c = 2;
        if (k==0 || k==a.size()-1)
        {
            if (bc==BoundaryCondition::Neumann)
            {
                c = 1;
            }
            else
            {
                c = 0;
            }
        }

        if (bc==BoundaryCondition::Neumann)
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
inline void HeatPlot(const ModalField<N1, N2, N3> &u, float L1, float L3, int j2, std::string filename)
{
    NodalField<N1, N2, N3> U(u.BC());
    u.ToNodalHorizontal(U);

    matplotlibcpp::figure();


    int scale = 2;

    int cols = N1*scale;
    int rows = N1*scale;

    ArrayXf x = ArrayXf::LinSpaced(rows, -0.5*L1*scale, 0.5*L1*scale);

    std::vector<float> imdata(rows*cols);

    for (int col=0; col<N1; col++)
    {
        ArrayXf y = Evaluate(U.stack(col, j2), x, L3, U.BC());

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
