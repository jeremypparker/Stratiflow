#pragma once

#include "Field.h"

#include <matplotlib-cpp.h>

template<int N1, int N2, int N3>
void QuiverPlot(const NodalField<N1, N2, N3>& u,
                const NodalField<N1, N2, N3>& v,
                double L1,
                double L3,
                int j2,
                std::string filename)
{
    // // plot some arrows
    // unsigned int skip1 = 5;
    // unsigned int skip3 = 1;

    // matplotlibcpp::figure();

    // auto x = ChebPoints(N3, L3);

    // for (unsigned int j1 = 0; j1 < N1; j1+=skip1)
    // {
    //     for (unsigned int j3 = 0; j3 < N3; j3+=skip3)
    //     {
    //         double v1 = 0.2*u.slice(j3)(j1, j2);
    //         double v3 = 0.2*v.slice(j3)(j1, j2);

    //         double x1 = j1*L1/static_cast<double>(N1);
    //         double x3 = x(j3);

    //         matplotlibcpp::plot({x1, x1+v1}, {x3, x3+v3}, "b-");
    //     }
    // }

    // matplotlibcpp::save(filename);
    // matplotlibcpp::close();
}

template<int N1, int N2, int N3>
inline void HeatPlot(const NodalField<N1, N2, N3> &u, double L1, double L3, int j2, std::string filename)
{
    matplotlibcpp::figure();

    int cols = N1;
    int rows = N3;

    std::vector<double> imdata(rows*cols);

    for (int col=0; col<cols; col++)
    {
        for (int row=0; row<rows; row++)
        {
            imdata[row*cols + col] = u.slice(row)(col, j2);
        }
    }

    matplotlibcpp::imshow(imdata, rows, cols);

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}

inline void Interpolate(const ArrayXd& y, double L3, BoundaryCondition bc, std::string filename)
{
    // ArrayXd xGrid = ArrayXd::LinSpaced(1000, -L3, L3);
    // Plot(xGrid, BarycentricEval(y, xGrid, L3), "r-");
    // matplotlibcpp::save(filename);
    // matplotlibcpp::close();
}