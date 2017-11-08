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
    // plot some arrows
    unsigned int skip1 = 5;
    unsigned int skip3 = 1;

    matplotlibcpp::figure();

    auto x = ChebPoints(N3, L3);

    for (unsigned int j1 = 0; j1 < N1; j1+=skip1)
    {
        for (unsigned int j3 = 0; j3 < N3; j3+=skip3)
        {
            double v1 = 0.2*u.slice(j3)(j1, j2);
            double v3 = 0.2*v.slice(j3)(j1, j2);

            double x1 = j1*L1/static_cast<double>(N1);
            double x3 = x(j3);

            matplotlibcpp::plot({x1, x1+v1}, {x3, x3+v3}, "b-");
        }
    }

    matplotlibcpp::save(filename);
    matplotlibcpp::close();
}


// namespace
// {
// ArrayXd BarycentricWeights(const ArrayXd& x)
// {
//     int N = x.rows()-1;
//     ArrayXd w(N+1);

//     w.setOnes();

//     for (int j=0; j<=N; j++)
//     {
//         for (int k=0; k<=N; k++)
//         {
//             if (j!=k)
//             {
//                 w(j) *= (x(j)-x(k))/5;
//             }
//         }
//     }
//     w = 1/w;
//     return w;
// }

// std::vector<double> ToVector(const ArrayXd& arr)
// {
//     return std::vector<double>(arr.data(), arr.data() + arr.rows());
// }

// ArrayXd BarycentricEval(const ArrayXd& x, const ArrayXd& y, const ArrayXd& at)
// {
//     assert(x.rows() == y.rows());

//     ArrayXd w = BarycentricWeights(x);

//     ArrayXd result(at.rows());
//     ArrayXd denom(at.rows());

//     for (int j=0; j<x.rows(); j++)
//     {
//         result += y(j)*w(j)/(at-x(j));
//         denom += w(j)/(at-x(j));
//     }

//     result /= denom;

//     for (int j=0; j<x.rows(); j++)
//     {
//         for (int k=0; k<at.rows(); k++)
//         {
//             if (std::abs(at(k)-x(j)) < 0.1)
//             {
//                 result(k) = y(j);
//             }
//         }
//     }

//     return result;
// }
// }

// inline void Interpolate(const ArrayXd& y, double L3, BoundaryCondition bc, std::string filename)
// {
//     int N = y.rows();
//     assert(N%2 == 1);

//     ArrayXd x = ChebPoints(N, L3);

//     // use two different polynomials, on extended domain
//     ArrayXd xLeft(N);
//     ArrayXd xRight(N);
//     ArrayXd yLeft(N);
//     ArrayXd yRight(N);

//     xLeft << x.segment(N/2, N/2)-2*L3, x.head(N/2+1);
//     xRight << x.tail(N/2+1), x.segment(1, N/2)+2*L3;

//     if (bc == BoundaryCondition::Neumann)
//     {
//         yLeft << y.segment(1, N/2).reverse(), y.head(N/2+1);
//         yRight << y.tail(N/2+1), y.segment(N/2, N/2).reverse();
//     }
//     else
//     {
//         yLeft << -y.segment(1, N/2).reverse(), y.head(N/2+1);
//         yRight << y.tail(N/2+1), -y.segment(N/2, N/2).reverse();
//     }

//     ArrayXd xGridLeft = ArrayXd::LinSpaced(59, -L3*2, 0);
//     ArrayXd xGridRight = ArrayXd::LinSpaced(59, 0, L3*2);

//     //ArrayXd plotLeft = BarycentricEval(xLeft, yLeft, xGridLeft);
//     ArrayXd plotRight = BarycentricEval(xRight, yRight, xGridRight);

//     //matplotlibcpp::plot(ToVector(xGridLeft), ToVector(plotLeft), "r-");
//     matplotlibcpp::plot(ToVector(xGridRight), ToVector(plotRight), "b-");

//     matplotlibcpp::xlim(-L3, L3);
//     matplotlibcpp::ylim(-2, 2);

//     matplotlibcpp::save(filename);
//     matplotlibcpp::close();
// }

namespace
{
    // ArrayXd BarycentricWeights(const ArrayXd& x)
    // {
    //     int N = x.rows()-1;
    //     ArrayXd w(N+1);

    //     w.setOnes();

    //     for (int j=0; j<=N; j++)
    //     {
    //         for (int k=0; k<=N; k++)
    //         {
    //             if (j!=k)
    //             {
    //                 w(j) *= x(j)-x(k);
    //             }
    //         }
    //     }
    //     w = 1/w;
    //     return w;
    // }
    // ArrayXd GaussLobattoNodes(int N)
    // {
    //     ArrayXd x(N);

    //     x = -cos(ArrayXd::LinSpaced(N+1, 0, pi));

    //     return x;
    // }

    // ArrayXd FullSymmetrisedNodes(int N, double L)
    // {
    //     ArrayXd x = GaussLobattoNodes(N);

    //     ArrayXd out(2*N+1);
    //     out << L*(x-1), L*(x.segment(1, N)+1);

    //     return out;
    // }

    // ArrayXd ApplyBC(const ArrayXd& f, BoundaryCondition bc)
    // {
    //     int N = f.rows()-1;

    //     ArrayXd out(2*N+1);

    //     if(bc == BoundaryCondition::Neumann)
    //     {
    //         out << f.segment(1, N/2).reverse(), f, f.segment(N/2, N/2).reverse();
    //     }
    //     else
    //     {
    //         out << -f.segment(1, N/2).reverse(), f, -f.segment(N/2, N/2).reverse();
    //     }

    //     return out;
    // }


    // std::vector<double> ToVector(const ArrayXd& arr)
    // {
    //     return std::vector<double>(arr.data(), arr.data() + arr.rows());
    // }

    // void Plot(const ArrayXd& x, const ArrayXd& y, std::string format="b-o")
    // {
    //     matplotlibcpp::plot(ToVector(x), ToVector(y), format);
    // }

    // ArrayXd BarycentricEval(ArrayXd& f, ArrayXd& at, double L)
    // {
    //     int N = f.rows()-1;
    //     ArrayXd x = FullSymmetrisedNodes(N, L);
    //     ArrayXd w1 = BarycentricWeights(x.head(N+1));
    //     ArrayXd w2 = BarycentricWeights(x.tail(N+1));
    //     ArrayXd ffull = ApplyBC(f, BoundaryCondition::Neumann);

    //     ArrayXd result(at.rows());
    //     result.setZero();

    //     ArrayXd denom(at.rows());
    //     denom.setZero();

    //     for (int j=0; j<at.rows(); j++)
    //     {
    //         ArrayXd w(N+1);
    //         int start;
    //         int end;
    //         if (at(j)<0)
    //         {
    //             start = 0;
    //             end = N;
    //             w = w1;
    //         }
    //         else
    //         {
    //             start =N;
    //             end = 2*N;
    //             w = w2;
    //         }
    //         for (int i=start; i<=end; i++)
    //         {
    //             result(j) += ffull(i)*w(i-start)/(at(j)-x(i));
    //             denom(j) += w(i-start)/(at(j)-x(i));
    //         }
    //     }

    //     result /= denom;

    //     return result;
    // }

}

inline void Interpolate(ArrayXd y, double L3, BoundaryCondition bc, std::string filename)
{
    // ArrayXd xGrid = ArrayXd::LinSpaced(1000, -L3, L3);
    // Plot(xGrid, BarycentricEval(y, xGrid, L3), "r-");
    // matplotlibcpp::save(filename);
    // matplotlibcpp::close();
}
