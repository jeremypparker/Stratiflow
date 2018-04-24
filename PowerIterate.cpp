#include "StateVector.h"

int main(int argc, char* argv[])
{
    const stratifloat T = 10;

    if (argc == 2)
    {
        Ri = std::stof(argv[1]);
    }
    else
    {
        Ri = std::stof(argv[2]);
    }

    StateVector stationaryPoint;

    if (argc == 3)
    {
        stationaryPoint.LoadFromFile(argv[1]);
    }

    StateVector stationaryPointEnd;

    stationaryPoint.FullEvolve(T, stationaryPointEnd, false, true);

    // separate real and imaginary parts
    StateVector b_kr;
    b_kr.Randomise(0.0001, true);
    StateVector b_ki;
    b_ki.Randomise(0.0001, true);

    StateVector b_kr1;
    StateVector b_ki1;

    int iterations = 0;
    while (true)
    {
        b_kr.PlotAll("real"+std::to_string(iterations));
        b_ki.PlotAll("imag"+std::to_string(iterations));

        // normalise
        stratifloat norm = sqrt(b_kr.Norm2() + b_ki.Norm2());
        b_kr *= 1/norm;
        b_ki *= 1/norm;

        // power iterate
        b_kr.LinearEvolve(T, stationaryPoint, stationaryPointEnd, b_kr1);
        b_ki.LinearEvolve(T, stationaryPoint, stationaryPointEnd, b_ki1);

        // largest eigenvalue (in magnitude) of exponential matrix
        complex mu = (b_kr.Dot(b_kr1) + b_ki.Dot(b_ki1));
                     + i*(b_kr.Dot(b_ki1) - b_ki.Dot(b_kr1));

        // abs and arg of exp(A) eigenvalue give real/imag of eigenvalue of A
        stratifloat growthGuess = log(abs(mu))/T;
        stratifloat phaseSpeed = arg(mu) / T; // only works if T sufficiently small

        iterations++;
        std::cout << "ITERATION " << iterations << ", growth rate: " << growthGuess << std::endl;
        std::cout << "phase speed: " << phaseSpeed << std::endl;

        // for next step
        b_kr = b_kr1;
        b_ki = b_ki1;
    }
}