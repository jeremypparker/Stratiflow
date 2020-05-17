#include "ExtendedStateVector.h"
#include <iomanip>

int main(int argc, char* argv[])
{
    if (argc == 4)
    {
        LoadParameters(argv[3]);
        StateVector::ResetForParams();
    }
    PrintParameters();


    ExtendedStateVector field;
    if (argc >= 3)
    {
        field.p = std::stof(argv[2]);
        field.x.LoadFromFile(argv[1]);
    }
    else
    {
        field.LoadFromFile(argv[1]);
    }

    flowParams.Ri = field.p;

    std::cout << std::setprecision(20);
    std::cout << "STATIONARY POINT "
              << field.p << " "
              << field.x.Norm() << " "
              << field.x.Energy() << " "
              << field.x.OtherAxis() << std::endl;

    //field.x.AddBackground();
    //field.x.RemovePhaseShift();
    //field.PlotAll("plots");
}
