#include "ExtendedStateVector.h"

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

    Ri = field.p;


    std::cout << "STATIONARY POINT "
              << field.p << " "
              << field.x.Norm() << " "
              << field.x.Energy() << " "
              << field.x.Enstrophy() << std::endl;

    field.x.AddBackground();
    field.PlotAll("plots");
}
