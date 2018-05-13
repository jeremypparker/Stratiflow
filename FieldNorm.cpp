#include "ExtendedStateVector.h"

int main(int argc, char* argv[])
{
    if (argc == 4)
    {
        LoadParameters(argv[3]);
    }
    PrintParameters();
    StateVector::ResetForParams();

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

    ExtendedStateVector endfield;
    field.FullEvolve(5, endfield);

    endfield -= field;

    std::cout << "STATIONARY POINT "
              << field.p << " "
              << field.x.Norm() << " "
              << endfield.x.Norm() << std::endl;
}
