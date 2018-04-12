#include "StateVector.h"

int main(int argc, char* argv[])
{
    Ri = std::stof(argv[2]);

    StateVector field;
    field.LoadFromFile(argv[1]);

    StateVector endfield;
    field.FullEvolve(5, endfield);

    endfield -= field;

    std::cout << field.Norm() << std::endl;
    std::cout << endfield.Norm() << std::endl;
}