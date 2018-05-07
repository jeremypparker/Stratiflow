#include "ExtendedStateVector.h"

int main(int argc, char* argv[])
{
    ExtendedStateVector field;
    if (argc == 3)
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

    std::cout << field.x.Norm() << std::endl;
    std::cout << endfield.x.Norm() << std::endl;
}
