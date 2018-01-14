#include "Field.h"
#include "Graph.h"
#include "IMEXRK.h"

#include <string>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <map>


int main(int argc, char *argv[])
{
    IMEXRK::NField loaded(BoundaryCondition::Decaying);

    IMEXRK::LoadVariable(argv[1], loaded, 2);
    IMEXRK::MField loadedModal(BoundaryCondition::Decaying);

    loaded.ToModal(loadedModal);

    HeatPlot(loadedModal, IMEXRK::L1, IMEXRK::L3, 0, "output.png");

    return 0;
}