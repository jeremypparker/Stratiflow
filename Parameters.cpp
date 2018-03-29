#include "Parameters.h"

// These are runtime parameters - could load from a file in future

stratifloat L1 = 18.0455;// size of domain streamwise
stratifloat L2 = 4.0f;  // size of domain spanwise
stratifloat L3 = 30.0f; // size of domain vertically
stratifloat Re = 1000;
stratifloat Ri = 0.16;
stratifloat R = 1;
stratifloat Pr = R*R;
stratifloat Pe = Re*Pr;
bool EnforceSymmetry = false;
