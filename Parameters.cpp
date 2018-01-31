#include "Parameters.h"

// These are runtime parameters - could load from a file in future

stratifloat L1 = 16.0298016098; // size of domain streamwise
stratifloat L2 = 4.0f;  // size of domain spanwise
stratifloat L3 = 3.0f; // vertical scaling factor
stratifloat Re = 1000;
stratifloat Ri = 0.16;
stratifloat R = 2;
stratifloat Pr = R*R;
stratifloat Pe = Re*Pr;
EnergyType EnergyConstraint = EnergyType::Correct;
