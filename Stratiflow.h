#pragma once
#include "Field.h"
#include "Parameters.h"

constexpr int M1 = N1/2 + 1;

using NField = NodalField<N1,N2,N3>;
using MField = ModalField<N1,N2,N3>;
using M1D = Modal1D<N1,N2,N3>;
using N1D = Nodal1D<N1,N2,N3>;