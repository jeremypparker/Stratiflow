#include "ExtendedStateVector.h"
#include "Arnoldi.h"

int main(int argc, char* argv[])
{
    // load a state
    ExtendedStateVector state;
    state.LoadAndInterpolate<256,1,384>(argv[1]);
    //state.x.RemovePhaseShift();
    Ri = state.p;

    DumpParameters();

    // Load basis vectors
    StateVector basis1, basis2;
    basis1.LoadAndInterpolate<256,1,384>("/nfs/st01/hpc-fluids-rrk26/jpp39/DiabloMatch/Re1000/after_6030/stab/eigReal.fields");
    basis2.LoadAndInterpolate<256,1,384>("/nfs/st01/hpc-fluids-rrk26/jpp39/DiabloMatch/Re1000/after_6030/stab/eig2Real.fields");

    basis1.RemovePhaseShift();
    basis2.RemovePhaseShift();

    basis1 *= 1/basis1.Norm();
    basis2 *= 1/basis2.Norm();

    basis2 += -basis1.Dot(basis2) * basis1;

    basis2 *= 1/basis2.Norm();

    //state.x += 0.0002*basis1;

    //state *= 0.01;

    StateVector::solver.SetBackground(InitialU, InitialB);

    // follow the trajectory, projected on basis
    StateVector saveState;
    NeumannModal z, tanhsech2;
    z.SetValue([](stratifloat z){return z;}, L3);
    tanhsech2.SetValue([](stratifloat z){return tanh(z)/cosh(z)/cosh(z);}, L3);

    stratifloat oldenergy = -1;
    for (int n=0; n<500; n++)
    {
        saveState = state.x;

        saveState.PlotAll(std::to_string(n));

        saveState.AddBackground();

        stratifloat energy = 0;
        energy += 0.5*InnerProd(saveState.u1,saveState.u1,L3);
        energy += 0.5*InnerProd(saveState.u3,saveState.u3,L3);
        energy += -Ri*InnerProd(saveState.b,z,L3);

        if(oldenergy == -1)
        {
            oldenergy = energy;
        }

        stratifloat energychange = (energy-oldenergy)/20.0;
        oldenergy = energy;

        NeumannModal ux, wz;
        DirichletModal uz, wx, bz;

        ux = ddx(saveState.u1);
        uz = ddz(saveState.u1);
        wx = ddx(saveState.u3);
        wz = ddz(saveState.u3);
        bz = ddz(saveState.b);

        stratifloat viscousdiss = InnerProd(ux,ux,L3) + InnerProd(uz,uz,L3)
                                + InnerProd(wx,wx,L3) + InnerProd(wz,wz,L3);

        stratifloat velforcing = 2*InnerProd(saveState.u1, tanhsech2, L3);

        stratifloat buoyancydiff = Ri*IntegrateAllSpace(bz, 1, 1, L3);

        stratifloat buoyforcing = 2*Ri*InnerProd(z, tanhsech2, L3);

        std::cout << "Step " << n << " " << energychange*Re << " " << -viscousdiss
                  << " " << velforcing << " " << buoyancydiff << " " << -buoyforcing << std::endl;

        state.FullEvolve(20, state, false, false);
    }

    state.SaveToFile("trackingresult");
}
