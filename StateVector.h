#pragma once

#include "IMEXRK.h"

// This class contains a full state's information
// its operations are not particularly efficient
// so it should only be used for high level algorithms
class StateVector
{
public:
    StateVector()
    {
        Zero();
    }

    StateVector(const StateVector& other)
    {
        u1 = other.u1;
        u2 = other.u2;
        u3 = other.u3;
        b = other.b;
    }

    Modal u1;
    Modal u2;
    Modal u3;
    Modal b;
    Modal p;

    stratifloat FullEvolve(stratifloat T, StateVector& result, bool snapshot = false, bool screenshot = true, bool calcmixing = false) const;

    void FixedEvolve(stratifloat deltaT, int steps, std::vector<StateVector>& result) const;

    void LinearEvolve(stratifloat T, const StateVector& about, StateVector& result) const;

    void AdjointEvolve(stratifloat deltaT, int steps, const std::vector<StateVector>& intermediate, StateVector& result) const;


    const StateVector& operator+=(const StateVector& other)
    {
        u1 += other.u1;
        if (gridParams.ThreeDimensional)
        {
            u2 += other.u2;
        }
        u3 += other.u3;
        b += other.b;

        return *this;
    }

    const StateVector& operator-=(const StateVector& other)
    {
        u1 -= other.u1;
        if (gridParams.ThreeDimensional)
        {
            u2 -= other.u2;
        }
        u3 -= other.u3;
        b  -= other.b;
        return *this;
    }

    const StateVector& MulAdd(stratifloat a, const StateVector& B)
    {
        u1 += a*B.u1;
        if (gridParams.ThreeDimensional)
        {
            u2 += a*B.u2;
        }
        u3 += a*B.u3;
        b  += a*B.b;
        return *this;
    }

    const StateVector& operator*=(stratifloat other)
    {
        u1 *= other;
        if (gridParams.ThreeDimensional)
        {
            u2 *= other;
        }
        u3 *= other;
        b  *= other;
        return *this;
    }

    const StateVector& operator=(const StateVector& other)
    {
        u1 = other.u1;
        u2 = other.u2;
        u3 = other.u3;
        b = other.b;
        return *this;
    }

    stratifloat RemovePhaseShift()
    {
        stratifloat shift = -std::arg(u1(1,0,gridParams.N3/2))+pi/2;
        return RemovePhaseShift(shift);
    }

    stratifloat RemovePhaseShift(stratifloat shift)
    {
        u1.PhaseShift(shift);
        if (gridParams.ThreeDimensional)
        {
            u2.PhaseShift(shift);
        }
        u3.PhaseShift(shift);
        b.PhaseShift(shift);

        return shift;
    }

    stratifloat Dot(const StateVector& other) const
    {
        stratifloat prod = InnerProd(u1, other.u1, flowParams.L3)
                         + InnerProd(u3, other.u3, flowParams.L3)
                         + flowParams.Ri*InnerProd(b, other.b, flowParams.L3); // TODO: is this correct PE?

        if (gridParams.ThreeDimensional)
        {
            prod += InnerProd(u2, other.u2, flowParams.L3);
        }

        return prod;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Energy() const
    {
        return 0.5*Norm2();
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    stratifloat Enstrophy() const
    {
        Modal FractionalTemp;
        FractionalTemp = ddz(u1)+-1.0*ddx(u3);

        return InnerProd(FractionalTemp, FractionalTemp, flowParams.L3);
    }

    void Zero()
    {
        u1.Zero();
        if (gridParams.ThreeDimensional)
        {
            u2.Zero();
        }
        u3.Zero();
        b.Zero();
        p.Zero();
    }

    void Rescale(stratifloat energy);

    void ExciteLowWavenumbers(stratifloat energy);

    void Randomise(stratifloat energy, bool restrictToMiddle = false)
    {
        u1.RandomizeCoefficients(0.3);
        if (gridParams.ThreeDimensional)
        {
            u2.RandomizeCoefficients(0.3);
        }
        u3.RandomizeCoefficients(0.3);
        b.RandomizeCoefficients(0.3);

        if (restrictToMiddle)
        {
            for (int j=0; j<gridParams.N3/4; j++)
            {
                u1.slice(j).setZero();
                u1.slice(gridParams.N3-j-1).setZero();

                u2.slice(j).setZero();
                u2.slice(gridParams.N3-j-1).setZero();

                u3.slice(j).setZero();
                u3.slice(gridParams.N3-j-1).setZero();

                b.slice(j).setZero();
                b.slice(gridParams.N3-j-1).setZero();
            }
        }

        Rescale(energy);
    }

    void LoadFromFile(const std::string& filename)
    {
        if (EndsWith(filename, ".fields"))
        {
            solver.LoadFlow(filename);
        }
        else
        {
            solver.LoadFlow(filename+".fields");
        }
        CopyFromSolver();
    }

    void SaveToFile(const std::string& filename) const
    {
        CopyToSolver();
        solver.PopulateNodalVariables();

        if (EndsWith(filename, ".fields"))
        {
            solver.SaveFlow(filename);
        }
        else
        {
            solver.SaveFlow(filename+".fields");
        }
    }

    template<int K1, int K2, int K3>
    void LoadAndInterpolate(const std::string& filename)
    {/*
        // Set everything to zero. It interpolating to more modes, higher modes will not be overwritten
        Zero();

        // Load in to modal fields
        NodalField<K1,K2,K3> U1Loaded;
        NodalField<K1,K2,K3> U2Loaded;
        NodalField<K1,K2,K3> U3Loaded;
        NodalField<K1,K2,K3> BLoaded;

        std::ifstream filestream(filename, std::ios::in | std::ios::binary);

        U1Loaded.Load(filestream);
        U2Loaded.Load(filestream);
        U3Loaded.Load(filestream);
        BLoaded.Load(filestream);

        ModalField<K1,K2,K3> u1Loaded;
        ModalField<K1,K2,K3> u2Loaded;
        ModalField<K1,K2,K3> u3Loaded;
        ModalField<K1,K2,K3> bLoaded;

        U1Loaded.ToModal(u1Loaded);
        U2Loaded.ToModal(u2Loaded);
        U3Loaded.ToModal(u3Loaded);
        BLoaded.ToModal(bLoaded);

        // just 2D for simplicity for now
        assert(K2==1);
        assert(gridParams.N2==1);
        assert(!gridParams.ThreeDimensional);

        for (int j1=0; j1<std::min(K1/2+1,gridParams.N1/2+1); j1++)BROKEN
        {
            // Neumann fields u1, u2 and b
            for (int j3=0; j3<gridParams.N3; j3++)
            {
                stratifloat z = newNeumannPoints(j3);

                // for each new gridpoint, find the old gridpoints either side
                int k3_below;
                int k3_above = 1;
                stratifloat z_below;
                stratifloat z_above;

                do
                {
                    k3_above++;
                    k3_below = k3_above-1;

                    z_below = oldNeumannPoints(k3_below);
                    z_above = oldNeumannPoints(k3_above);
                } while(z_above<z && k3_above<K3-2);

                // linearly interpolate between these points
                stratifloat weight_above = (z-z_below)/(z_above-z_below);
                stratifloat weight_below = (z_above-z)/(z_above-z_below);

                if(weight_above > 1)
                {
                    weight_above = 1;
                    weight_below = 0;
                }

                if (weight_below > 1)
                {
                    weight_below = 1;
                    weight_above = 0;
                }

                u1(j1,0,j3) = weight_below*u1Loaded(j1,0,k3_below)
                            + weight_above*u1Loaded(j1,0,k3_above);

                u2(j1,0,j3) = weight_below*u2Loaded(j1,0,k3_below)
                            + weight_above*u2Loaded(j1,0,k3_above);

                b(j1,0,j3) = weight_below*bLoaded(j1,0,k3_below)
                           + weight_above*bLoaded(j1,0,k3_above);
            }

            // neumann conditions
            u1(j1,0,0) = u1(j1,0,1);
            u2(j1,0,0) = u2(j1,0,1);
            b(j1,0,0)   = b(j1,0,1);

            u1(j1,0,gridParams.N3-1)=u1(j1,0,gridParams.N3-2);
            u2(j1,0,gridParams.N3-1)=u2(j1,0,gridParams.N3-2);
            b(j1,0,gridParams.N3-1) = b(j1,0,gridParams.N3-2);

            // same for dirichlet field u3
            for (int j3=1; j3<gridParams.N3; j3++)
            {
                stratifloat z = newDirichletPoints(j3);

                int k3_below;
                int k3_above = 1;
                stratifloat z_below;
                stratifloat z_above;

                do
                {
                    k3_above++;
                    k3_below = k3_above-1;

                    z_below = oldDirichletPoints(k3_below);
                    z_above = oldDirichletPoints(k3_above);
                } while(z_above<z && k3_above<K3-1);

                stratifloat weight_above = (z-z_below)/(z_above-z_below);
                stratifloat weight_below = (z_above-z)/(z_above-z_below);

                if(weight_above > 1)
                {
                    weight_above = 1;
                    weight_below = 0;
                }

                if (weight_below > 1)
                {
                    weight_below = 1;
                    weight_above = 0;
                }

                u3(j1,0,j3) = weight_below*u3Loaded(j1,0,k3_below)
                            + weight_above*u3Loaded(j1,0,k3_above);
            }
        }

*/assert(0);    }

    void MakeMode2()
    {
        u1.MakeMode2();
        u2.MakeMode2();
        u3.MakeMode2();
        b.MakeMode2();
    }

    void PlotAll(std::string directory) const
    {
        MakeCleanDir(directory);

        HeatPlot(u1, flowParams.L1, flowParams.L3, 0, directory+"/u1.png");
        if (gridParams.ThreeDimensional)
        {
            HeatPlot(u2, flowParams.L1, flowParams.L3, 0, directory+"/u2.png");
        }
        HeatPlot(u3, flowParams.L1, flowParams.L3, 0, directory+"/u3.png");
        HeatPlot(b, flowParams.L1, flowParams.L3, 0, directory+"/b.png");

        Modal FractionalTemp;
        FractionalTemp = -1.0*ddz(u1)+ddx(u3);
        HeatPlot(FractionalTemp, flowParams.L1, flowParams.L3, 0, directory+"/vorticity.png");
        HeatPlot(FractionalTemp, flowParams.L1, flowParams.L3, 0, directory+"/vorticity.eps");
    }


    static void ResetForParams()
    {
        solver = IMEXRK();
    }

private:
    void CopyToSolver() const
    {
        solver.u1 = u1;
        if (gridParams.ThreeDimensional)
        {
            solver.u2 = u2;
        }
        else
        {
            solver.u2.Zero();
        }
        solver.u3 = u3;
        solver.b = b;
        solver.p = p;
    }

    void CopyFromSolver()
    {
        CopyFromSolver(*this);
    }

    void CopyFromSolver(StateVector& into) const
    {
        into.u1 = solver.u1;
        if (gridParams.ThreeDimensional)
        {
            into.u2 = solver.u2;
        }
        into.u3 = solver.u3;
        into.b = solver.b;
        into.p = solver.p;
    }

public:
    static IMEXRK solver;
};

StateVector operator+(const StateVector& lhs, const StateVector& rhs);
StateVector operator-(const StateVector& lhs, const StateVector& rhs);
StateVector operator*(stratifloat scalar, const StateVector& vector);
StateVector operator*(const StateVector& vector, stratifloat scalar);
