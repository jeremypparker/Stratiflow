#pragma once

#include "IMEXRK.h"
#include "OrrSommerfeld.h"

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
        EnforceBCs();
    }

    NeumannModal u1;
    NeumannModal u2;
    DirichletModal u3;
    NeumannModal b;
    NeumannModal p;

    stratifloat FullEvolve(stratifloat T, StateVector& result, bool snapshot = false, bool screenshot = true, bool calcmixing = false) const;

    void FixedEvolve(stratifloat deltaT, int steps, std::vector<StateVector>& result) const;

    void LinearEvolve(stratifloat T, const StateVector& about, StateVector& result) const;

    void AdjointEvolve(stratifloat deltaT, int steps, const std::vector<StateVector>& intermediate, StateVector& result) const;


    const StateVector& operator+=(const StateVector& other)
    {
        u1 += other.u1;
        if (ThreeDimensional)
        {
            u2 += other.u2;
        }
        u3 += other.u3;
        b += other.b;
        EnforceBCs();

        return *this;
    }

    const StateVector& operator-=(const StateVector& other)
    {
        u1 -= other.u1;
        if (ThreeDimensional)
        {
            u2 -= other.u2;
        }
        u3 -= other.u3;
        b  -= other.b;
        EnforceBCs();
        return *this;
    }

    const StateVector& MulAdd(stratifloat a, const StateVector& B)
    {
        u1 += a*B.u1;
        if (ThreeDimensional)
        {
            u2 += a*B.u2;
        }
        u3 += a*B.u3;
        b  += a*B.b;
        EnforceBCs();
        return *this;
    }

    const StateVector& operator*=(stratifloat other)
    {
        u1 *= other;
        if (ThreeDimensional)
        {
            u2 *= other;
        }
        u3 *= other;
        b  *= other;
        EnforceBCs();
        return *this;
    }

    const StateVector& operator=(const StateVector& other)
    {
        u1 = other.u1;
        u2 = other.u2;
        u3 = other.u3;
        b = other.b;
        EnforceBCs();
        return *this;
    }

    void RemovePhaseShift()
    {
        stratifloat shift = -std::arg(u1(1,0,N3/2));
        u1.PhaseShift(shift);
        if (ThreeDimensional)
        {
            u2.PhaseShift(shift);
        }
        u3.PhaseShift(shift);
        b.PhaseShift(shift);
    }

    stratifloat Dot(const StateVector& other) const
    {
        stratifloat prod = InnerProd(u1, other.u1, L3)
                         + InnerProd(u3, other.u3, L3)
                         + Ri*InnerProd(b, other.b, L3); // TODO: is this correct PE?

        if (ThreeDimensional)
        {
            prod += InnerProd(u2, other.u2, L3);
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
        DirichletModal FractionalTemp;
        FractionalTemp = ddz(u1)+-1.0*ddx(u3);

        return InnerProd(FractionalTemp, FractionalTemp, L3);
    }

    void Zero()
    {
        u1.Zero();
        if (ThreeDimensional)
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
        if (ThreeDimensional)
        {
            u2.RandomizeCoefficients(0.3);
        }
        u3.RandomizeCoefficients(0.3);
        b.RandomizeCoefficients(0.3);

        if (restrictToMiddle)
        {
            for (int j=0; j<N3/4; j++)
            {
                u1.slice(j).setZero();
                u1.slice(N3-j-1).setZero();

                u2.slice(j).setZero();
                u2.slice(N3-j-1).setZero();

                u3.slice(j).setZero();
                u3.slice(N3-j-1).setZero();

                b.slice(j).setZero();
                b.slice(N3-j-1).setZero();
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
    {
        // Set everything to zero. It interpolating to more modes, higher modes will not be overwritten
        Zero();

        // Load in to modal fields
        NodalField<K1,K2,K3> U1Loaded(BoundaryCondition::Neumann);
        NodalField<K1,K2,K3> U2Loaded(BoundaryCondition::Neumann);
        NodalField<K1,K2,K3> U3Loaded(BoundaryCondition::Dirichlet);
        NodalField<K1,K2,K3> BLoaded(BoundaryCondition::Neumann);

        std::ifstream filestream(filename, std::ios::in | std::ios::binary);

        U1Loaded.Load(filestream);
        U2Loaded.Load(filestream);
        U3Loaded.Load(filestream);
        BLoaded.Load(filestream);

        ModalField<K1,K2,K3> u1Loaded(BoundaryCondition::Neumann);
        ModalField<K1,K2,K3> u2Loaded(BoundaryCondition::Neumann);
        ModalField<K1,K2,K3> u3Loaded(BoundaryCondition::Dirichlet);
        ModalField<K1,K2,K3> bLoaded(BoundaryCondition::Neumann);

        U1Loaded.ToModal(u1Loaded);
        U2Loaded.ToModal(u2Loaded);
        U3Loaded.ToModal(u3Loaded);
        BLoaded.ToModal(bLoaded);

        // Vertical points to interpolate to/from
        ArrayX oldNeumannPoints = VerticalPointsFractional(L3, K3);
        ArrayX oldDirichletPoints = VerticalPoints(L3, K3);

        ArrayX newNeumannPoints = VerticalPointsFractional(L3, N3);
        ArrayX newDirichletPoints = VerticalPoints(L3, N3);

        // just 2D for simplicity for now
        assert(K2==1);
        assert(N2==1);
        assert(!ThreeDimensional);

        for (int j1=0; j1<std::min(K1/2+1,N1/2+1); j1++)
        {
            // Neumann fields u1, u2 and b
            for (int j3=0; j3<N3; j3++)
            {
                stratifloat z = newNeumannPoints(j3);

                // for each new gridpoint, find the old gridpoints either side
                int k3_below;
                int k3_above = 0;
                stratifloat z_below;
                stratifloat z_above;

                do
                {
                    k3_above++;
                    k3_below = k3_above-1;

                    z_below = oldNeumannPoints(k3_below);
                    z_above = oldNeumannPoints(k3_above);
                } while(z_above<z);

                // linearly interpolate between these points
                stratifloat weight_above = (z-z_below)/(z_above-z_below);
                stratifloat weight_below = (z_above-z)/(z_above-z_below);

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

            u1(j1,0,N3-1)=u1(j1,0,N3-2);
            u2(j1,0,N3-1)=u2(j1,0,N3-2);
            b(j1,0,N3-1) = b(j1,0,N3-2);

            // same for dirichlet field u3
            for (int j3=1; j3<N3; j3++)
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
                } while(z_above<z);

                stratifloat weight_above = (z-z_below)/(z_above-z_below);
                stratifloat weight_below = (z_above-z)/(z_above-z_below);

                u3(j1,0,j3) = weight_below*u3Loaded(j1,0,k3_below)
                            + weight_above*u3Loaded(j1,0,k3_above);
            }
        }

        EnforceBCs();
    }

    void EnforceBCs()
    {
        u3.ZeroEnds();
        u1.NeumannEnds();
        u2.NeumannEnds();
        b.NeumannEnds();
    }

    void AddBackground()
    {
        NeumannNodal U1;
        NeumannNodal B;

        Neumann1D U_;
        Neumann1D B_;

        U_.SetValue(InitialU, L3);
        B_.SetValue(InitialB, L3);



        u1.ToNodal(U1);
        b.ToNodal(B);

        U1 += U_;
        B += B_;

        U1.ToModal(u1);
        B.ToModal(b);
    }

    void RemoveBackground()
    {
        NeumannNodal U1;
        NeumannNodal B;

        Neumann1D U_;
        Neumann1D B_;

        U_.SetValue(InitialU, L3);
        B_.SetValue(InitialB, L3);



        u1.ToNodal(U1);
        b.ToNodal(B);

        U1 -= U_;
        B -= B_;

        U1.ToModal(u1);
        B.ToModal(b);

    }

    void PlotAll(std::string directory) const
    {
        MakeCleanDir(directory);

        HeatPlot(u1, L1, L3, 0, directory+"/u1.png");
        if (ThreeDimensional)
        {
            HeatPlot(u2, L1, L3, 0, directory+"/u2.png");
        }
        HeatPlot(u3, L1, L3, 0, directory+"/u3.png");
        HeatPlot(b, L1, L3, 0, directory+"/b.png");

        DirichletModal FractionalTemp;
        FractionalTemp = ddz(u1)+-1.0*ddx(u3);
        HeatPlot(FractionalTemp, L1, L3, 0, directory+"/vorticity.png");
    }

    stratifloat ToEigenMode(stratifloat energy, int mode=1)
    {
        stratifloat growth = EigenModes(2*pi*mode/L1, u1, u2, u3, b);
        p.Zero();
        Rescale(energy);

        return growth;
    }

    static void ResetForParams()
    {
        solver = IMEXRK();
    }

private:
    void CopyToSolver() const
    {
        solver.u1 = u1;
        if (ThreeDimensional)
        {
            solver.u2 = u2;
        }
        solver.u3 = u3;
        solver.b = b;
        solver.u2.Zero();
        solver.p = p;
    }

    void CopyFromSolver()
    {
        CopyFromSolver(*this);
    }

    void CopyFromSolver(StateVector& into) const
    {
        into.u1 = solver.u1;
        if (ThreeDimensional)
        {
            into.u2 = solver.u2;
        }
        into.u3 = solver.u3;
        into.b = solver.b;
        into.p = solver.p;
        into.EnforceBCs();
    }

    static IMEXRK solver;
};

StateVector operator+(const StateVector& lhs, const StateVector& rhs);
StateVector operator-(const StateVector& lhs, const StateVector& rhs);
StateVector operator*(stratifloat scalar, const StateVector& vector);
StateVector operator*(const StateVector& vector, stratifloat scalar);
