#pragma once

#include "IMEXRK.h"
#include "OrrSommerfeld.h"

// This class contains a full state's information
// its operations are not particularly efficient
// so it should only be used for high level algorithms
class StateVector
{
public:
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
        return *this;
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
        NodalField<K1,K2,K3> U1Loaded(BoundaryCondition::Neumann);
        NodalField<K1,K2,K3> U2Loaded(BoundaryCondition::Neumann);
        NodalField<K1,K2,K3> U3Loaded(BoundaryCondition::Dirichlet);
        NodalField<K1,K2,K3> BLoaded(BoundaryCondition::Neumann);

        std::ifstream filestream(filename, std::ios::in | std::ios::binary);

        U1Loaded.Load(filestream);
        U2Loaded.Load(filestream);
        U3Loaded.Load(filestream);
        BLoaded.Load(filestream);

        NeumannNodal U1;
        NeumannNodal U2;
        DirichletNodal U3;
        NeumannNodal B;

        ArrayX oldNeumannPoints = VerticalPoints(L3, K3);
        ArrayX oldDirichletPoints = VerticalPointsFractional(L3, K3);

        ArrayX newNeumannPoints = VerticalPoints(L3, N3);
        ArrayX newDirichletPoints = VerticalPointsFractional(L3, N3);

        // just 2D for simplicity for now
        assert(K2==1);
        assert(N2==1);
        assert(ThreeDimensional);

        for (int j1=0; j1<N1; j1++)
        {
            stratifloat x = j1*L1/N1;

            int k1_left = static_cast<int>(x/(L1/K1));
            int k1_right = k1_left+1;

            stratifloat x_left = k1_left*L1/K1;
            stratifloat x_right = k1_right*L1/K1;

            stratifloat weight_left = (x_right-x)/(x_right-x_left);
            stratifloat weight_right = (x-x_left)/(x_right-x_left);

            if (k1_left<0) k1_left += K1;
            if (k1_right>=K1) k1_right -= K1;

            for (int j3=1; j3<N3-1; j3++)
            {
                stratifloat z = newNeumannPoints(j3);

                int k3_above;
                int k3_below = 0;
                stratifloat z_below;
                stratifloat z_above;

                do
                {
                    k3_below++;
                    k3_above = k3_below-1;

                    z_below = oldNeumannPoints(k3_below);
                    z_above = oldNeumannPoints(k3_above);
                } while(z_below>z);

                stratifloat weight_above = (z-z_below)/(z_above-z_below);
                stratifloat weight_below = (z_above-z)/(z_above-z_below);

                U1(j1,0,j3) = weight_left*weight_below*U1Loaded(k1_left,0,k3_below)
                            + weight_left*weight_above*U1Loaded(k1_left,0,k3_above)
                            + weight_right*weight_below*U1Loaded(k1_right,0,k3_below)
                            + weight_right*weight_above*U1Loaded(k1_right,0,k3_above);

                U2(j1,0,j3) = weight_left*weight_below*U2Loaded(k1_left,0,k3_below)
                            + weight_left*weight_above*U2Loaded(k1_left,0,k3_above)
                            + weight_right*weight_below*U2Loaded(k1_right,0,k3_below)
                            + weight_right*weight_above*U2Loaded(k1_right,0,k3_above);

                B(j1,0,j3) = weight_left*weight_below*BLoaded(k1_left,0,k3_below)
                            + weight_left*weight_above*BLoaded(k1_left,0,k3_above)
                            + weight_right*weight_below*BLoaded(k1_right,0,k3_below)
                            + weight_right*weight_above*BLoaded(k1_right,0,k3_above);
            }

            // neumann conditions
            U1(j1,0,0) = U1(j1,0,1);
            U2(j1,0,0) = U2(j1,0,1);
            B(j1,0,0)   = B(j1,0,1);

            U1(j1,0,N3-1)=U1(j1,0,N3-2);
            U2(j1,0,N3-1)=U2(j1,0,N3-2);
            B(j1,0,N3-1) = B(j1,0,N3-2);

            for (int j3=1; j3<N3-2; j3++)
            {
                stratifloat z = newDirichletPoints(j3);

                int k3_above;
                int k3_below = 0;
                stratifloat z_below;
                stratifloat z_above;

                do
                {
                    k3_below++;
                    k3_above = k3_below-1;

                    z_below = oldDirichletPoints(k3_below);
                    z_above = oldDirichletPoints(k3_above);
                } while(z_below>z);

                stratifloat weight_above = (z-z_below)/(z_above-z_below);
                stratifloat weight_below = (z_above-z)/(z_above-z_below);

                U3(j1,0,j3) = weight_left*weight_below*U3Loaded(k1_left,0,k3_below)
                            + weight_left*weight_above*U3Loaded(k1_left,0,k3_above)
                            + weight_right*weight_below*U3Loaded(k1_right,0,k3_below)
                            + weight_right*weight_above*U3Loaded(k1_right,0,k3_above);
            }

            // dirichlet endpoints match up exactly
            U3(j1,0,0)    = weight_left*U3Loaded(k1_left,0,0)    + weight_right*U3Loaded(k1_right,0,0);
            U3(j1,0,N3-2) = weight_left*U3Loaded(k1_left,0,K3-2) + weight_right*U3Loaded(k1_right,0,K3-2);
        }

        U1.ToModal(u1);
        U2.ToModal(u2);
        U3.ToModal(u3);
        B.ToModal(b);
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
    }

    static IMEXRK solver;
};

StateVector operator+(const StateVector& lhs, const StateVector& rhs);
StateVector operator-(const StateVector& lhs, const StateVector& rhs);
StateVector operator*(stratifloat scalar, const StateVector& vector);
StateVector operator*(const StateVector& vector, stratifloat scalar);
