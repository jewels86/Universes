using Celestite;
using Jewels.Lazulite;
using static Sphaera.Kernels;

namespace Sphaera;

public class Sphaera : Simulation<SphaeraConfiguration, SphaeraContributor>
{
    public Sphaera(SphaeraConfiguration config, List<SphaeraContributor> contributors) : base(config, contributors)
    {
        Positions = new VectorValue[TotalSteps];
        Velocities = new VectorValue[TotalSteps];
        
        float[] stridedPositions = new float[TotalContributors * 3];
        float[] stridedVelocities = new float[TotalContributors * 3];

        for (int i = 0; i < TotalContributors; i++)
        {
            stridedPositions[KernelProgramming.StridedIndexOf(i, 3)] = Contributors[i].InitialPosition[0];
            stridedPositions[KernelProgramming.StridedIndexOf(i, 3, 1)] = Contributors[i].InitialPosition[1];
            stridedPositions[KernelProgramming.StridedIndexOf(i, 3, 2)] = Contributors[i].InitialPosition[2];
            
            stridedVelocities[KernelProgramming.StridedIndexOf(i, 3)] = Contributors[i].InitialVelocity[0];
            stridedVelocities[KernelProgramming.StridedIndexOf(i, 3, 1)] = Contributors[i].InitialVelocity[1];
            stridedVelocities[KernelProgramming.StridedIndexOf(i, 3, 2)] = Contributors[i].InitialVelocity[2];
        }

        Positions[0] = new(stridedPositions, Configuration.AcceleratorIndex);
        Velocities[0] = new(stridedVelocities, Configuration.AcceleratorIndex);
        
        Masses = new(Contributors.Select(c => c.Mass).ToArray(), Configuration.AcceleratorIndex);
    }

    public VectorValue[] Positions { get; private set; }
    public VectorValue[] Velocities { get; private set; }
    public VectorValue Masses { get; private set; }
    
    public int TotalGridVolume => Configuration.GridSize * Configuration.GridSize * Configuration.GridSize;
    
    public override void Step()
    {
        var alpha = Compute.Get(Configuration.AcceleratorIndex, TotalGridVolume);
        var gradAlpha = Compute.Get(Configuration.AcceleratorIndex, TotalGridVolume * 3);
        var hessian = Compute.Get(Configuration.AcceleratorIndex, TotalGridVolume * 9);
        var resultPositions = Compute.Get(Configuration.AcceleratorIndex, TotalContributors * 3);
        var resultVelocities = Compute.Get(Configuration.AcceleratorIndex, TotalContributors * 3);
        
        Compute.Call(Configuration.AcceleratorIndex, AlphaComputationKernels, TotalGridVolume, alpha, Positions[^1], Masses, Configuration.GridSize, Configuration.GridSpacing);
        Compute.Call(Configuration.AcceleratorIndex, GradAlphaComputationKernels, TotalGridVolume, gradAlpha, alpha, Configuration.GridSize, Configuration.GridSpacing);
        Compute.Call(Configuration.AcceleratorIndex, HessianComputationKernels, TotalGridVolume, hessian, alpha, Configuration.GridSize, Configuration.GridSpacing);
        Compute.Call(Configuration.AcceleratorIndex, StepKernels, TotalContributors,
            resultPositions, resultVelocities,
            Positions[^1], Velocities[^1], Positions[^2], 
            gradAlpha, hessian,
            Configuration.GridSize, Configuration.GridSpacing, Configuration.DeltaTime);
        
        Positions[CurrentStep] = new(resultPositions);
        Velocities[CurrentStep] = new(resultVelocities);
        CurrentStep++;
    }

    public override void Compile()
    {
        foreach (var contributor in Contributors) (contributor.Positions, contributor.Velocities) = (new float[TotalSteps][], new float[TotalSteps][]); 
        
        float[][] allPositions = Positions.Select(pos => pos.ToHost()).ToArray();
        float[][] allVelocities = Velocities.Select(vel => vel.ToHost()).ToArray();

        for (int t = 0; t < TotalSteps; t++)
        {
            float[] stridedPositions = allPositions[t];
            float[] stridedVelocities = allVelocities[t];

            for (int i = 0; i < TotalContributors; i++)
            {
                var contributor = Contributors[i];
                var (rx, ry, rz) = (
                    stridedPositions[KernelProgramming.StridedIndexOf(i, 3)], 
                    stridedPositions[KernelProgramming.StridedIndexOf(i, 3, 1)], 
                    stridedPositions[KernelProgramming.StridedIndexOf(i, 3, 2)]);
                var (vx, vy, vz) = (
                    stridedVelocities[KernelProgramming.StridedIndexOf(i, 3)], 
                    stridedVelocities[KernelProgramming.StridedIndexOf(i, 3, 1)], 
                    stridedVelocities[KernelProgramming.StridedIndexOf(i, 3, 2)]);
                
                contributor.Positions[t] = [rx, ry, rz];
                contributor.Velocities[t] = [vx, vy, vz];
            }
        }
    }
    
    public override void Dispose()
    {
        foreach (var position in Positions) position.Dispose();
        foreach (var velocity in Velocities) velocity.Dispose();
        Masses.Dispose();
    }
}