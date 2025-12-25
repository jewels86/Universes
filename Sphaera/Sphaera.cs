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
        float[] stridedPrevPositions = new float[TotalContributors * 3];

        for (int i = 0; i < TotalContributors; i++)
        {
            float px = Contributors[i].InitialPosition[0];
            float py = Contributors[i].InitialPosition[1];
            float pz = Contributors[i].InitialPosition[2];
            
            float vx = Contributors[i].InitialVelocity[0];
            float vy = Contributors[i].InitialVelocity[1];
            float vz = Contributors[i].InitialVelocity[2];
            
            stridedPositions[KernelProgramming.StridedIndexOf(i, 3)] = px;
            stridedPositions[KernelProgramming.StridedIndexOf(i, 3, 1)] = py;
            stridedPositions[KernelProgramming.StridedIndexOf(i, 3, 2)] = pz;
            
            stridedVelocities[KernelProgramming.StridedIndexOf(i, 3)] = vx;
            stridedVelocities[KernelProgramming.StridedIndexOf(i, 3, 1)] = vy;
            stridedVelocities[KernelProgramming.StridedIndexOf(i, 3, 2)] = vz;
            
            stridedPrevPositions[KernelProgramming.StridedIndexOf(i, 3)] = px - vx * Configuration.DeltaTime;
            stridedPrevPositions[KernelProgramming.StridedIndexOf(i, 3, 1)] = py - vy * Configuration.DeltaTime;
            stridedPrevPositions[KernelProgramming.StridedIndexOf(i, 3, 2)] = pz - vz * Configuration.DeltaTime;
        }
        
        Positions[0] = new(stridedPrevPositions, Configuration.AcceleratorIndex);
        Positions[1] = new(stridedPositions, Configuration.AcceleratorIndex);
        Velocities[0] = new(stridedVelocities, Configuration.AcceleratorIndex);
        Velocities[1] = new(stridedVelocities, Configuration.AcceleratorIndex);
        CurrentStep = 2;
        
        Masses = new(Contributors.Select(c => c.Mass).ToArray(), Configuration.AcceleratorIndex);
    }

    public VectorValue[] Positions { get; private set; }
    public VectorValue[] Velocities { get; private set; }
    public VectorValue Masses { get; private set; }
    
    public int TotalGridVolume => Configuration.GridSize * Configuration.GridSize * Configuration.GridSize;
    
    public override void Step()
    {
        Console.WriteLine(CurrentStep);
        var alpha = Compute.Get(Configuration.AcceleratorIndex, TotalGridVolume);
        var gradAlpha = Compute.Get(Configuration.AcceleratorIndex, TotalGridVolume * 3);
        var hessian = Compute.Get(Configuration.AcceleratorIndex, TotalGridVolume * 9);
        var resultPositions = Compute.Get(Configuration.AcceleratorIndex, TotalContributors * 3);
        var resultVelocities = Compute.Get(Configuration.AcceleratorIndex, TotalContributors * 3);
        
        Compute.Call(Configuration.AcceleratorIndex, AlphaComputationKernels, TotalGridVolume, alpha, Positions[CurrentStep - 1], Masses, Configuration.GridSize, Configuration.GridSpacing);
        Compute.Call(Configuration.AcceleratorIndex, GradAlphaComputationKernels, TotalGridVolume, gradAlpha, alpha, Configuration.GridSize, Configuration.GridSpacing);
        Compute.Call(Configuration.AcceleratorIndex, HessianComputationKernels, TotalGridVolume, hessian, alpha, Configuration.GridSize, Configuration.GridSpacing);
        Compute.Call(Configuration.AcceleratorIndex, StepKernels, TotalContributors,
            resultPositions, resultVelocities,
            Positions[CurrentStep - 1], Velocities[CurrentStep - 1], Positions[CurrentStep - 2], 
            gradAlpha, hessian,
            Configuration.GridSize, Configuration.GridSpacing, Configuration.DeltaTime);
        
        Positions[CurrentStep] = new(resultPositions);
        Velocities[CurrentStep] = new(resultVelocities);
        CurrentStep++;
        
        Compute.Return([alpha, gradAlpha, hessian]);
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