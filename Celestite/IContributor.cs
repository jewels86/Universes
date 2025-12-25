namespace Celestite;

public interface IContributor
{
    public float[] InitialPosition { get; }
    public float[] InitialVelocity { get; }
    
    public float[][] Positions { get; }
    public float[][] Velocities { get; }
}

public abstract class BaseContributor(float[] initialPosition, float[] initialVelocity) : IContributor
{
    public float[] InitialPosition { get; set; } = initialPosition;
    public float[] InitialVelocity { get; set; } = initialVelocity;

    public float[][] Positions { get; set; } = [];
    public float[][] Velocities { get; set; } = [];
}

public interface IMassiveContributor : IContributor
{
    public float Mass { get; }
}