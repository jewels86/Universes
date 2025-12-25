namespace Celestite;

public interface ISimulationConfiguration
{
    public bool SaveVelocities { get; set; }
    public int TotalSteps { get; set; }
    public float DeltaTime { get; set; }
    public int TotalContributors { get; set; }
}

public abstract class BaseSimulationConfiguration : ISimulationConfiguration
{
    public bool SaveVelocities { get; set; }
    public int TotalSteps { get; set; }
    public float DeltaTime { get; set; }
    public int TotalContributors { get; set; }
}