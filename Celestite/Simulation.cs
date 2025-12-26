using Jewels.Lazulite;
using ScottPlot;

namespace Celestite;

public abstract class Simulation<TConfiguration, TContributor>(TConfiguration configuration, List<TContributor> contributors) : ISimulation
    where TConfiguration : ISimulationConfiguration
    where TContributor : IContributor
{
    public TConfiguration Configuration { get; protected set; } = configuration;
    public List<TContributor> Contributors { get; protected set; } = contributors;

    public abstract void Step();
    public abstract void Compile();
    public abstract void Dispose();

    public int CurrentStep { get; protected set; } = 0;
    public int TotalSteps => Configuration.TotalSteps;
    public float DeltaTime => Configuration.DeltaTime;
    public int TotalContributors => Configuration.TotalContributors;

    public ISimulationConfiguration GetConfiguration() => Configuration;
    public List<IContributor> GetContributors() => Contributors.Cast<IContributor>().ToList();
    
    public bool ShouldContinue() => CurrentStep < TotalSteps;

    public void QuickPlot(string path, int width = 800, int height = 800)
    {
        var plot = new Plot();

        foreach (var contributor in Contributors)
        {
            var xPositions = contributor.Positions.Select(pos => pos[0]).ToArray();
            var yPositions = contributor.Positions.Select(pos => pos[1]).ToArray();

            var start = plot.Add.Scatter(xPositions[0], yPositions[0]);
            start.MarkerSize = 20;
            
            var scatter = plot.Add.Scatter(xPositions, yPositions);
            scatter.Color = start.Color;
        }
        
        plot.Axes.AutoScale();
        plot.SavePng(path, width, height);
    }
}

public interface ISimulation : IDisposable
{
    public void Step();
    public List<IContributor> GetContributors();
    public ISimulationConfiguration GetConfiguration();
    public bool ShouldContinue();
}