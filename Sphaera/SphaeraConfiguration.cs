using Celestite;

namespace Sphaera;

public class SphaeraConfiguration : BaseSimulationConfiguration
{
    public int GridSize { get; set; }
    public float GridSpacing { get; set; }
    
    public int AcceleratorIndex { get; set; } = 0;
}