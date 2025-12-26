using Jewels.Lazulite;
using Sphaera;

namespace Testing;

public static class SphaeraTest
{
    public static SphaeraConfiguration TwoBodyConfiguration => new()
    {
        AcceleratorIndex = Compute.RequestAccelerator(true),
        DeltaTime = 0.01f,
        GridSize = 64,
        GridSpacing = 0.1f,
        TotalSteps = 1000,
        SaveVelocities = true,
        TotalContributors = 2
    };

    public static List<SphaeraContributor> TwoBodyContributors =>
    [
        new([3, 3, 0], [0, 0, 0], 10),
        new([4, 3, 0], [0, 0.5f, 0], 1)
    ];
    
    public static void TestTwoBody()
    {
        using Sphaera.Sphaera sphaera = new(TwoBodyConfiguration, TwoBodyContributors);
        
        while (sphaera.ShouldContinue()) sphaera.Step();
        sphaera.Compile();
        sphaera.QuickPlot("sphaera.png");
    }
}