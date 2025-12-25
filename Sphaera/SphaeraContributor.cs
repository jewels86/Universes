using Celestite;

namespace Sphaera;

public class SphaeraContributor(float[] initialPosition, float[] initialVelocity, float mass) : 
    BaseContributor(initialPosition, initialVelocity), 
    IMassiveContributor
{
    public float Mass => mass;
}