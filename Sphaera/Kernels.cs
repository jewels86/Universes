using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using Jewels.Lazulite;

namespace Sphaera;

public static class Kernels
{
    internal const float Epsilon = 0.00001f;
    internal const float G = 0.2f;
    
    internal static float GaussianContribution(float dist, float mass) => mass * XMath.Exp(-(dist * dist));
    internal static float SoftenedInverseContribution(float dist, float mass) => mass / (dist + Epsilon);
    
    #region Helpers
    internal static (float x, float y, float z) GetPosition(int i, int size, float spacing)
    {
        int gridX = i % size;
        int gridY = (i / size) % size;
        int gridZ = i / (size * size);
        
        return (gridX * spacing, gridY * spacing, gridZ * spacing);
    }

    internal static int GetIndex((float x, float y, float z) pos, int size, float spacing)
    {
        int gridX = (int)(pos.x / spacing);
        int gridY = (int)(pos.y / spacing);
        int gridZ = (int)(pos.z / spacing);
    
        if (gridX < 0 || gridX >= size || 
            gridY < 0 || gridY >= size || 
            gridZ < 0 || gridZ >= size)
            return -1;
    
        return gridX + gridY * size + gridZ * size * size;
    }

    internal static float GetAlphaFrom(int x, int y, int z, int size, ArrayView1D<float, Stride1D.Dense> alpha)
    {
        if (x < 0 || x >= size || y < 0 || y >= size || z < 0 || z >= size) return 0;
        return alpha[x + y * size + z * size * size];
    }
    
    internal static (int x, int y, int z) GetGridIndex(int i, int size) => (i % size, (i / size) % size, i / (size * size));
    #endregion
    #region Kernels
    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>[] AlphaComputationKernels { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> positions,
        ArrayView1D<float, Stride1D.Dense> masses, int size, float spacing) =>
    {
        var pos = GetPosition(i, size, spacing);
        float alpha = 0;

        for (int j = 0; j < positions.Length / 3; j++)
        {
            var (other, mass) = (KernelProgramming.Vector3Get(positions, j), masses[j]);
            var diff = KernelProgramming.Vector3Subtract(other, pos);
            float dist = XMath.Sqrt(KernelProgramming.Vector3Magnitude2(diff));
            alpha += GaussianContribution(dist, mass);
        }

        result[i] = alpha;
    });

    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>[] GradAlphaComputationKernels { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> alpha, int size, float spacing) =>
    {
        var (gridX, gridY, gridZ) = GetGridIndex(i, size);

        float wrtX =
            (GetAlphaFrom(gridX + 1, gridY, gridZ, size, alpha)
             - GetAlphaFrom(gridX - 1, gridY, gridZ, size, alpha)) / (2 * spacing);
        float wrtY =
            (GetAlphaFrom(gridX, gridY + 1, gridZ, size, alpha)
             - GetAlphaFrom(gridX, gridY - 1, gridZ, size, alpha)) / (2 * spacing);
        float wrtZ =
            (GetAlphaFrom(gridX, gridY, gridZ + 1, size, alpha)
             - GetAlphaFrom(gridX, gridY, gridZ - 1, size, alpha)) / (2 * spacing);

        KernelProgramming.Vector3Set(result, i, (wrtX, wrtY, wrtZ));
    });

    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        int, float>[] HessianComputationKernels { get; } = Compute.Load((Index1D i,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> alpha,
        int size, float spacing) =>
    {
        int gridX = i % size;
        int gridY = (i / size) % size;
        int gridZ = i / (size * size);

        result[KernelProgramming.StridedIndexOf(i, 6)] = // Hxx
            (GetAlphaFrom(gridX + 1, gridY, gridZ, size, alpha)
             - 2 * GetAlphaFrom(gridX, gridY, gridZ, size, alpha)
             + GetAlphaFrom(gridX - 1, gridY, gridZ, size, alpha)) / (spacing * spacing);

        result[KernelProgramming.StridedIndexOf(i, 6, 1)] = // Hxy
            (GetAlphaFrom(gridX + 1, gridY + 1, gridZ, size, alpha)
             - GetAlphaFrom(gridX + 1, gridY - 1, gridZ, size, alpha)
             - GetAlphaFrom(gridX - 1, gridY + 1, gridZ, size, alpha)
             + GetAlphaFrom(gridX - 1, gridY - 1, gridZ, size, alpha)) / (4 * spacing * spacing);

        result[KernelProgramming.StridedIndexOf(i, 6, 2)] = // Hxz
            (GetAlphaFrom(gridX + 1, gridY, gridZ + 1, size, alpha)
             - GetAlphaFrom(gridX + 1, gridY, gridZ - 1, size, alpha)
             - GetAlphaFrom(gridX - 1, gridY, gridZ + 1, size, alpha)
             + GetAlphaFrom(gridX - 1, gridY, gridZ - 1, size, alpha)) / (4 * spacing * spacing);

        result[KernelProgramming.StridedIndexOf(i, 6, 3)] = // Hyy
            (GetAlphaFrom(gridX, gridY + 1, gridZ, size, alpha)
             - 2 * GetAlphaFrom(gridX, gridY, gridZ, size, alpha)
             + GetAlphaFrom(gridX, gridY - 1, gridZ, size, alpha)) / (spacing * spacing);

        result[KernelProgramming.StridedIndexOf(i, 6, 4)] = // Hyz
            (GetAlphaFrom(gridX, gridY + 1, gridZ + 1, size, alpha)
             - GetAlphaFrom(gridX, gridY + 1, gridZ - 1, size, alpha)
             - GetAlphaFrom(gridX, gridY - 1, gridZ + 1, size, alpha)
             + GetAlphaFrom(gridX, gridY - 1, gridZ - 1, size, alpha)) / (4 * spacing * spacing);

        result[KernelProgramming.StridedIndexOf(i, 6, 5)] = // Hzz
            (GetAlphaFrom(gridX, gridY, gridZ + 1, size, alpha) - 2
             * GetAlphaFrom(gridX, gridY, gridZ, size, alpha)
             + GetAlphaFrom(gridX, gridY, gridZ - 1, size, alpha)) / (spacing * spacing);
    });

    public static Action<Index1D,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, float, float>[] StepKernels { get; }
        = Compute.Load((Index1D i,
            ArrayView1D<float, Stride1D.Dense> resultPositions,
            ArrayView1D<float, Stride1D.Dense> resultVelocities,
            ArrayView1D<float, Stride1D.Dense> positions,
            ArrayView1D<float, Stride1D.Dense> velocities,
            ArrayView1D<float, Stride1D.Dense> prevPositions,
            ArrayView1D<float, Stride1D.Dense> gradAlpha,
            ArrayView1D<float, Stride1D.Dense> hessian, int size, float spacing, float dt) =>
        {
            var r = KernelProgramming.Vector3Get(positions, i);
            var v = KernelProgramming.Vector3Get(velocities, i);
            var prevR = KernelProgramming.Vector3Get(prevPositions, i);
            
            var gridIndex = GetIndex(r, size, spacing);
            var grad = KernelProgramming.Vector3Get(gradAlpha, gridIndex);
            var f = KernelProgramming.Vector3Multiply(grad, G);

            var a = Acceleration(gradAlpha, hessian, r, v, f, size, spacing);
            Verlet(resultPositions, resultVelocities, r, prevR, a, dt, i);
        });
    
    #endregion
    #region Math
    internal static float CreateKappa(
        (float x, float y, float z) r,
        (float x, float y, float z) v,
        (float x, float y, float z) grad,
        int size, float spacing,
        ArrayView1D<float, Stride1D.Dense> hessian)
    {
        // kappa = (grad alpha dot v hat) dot (v dot nabla)(grad alpha dot v hat)
        int i = GetIndex(r, size, spacing);
        if (i < 0) return 0;

        float vMag = XMath.Sqrt(KernelProgramming.Vector3Magnitude2(v));
        if (vMag < Epsilon) return 0;
        var vHat = KernelProgramming.Vector3Divide(v, vMag);
        var dot1 = grad.x * vHat.x + grad.y * vHat.y + grad.z * vHat.z;
        
        float Hxx = hessian[KernelProgramming.StridedIndexOf(i, 6)];
        float Hxy = hessian[KernelProgramming.StridedIndexOf(i, 6, 1)];
        float Hxz = hessian[KernelProgramming.StridedIndexOf(i, 6, 2)];
        float Hyy = hessian[KernelProgramming.StridedIndexOf(i, 6, 3)];
        float Hyz = hessian [KernelProgramming.StridedIndexOf(i, 6, 4)];
        float Hzz = hessian [KernelProgramming.StridedIndexOf(i, 6, 5)];
        
        float HvhatX = Hxx * vHat.x + Hxy * vHat.y + Hxz * vHat.z;
        float HvhatY = Hxy * vHat.x + Hyy * vHat.y + Hyz * vHat.z;
        float HvhatZ = Hxz * vHat.x + Hyz * vHat.y + Hzz * vHat.z;
        
        float dot2 = v.x * HvhatX + v.y * HvhatY + v.z * HvhatZ;
        return dot1 * dot2;
    }

    internal static float CreateGamma(
        (float x, float y, float z) v,
        (float x, float y, float z) gradAlpha)
    {
        // Gamma = 1/sqrt(1 + (grad alpha dot v hat)^2)
        var vMag = XMath.Sqrt(KernelProgramming.Vector3Magnitude2(v));
        if (vMag < Epsilon) return 0;
        
        var vHat = KernelProgramming.Vector3Divide(v, vMag);
        float dot = gradAlpha.x * vHat.x + gradAlpha.y * vHat.y + gradAlpha.z * vHat.z;
        return 1 / XMath.Sqrt(1 + dot * dot);
    }
    
    internal static void Verlet(
        ArrayView1D<float, Stride1D.Dense> positions,
        ArrayView1D<float, Stride1D.Dense> velocities,
        (float x, float y, float z) r,
        (float x, float y, float z) prevR,
        (float x, float y, float z) a,
        float dt, int i)
    {
        // r_{n+1} = 2r - r_{n-1} + a dt^2
        // v_{n+1} = (r_{n+1} - r) / 2dt
        var diff = KernelProgramming.Vector3Subtract(KernelProgramming.Vector3Multiply(r, 2), prevR);
        var adt2 = KernelProgramming.Vector3Multiply(a, dt * dt);
        var rNext = KernelProgramming.Vector3Add(diff, adt2);
        var vNext = KernelProgramming.Vector3Divide(KernelProgramming.Vector3Subtract(rNext, r), dt * 2);
        
        KernelProgramming.Vector3Set(positions, i, rNext);
        KernelProgramming.Vector3Set(velocities, i, vNext);
    }

    internal static (float x, float y, float z) Acceleration(
        ArrayView1D<float, Stride1D.Dense> gradAlpha,
        ArrayView1D<float, Stride1D.Dense> hessian,
        (float x, float y, float z) r,
        (float x, float y, float z) v,
        (float x, float y, float z) a,
        int size, float spacing)
    {
        // a_s = Gamma(a - v kappa Gamma^2)
        int i = GetIndex(r, size, spacing);
        if (i < 0) return (0, 0, 0);
        var grad = KernelProgramming.Vector3Get(gradAlpha, i);
        
        var kappa = CreateKappa(r, v, grad, size, spacing, hessian);
        var gamma = CreateGamma(v, grad);
        
        var vKappaGamma2 = KernelProgramming.Vector3Multiply(v, kappa * gamma * gamma);
        var forceLike = KernelProgramming.Vector3Subtract(a, vKappaGamma2);
        return KernelProgramming.Vector3Multiply(forceLike, gamma);
    }
    #endregion
}