#!/usr/bin/env python3
"""
Performance Profiling for BooFun

This script profiles critical operations to identify bottlenecks:
1. Walsh-Hadamard transform (Fourier computation)
2. Influence computation
3. Noise stability calculation
4. Truth table operations

Usage:
    python scripts/profile_performance.py [--detailed]
"""

import time
import sys
import cProfile
import pstats
from io import StringIO
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

import boofun as bf


def time_operation(func, *args, iterations=10, **kwargs):
    """Time an operation with multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'result': result
    }


def profile_fourier_transform():
    """Profile Walsh-Hadamard (Fourier) transform for various n."""
    print("\n" + "="*70)
    print("FOURIER TRANSFORM (WALSH-HADAMARD) PROFILING")
    print("="*70)
    
    results = []
    
    for n in [8, 10, 12, 14, 16, 18, 20]:
        # Create a majority function
        f = bf.majority(n)
        
        # Time the Fourier transform
        timing = time_operation(f.fourier, iterations=5)
        
        result = {
            'n': n,
            'size': 2**n,
            'mean_time': timing['mean'],
            'std_time': timing['std']
        }
        results.append(result)
        
        print(f"n={n:2d} (2^{n}={2**n:>8d} points): "
              f"{timing['mean']*1000:>8.2f} ms ± {timing['std']*1000:>6.2f} ms")
    
    # Check scaling
    print("\nScaling analysis:")
    for i in range(1, len(results)):
        ratio = results[i]['mean_time'] / results[i-1]['mean_time']
        size_ratio = results[i]['size'] / results[i-1]['size']
        print(f"  n={results[i-1]['n']}→{results[i]['n']}: "
              f"time ratio={ratio:.2f}x (size ratio={size_ratio:.0f}x)")
    
    return results


def profile_influences():
    """Profile influence computation for various n."""
    print("\n" + "="*70)
    print("INFLUENCE COMPUTATION PROFILING")
    print("="*70)
    
    for n in [8, 10, 12, 14, 16]:
        f = bf.majority(n)
        
        # Individual influences
        timing = time_operation(f.influences, iterations=5)
        print(f"n={n:2d}: f.influences()    = {timing['mean']*1000:>8.2f} ms")
        
        # Total influence
        timing = time_operation(f.total_influence, iterations=5)
        print(f"n={n:2d}: f.total_influence() = {timing['mean']*1000:>8.2f} ms")
        print()


def profile_noise_stability():
    """Profile noise stability computation."""
    print("\n" + "="*70)
    print("NOISE STABILITY PROFILING")
    print("="*70)
    
    rho = 0.5
    
    for n in [8, 10, 12, 14, 16]:
        f = bf.majority(n)
        
        timing = time_operation(f.noise_stability, rho, iterations=5)
        print(f"n={n:2d}: f.noise_stability({rho}) = {timing['mean']*1000:>8.2f} ms")


def profile_function_creation():
    """Profile function creation overhead."""
    print("\n" + "="*70)
    print("FUNCTION CREATION PROFILING")
    print("="*70)
    
    for n in [8, 10, 12, 14, 16]:
        # Majority
        timing = time_operation(bf.majority, n, iterations=10)
        print(f"n={n:2d}: bf.majority({n})  = {timing['mean']*1000:>6.2f} ms")
        
        # Parity
        timing = time_operation(bf.parity, n, iterations=10)
        print(f"n={n:2d}: bf.parity({n})    = {timing['mean']*1000:>6.2f} ms")
        
        # AND
        timing = time_operation(bf.AND, n, iterations=10)
        print(f"n={n:2d}: bf.AND({n})       = {timing['mean']*1000:>6.2f} ms")
        print()


def detailed_profile(func_name="fourier"):
    """Run detailed cProfile analysis."""
    print("\n" + "="*70)
    print(f"DETAILED PROFILING: {func_name}")
    print("="*70)
    
    profiler = cProfile.Profile()
    
    if func_name == "fourier":
        f = bf.majority(14)
        profiler.enable()
        for _ in range(10):
            _ = f.fourier()
        profiler.disable()
    elif func_name == "influences":
        f = bf.majority(14)
        profiler.enable()
        for _ in range(10):
            _ = f.influences()
        profiler.disable()
    else:
        print(f"Unknown function: {func_name}")
        return
    
    # Print stats
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    stats.print_stats(30)
    print(s.getvalue())


def memory_profile():
    """Profile memory usage for large n."""
    print("\n" + "="*70)
    print("MEMORY ESTIMATION")
    print("="*70)
    
    for n in [10, 14, 18, 20, 22, 24]:
        tt_size = 2**n  # Number of entries in truth table
        tt_bytes = tt_size * 1  # Boolean array (1 byte per entry typically)
        fourier_bytes = tt_size * 8  # Float64 array
        
        print(f"n={n:2d}: Truth table = {tt_bytes/1024/1024:>8.2f} MB, "
              f"Fourier = {fourier_bytes/1024/1024:>8.2f} MB")


def main():
    detailed = "--detailed" in sys.argv
    
    print("BooFun Performance Profiling")
    print("="*70)
    
    # Run basic profiling
    profile_function_creation()
    profile_fourier_transform()
    profile_influences()
    profile_noise_stability()
    memory_profile()
    
    if detailed:
        detailed_profile("fourier")
        detailed_profile("influences")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on profiling results, consider these optimizations:

1. WALSH-HADAMARD TRANSFORM (n > 16):
   - Use in-place FFT to reduce memory allocation
   - Consider sparse representation for functions with few non-zero coefficients
   - GPU acceleration via CuPy for n > 20

2. INFLUENCE COMPUTATION:
   - Cache Fourier coefficients (they're reused)
   - Vectorize the summation over subsets
   - Use bit manipulation for subset enumeration

3. GENERAL:
   - Lazy evaluation for chained operations
   - Memoization for repeated computations
   - Sparse representations for large n with structure
""")


if __name__ == "__main__":
    main()
