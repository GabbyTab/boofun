# Performance Guide

This document describes BooFun's performance characteristics, optimization strategies, and benchmarks.

## The Bottleneck: Exponential Truth Tables

Most Boolean function analysis is O(2^n) or O(n * 2^n) because it touches every input. For n = 20 the truth table has ~1 million entries; for n = 25 it has ~33 million. The techniques below help push the practical limit higher.

## Quick Summary

| Operation | Complexity | n=10 | n=14 | n=18 | n=20 |
|-----------|------------|------|------|------|------|
| Truth table creation | O(2^n) | <1ms | ~10ms | ~200ms | ~1s |
| Walsh-Hadamard Transform | O(n·2^n) | <1ms | ~5ms | ~100ms | ~500ms |
| Influence computation | O(2^n) | <1ms | ~5ms | ~80ms | ~350ms |
| Property testing | O(queries) | <1ms | <1ms | <5ms | <10ms |

## Optimization Tiers

BooFun uses multiple optimization strategies, automatically selecting the best available.

### Tier 1: NumPy Vectorization (Default)

Always available. 10-100x faster than pure Python. Used for all array operations.

Avoid Python-level loops over truth tables. Use `f.fourier()` (vectorised WHT) rather than iterating over inputs manually. Batch evaluation is faster than per-input calls:

```python
# Slow
results = [f.evaluate(x) for x in range(2**n)]

# Fast
results = f.evaluate(np.arange(2**n))
```

### Tier 2: Numba JIT Compilation (Recommended)

Install the performance extras:

```bash
pip install boofun[performance]
```

2-10x faster than NumPy for iterative operations (WHT, influences, sensitivity). JIT compilation of hot paths, with a one-time compilation cost on first call.

Check if Numba is active:

```python
from boofun.core.numba_optimizations import is_numba_available
print(is_numba_available())

# Or check which backends are in use:
from boofun.core.optimizations import HAS_NUMBA, INFLUENCES_BACKEND
print(f"Numba available: {HAS_NUMBA}")
print(f"Influences backend: {INFLUENCES_BACKEND}")
```

### Tier 3: GPU Acceleration via CuPy (Optional)

For n > 14, GPU parallelism can significantly accelerate spectral operations.

```bash
pip install cupy-cuda12x  # adjust for your CUDA version
```

Or on Google Colab, just select a GPU runtime.

**What's accelerated:**

| Operation | Module | GPU benefit threshold |
|-----------|--------|----------------------|
| Walsh-Hadamard Transform | `gpu.gpu_walsh_hadamard` | n > 14 |
| Influence computation | `gpu.gpu_influences` | n > 12 |
| Noise stability | `gpu.gpu_noise_stability` | n > 12 |
| Spectral weight by degree | `gpu.gpu_spectral_weight_by_degree` | n > 12 |
| Batch truth table lookup | `gpu.gpu_accelerate('truth_table_batch', ...)` | > 10K inputs |

**Usage:**

```python
from boofun.core.gpu import (
    is_gpu_available,
    gpu_walsh_hadamard,
    gpu_influences,
    GPUBooleanFunctionOps,
)

# Low-level
if is_gpu_available():
    fourier = gpu_walsh_hadamard(pm_values)
    influences = gpu_influences(fourier, n_vars=n)

# High-level wrapper
ops = GPUBooleanFunctionOps(truth_table)
fourier = ops.fourier()
influences = ops.influences()
stability = ops.noise_stability(rho=0.5)
```

For small n (< 12), the overhead of transferring data to/from the GPU exceeds the computation time. The `should_use_gpu()` heuristic handles this automatically in batch processing.

See `notebooks/gpu_performance.ipynb` for an interactive Colab benchmark comparing CPU vs GPU across different n.

## Memory Optimization

### Truth Table Representations

| Format | Memory (n=20) | Access Time | Best For |
|--------|---------------|-------------|----------|
| numpy bool | 1 MB | O(1) | n ≤ 14 |
| packed bitarray | 128 KB | O(1) | 14 < n ≤ 20 |
| sparse | ~k·12 bytes | O(1) | High sparsity |

### Auto-Selection

```python
from boofun.core.auto_representation import recommend_representation

rec = recommend_representation(n_vars=18, sparsity=0.1)
print(rec)
# {'representation': 'sparse_truth_table', 'reason': 'Sparsity 10.0% < 30%'}
```

## Batch Processing

The batch processing module handles large sets of inputs efficiently:

```python
from boofun.core.batch_processing import BatchProcessorManager

manager = BatchProcessorManager()
results = manager.process_batch(function_data, inputs, n_vars, rep_type="truth_table")
```

It automatically selects between CPU and GPU based on input size and available hardware.

## Caching and Lazy Conversion

### Instance-Level Caching

BooleanFunction instances cache Fourier coefficients, influences, and other computed values:

```python
f = bf.majority(15)

# First call: computes WHT (slow)
fourier = f.fourier()

# Second call: returns cached (instant)
fourier = f.fourier()
```

### Global Compute Cache

```python
from boofun.core.optimizations import get_global_cache

cache = get_global_cache()
print(cache.stats())
# {'size': 42, 'max_size': 500, 'hits': 156, 'misses': 42, 'hit_rate': 0.79}
```

### Lazy Conversion

Representations are computed lazily through the conversion graph. If you create a function from a truth table and request Fourier coefficients, only the truth_table -> fourier_expansion conversion runs.

## Benchmarks

### Walsh-Hadamard Transform

```
n=10:  NumPy: 0.5ms, Numba: 0.2ms, GPU: 0.1ms
n=14:  NumPy: 8ms,   Numba: 3ms,   GPU: 0.5ms
n=18:  NumPy: 150ms, Numba: 50ms,  GPU: 5ms
n=20:  NumPy: 700ms, Numba: 200ms, GPU: 15ms
```

### Influence Computation

```
n=10:  NumPy: 0.3ms, Numba: 0.1ms
n=14:  NumPy: 5ms,   Numba: 1ms
n=18:  NumPy: 90ms,  Numba: 20ms
n=20:  NumPy: 400ms, Numba: 80ms
```

### Property Testing (1000 queries)

```
BLR linearity:  ~2ms (independent of n)
Junta test:     ~5ms (for k-junta)
Monotonicity:   ~3ms
```

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# With comparison
pytest tests/benchmarks/ --benchmark-compare

# In Docker
docker-compose run benchmark
```

## Profiling

```python
import time
f = bf.majority(15)

start = time.perf_counter()
fourier = f.fourier()
print(f"WHT: {time.perf_counter() - start:.3f}s")

start = time.perf_counter()
inf = [f.influence(i) for i in range(15)]
print(f"Influences: {time.perf_counter() - start:.3f}s")
```

Or use the built-in profiling script:

```bash
python scripts/profile_performance.py
```

## Working with Large n (> 20 variables)

Most BooFun methods (`.fourier()`, `.influences()`, `.noise_stability()`) materialise the full truth table (2^n entries). This is fine for n <= 20 but becomes a problem beyond that. Here's what to do.

### What works at any n (no truth table needed)

These operations use **oracle access** -- they call `f.evaluate(x)` on individual or sampled inputs and never build a 2^n array:

```python
import boofun as bf
from boofun.analysis.sampling import (
    estimate_fourier_coefficient,
    estimate_fourier_adaptive,
    estimate_influence,
    estimate_total_influence,
    estimate_expectation,
    RandomVariableView,
)
from boofun.core.adapters import adapt_callable

# Create a function from a callable (no truth table)
f = adapt_callable(lambda x: x[0] & (x[1] | x[2]), n_vars=30)

# Estimate Fourier coefficients by sampling
f_hat_S = estimate_fourier_coefficient(f, S=0b101, n_samples=10000)

# Adaptive estimation with target error
est, std_err, n_used = estimate_fourier_adaptive(f, S=0b101, target_error=0.01)

# Estimate influences
inf_0 = estimate_influence(f, i=0, n_samples=5000)

# Property testing works at any n (query complexity, not n)
from boofun.analysis import PropertyTester
tester = PropertyTester(f)
tester.blr_linearity_test()    # O(1/epsilon) queries
tester.monotonicity_test()     # O(n/epsilon) queries
```

### What triggers truth table materialisation

These operations need the full truth table and will emit a **warning** for n > 25:

- `f.fourier()` (exact Fourier coefficients)
- `f.influences()` (exact influences)
- `f.noise_stability(rho)` (exact noise stability)
- Any representation conversion that routes through `truth_table`

### Controlling the large-n safety check

```python
from boofun.core.conversion_graph import set_large_n_policy

# Default: warn but proceed
set_large_n_policy("warn", threshold=25)

# Hard error (good for automated pipelines)
set_large_n_policy("raise", threshold=22)

# I know what I'm doing -- go up to n=28
set_large_n_policy("warn", threshold=28)

# Disable entirely
set_large_n_policy("off")
```

### Future plans (v2.0.0)

- Symbolic/oracle representations (BDD, ZDD) for n > 25
- Lazy evaluation that avoids materialisation entirely
- Large-scale research mode for conjecture-checking at n = 30+

See the [ROADMAP](../ROADMAP.md) for details.

## Quick Reference

| n range | Recommended approach |
|---------|---------------------|
| 1-14 | Default (NumPy), exact methods |
| 15-20 | `pip install boofun[performance]` (Numba), exact methods |
| 20-25 | Add CuPy for GPU, or use Colab. Exact methods still feasible. |
| 25-30 | Use sampling/estimation (`estimate_fourier_coefficient`, `PropertyTester`). Avoid `.fourier()`. |
| 30+ | Oracle access only. Wait for v2.0.0 symbolic representations. |
