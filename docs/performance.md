# Performance Guide

This document describes BooFun's performance characteristics, optimization strategies, and benchmarks.

## Quick Summary

| Operation | Complexity | n=10 | n=14 | n=18 | n=20 |
|-----------|------------|------|------|------|------|
| Truth table creation | O(2^n) | <1ms | ~10ms | ~200ms | ~1s |
| Walsh-Hadamard Transform | O(n·2^n) | <1ms | ~5ms | ~100ms | ~500ms |
| Influence computation | O(2^n) | <1ms | ~5ms | ~80ms | ~350ms |
| Property testing | O(queries) | <1ms | <1ms | <5ms | <10ms |

## Optimization Tiers

BooFun uses multiple optimization strategies, automatically selecting the best available:

### Tier 1: NumPy Vectorization (Default)
- Always available
- 10-100x faster than pure Python
- Used for all array operations

### Tier 2: Numba JIT Compilation (Recommended)
- Requires: `pip install numba`
- 2-10x faster than NumPy for iterative operations
- JIT compilation of hot paths
- Used for: WHT, influences, sensitivity

### Tier 3: GPU Acceleration (Optional)
- Requires: `pip install cupy-cuda11x` (match your CUDA version)
- 10-100x faster for large n (n > 16)
- Used for: WHT, batch operations

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

# Get recommendation for your use case
rec = recommend_representation(n_vars=18, sparsity=0.1)
print(rec)
# {'representation': 'sparse_truth_table', 
#  'reason': 'Sparsity 10.0% < 30%'}
```

### Using Packed Truth Tables

```python
from boofun.core.representations.packed_truth_table import create_packed_truth_table

# Convert existing truth table
packed = create_packed_truth_table(truth_table)

# Memory savings
from boofun.core.representations.packed_truth_table import memory_comparison
print(memory_comparison(20))
# packed_bitarray: 131,072 bytes (128.0 KB)
# numpy_bool: 1,048,576 bytes (1024.0 KB)
# savings: 8x
```

## Parallelization

### Batch Operations

```python
from boofun.core.optimizations import parallel_batch_influences, parallel_batch_fourier

# Compute influences for many functions at once
functions = [bf.random(n=10) for _ in range(100)]
all_influences = parallel_batch_influences(functions)

# Fourier coefficients in parallel
all_fourier = parallel_batch_fourier(functions)
```

### Numba Parallel Loops

Numba functions automatically use all CPU cores:

```python
# These are JIT-compiled with Numba:
# - _vectorized_influences_numba
# - _total_influence_numba
# - _fast_wht_numba

# Check if Numba is being used:
from boofun.core.optimizations import HAS_NUMBA, INFLUENCES_BACKEND
print(f"Numba available: {HAS_NUMBA}")
print(f"Influences backend: {INFLUENCES_BACKEND}")
```

## Caching and Memoization

### Global Compute Cache

```python
from boofun.core.optimizations import get_global_cache

cache = get_global_cache()

# Check cache statistics
print(cache.stats())
# {'size': 42, 'max_size': 500, 'hits': 156, 'misses': 42, 'hit_rate': 0.79}

# Clear cache if needed
cache.clear()
```

### Instance-Level Caching

BooleanFunction instances cache:
- Fourier coefficients (`_fourier_cache`)
- Influences (`_influences_cache`)
- Decision tree depth (`_dt_cache`)

```python
f = bf.majority(5)

# First call computes
_ = f.fourier()  # ~1ms

# Second call returns cached
_ = f.fourier()  # ~0.001ms
```

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

## Best Practices

### 1. Install Numba

```bash
pip install numba
```

This alone provides 2-10x speedup for most operations.

### 2. Use Appropriate Representations

```python
# For n > 14, consider sparse or packed
from boofun.core.auto_representation import AdaptiveFunction

# Automatically chooses best format
f = AdaptiveFunction(truth_table, n_vars=18)
print(f.format)  # 'packed' or 'sparse'
```

### 3. Batch Operations

```python
# Bad: sequential
results = [f.influences() for f in functions]

# Good: parallel
from boofun.core.optimizations import parallel_batch_influences
results = parallel_batch_influences(functions)
```

### 4. Reuse Functions

```python
# Bad: recreate function each time
for _ in range(100):
    f = bf.AND(10)
    print(f.total_influence())

# Good: reuse function
f = bf.AND(10)
for _ in range(100):
    print(f.total_influence())  # Uses cached values
```

### 5. Profile Your Code

```python
from boofun.core.optimizations import WHT_BACKEND, INFLUENCES_BACKEND

print(f"WHT backend: {WHT_BACKEND}")
print(f"Influences backend: {INFLUENCES_BACKEND}")

# Time specific operations
import time
f = bf.random(n=16)

start = time.time()
_ = f.fourier()
print(f"WHT: {time.time() - start:.3f}s")

start = time.time()
_ = f.influences()
print(f"Influences: {time.time() - start:.3f}s")
```

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# Specific benchmark
pytest tests/benchmarks/test_external_benchmarks.py -v

# With comparison
pytest tests/benchmarks/ --benchmark-compare
```

## Docker Performance

The Docker images include Numba by default:

```bash
# Run benchmarks in Docker
docker-compose run benchmark
```

## Scaling Guidelines

| n | Recommended Approach |
|---|---------------------|
| ≤ 14 | Dense truth table, Numba |
| 15-18 | Packed/sparse, Numba, consider GPU |
| 19-22 | GPU required, sparse representation |
| > 22 | Consider sampling/approximation algorithms |

## Future Optimizations

- Distributed computation with Dask
- Further GPU kernel optimization
- Memory-mapped truth tables for very large n
