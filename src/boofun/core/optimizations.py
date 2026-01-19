"""
Performance Optimizations for BooFun

This module provides optimized implementations of critical operations:
1. Fast Walsh-Hadamard Transform (in-place, vectorized)
2. Vectorized influence computation
3. Lazy evaluation helpers

These optimizations are automatically used when available.
"""

import numpy as np
from typing import Optional, Callable, Any
import warnings

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def fast_walsh_hadamard(values: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Fast in-place Walsh-Hadamard Transform.
    
    This is the optimized O(n * 2^n) algorithm using butterfly operations.
    Works in-place to minimize memory allocations.
    
    Args:
        values: Input array of length 2^n (will be modified in-place)
        normalize: If True, divide by 2^n at the end
        
    Returns:
        Transformed array (same object as input if in-place)
    """
    n = int(np.log2(len(values)))
    size = len(values)
    
    # Validate input
    if size != (1 << n):
        raise ValueError(f"Input length must be power of 2, got {size}")
    
    # Make a copy to avoid modifying input
    result = values.astype(np.float64).copy()
    
    # Butterfly operations
    step = 1
    while step < size:
        for i in range(0, size, step * 2):
            for j in range(step):
                u = result[i + j]
                v = result[i + j + step]
                result[i + j] = u + v
                result[i + j + step] = u - v
        step *= 2
    
    if normalize:
        result /= size
    
    return result


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _fast_wht_numba(values: np.ndarray) -> np.ndarray:
        """Numba-accelerated Walsh-Hadamard Transform."""
        n = len(values)
        result = values.copy()
        
        step = 1
        while step < n:
            half_step = step
            step *= 2
            for i in prange(0, n, step):
                for j in range(half_step):
                    u = result[i + j]
                    v = result[i + j + half_step]
                    result[i + j] = u + v
                    result[i + j + half_step] = u - v
        
        return result / n
    
    def fast_walsh_hadamard_numba(values: np.ndarray) -> np.ndarray:
        """Use Numba-accelerated WHT if available."""
        return _fast_wht_numba(values.astype(np.float64))


def vectorized_truth_table_to_pm(truth_table: np.ndarray) -> np.ndarray:
    """
    Convert {0,1} truth table to {-1,+1} representation.
    
    This is vectorized for speed.
    """
    return 1.0 - 2.0 * truth_table.astype(np.float64)


def vectorized_influences_from_fourier(fourier_coeffs: np.ndarray, n_vars: int) -> np.ndarray:
    """
    Compute influences from Fourier coefficients using vectorization.
    
    Influence of variable i: Inf_i(f) = Σ_{S∋i} f̂(S)²
    
    This is faster than iterating over all subsets for each variable.
    """
    influences = np.zeros(n_vars, dtype=np.float64)
    size = len(fourier_coeffs)
    
    # Precompute squared coefficients
    squared = fourier_coeffs ** 2
    
    # For each variable i, sum squared coefficients of subsets containing i
    for i in range(n_vars):
        bit_mask = 1 << i
        # Find all subset indices where bit i is set
        mask = np.arange(size, dtype=np.int64) & bit_mask
        influences[i] = np.sum(squared[mask > 0])
    
    return influences


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _vectorized_influences_numba(fourier_coeffs: np.ndarray, n_vars: int) -> np.ndarray:
        """Numba-accelerated influence computation."""
        size = len(fourier_coeffs)
        influences = np.zeros(n_vars, dtype=np.float64)
        squared = fourier_coeffs ** 2
        
        for i in prange(n_vars):
            bit_mask = 1 << i
            total = 0.0
            for s in range(size):
                if s & bit_mask:
                    total += squared[s]
            influences[i] = total
        
        return influences
    
    def vectorized_influences_numba(fourier_coeffs: np.ndarray, n_vars: int) -> np.ndarray:
        """Use Numba-accelerated influence computation if available."""
        return _vectorized_influences_numba(fourier_coeffs.astype(np.float64), n_vars)


def vectorized_total_influence_from_fourier(fourier_coeffs: np.ndarray, n_vars: int) -> float:
    """
    Compute total influence from Fourier coefficients.
    
    Total influence = Σ_S |S| · f̂(S)² = Σ_i Inf_i(f)
    
    Uses the formula: I[f] = Σ_S |S| · f̂(S)²
    """
    size = len(fourier_coeffs)
    squared = fourier_coeffs ** 2
    
    # Compute |S| for each subset S (popcount)
    subset_sizes = np.array([bin(s).count('1') for s in range(size)], dtype=np.float64)
    
    return np.dot(subset_sizes, squared)


if HAS_NUMBA:
    @njit(cache=True)
    def _popcount(x: int) -> int:
        """Count set bits in integer."""
        count = 0
        while x:
            count += 1
            x &= x - 1
        return count
    
    @njit(parallel=True, cache=True)
    def _total_influence_numba(fourier_coeffs: np.ndarray) -> float:
        """Numba-accelerated total influence."""
        size = len(fourier_coeffs)
        total = 0.0
        for s in prange(size):
            total += _popcount(s) * fourier_coeffs[s] ** 2
        return total
    
    def vectorized_total_influence_numba(fourier_coeffs: np.ndarray) -> float:
        """Use Numba-accelerated total influence if available."""
        return _total_influence_numba(fourier_coeffs.astype(np.float64))


def noise_stability_from_fourier(fourier_coeffs: np.ndarray, rho: float) -> float:
    """
    Compute noise stability from Fourier coefficients.
    
    Stab_ρ[f] = Σ_S ρ^|S| · f̂(S)²
    """
    size = len(fourier_coeffs)
    squared = fourier_coeffs ** 2
    
    # Compute ρ^|S| for each subset
    subset_sizes = np.array([bin(s).count('1') for s in range(size)])
    rho_powers = rho ** subset_sizes
    
    return np.dot(rho_powers, squared)


class LazyFourierCoefficients:
    """
    Lazy wrapper for Fourier coefficients.
    
    Delays computation until coefficients are actually needed,
    and caches the result.
    """
    
    def __init__(self, compute_func: Callable[[], np.ndarray]):
        """
        Args:
            compute_func: Function that computes the Fourier coefficients
        """
        self._compute_func = compute_func
        self._coeffs: Optional[np.ndarray] = None
        self._computed = False
    
    def get(self) -> np.ndarray:
        """Get coefficients, computing if necessary."""
        if not self._computed:
            self._coeffs = self._compute_func()
            self._computed = True
        return self._coeffs
    
    def is_computed(self) -> bool:
        """Check if coefficients have been computed."""
        return self._computed
    
    def clear(self):
        """Clear cached coefficients."""
        self._coeffs = None
        self._computed = False


def get_best_wht_implementation():
    """
    Get the best available Walsh-Hadamard Transform implementation.
    
    Returns tuple of (function, name) where function is the WHT implementation
    and name is a description.
    """
    # Try GPU-accelerated pyfwht first
    try:
        from pyfwht import fwht
        def pyfwht_wrapper(values):
            return fwht(values.astype(np.float64)) / len(values)
        return pyfwht_wrapper, "pyfwht (GPU-accelerated)"
    except ImportError:
        pass
    
    # Try Numba-accelerated version
    if HAS_NUMBA:
        return fast_walsh_hadamard_numba, "Numba JIT-compiled"
    
    # Fall back to pure NumPy
    return fast_walsh_hadamard, "NumPy (pure Python)"


# Export best implementations
BEST_WHT, WHT_BACKEND = get_best_wht_implementation()

# Choose best influence computation
if HAS_NUMBA:
    BEST_INFLUENCES = vectorized_influences_numba
    INFLUENCES_BACKEND = "Numba"
else:
    BEST_INFLUENCES = vectorized_influences_from_fourier
    INFLUENCES_BACKEND = "NumPy"


# =============================================================================
# Parallel Computation Support
# =============================================================================

try:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import multiprocessing
    HAS_PARALLEL = True
    MAX_WORKERS = multiprocessing.cpu_count()
except ImportError:
    HAS_PARALLEL = False
    MAX_WORKERS = 1


def parallel_batch_influences(
    functions: list,
    max_workers: Optional[int] = None,
    use_threads: bool = True
) -> list:
    """
    Compute influences for multiple Boolean functions in parallel.
    
    Args:
        functions: List of BooleanFunction objects
        max_workers: Number of parallel workers (default: CPU count)
        use_threads: Use threads instead of processes (faster for small tasks)
        
    Returns:
        List of influence arrays
    """
    if max_workers is None:
        max_workers = min(MAX_WORKERS, len(functions))
    
    if not HAS_PARALLEL or len(functions) <= 1:
        return [f.influences() for f in functions]
    
    def compute_influences(f):
        return f.influences()
    
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    with ExecutorClass(max_workers=max_workers) as executor:
        results = list(executor.map(compute_influences, functions))
    
    return results


def parallel_batch_fourier(
    functions: list,
    max_workers: Optional[int] = None
) -> list:
    """
    Compute Fourier coefficients for multiple Boolean functions in parallel.
    
    Args:
        functions: List of BooleanFunction objects
        max_workers: Number of parallel workers
        
    Returns:
        List of Fourier coefficient arrays
    """
    if max_workers is None:
        max_workers = min(MAX_WORKERS, len(functions))
    
    if not HAS_PARALLEL or len(functions) <= 1:
        return [f.fourier() for f in functions]
    
    def compute_fourier(f):
        return f.fourier()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_fourier, functions))
    
    return results


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _parallel_sensitivity_batch(truth_tables: np.ndarray) -> np.ndarray:
        """
        Compute sensitivity for multiple functions in parallel.
        
        Args:
            truth_tables: 2D array where each row is a truth table
            
        Returns:
            Array of sensitivity values
        """
        num_funcs = truth_tables.shape[0]
        size = truth_tables.shape[1]
        n = 0
        temp = size
        while temp > 1:
            temp //= 2
            n += 1
        
        sensitivities = np.zeros(num_funcs, dtype=np.float64)
        
        for f_idx in prange(num_funcs):
            max_sens = 0
            for x in range(size):
                sens = 0
                for i in range(n):
                    neighbor = x ^ (1 << i)
                    if truth_tables[f_idx, x] != truth_tables[f_idx, neighbor]:
                        sens += 1
                if sens > max_sens:
                    max_sens = sens
            sensitivities[f_idx] = max_sens
        
        return sensitivities


# =============================================================================
# Aggressive Memoization System
# =============================================================================

import functools
import hashlib
from typing import Dict, Tuple
import weakref


class ComputeCache:
    """
    LRU cache for expensive Boolean function computations.
    
    Caches results keyed by function hash and computation type.
    Automatically evicts least-recently-used entries when full.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: list = []
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, func_hash: str, computation: str, *args) -> str:
        """Create cache key from function hash and computation."""
        args_str = "_".join(str(a) for a in args)
        return f"{func_hash}_{computation}_{args_str}"
    
    def get(self, func_hash: str, computation: str, *args) -> Tuple[bool, Any]:
        """
        Try to get cached result.
        
        Returns:
            Tuple of (found, value)
        """
        key = self._make_key(func_hash, computation, *args)
        
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return (True, self._cache[key])
        
        self._misses += 1
        return (False, None)
    
    def put(self, func_hash: str, computation: str, value: Any, *args):
        """Store result in cache."""
        key = self._make_key(func_hash, computation, *args)
        
        # Evict if necessary
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0
        }


# Global compute cache
_GLOBAL_CACHE = ComputeCache(max_size=500)


def get_global_cache() -> ComputeCache:
    """Get the global computation cache."""
    return _GLOBAL_CACHE


def cached_computation(computation_name: str):
    """
    Decorator for caching expensive computations on BooleanFunction.
    
    Usage:
        @cached_computation("influences")
        def compute_influences(self):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get function hash
            func_hash = getattr(self, '_cache_hash', None)
            if func_hash is None:
                # Compute hash from truth table
                tt = self.get_representation("truth_table")
                func_hash = hashlib.md5(tt.tobytes()).hexdigest()[:16]
                self._cache_hash = func_hash
            
            # Check cache
            found, value = _GLOBAL_CACHE.get(func_hash, computation_name, *args)
            if found:
                return value
            
            # Compute and cache
            result = func(self, *args, **kwargs)
            _GLOBAL_CACHE.put(func_hash, computation_name, result, *args)
            return result
        
        return wrapper
    return decorator


def memoize_method(func):
    """
    Simple memoization for instance methods.
    
    Caches results in the instance's __dict__.
    """
    cache_name = f"_memo_{func.__name__}"
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create cache if needed
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})
        
        cache = getattr(self, cache_name)
        
        # Make key from args
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
        
        return cache[key]
    
    return wrapper


# =============================================================================
# Batch Operations
# =============================================================================

def batch_evaluate(f, inputs: np.ndarray) -> np.ndarray:
    """
    Efficiently evaluate a Boolean function on a batch of inputs.
    
    Args:
        f: BooleanFunction
        inputs: 2D array of shape (batch_size, n_vars) or 1D array of indices
        
    Returns:
        Boolean array of results
    """
    if inputs.ndim == 1:
        # Array of indices
        tt = f.get_representation("truth_table")
        return tt[inputs].astype(bool)
    else:
        # Array of bit vectors
        n = f.n_vars
        # Convert each row to index
        powers = 2 ** np.arange(n - 1, -1, -1)
        indices = np.dot(inputs.astype(int), powers)
        tt = f.get_representation("truth_table")
        return tt[indices].astype(bool)


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _batch_evaluate_numba(truth_table: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch evaluation."""
        result = np.empty(len(indices), dtype=np.bool_)
        for i in prange(len(indices)):
            result[i] = truth_table[indices[i]]
        return result
