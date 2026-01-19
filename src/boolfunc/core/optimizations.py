"""
Performance Optimizations for BoolFunc

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
