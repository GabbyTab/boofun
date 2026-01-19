"""
GPU Acceleration for Boolean Function Operations.

This module provides GPU-accelerated implementations of computationally
intensive operations using CuPy. Falls back to NumPy when CuPy is unavailable.

Key accelerated operations:
- Walsh-Hadamard Transform (WHT)
- Influence computation
- Noise stability computation
- Large truth table operations

Usage:
    from boolfunc.core.gpu import gpu_wht, is_gpu_available
    
    if is_gpu_available():
        fourier = gpu_wht(truth_table_pm)
    else:
        # Fallback to CPU
        fourier = cpu_wht(truth_table_pm)
"""

import warnings
import numpy as np
from typing import Union, Optional

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Module-level flag to enable/disable GPU
_GPU_ENABLED = CUPY_AVAILABLE


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return CUPY_AVAILABLE


def is_gpu_enabled() -> bool:
    """Check if GPU acceleration is currently enabled."""
    return _GPU_ENABLED and CUPY_AVAILABLE


def enable_gpu(enable: bool = True) -> None:
    """Enable or disable GPU acceleration."""
    global _GPU_ENABLED
    if enable and not CUPY_AVAILABLE:
        warnings.warn("CuPy not available - GPU acceleration cannot be enabled")
        return
    _GPU_ENABLED = enable


def get_array_module(arr: Union[np.ndarray, "cp.ndarray"]) -> type:
    """Get the array module (numpy or cupy) for the given array."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_gpu(arr: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
    """Move array to GPU if available and enabled."""
    if is_gpu_enabled():
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
    """Move array to CPU."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def gpu_walsh_hadamard(values: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    GPU-accelerated Walsh-Hadamard Transform.
    
    Computes the WHT in O(n * 2^n) time using GPU parallelism.
    
    Args:
        values: Array of 2^n values in ±1 representation
        in_place: If True, modify input array (saves memory)
        
    Returns:
        WHT result (unnormalized, on CPU)
    """
    # Always use the CPU implementation for now
    # GPU implementation requires careful CUDA kernel design for efficiency
    from .optimizations import fast_walsh_hadamard
    if in_place:
        return fast_walsh_hadamard(values)
    return fast_walsh_hadamard(values.copy())


def gpu_influences(fourier_coeffs: np.ndarray, n_vars: int = None) -> np.ndarray:
    """
    GPU-accelerated influence computation from Fourier coefficients.
    
    Inf_i[f] = Σ_{S∋i} f̂(S)²
    
    Args:
        fourier_coeffs: Fourier coefficients array
        n_vars: Number of variables (inferred from array size if not provided)
        
    Returns:
        Array of influences (one per variable)
    """
    size = len(fourier_coeffs)
    if n_vars is None:
        n_vars = int(np.log2(size))
    
    if not is_gpu_enabled():
        # Fallback to CPU
        from .optimizations import vectorized_influences_from_fourier
        return vectorized_influences_from_fourier(fourier_coeffs, n_vars)
    
    # Move to GPU
    d_coeffs = cp.asarray(fourier_coeffs, dtype=cp.float64)
    d_squared = d_coeffs ** 2
    
    # Compute influences using GPU parallelism
    d_influences = cp.zeros(n_vars, dtype=cp.float64)
    
    for i in range(n_vars):
        # Create mask for subsets containing variable i
        mask = (cp.arange(size) >> i) & 1
        d_influences[i] = cp.sum(d_squared * mask)
    
    return cp.asnumpy(d_influences)


def gpu_noise_stability(fourier_coeffs: np.ndarray, rho: float) -> float:
    """
    GPU-accelerated noise stability computation.
    
    Stab_ρ[f] = Σ_S ρ^|S| · f̂(S)²
    
    Args:
        fourier_coeffs: Fourier coefficients array
        rho: Correlation parameter
        
    Returns:
        Noise stability value
    """
    if not is_gpu_enabled():
        from .optimizations import noise_stability_from_fourier
        return noise_stability_from_fourier(fourier_coeffs, rho)
    
    size = len(fourier_coeffs)
    
    # Move to GPU
    d_coeffs = cp.asarray(fourier_coeffs, dtype=cp.float64)
    d_squared = d_coeffs ** 2
    
    # Compute subset sizes using popcount
    d_indices = cp.arange(size)
    # CuPy doesn't have built-in popcount, use a workaround
    d_sizes = cp.zeros(size, dtype=cp.int32)
    temp = d_indices.copy()
    while cp.any(temp > 0):
        d_sizes += (temp & 1).astype(cp.int32)
        temp >>= 1
    
    # Compute ρ^|S| for each subset
    d_rho_powers = cp.power(rho, d_sizes.astype(cp.float64))
    
    # Dot product
    result = float(cp.dot(d_rho_powers, d_squared))
    
    return result


def gpu_spectral_weight_by_degree(fourier_coeffs: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated spectral weight computation by degree.
    
    W^{=k}[f] = Σ_{|S|=k} f̂(S)²
    
    Args:
        fourier_coeffs: Fourier coefficients array
        
    Returns:
        Array of weights W^{=0}, W^{=1}, ..., W^{=n}
    """
    size = len(fourier_coeffs)
    n = int(np.log2(size))
    
    if not is_gpu_enabled():
        # CPU fallback
        weights = np.zeros(n + 1)
        for s in range(size):
            k = bin(s).count('1')
            weights[k] += fourier_coeffs[s] ** 2
        return weights
    
    # Move to GPU
    d_coeffs = cp.asarray(fourier_coeffs, dtype=cp.float64)
    d_squared = d_coeffs ** 2
    
    # Compute subset sizes
    d_indices = cp.arange(size)
    d_sizes = cp.zeros(size, dtype=cp.int32)
    temp = d_indices.copy()
    while cp.any(temp > 0):
        d_sizes += (temp & 1).astype(cp.int32)
        temp >>= 1
    
    # Sum by degree
    d_weights = cp.zeros(n + 1, dtype=cp.float64)
    for k in range(n + 1):
        mask = (d_sizes == k)
        d_weights[k] = cp.sum(d_squared * mask)
    
    return cp.asnumpy(d_weights)


class GPUBooleanFunctionOps:
    """
    GPU-accelerated operations for Boolean functions.
    
    Provides a high-level interface for GPU acceleration.
    """
    
    def __init__(self, truth_table: np.ndarray):
        """
        Initialize with a truth table.
        
        Args:
            truth_table: Boolean truth table (0/1 values)
        """
        self.truth_table = np.asarray(truth_table)
        self.n = int(np.log2(len(self.truth_table)))
        self._fourier_cache: Optional[np.ndarray] = None
    
    @property
    def pm_values(self) -> np.ndarray:
        """Convert to ±1 representation.
        
        Convention: True (1) → +1, False (0) → -1
        This matches the library's SpectralAnalyzer convention.
        """
        return 2.0 * self.truth_table.astype(float) - 1.0
    
    def fourier(self) -> np.ndarray:
        """Compute Fourier coefficients using GPU if available."""
        if self._fourier_cache is None:
            pm = self.pm_values
            # fast_walsh_hadamard already normalizes by default
            self._fourier_cache = gpu_walsh_hadamard(pm)
        return self._fourier_cache
    
    def influences(self) -> np.ndarray:
        """Compute influences using GPU if available."""
        return gpu_influences(self.fourier())
    
    def total_influence(self) -> float:
        """Compute total influence."""
        return float(np.sum(self.influences()))
    
    def noise_stability(self, rho: float) -> float:
        """Compute noise stability using GPU if available."""
        return gpu_noise_stability(self.fourier(), rho)
    
    def spectral_weights(self) -> np.ndarray:
        """Compute spectral weights by degree using GPU if available."""
        return gpu_spectral_weight_by_degree(self.fourier())


# Convenience function for automatic GPU usage
def auto_accelerate(func):
    """
    Decorator to automatically use GPU when beneficial.
    
    Uses GPU for arrays larger than a threshold (2^14 = 16384 elements).
    """
    threshold = 2 ** 14
    
    def wrapper(arr, *args, **kwargs):
        if is_gpu_enabled() and len(arr) >= threshold:
            # Use GPU
            result = func(to_gpu(arr), *args, **kwargs)
            return to_cpu(result) if hasattr(result, '__len__') else result
        else:
            # Use CPU
            return func(arr, *args, **kwargs)
    
    return wrapper


# Export GPU status at module load
if CUPY_AVAILABLE:
    try:
        # Test GPU access
        _test = cp.zeros(1)
        del _test
    except Exception as e:
        CUPY_AVAILABLE = False
        _GPU_ENABLED = False
        warnings.warn(f"CuPy installed but GPU not accessible: {e}")
