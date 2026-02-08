"""
GPU acceleration for Boolean function operations.

Provides CuPy-accelerated implementations of computationally intensive
operations with automatic fallback to NumPy on CPU.

Accelerated operations:
- Walsh-Hadamard Transform (WHT)
- Influence computation from Fourier coefficients
- Noise stability computation
- Spectral weight by degree
- Batch truth table / Fourier evaluation

Usage::

    from boofun.core.gpu import is_gpu_available, gpu_walsh_hadamard

    if is_gpu_available():
        fourier = gpu_walsh_hadamard(truth_table_pm)

Install CuPy for GPU support::

    pip install cupy-cuda12x   # adjust for your CUDA version

.. note::
    This module was consolidated from ``gpu.py`` and ``gpu_acceleration.py``
    in v1.3.0.  The old ``gpu_acceleration`` module is removed; all public
    names are available here.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# CuPy import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False

# Runtime toggle
_GPU_ENABLED = CUPY_AVAILABLE


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available (CuPy + working CUDA)."""
    return CUPY_AVAILABLE


def is_gpu_enabled() -> bool:
    """Check if GPU acceleration is currently enabled."""
    return _GPU_ENABLED and CUPY_AVAILABLE


def enable_gpu(enable: bool = True) -> None:
    """Enable or disable GPU acceleration at runtime."""
    global _GPU_ENABLED
    if enable and not CUPY_AVAILABLE:
        warnings.warn("CuPy not available -- GPU acceleration cannot be enabled")
        return
    _GPU_ENABLED = enable


def get_gpu_info() -> Dict[str, Any]:
    """Return information about available GPU resources."""
    info: Dict[str, Any] = {
        "gpu_available": CUPY_AVAILABLE,
        "gpu_enabled": is_gpu_enabled(),
        "backend": "cupy" if CUPY_AVAILABLE else None,
        "devices": [],
    }
    if CUPY_AVAILABLE:
        try:
            count = cp.cuda.runtime.getDeviceCount()
            for i in range(count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                info["devices"].append(
                    {
                        "id": i,
                        "name": props["name"].decode("utf-8"),
                        "memory_gb": props["totalGlobalMem"] / (1024**3),
                        "compute_capability": f"{props['major']}.{props['minor']}",
                    }
                )
        except Exception:
            pass
    return info


def should_use_gpu(operation: str, data_size: int, n_vars: int) -> bool:
    """
    Heuristic: should we use GPU for this operation?

    Args:
        operation: One of 'truth_table', 'fourier', 'walsh_hadamard', 'wht',
                   'influences', etc.
        data_size: Number of elements in the input.
        n_vars: Number of Boolean variables.

    Returns:
        True if GPU acceleration is recommended.
    """
    if not is_gpu_enabled():
        return False

    if operation in ("truth_table", "truth_table_batch"):
        return data_size > 10_000
    if operation == "fourier":
        return data_size > 5_000 or (2**n_vars) > 1_000
    if operation in ("walsh_hadamard", "wht"):
        return n_vars > 10 or data_size > 2_000
    # Default: GPU for large data
    return data_size > 10_000


# ---------------------------------------------------------------------------
# Array transfer helpers
# ---------------------------------------------------------------------------


def get_array_module(arr: Union[np.ndarray, Any]) -> Any:
    """Get the array module (numpy or cupy) for the given array."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np


def to_gpu(arr: np.ndarray) -> Union[np.ndarray, Any]:
    """Move *arr* to GPU memory if available and enabled."""
    if is_gpu_enabled():
        return cp.asarray(arr)
    return arr


def to_cpu(arr: Union[np.ndarray, Any]) -> np.ndarray:
    """Move *arr* to CPU memory."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Core accelerated operations
# ---------------------------------------------------------------------------


def gpu_walsh_hadamard(values: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Walsh-Hadamard Transform with optional GPU acceleration.

    When CuPy is available and enabled the iterative butterfly is run on GPU;
    otherwise delegates to the CPU ``fast_walsh_hadamard`` in *optimizations*.

    Args:
        values: Array of 2^n values in +/-1 representation.
        in_place: Modify *values* in place (saves memory on CPU path).

    Returns:
        WHT result (normalised, on CPU).
    """
    if is_gpu_enabled():
        return _gpu_wht_cupy(values)

    from .optimizations import fast_walsh_hadamard

    if in_place:
        return fast_walsh_hadamard(values)
    return fast_walsh_hadamard(values.copy())


def _gpu_wht_cupy(values: np.ndarray) -> np.ndarray:
    """Iterative Walsh-Hadamard butterfly on GPU via CuPy."""
    n_vars = int(np.log2(len(values)))
    d = cp.asarray(values, dtype=cp.float64)

    for i in range(n_vars):
        step = 1 << i
        for j in range(0, len(d), step * 2):
            u = d[j : j + step].copy()
            v = d[j + step : j + 2 * step].copy()
            d[j : j + step] = u + v
            d[j + step : j + 2 * step] = u - v

    d /= len(values)
    return cp.asnumpy(d)


def gpu_influences(fourier_coeffs: np.ndarray, n_vars: Optional[int] = None) -> np.ndarray:
    """
    Influence computation: Inf_i[f] = sum_{S containing i} f_hat(S)^2.

    Uses GPU when enabled, otherwise falls back to CPU vectorised code.
    """
    size = len(fourier_coeffs)
    if n_vars is None:
        n_vars = int(np.log2(size))

    if not is_gpu_enabled():
        from .optimizations import vectorized_influences_from_fourier

        return vectorized_influences_from_fourier(fourier_coeffs, n_vars)

    d_squared = cp.asarray(fourier_coeffs, dtype=cp.float64) ** 2
    influences = cp.zeros(n_vars, dtype=cp.float64)
    indices = cp.arange(size)

    for i in range(n_vars):
        mask = (indices >> i) & 1
        influences[i] = cp.sum(d_squared * mask)

    return cp.asnumpy(influences)


def gpu_noise_stability(fourier_coeffs: np.ndarray, rho: float) -> float:
    """
    Noise stability: Stab_rho[f] = sum_S rho^|S| f_hat(S)^2.

    Uses GPU when enabled, otherwise falls back to CPU.
    """
    if not is_gpu_enabled():
        from .optimizations import noise_stability_from_fourier

        return noise_stability_from_fourier(fourier_coeffs, rho)

    size = len(fourier_coeffs)
    d_squared = cp.asarray(fourier_coeffs, dtype=cp.float64) ** 2

    # Popcount via bit-stripping
    indices = cp.arange(size, dtype=cp.int64)
    sizes = cp.zeros(size, dtype=cp.int32)
    temp = indices.copy()
    while cp.any(temp > 0):
        sizes += (temp & 1).astype(cp.int32)
        temp >>= 1

    rho_powers = cp.power(float(rho), sizes.astype(cp.float64))
    return float(cp.dot(rho_powers, d_squared))


def gpu_spectral_weight_by_degree(fourier_coeffs: np.ndarray) -> np.ndarray:
    """
    Spectral weight by degree: W^{=k}[f] = sum_{|S|=k} f_hat(S)^2.

    Uses GPU when enabled, otherwise falls back to CPU.
    """
    size = len(fourier_coeffs)
    n = int(np.log2(size))

    if not is_gpu_enabled():
        weights = np.zeros(n + 1)
        for s in range(size):
            k = bin(s).count("1")
            weights[k] += fourier_coeffs[s] ** 2
        return weights

    d_squared = cp.asarray(fourier_coeffs, dtype=cp.float64) ** 2
    indices = cp.arange(size, dtype=cp.int64)
    sizes = cp.zeros(size, dtype=cp.int32)
    temp = indices.copy()
    while cp.any(temp > 0):
        sizes += (temp & 1).astype(cp.int32)
        temp >>= 1

    weights = cp.zeros(n + 1, dtype=cp.float64)
    for k in range(n + 1):
        weights[k] = cp.sum(d_squared[sizes == k])

    return cp.asnumpy(weights)


# ---------------------------------------------------------------------------
# Batch evaluation (migrated from gpu_acceleration.py)
# ---------------------------------------------------------------------------


def gpu_accelerate(operation: str, *args: Any, **kwargs: Any) -> np.ndarray:
    """
    Run a named operation on GPU if available, otherwise raise.

    Supported operations:
    - ``truth_table_batch``: args = (inputs, truth_table)
    - ``fourier_batch``: args = (inputs, coefficients)
    - ``walsh_hadamard``: args = (function_values,)
    """
    if not is_gpu_enabled():
        raise RuntimeError("GPU acceleration is not available")

    if operation == "truth_table_batch":
        inputs, truth_table = args[:2]
        d_inputs = cp.asarray(inputs)
        d_tt = cp.asarray(truth_table)
        return cp.asnumpy(d_tt[d_inputs])

    if operation == "fourier_batch":
        inputs, coefficients = args[:2]
        # Delegate to CPU -- the custom CUDA kernel is fragile and the
        # element-wise Python fallback is slower than NumPy.  GPU Fourier
        # batch evaluation can be added back when a proper kernel is tested.
        from .representations.fourier_expansion import FourierExpansionRepresentation

        rep = FourierExpansionRepresentation()
        return rep._evaluate_batch(inputs, coefficients)

    if operation == "walsh_hadamard":
        return gpu_walsh_hadamard(args[0])

    raise ValueError(f"Unknown GPU operation: {operation}")


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


class GPUBooleanFunctionOps:
    """
    GPU-accelerated operations for a single Boolean function.

    Wraps a truth table and caches the Fourier transform.
    """

    def __init__(self, truth_table: np.ndarray) -> None:
        self.truth_table = np.asarray(truth_table)
        self.n = int(np.log2(len(self.truth_table)))
        self._fourier_cache: Optional[np.ndarray] = None

    @property
    def pm_values(self) -> np.ndarray:
        """±1 representation (O'Donnell convention: 0->+1, 1->-1)."""
        return 1.0 - 2.0 * self.truth_table.astype(float)

    def fourier(self) -> np.ndarray:
        if self._fourier_cache is None:
            self._fourier_cache = gpu_walsh_hadamard(self.pm_values)
        return self._fourier_cache

    def influences(self) -> np.ndarray:
        return gpu_influences(self.fourier())

    def total_influence(self) -> float:
        return float(np.sum(self.influences()))

    def noise_stability(self, rho: float) -> float:
        return gpu_noise_stability(self.fourier(), rho)

    def spectral_weights(self) -> np.ndarray:
        return gpu_spectral_weight_by_degree(self.fourier())


# ---------------------------------------------------------------------------
# Convenience decorator
# ---------------------------------------------------------------------------


def auto_accelerate(func):  # type: ignore[no-untyped-def]
    """Decorator: route to GPU for arrays >= 2^14 elements."""
    threshold = 2**14

    def wrapper(arr, *args, **kwargs):  # type: ignore[no-untyped-def]
        if is_gpu_enabled() and len(arr) >= threshold:
            result = func(to_gpu(arr), *args, **kwargs)
            return to_cpu(result) if hasattr(result, "__len__") else result
        return func(arr, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Validate GPU access at import time
# ---------------------------------------------------------------------------
if CUPY_AVAILABLE:
    try:
        _test = cp.zeros(1)
        del _test
    except Exception as e:
        CUPY_AVAILABLE = False
        _GPU_ENABLED = False
        warnings.warn(f"CuPy installed but GPU not accessible: {e}")
