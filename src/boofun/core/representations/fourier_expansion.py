from typing import Any, Dict

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


@register_strategy("fourier_expansion")
class FourierExpansionRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Fourier expansion representation of Boolean functions"""

    def evaluate(
        self, inputs: np.ndarray, data: np.ndarray, space: Space, n_vars: int
    ) -> np.ndarray:
        """
        Evaluate the Fourier expansion at given inputs.

        Args:
            inputs: Binary input vectors (shape: (m, n) or (n,))
            data: Fourier coefficients (1D array of length 2**n)

        Returns:
            Fourier expansion values (real numbers)
        """

        # Convert to ±1 domain

        if not isinstance(inputs, np.ndarray):
            return self._evaluate_single(inputs, data)  # type: ignore[unreachable]
        return self._evaluate_batch(inputs, data)

    def _evaluate_single(self, x: int, coeffs: np.ndarray) -> float:
        """Evaluate f(x) given x as an integer bitstring, and coeffs of Fourier expansion over {0,1}^n"""
        result = 0.0
        for j in range(len(coeffs)):
            parity = bin(x & j).count("1") % 2
            char_val = (-1) ** parity
            result += coeffs[j] * char_val
        return result

    def _evaluate_batch(self, X: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate a batch of inputs X, where each x is an integer bitstring"""
        results = np.zeros(len(X))
        for idx, x in enumerate(X):
            results[idx] = self._evaluate_single(x, coeffs)
        return results

    def dump(self, data: np.ndarray, space=None, **kwargs) -> Dict[str, Any]:
        """Export Fourier coefficients in serializable format"""
        return {
            "coefficients": data.tolist(),
            "type": "fourier_expansion",
            "metadata": {
                "num_vars": int(np.log2(len(data))),
                "norm": float(np.linalg.norm(data)),
            },
        }

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute Fourier coefficients from any evaluable Boolean representation.
        Fourier basis: {(-1)^{x·s}} for s ⊆ [n]
        """

        size = 1 << n_vars  # 2^n

        # Evaluate function f on all inputs
        x_indices = np.arange(size, dtype=int)
        f_vals = source_repr.evaluate(x_indices, source_data, space, n_vars)
        f_vals = np.asarray(f_vals, dtype=float)

        # Map {0,1} to {+1,-1} (O'Donnell convention: 0->+1, 1->-1)
        pm_vals = 1.0 - 2.0 * f_vals

        # Use the Walsh-Hadamard Transform in O(n * 2^n) instead of
        # the naive O(4^n) double loop.
        try:
            from ..optimizations import fast_walsh_hadamard

            coeffs = fast_walsh_hadamard(pm_vals.copy())
        except ImportError:
            # Fallback: in-place butterfly WHT
            coeffs = pm_vals.copy()
            h = 1
            while h < size:
                for i in range(0, size, h * 2):
                    for j in range(i, i + h):
                        x = coeffs[j]
                        y = coeffs[j + h]
                        coeffs[j] = x + y
                        coeffs[j + h] = x - y
                h *= 2
            coeffs /= size

        return coeffs

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert to another representation from Fourier expansion"""
        # Placeholder: Actual conversion requires inverse transform
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """Create zero-initialized Fourier coefficients array"""
        return np.zeros(2**n_vars, dtype=float)

    def is_complete(self, data: np.ndarray) -> bool:
        """Check if representation contains non-zero coefficients"""
        return bool(np.any(data != 0))

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity estimates for Fourier operations."""
        size = 2**n_vars
        return {
            "evaluation": n_vars * size,  # O(n * 2^n) for single evaluation
            "batch_evaluation": size,  # O(2^n) per input in batch
            "fft_computation": n_vars * size,  # O(n * 2^n) for Walsh-Hadamard transform
            "coefficient_access": 1,  # O(1) for accessing coefficients
            "creation_from_tt": n_vars * size,  # FFT complexity
            "storage": size,  # O(2^n) storage
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, Any]:
        """Return memory requirements for n variables"""
        num_coeffs = 2**n_vars
        return {
            "dtype": "float64",
            "elements": num_coeffs,
            "bytes": num_coeffs * 8,  # 8 bytes per float
            "human_readable": (
                f"{num_coeffs * 8 / 1024:.2f} KB"
                if num_coeffs > 1024
                else f"{num_coeffs * 8} bytes"
            ),
        }
