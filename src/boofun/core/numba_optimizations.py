"""
Numba JIT optimizations for Boolean function operations.

This module provides JIT-compiled versions of critical operations for
significant performance improvements in compute-intensive scenarios.
"""

import warnings
from typing import Any, Dict

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available - optimizations disabled")


# JIT-compiled utility functions
if HAS_NUMBA:

    @njit
    def popcount(x):
        """Count number of set bits in integer (population count)."""
        count = 0
        while x:
            count += 1
            x &= x - 1  # Clear lowest set bit
        return count

    @njit
    def binary_to_index(binary_vec):
        """Convert binary vector to integer index (LSB convention: vec[i] = x_i)."""
        result = 0
        for i in range(len(binary_vec)):
            if binary_vec[i]:
                result += 1 << i
        return result

    @njit(parallel=True)
    def batch_binary_to_indices(binary_matrix):
        """Convert batch of binary vectors to indices (LSB convention)."""
        n_inputs, n_vars = binary_matrix.shape
        indices = np.zeros(n_inputs, dtype=np.int32)

        for i in prange(n_inputs):
            result = 0
            for j in range(n_vars):
                if binary_matrix[i, j]:
                    result += 1 << j
            indices[i] = result

        return indices

    @njit(parallel=True)
    def truth_table_batch_eval(inputs, truth_table):
        """JIT-compiled batch truth table evaluation."""
        results = np.zeros(len(inputs), dtype=np.bool_)

        for i in prange(len(inputs)):
            idx = inputs[i]
            if 0 <= idx < len(truth_table):
                results[i] = truth_table[idx]

        return results

    @njit(parallel=True)
    def fourier_batch_eval(inputs, coefficients):
        """JIT-compiled batch Fourier expansion evaluation."""
        results = np.zeros(len(inputs), dtype=np.bool_)

        for i in prange(len(inputs)):
            x = inputs[i]
            result = 0.0

            for j in range(len(coefficients)):
                # Compute parity of x & j
                parity = popcount(x & j) & 1
                char_val = -1.0 if parity else 1.0
                result += coefficients[j] * char_val

            results[i] = result > 0.0

        return results

    @njit
    def walsh_hadamard_transform_inplace(values):
        """In-place Walsh-Hadamard transform."""
        n = len(values)
        n_vars = 0
        temp = n
        while temp > 1:
            temp //= 2
            n_vars += 1

        # Iterative Walsh-Hadamard transform
        for i in range(n_vars):
            step = 1 << i  # 2^i
            for j in range(0, n, step * 2):
                for k in range(step):
                    if j + k + step < n:
                        u = values[j + k]
                        v = values[j + k + step]
                        values[j + k] = u + v
                        values[j + k + step] = u - v

        # Normalize
        for i in range(n):
            values[i] /= n

    @njit(parallel=True)
    def anf_batch_eval(inputs, monomial_arrays, monomial_lengths, coefficients):
        """JIT-compiled batch ANF evaluation."""
        results = np.zeros(len(inputs), dtype=np.bool_)

        for i in prange(len(inputs)):
            x = inputs[i]
            result = 0

            # Convert integer to binary representation
            binary = np.zeros(32, dtype=np.bool_)  # Assume max 32 variables
            temp = x
            for bit in range(32):
                binary[bit] = (temp & 1) == 1
                temp >>= 1

            # Evaluate each monomial
            for mono_idx in range(len(coefficients)):
                if coefficients[mono_idx] == 0:
                    continue

                # Compute monomial value
                monomial_val = 1
                mono_len = monomial_lengths[mono_idx]
                for var_idx in range(mono_len):
                    var = monomial_arrays[mono_idx, var_idx]
                    if var < 32:  # Bounds check
                        monomial_val *= binary[var]

                result ^= coefficients[mono_idx] * monomial_val

            results[i] = bool(result)

        return results

    @njit
    def influences_computation(truth_table, n_vars):
        """JIT-compiled influence computation."""
        influences = np.zeros(n_vars, dtype=np.float64)
        size = len(truth_table)

        for var in range(n_vars):
            disagreements = 0

            for x in range(size):
                # Flip the var-th bit (LSB=x₀ convention)
                x_flipped = x ^ (1 << var)

                if x_flipped < size:  # Bounds check
                    if truth_table[x] != truth_table[x_flipped]:
                        disagreements += 1

            influences[var] = disagreements / size

        return influences

    @njit
    def noise_stability_computation(fourier_coeffs, rho):
        """JIT-compiled noise stability computation."""
        stability = 0.0

        for s in range(len(fourier_coeffs)):
            # Count bits in s to get subset size
            subset_size = popcount(s)
            stability += fourier_coeffs[s] ** 2 * (rho**subset_size)

        return stability

    @njit(parallel=True)
    def polynomial_batch_eval(inputs, monomial_powers, coefficients):
        """JIT-compiled polynomial evaluation."""
        results = np.zeros(len(inputs), dtype=np.float64)

        for i in prange(len(inputs)):
            x = inputs[i]
            result = 0.0

            # Convert to binary for polynomial evaluation
            binary = np.zeros(32, dtype=np.int32)
            temp = x
            for bit in range(32):
                binary[bit] = temp & 1
                temp >>= 1

            # Evaluate polynomial terms
            for term_idx in range(len(coefficients)):
                if coefficients[term_idx] == 0:
                    continue

                term_value = 1.0
                powers = monomial_powers[term_idx]

                for var in range(len(powers)):
                    if powers[var] > 0:
                        term_value *= binary[var] ** powers[var]

                result += coefficients[term_idx] * term_value

            results[i] = result

        return results

    @njit
    def degree_computation(monomial_powers):
        """Compute degree of polynomial representation."""
        max_degree = 0

        for i in range(len(monomial_powers)):
            degree = 0
            for j in range(len(monomial_powers[i])):
                degree += monomial_powers[i][j]
            max_degree = max(max_degree, degree)

        return max_degree


class NumbaOptimizer:
    """
    Manager for Numba JIT optimizations.

    Provides optimized versions of critical Boolean function operations
    with automatic fallback to pure Python/NumPy implementations.
    """

    def __init__(self):
        """Initialize Numba optimizer."""
        self.available = HAS_NUMBA
        self.compiled_functions = {}

        if self.available:
            self._warm_up_functions()

    def _warm_up_functions(self):
        """Warm up JIT-compiled functions with small test cases."""
        try:
            # Test data for warm-up
            test_inputs = np.array([0, 1, 2, 3], dtype=np.int32)
            test_truth_table = np.array([False, True, True, False])
            test_coeffs = np.array([0.5, 0.3, -0.2, 0.1])

            # Warm up functions
            truth_table_batch_eval(test_inputs, test_truth_table)
            fourier_batch_eval(test_inputs, test_coeffs)
            influences_computation(test_truth_table, 2)

            self.compiled_functions["truth_table_batch"] = truth_table_batch_eval
            self.compiled_functions["fourier_batch"] = fourier_batch_eval
            self.compiled_functions["influences"] = influences_computation
            self.compiled_functions["noise_stability"] = noise_stability_computation
            self.compiled_functions["walsh_hadamard"] = walsh_hadamard_transform_inplace

        except Exception as e:
            warnings.warn(f"Numba warm-up failed: {e}")
            self.available = False

    def is_available(self) -> bool:
        """Check if Numba optimization is available."""
        return self.available

    def optimize_truth_table_batch(self, inputs: np.ndarray, truth_table: np.ndarray) -> np.ndarray:
        """Optimized batch truth table evaluation."""
        if not self.available:
            raise RuntimeError("Numba not available")

        return truth_table_batch_eval(inputs.astype(np.int32), truth_table)

    def optimize_fourier_batch(self, inputs: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """Optimized batch Fourier evaluation."""
        if not self.available:
            raise RuntimeError("Numba not available")

        return fourier_batch_eval(inputs.astype(np.int32), coefficients.astype(np.float64))

    def optimize_influences(self, truth_table: np.ndarray, n_vars: int) -> np.ndarray:
        """Optimized influence computation."""
        if not self.available:
            raise RuntimeError("Numba not available")

        return influences_computation(truth_table, n_vars)

    def optimize_noise_stability(self, fourier_coeffs: np.ndarray, rho: float) -> float:
        """Optimized noise stability computation."""
        if not self.available:
            raise RuntimeError("Numba not available")

        return noise_stability_computation(fourier_coeffs, rho)

    def optimize_walsh_hadamard(self, values: np.ndarray) -> np.ndarray:
        """Optimized Walsh-Hadamard transform."""
        if not self.available:
            raise RuntimeError("Numba not available")

        # Make a copy since transform is in-place
        result = values.copy().astype(np.float64)
        walsh_hadamard_transform_inplace(result)
        return result

    def optimize_anf_batch(self, inputs: np.ndarray, anf_data: Dict) -> np.ndarray:
        """Optimized batch ANF evaluation."""
        if not self.available:
            raise RuntimeError("Numba not available")

        # Convert ANF data to Numba-compatible format
        monomials = []
        coeffs = []

        for monomial, coeff in anf_data.items():
            if coeff != 0:
                monomials.append(sorted(list(monomial)))
                coeffs.append(coeff)

        if not monomials:
            return np.zeros(len(inputs), dtype=bool)

        # Pad monomials to same length
        max_len = max(len(m) for m in monomials)
        monomial_arrays = np.zeros((len(monomials), max_len), dtype=np.int32)
        monomial_lengths = np.zeros(len(monomials), dtype=np.int32)

        for i, monomial in enumerate(monomials):
            monomial_lengths[i] = len(monomial)
            for j, var in enumerate(monomial):
                monomial_arrays[i, j] = var

        return anf_batch_eval(
            inputs.astype(np.int32),
            monomial_arrays,
            monomial_lengths,
            np.array(coeffs, dtype=np.int32),
        )

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "numba_available": self.available,
            "compiled_functions": list(self.compiled_functions.keys()),
            "optimization_enabled": self.available and len(self.compiled_functions) > 0,
        }


# Global Numba optimizer instance
_numba_optimizer = NumbaOptimizer()


def is_numba_available() -> bool:
    """Check if Numba optimization is available."""
    return _numba_optimizer.is_available()


def numba_optimize(operation: str, *args, **kwargs) -> Any:
    """
    Apply Numba optimization to an operation.

    Args:
        operation: Operation name
        *args, **kwargs: Operation arguments

    Returns:
        Optimized operation result
    """
    if not _numba_optimizer.is_available():
        raise RuntimeError("Numba optimization not available")

    if operation == "truth_table_batch":
        return _numba_optimizer.optimize_truth_table_batch(*args)
    elif operation == "fourier_batch":
        return _numba_optimizer.optimize_fourier_batch(*args)
    elif operation == "influences":
        return _numba_optimizer.optimize_influences(*args)
    elif operation == "noise_stability":
        return _numba_optimizer.optimize_noise_stability(*args)
    elif operation == "walsh_hadamard":
        return _numba_optimizer.optimize_walsh_hadamard(*args)
    elif operation == "anf_batch":
        return _numba_optimizer.optimize_anf_batch(*args)
    else:
        raise ValueError(f"Unknown optimization: {operation}")


def get_numba_stats() -> Dict[str, Any]:
    """Get Numba optimization statistics."""
    return _numba_optimizer.get_optimization_stats()


# Fallback functions for when Numba is not available
def fallback_truth_table_batch(inputs: np.ndarray, truth_table: np.ndarray) -> np.ndarray:
    """Fallback truth table batch evaluation."""
    results = np.zeros(len(inputs), dtype=bool)
    for i, idx in enumerate(inputs):
        if 0 <= idx < len(truth_table):
            results[i] = truth_table[idx]
    return results


def fallback_fourier_batch(inputs: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Fallback Fourier batch evaluation."""
    results = np.zeros(len(inputs), dtype=bool)

    for i, x in enumerate(inputs):
        result = 0.0
        for j, coeff in enumerate(coefficients):
            parity = bin(x & j).count("1") % 2
            char_val = -1.0 if parity else 1.0
            result += coeff * char_val
        results[i] = result > 0.0

    return results


def fallback_influences(truth_table: np.ndarray, n_vars: int) -> np.ndarray:
    """Fallback influence computation."""
    influences = np.zeros(n_vars)
    size = len(truth_table)

    for var in range(n_vars):
        disagreements = 0
        for x in range(size):
            # Flip the var-th bit (LSB=x₀ convention)
            x_flipped = x ^ (1 << var)
            if x_flipped < size and truth_table[x] != truth_table[x_flipped]:
                disagreements += 1
        influences[var] = disagreements / size

    return influences
