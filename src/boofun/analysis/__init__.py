# src/boofun/analysis/__init__.py
"""
Boolean function analysis module providing spectral analysis tools.

This module implements fundamental algorithms for analyzing Boolean functions
including Fourier analysis, influence computation, and noise stability.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

from ..core.numba_optimizations import is_numba_available, numba_optimize


class SpectralAnalyzer:
    """
    Spectral analysis tools for Boolean functions.

    Provides methods for computing Fourier coefficients, variable influences,
    total influence, noise stability, and other spectral properties.
    """

    def __init__(self, function: "BooleanFunction"):
        """
        Initialize analyzer with a Boolean function.

        Args:
            function: BooleanFunction instance to analyze
        """
        self.function = function
        self.n_vars = function.n_vars
        if self.n_vars is None:
            raise ValueError("Function must have defined number of variables")

        # Cache for expensive computations
        self._fourier_coeffs = None
        self._influences = None

        # Track error model for uncertainty propagation
        self.error_model = function.error_model

    def fourier_expansion(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute Fourier expansion coefficients.

        The Fourier expansion represents f: {0,1}^n → {0,1} as:
        f(x) = Σ_S f̂(S) * χ_S(x)
        where χ_S(x) = (-1)^(Σ_{i∈S} x_i) are the Walsh functions.

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            Array of Fourier coefficients indexed by subsets
        """
        if self._fourier_coeffs is not None and not force_recompute:
            return self._fourier_coeffs

        # Get function values on all inputs
        size = 1 << self.n_vars  # 2^n

        # Convert function to {-1, +1} representation for Fourier analysis
        # O'Donnell convention (Analysis of Boolean Functions, Chapter 1):
        #   f(x) = 0  →  +1  (in ±1 domain)
        #   f(x) = 1  →  -1  (in ±1 domain)
        # This ensures the fundamental identity f̂(∅) = E[f] holds.
        f_vals = np.zeros(size, dtype=float)
        for i in range(size):
            val = self.function.evaluate(np.array(i))
            f_vals[i] = -1.0 if val else 1.0

        # Compute Fourier coefficients using Walsh-Hadamard transform
        self._fourier_coeffs = self._walsh_hadamard_transform(f_vals)
        return self._fourier_coeffs

    def _walsh_hadamard_transform(self, f_vals: np.ndarray) -> np.ndarray:
        """
        Compute Walsh-Hadamard transform (fast Fourier transform for Boolean functions).

        Uses the fastest available backend:
        1. pyfwht (GPU-accelerated, if installed)
        2. scipy.linalg.hadamard (CPU-optimized)
        3. Pure numpy iterative (fallback)

        Args:
            f_vals: Function values in {-1, 1} representation

        Returns:
            Fourier coefficients
        """
        n = self.n_vars
        size = len(f_vals)

        # Try pyfwht for GPU-accelerated transform (fastest for large n)
        try:
            from pyfwht import fwht

            # pyfwht expects natural order, returns Walsh-Hadamard coefficients
            result = fwht(f_vals.astype(np.float64))
            return result / size
        except ImportError:
            pass
        except Exception:
            pass  # Fall back if pyfwht fails

        # Try to use scipy.linalg.hadamard for better performance
        try:
            from scipy.linalg import hadamard

            # Hadamard matrix multiply is more cache-friendly for medium n
            if n <= 14:  # scipy.linalg.hadamard has size limitations
                H = hadamard(size, dtype=float)
                return (H @ f_vals) / size
        except (ImportError, ValueError):
            pass

        # Fall back to iterative implementation (in-place, memory-efficient)
        coeffs = f_vals.copy()

        # Iterative Walsh-Hadamard transform
        # This is equivalent to matrix multiplication with Hadamard matrix
        for i in range(n):
            step = 1 << i  # 2^i
            for j in range(0, size, step * 2):
                for k in range(step):
                    u = coeffs[j + k]
                    v = coeffs[j + k + step]
                    coeffs[j + k] = u + v
                    coeffs[j + k + step] = u - v

        return coeffs / size  # Normalize

    def influences(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute variable influences (also called sensitivities).

        The influence of variable i is:
        Inf_i(f) = Pr[f(x) ≠ f(x ⊕ e_i)]
        where e_i is the i-th unit vector.

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            Array of influences for each variable
        """
        if self._influences is not None and not force_recompute:
            return self._influences

        # Try Numba optimization if available and function has truth table
        if is_numba_available() and self.function.has_rep("truth_table"):
            try:
                truth_table = self.function.get_representation("truth_table")
                influences = numba_optimize("influences", truth_table, self.n_vars)
                self._influences = influences
                return influences
            except Exception as e:
                warnings.warn(f"Numba optimization failed, using fallback: {e}")

        # Fallback to standard computation
        influences = np.zeros(self.n_vars)
        size = 1 << self.n_vars  # 2^n

        for i in range(self.n_vars):
            # Count disagreements when flipping bit i
            disagreements = 0

            for x in range(size):
                # Flip the i-th bit (LSB=x₀ convention)
                x_flipped = x ^ (1 << i)

                # Evaluate function at both points
                f_x = self.function.evaluate(np.array(x))
                f_x_flipped = self.function.evaluate(np.array(x_flipped))

                if f_x != f_x_flipped:
                    disagreements += 1

            # Influence is fraction of inputs where flipping bit i changes output
            influences[i] = disagreements / size

        self._influences = influences
        return influences

    def total_influence(self) -> float:
        """
        Compute total influence (sum of all variable influences).

        Returns:
            Total influence = Σ_i Inf_i(f)
        """
        return np.sum(self.influences())

    def noise_stability(self, rho: float) -> float:
        """
        Compute noise stability at correlation ρ.

        Noise stability is the probability that f(x) = f(y) where
        y is obtained from x by flipping each bit independently with
        probability (1-ρ)/2.

        Args:
            rho: Correlation parameter in [-1, 1]

        Returns:
            Noise stability value
        """
        if not -1 <= rho <= 1:
            raise ValueError("Correlation rho must be in [-1, 1]")

        # Use Fourier expansion to compute noise stability
        fourier_coeffs = self.fourier_expansion()

        # Try Numba optimization
        if is_numba_available():
            try:
                return numba_optimize("noise_stability", fourier_coeffs, rho)
            except Exception as e:
                warnings.warn(f"Numba noise stability optimization failed: {e}")

        # Fallback: Noise stability = Σ_S f̂(S)² * ρ^|S|
        stability = 0.0
        size = len(fourier_coeffs)

        for s in range(size):
            # Count number of bits in subset s (|S|)
            subset_size = bin(s).count("1")
            stability += fourier_coeffs[s] ** 2 * (rho**subset_size)

        return stability

    def spectral_concentration(self, degree: int) -> float:
        """
        Compute spectral concentration at given degree.

        This is the fraction of Fourier weight on coefficients
        corresponding to sets of size at most 'degree'.

        Args:
            degree: Maximum subset size to include

        Returns:
            Fraction of spectral weight on low-degree coefficients
        """
        fourier_coeffs = self.fourier_expansion()

        total_weight = np.sum(fourier_coeffs**2)
        low_degree_weight = 0.0

        for s in range(len(fourier_coeffs)):
            subset_size = bin(s).count("1")
            if subset_size <= degree:
                low_degree_weight += fourier_coeffs[s] ** 2

        return low_degree_weight / total_weight if total_weight > 0 else 0.0

    def get_fourier_coefficient(self, subset: Union[int, List[int]]) -> float:
        """
        Get Fourier coefficient for a specific subset.

        Args:
            subset: Either integer index or list of variable indices

        Returns:
            Fourier coefficient for the subset
        """
        fourier_coeffs = self.fourier_expansion()

        if isinstance(subset, list):
            # Convert list of indices to integer representation (LSB=x₀ convention)
            index = 0
            for i in subset:
                if 0 <= i < self.n_vars:
                    index |= 1 << i
                else:
                    raise ValueError(f"Variable index {i} out of range")
        else:
            index = subset

        if 0 <= index < len(fourier_coeffs):
            return fourier_coeffs[index]
        else:
            raise ValueError(f"Subset index {index} out of range")

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the Boolean function.

        Returns:
            Dictionary with various spectral properties and confidence information
        """
        influences = self.influences()

        # Basic statistics (bias first as it's fundamental)
        summary = {
            "bias": self.function.bias(),
            "total_influence": self.total_influence(),
            "max_influence": np.max(influences),
            "min_influence": np.min(influences),
            "avg_influence": np.mean(influences),
            "noise_stability_0.9": self.noise_stability(0.9),
            "noise_stability_0.5": self.noise_stability(0.5),
            "spectral_concentration_1": self.spectral_concentration(1),
            "spectral_concentration_2": self.spectral_concentration(2),
        }

        # Add error model information
        if hasattr(self.error_model, "get_confidence"):
            summary["analysis_confidence"] = self.error_model.get_confidence(influences)
            summary["error_model_type"] = type(self.error_model).__name__

            if hasattr(self.error_model, "is_reliable"):
                summary["analysis_reliable"] = self.error_model.is_reliable(influences)

        return summary


class PropertyTester:
    """
    Property testing algorithms for Boolean functions.

    Implements various randomized and deterministic tests to check
    if Boolean functions satisfy specific properties.
    """

    def __init__(self, function: "BooleanFunction", random_seed: Optional[int] = None):
        self.function = function
        n_vars = function.n_vars
        if n_vars is None:
            raise ValueError("Function must have defined number of variables")
        self.n_vars: int = n_vars

        # Set random seed for reproducible testing
        self.rng = np.random.RandomState(random_seed)

    def constant_test(self) -> bool:
        """
        Test if function is constant.

        Deterministic test that checks if all outputs are identical.
        Time complexity: O(2^n) for exhaustive check.

        Returns:
            True if function is constant, False otherwise
        """
        if self.n_vars == 0:
            return True

        first_val = self.function.evaluate(np.array(0))
        for i in range(1, 1 << self.n_vars):
            if self.function.evaluate(np.array(i)) != first_val:
                return False
        return True

    def blr_linearity_test(self, num_queries: int = 100, epsilon: float = 0.1) -> bool:
        """
        BLR (Blum-Luby-Rubinfeld) linearity test.

        Tests if a function f: {0,1}^n → {0,1} is linear (i.e., f(x ⊕ y) = f(x) ⊕ f(y)).
        Uses randomized queries to test the linearity property.

        QUERY COMPLEXITY: O(3 * num_queries) - safe for arbitrarily large n.

        Args:
            num_queries: Number of random queries to perform
            epsilon: Error tolerance (function passes if error rate < epsilon)

        Returns:
            True if function appears linear, False otherwise
        """
        import random

        if self.n_vars == 0:
            return True

        violations = 0
        n = self.n_vars

        # Use Python's random for arbitrary precision integers (handles n > 63)
        py_rng = random.Random(self.rng.randint(0, 2**31))

        for _ in range(num_queries):
            # Generate random inputs x and y using arbitrary precision
            x = py_rng.getrandbits(n)
            y = py_rng.getrandbits(n)

            # Compute x ⊕ y
            x_xor_y = x ^ y

            # Test linearity condition: f(x) ⊕ f(y) = f(x ⊕ y)
            f_x = self.function.evaluate(x)
            f_y = self.function.evaluate(y)
            f_x_xor_y = self.function.evaluate(x_xor_y)

            expected = f_x ^ f_y  # XOR of outputs
            if f_x_xor_y != expected:
                violations += 1

        error_rate = violations / num_queries
        return error_rate < epsilon

    def junta_test(self, k: int, num_queries: int = 1000, confidence: float = 0.9) -> bool:
        """
        Test if function is a k-junta (depends on at most k variables).

        A function is a k-junta if there exists a set S ⊆ [n] with |S| ≤ k
        such that f(x) depends only on coordinates in S.

        Args:
            k: Maximum number of relevant variables
            num_queries: Number of random queries
            confidence: Confidence level for the test

        Returns:
            True if function appears to be a k-junta, False otherwise
        """
        if k >= self.n_vars:
            return True  # Trivially true

        # Use influence-based approach: if function is k-junta,
        # at most k variables can have non-zero influence
        analyzer = SpectralAnalyzer(self.function)
        influences = analyzer.influences()

        # Count variables with significant influence
        threshold = 1.0 / (2**self.n_vars)  # Minimum detectable influence
        influential_vars = np.sum(influences > threshold)

        return influential_vars <= k

    def monotonicity_test(self, num_queries: int = 1000) -> bool:
        """
        Test if function is monotone (f(x) ≤ f(y) whenever x ≤ y coordinate-wise).

        QUERY COMPLEXITY: O(2 * num_queries) - safe for arbitrarily large n.

        Args:
            num_queries: Number of random pairs to test

        Returns:
            True if function appears monotone, False otherwise
        """
        import random

        violations = 0
        n = self.n_vars

        # Use Python's random for arbitrary precision
        py_rng = random.Random(self.rng.randint(0, 2**31))

        for _ in range(num_queries):
            # Generate random x using arbitrary precision
            x = py_rng.getrandbits(n)

            # Generate y ≥ x by flipping some 0 bits to 1
            y = x
            for i in range(n):
                if ((x >> i) & 1) == 0 and py_rng.random() < 0.3:
                    y |= 1 << i

            # Check monotonicity: f(x) ≤ f(y)
            f_x = self.function.evaluate(x)
            f_y = self.function.evaluate(y)

            if f_x > f_y:  # Violation of monotonicity
                violations += 1

        # Function is monotone if no violations found
        return violations == 0

    def unateness_test(self, num_queries: int = 1000) -> bool:
        """
        Test if function is unate (monotone in each variable, possibly negated).

        A function is unate if for each variable x_i, either:
        - f is monotone increasing in x_i, OR
        - f is monotone decreasing in x_i

        Unate functions generalize monotone functions - they can be monotone
        in different "directions" for different variables.

        QUERY COMPLEXITY: O(2 * num_queries) - safe for arbitrarily large n.

        Args:
            num_queries: Number of random pairs to test per variable

        Returns:
            True if function appears unate, False otherwise
        """
        import random

        n = self.n_vars
        py_rng = random.Random(self.rng.randint(0, 2**31))

        queries_per_var = max(num_queries // n, 10)

        for i in range(n):
            # Track if we've seen increasing or decreasing behavior
            saw_increase = False
            saw_decrease = False

            for _ in range(queries_per_var):
                # Generate random x
                x = py_rng.getrandbits(n)

                # Create x' by flipping bit i
                x_prime = x ^ (1 << i)

                f_x = self.function.evaluate(x)
                f_x_prime = self.function.evaluate(x_prime)

                # Check the direction of change when x_i goes 0→1
                if ((x >> i) & 1) == 0:
                    # x has 0 in position i, x' has 1
                    if f_x_prime > f_x:
                        saw_increase = True
                    elif f_x_prime < f_x:
                        saw_decrease = True
                else:
                    # x has 1 in position i, x' has 0
                    if f_x > f_x_prime:
                        saw_increase = True
                    elif f_x < f_x_prime:
                        saw_decrease = True

                # If both directions seen, not unate in this variable
                if saw_increase and saw_decrease:
                    return False

        return True

    def symmetry_test(self, num_queries: int = 1000) -> bool:
        """
        Test if function is symmetric (invariant under permutations of variables).

        QUERY COMPLEXITY: O(2 * num_queries) - safe for arbitrarily large n.

        Args:
            num_queries: Number of random permutations to test

        Returns:
            True if function appears symmetric, False otherwise
        """
        import random

        n = self.n_vars

        # Use Python's random for arbitrary precision
        py_rng = random.Random(self.rng.randint(0, 2**31))

        for _ in range(num_queries):
            # Generate random input using arbitrary precision
            x = py_rng.getrandbits(n)

            # Generate random permutation
            perm = list(range(n))
            py_rng.shuffle(perm)

            # Apply permutation efficiently
            y = 0
            for i in range(n):
                if (x >> perm[i]) & 1:
                    y |= 1 << i

            # Check if f(x) = f(permuted(x))
            f_x = self.function.evaluate(x)
            f_y = self.function.evaluate(y)

            if f_x != f_y:
                return False

        return True

    def balanced_test(self) -> bool:
        """
        Test if function is balanced (outputs 0 and 1 equally often).

        Returns:
            True if function is balanced, False otherwise
        """
        if self.n_vars == 0:
            return False

        ones_count = 0
        total = 1 << self.n_vars

        for i in range(total):
            if self.function.evaluate(np.array(i)):
                ones_count += 1

        return ones_count == total // 2

    def dictator_test(
        self, num_queries: int = 1000, epsilon: float = 0.1
    ) -> Tuple[bool, Optional[int]]:
        """
        Test if function is a dictator or anti-dictator.

        A dictator function is f(x) = x_i for some i.
        An anti-dictator is f(x) = ¬x_i = 1 - x_i.

        From O'Donnell Chapter 7: The FKN theorem states that if f is
        Boolean and has small total influence, it's close to a dictator.

        Args:
            num_queries: Number of random queries
            epsilon: Distance threshold

        Returns:
            Tuple of (is_dictator_like, dictator_index) where:
            - is_dictator_like: True if f is close to a dictator/anti-dictator
            - dictator_index: The index of the dictator variable (or None)
        """
        # Compute influences
        analyzer = SpectralAnalyzer(self.function)
        influences = analyzer.influences()

        # Find the variable with maximum influence
        max_inf_idx = int(np.argmax(influences))
        max_inf = influences[max_inf_idx]

        # For a dictator, one variable has influence 1, others have 0
        total_inf = np.sum(influences)

        # Check if it's close to a dictator
        if max_inf > 1 - epsilon and total_inf < 1 + epsilon:
            # Verify by checking function values
            is_dictator = True
            is_anti_dictator = True

            for _ in range(min(num_queries, 1 << self.n_vars)):
                x = self.rng.randint(0, 1 << self.n_vars)
                f_x = int(self.function.evaluate(np.array(x)))

                # Extract the i-th bit (LSB=x₀ convention)
                x_i = (x >> max_inf_idx) & 1

                if f_x != x_i:
                    is_dictator = False
                if f_x != (1 - x_i):
                    is_anti_dictator = False

                if not is_dictator and not is_anti_dictator:
                    break

            if is_dictator or is_anti_dictator:
                return (True, max_inf_idx)

        return (False, None)

    def affine_test(self, num_queries: int = 1000, epsilon: float = 0.1) -> bool:
        """
        4-query test for affine functions over GF(2).

        From HW1 Problem 4: A function f: F_2^n → F_2 is affine iff
        f(x) + f(y) + f(z) = f(x + y + z) for all x, y, z.

        This is a generalization of the BLR linearity test.

        Args:
            num_queries: Number of random queries
            epsilon: Error tolerance

        Returns:
            True if function appears to be affine
        """
        violations = 0

        for _ in range(num_queries):
            # Generate three random inputs
            x = self.rng.randint(0, 1 << self.n_vars)
            y = self.rng.randint(0, 1 << self.n_vars)
            z = self.rng.randint(0, 1 << self.n_vars)

            # Compute x + y + z (XOR in GF(2))
            xyz = x ^ y ^ z

            # Test: f(x) + f(y) + f(z) = f(x + y + z)
            f_x = int(self.function.evaluate(np.array(x)))
            f_y = int(self.function.evaluate(np.array(y)))
            f_z = int(self.function.evaluate(np.array(z)))
            f_xyz = int(self.function.evaluate(np.array(xyz)))

            # XOR of three values should equal fourth
            if (f_x ^ f_y ^ f_z) != f_xyz:
                violations += 1

        return violations / num_queries < epsilon

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all available property tests.

        Returns:
            Dictionary mapping test names to results
        """
        results: Dict[str, Any] = {}

        try:
            results["constant"] = self.constant_test()
        except Exception as e:
            results["constant"] = f"Error: {e}"

        try:
            results["linear"] = self.blr_linearity_test()
        except Exception as e:
            results["linear"] = f"Error: {e}"

        try:
            results["balanced"] = self.balanced_test()
        except Exception as e:
            results["balanced"] = f"Error: {e}"

        try:
            results["monotone"] = self.monotonicity_test()
        except Exception as e:
            results["monotone"] = f"Error: {e}"

        try:
            results["unate"] = self.unateness_test()
        except Exception as e:
            results["unate"] = f"Error: {e}"

        try:
            results["symmetric"] = self.symmetry_test()
        except Exception as e:
            results["symmetric"] = f"Error: {e}"

        try:
            results["affine"] = self.affine_test()
        except Exception as e:
            results["affine"] = f"Error: {e}"

        # Test for small juntas
        for k in [1, 2, 3]:
            if k < self.n_vars:
                try:
                    results[f"{k}-junta"] = self.junta_test(k)
                except Exception as e:
                    results[f"{k}-junta"] = f"Error: {e}"

        return results


# Export main classes and submodules
from . import (  # noqa: E402
    basic_properties,
    block_sensitivity,
    certificates,
    complexity,
    cryptographic,
    decision_trees,
    equivalence,
    fourier,
    gaussian,
    gf2,
    global_hypercontractivity,
    hypercontractivity,
    invariance,
    learning,
    ltf_analysis,
    p_biased,
    query_complexity,
    restrictions,
    sampling,
    sensitivity,
    sparsity,
    symmetry,
)

__all__ = [
    "SpectralAnalyzer",
    "PropertyTester",
    "sensitivity",
    "block_sensitivity",
    "certificates",
    "symmetry",
    "complexity",
    "gf2",
    "equivalence",
    "p_biased",
    "learning",
    "fourier",
    "restrictions",
    "hypercontractivity",
    # BFW-style analysis (Scott Aaronson's Boolean Function Wizard)
    "query_complexity",
    "basic_properties",
    # Decision tree analysis (Tal's algorithms)
    "decision_trees",
    # Fourier sparsity analysis (Tal's algorithms)
    "sparsity",
    # Boolean functions as random variables (O'Donnell Chapters 1-3)
    "sampling",
    # Cryptographic analysis (nonlinearity, bent functions, SAC)
    "cryptographic",
    # O'Donnell Chapters 10-11: Gaussian analysis and Invariance Principle
    "gaussian",
    "invariance",
    # O'Donnell Chapter 5: Linear Threshold Functions
    "ltf_analysis",
    # Keevash et al.: Global Hypercontractivity
    "global_hypercontractivity",
]
