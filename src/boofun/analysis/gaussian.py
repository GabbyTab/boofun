"""
Gaussian analysis of Boolean functions (O'Donnell Chapter 10).

This module implements the connection between Boolean functions and Gaussian
space, which is fundamental for understanding the behavior of Boolean functions
under noise and for proving central limit-type theorems.

Key concepts:
- Hermite polynomials: The Gaussian analog of Fourier characters
- Gaussian noise stability: Stability under correlated Gaussian noise
- Ornstein-Uhlenbeck operator: The Gaussian analog of the noise operator
- Central Limit Theorem for Boolean functions

The connection to Gaussians allows us to:
1. Understand the "typical" behavior of low-degree Boolean functions
2. Prove hypercontractivity results
3. Apply invariance principles (connecting discrete and continuous)

Mathematical background:
- Standard Gaussian space: (ℝ^n, γ_n) where γ_n is the n-dimensional Gaussian
- Hermite polynomials H_k(x) form an orthonormal basis for L^2(ℝ, γ)
- The Hermite expansion generalizes Fourier expansion to Gaussians

References:
- O'Donnell, "Analysis of Boolean Functions", Chapter 10
- Mossel, O'Donnell, Oleszkiewicz, "Noise stability of functions with low influences"
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Hermite polynomials
    "hermite_polynomial",
    "hermite_coefficients",
    "probabilists_hermite",
    "physicists_hermite",
    # Gaussian noise
    "gaussian_noise_stability",
    "ornstein_uhlenbeck_operator",
    "gaussian_noise_sensitivity",
    # Central Limit Theorems
    "berry_esseen_bound",
    "clt_approximation",
    # Gaussian correlations
    "gaussian_inner_product",
    "multilinear_extension",
    # Utility
    "GaussianAnalyzer",
]


@lru_cache(maxsize=128)
def hermite_polynomial(n: int, variant: str = "probabilist") -> Callable[[float], float]:
    """
    Return the nth Hermite polynomial.

    There are two common normalizations:
    - Probabilist's: He_n(x) where E[He_n(X) He_m(X)] = δ_{nm} for X ~ N(0,1)
    - Physicist's: H_n(x) with different normalization

    The probabilist's Hermite polynomials satisfy:
        He_0(x) = 1
        He_1(x) = x
        He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)

    Args:
        n: Degree of the polynomial
        variant: "probabilist" or "physicist"

    Returns:
        Function computing He_n(x) or H_n(x)
    """
    if variant == "probabilist":
        return lambda x: probabilists_hermite(n, x)
    else:
        return lambda x: physicists_hermite(n, x)


def probabilists_hermite(n: int, x: float) -> float:
    """
    Evaluate the nth probabilist's Hermite polynomial at x.

    The probabilist's Hermite polynomials are orthonormal under the
    standard Gaussian measure: E[He_n(X) He_m(X)] = δ_{nm} for X ~ N(0,1).

    Uses the recurrence: He_n(x) = x * He_{n-1}(x) - (n-1) * He_{n-2}(x)

    Args:
        n: Degree
        x: Point to evaluate

    Returns:
        He_n(x)
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return x

    he_prev2 = 1.0  # He_0
    he_prev1 = x  # He_1

    for k in range(2, n + 1):
        he_curr = x * he_prev1 - (k - 1) * he_prev2
        he_prev2 = he_prev1
        he_prev1 = he_curr

    return he_prev1


def physicists_hermite(n: int, x: float) -> float:
    """
    Evaluate the nth physicist's Hermite polynomial at x.

    The physicist's normalization satisfies:
        H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}

    Recurrence: H_n(x) = 2x * H_{n-1}(x) - 2(n-1) * H_{n-2}(x)

    Args:
        n: Degree
        x: Point to evaluate

    Returns:
        H_n(x)
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return 2 * x

    h_prev2 = 1.0  # H_0
    h_prev1 = 2 * x  # H_1

    for k in range(2, n + 1):
        h_curr = 2 * x * h_prev1 - 2 * (k - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr

    return h_prev1


def hermite_coefficients(f: "BooleanFunction") -> Dict[Tuple[int, ...], float]:
    """
    Compute the Hermite expansion of the multilinear extension of f.

    For a Boolean function f: {-1,1}^n → ℝ, the multilinear extension
    p(x) has a Hermite expansion:
        p(x) = Σ_α ĥ_α H_α(x)

    where H_α(x) = ∏_i H_{α_i}(x_i).

    For multilinear polynomials (degree 1 in each variable):
        ĥ_S = f̂(S) for |S| = k (Fourier = Hermite for multilinear)

    Args:
        f: BooleanFunction to analyze

    Returns:
        Dictionary mapping multi-indices to Hermite coefficients
    """
    # For multilinear functions, Hermite = Fourier coefficients
    from ..analysis import SpectralAnalyzer

    analyzer = SpectralAnalyzer(f)
    fourier = analyzer.fourier_expansion()

    n = f.n_vars or 0
    coeffs = {}

    for s in range(len(fourier)):
        if abs(fourier[s]) > 1e-10:
            # Convert bitmask to multi-index
            alpha = tuple(1 if (s >> i) & 1 else 0 for i in range(n))
            coeffs[alpha] = float(fourier[s])

    return coeffs


def gaussian_noise_stability(f: "BooleanFunction", rho: float) -> float:
    """
    Compute the Gaussian noise stability of f at correlation rho.

    The Gaussian noise stability is:
        Stab_ρ[f] = E_{(x,y) ρ-correlated Gaussians}[f̃(x) f̃(y)]

    where f̃ is the multilinear extension of f.

    For the discrete case, this equals the standard noise stability:
        Stab_ρ[f] = Σ_S ρ^|S| f̂(S)²

    Args:
        f: BooleanFunction to analyze
        rho: Correlation parameter in [-1, 1]

    Returns:
        Gaussian noise stability
    """
    from ..analysis import SpectralAnalyzer

    analyzer = SpectralAnalyzer(f)
    fourier = analyzer.fourier_expansion()

    stability = 0.0
    for s, coeff in enumerate(fourier):
        k = bin(s).count("1")
        stability += (rho**k) * (coeff**2)

    return stability


def gaussian_noise_sensitivity(f: "BooleanFunction", rho: float) -> float:
    """
    Compute the Gaussian noise sensitivity: 1 - Stab_ρ[f].

    This measures how much the function changes under Gaussian noise.

    Args:
        f: BooleanFunction to analyze
        rho: Correlation parameter

    Returns:
        Gaussian noise sensitivity
    """
    return 1.0 - gaussian_noise_stability(f, rho)


def ornstein_uhlenbeck_operator(f: "BooleanFunction", rho: float) -> np.ndarray:
    """
    Apply the Ornstein-Uhlenbeck (noise) operator T_ρ to f.

    The O-U operator is defined as:
        (T_ρ f)(x) = E_y[f(y)] where y_i = ρ x_i + sqrt(1-ρ²) z_i

    and z_i are independent standard Gaussians.

    In Fourier domain: (T_ρ f)^(S) = ρ^|S| f̂(S)

    Args:
        f: BooleanFunction to apply operator to
        rho: Noise parameter

    Returns:
        Truth table of T_ρ f (as real values, not Boolean)
    """
    from ..analysis import SpectralAnalyzer

    n = f.n_vars or 0
    if n == 0:
        return np.array([float(f.evaluate(0))])

    analyzer = SpectralAnalyzer(f)
    fourier = analyzer.fourier_expansion()

    # Apply damping in Fourier domain
    damped_fourier = np.zeros_like(fourier)
    for s in range(len(fourier)):
        k = bin(s).count("1")
        damped_fourier[s] = (rho**k) * fourier[s]

    # Inverse transform (WHT is self-inverse up to normalization)
    size = 1 << n
    result = np.zeros(size)
    for x in range(size):
        for s in range(size):
            chi_val = 1 - 2 * (bin(x & s).count("1") % 2)
            result[x] += damped_fourier[s] * chi_val

    return result


def berry_esseen_bound(f: "BooleanFunction") -> float:
    """
    Compute the Berry-Esseen bound for the CLT approximation of f.

    For a Boolean function f with low influences, the distribution of
    f(x) for random x approaches Gaussian. The Berry-Esseen theorem
    quantifies this:

        sup_t |Pr[f(x) <= t] - Φ(t)| <= C * Σ_i Inf_i[f]^3 / Var[f]^{3/2}

    where Φ is the standard normal CDF.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Berry-Esseen bound (smaller = better Gaussian approximation)
    """
    from ..analysis import SpectralAnalyzer

    analyzer = SpectralAnalyzer(f)
    influences = analyzer.influences()
    total_inf = analyzer.total_influence()

    if total_inf < 1e-10:
        return 0.0

    # Variance = total influence for ±1 valued functions
    var = total_inf

    # Sum of cubed influences
    sum_inf_cubed = sum(inf**3 for inf in influences)

    # Berry-Esseen constant (sharp constant is about 0.4748)
    C = 0.5

    bound = C * sum_inf_cubed / (var**1.5)

    return bound


def clt_approximation(f: "BooleanFunction", num_samples: int = 10000) -> Tuple[float, float]:
    """
    Estimate how well f's distribution is approximated by a Gaussian.

    Uses Monte Carlo sampling to estimate the mean and variance of f,
    then computes the Kolmogorov-Smirnov distance to a Gaussian.

    Args:
        f: BooleanFunction to analyze
        num_samples: Number of random samples

    Returns:
        Tuple of (mean, variance) of f's distribution
    """
    n = f.n_vars
    if n is None or n == 0:
        return (float(f.evaluate(0)), 0.0)

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    pm_values = 1.0 - 2.0 * truth_table  # Convert to ±1

    mean = np.mean(pm_values)
    var = np.var(pm_values)

    return (float(mean), float(var))


def gaussian_inner_product(f: "BooleanFunction", g: "BooleanFunction") -> float:
    """
    Compute the Gaussian inner product ⟨f, g⟩_γ.

    For the discrete case with ±1 values, this equals:
        ⟨f, g⟩_γ = E[f(x)g(x)] = Σ_S f̂(S) ĝ(S)

    (Plancherel's theorem)

    Args:
        f, g: BooleanFunctions to compute inner product of

    Returns:
        Gaussian inner product
    """
    from ..analysis.fourier import plancherel_inner_product

    return plancherel_inner_product(f, g)


def multilinear_extension(f: "BooleanFunction") -> Callable[[np.ndarray], float]:
    """
    Return the multilinear extension of f.

    The multilinear extension p: ℝ^n → ℝ is the unique multilinear polynomial
    that agrees with f on {-1, 1}^n:

        p(x) = Σ_S f̂(S) ∏_{i∈S} x_i

    This extends f from the Boolean hypercube to all of ℝ^n.

    Args:
        f: BooleanFunction to extend

    Returns:
        Function computing p(x) for any x ∈ ℝ^n
    """
    from ..analysis import SpectralAnalyzer

    analyzer = SpectralAnalyzer(f)
    fourier = analyzer.fourier_expansion()
    n = f.n_vars or 0

    def p(x: np.ndarray) -> float:
        result = 0.0
        for s in range(len(fourier)):
            if abs(fourier[s]) < 1e-10:
                continue

            # Compute ∏_{i∈S} x_i
            product = 1.0
            for i in range(n):
                if (s >> i) & 1:
                    product *= x[i]

            result += fourier[s] * product

        return result

    return p


class GaussianAnalyzer:
    """
    Comprehensive Gaussian analysis for Boolean functions.

    Provides methods for analyzing Boolean functions in the context of
    Gaussian space and central limit theorems.
    """

    def __init__(self, f: "BooleanFunction"):
        """
        Initialize Gaussian analyzer.

        Args:
            f: BooleanFunction to analyze
        """
        self.function = f
        self.n_vars = f.n_vars
        self._hermite_coeffs: Optional[Dict] = None

    @property
    def hermite_coefficients(self) -> Dict[Tuple[int, ...], float]:
        """Get cached Hermite coefficients."""
        if self._hermite_coeffs is None:
            self._hermite_coeffs = hermite_coefficients(self.function)
        return self._hermite_coeffs

    def noise_stability(self, rho: float) -> float:
        """Compute Gaussian noise stability at correlation rho."""
        return gaussian_noise_stability(self.function, rho)

    def noise_sensitivity(self, rho: float) -> float:
        """Compute Gaussian noise sensitivity at correlation rho."""
        return gaussian_noise_sensitivity(self.function, rho)

    def berry_esseen(self) -> float:
        """Compute Berry-Esseen bound for Gaussian approximation."""
        return berry_esseen_bound(self.function)

    def multilinear_extension(self) -> Callable[[np.ndarray], float]:
        """Get the multilinear extension of f."""
        return multilinear_extension(self.function)

    def apply_noise_operator(self, rho: float) -> np.ndarray:
        """Apply Ornstein-Uhlenbeck operator T_ρ to f."""
        return ornstein_uhlenbeck_operator(self.function, rho)

    def is_approximately_gaussian(self, threshold: float = 0.1) -> bool:
        """
        Check if f's distribution is approximately Gaussian.

        Uses the Berry-Esseen bound to check if the function has
        low enough influences to be well-approximated by a Gaussian.

        Args:
            threshold: Maximum Berry-Esseen bound for "Gaussian"

        Returns:
            True if distribution is approximately Gaussian
        """
        return self.berry_esseen() < threshold

    def summary(self) -> str:
        """Return a summary of Gaussian properties."""
        be = self.berry_esseen()
        mean, var = clt_approximation(self.function)
        stab_09 = self.noise_stability(0.9)
        stab_05 = self.noise_stability(0.5)

        lines = [
            "Gaussian Analysis",
            "=" * 40,
            f"Berry-Esseen bound: {be:.6f}",
            f"Approximately Gaussian: {self.is_approximately_gaussian()}",
            f"Mean (E[f]): {mean:.6f}",
            f"Variance (Var[f]): {var:.6f}",
            f"Noise stability (ρ=0.9): {stab_09:.6f}",
            f"Noise stability (ρ=0.5): {stab_05:.6f}",
            f"Non-zero Hermite coefficients: {len(self.hermite_coefficients)}",
        ]

        return "\n".join(lines)
