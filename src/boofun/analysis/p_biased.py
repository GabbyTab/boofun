"""
P-biased Fourier analysis for Boolean functions.

This module implements Fourier analysis over p-biased product distributions,
as described in O'Donnell's "Analysis of Boolean Functions" Chapter 8.

In the standard (uniform) setting, inputs are drawn from {-1,+1}^n uniformly.
In the p-biased setting, each coordinate is independently:
    x_i = -1 with probability p
    x_i = +1 with probability 1-p

The p-biased Fourier basis uses orthogonal polynomials called the:
    φ_S^(p)(x) = ∏_{i∈S} φ^(p)(x_i)

where φ^(p)(x) = (x - μ)/σ is the normalized basis function, with:
    μ = E[x] = 1 - 2p
    σ = √Var(x) = 2√(p(1-p))

Key concepts:
- p-biased measure μ_p: Pr[x_i = -1] = p, Pr[x_i = +1] = 1-p
- p-biased inner product: ⟨f,g⟩_p = E_{x~μ_p}[f(x)g(x)]
- p-biased Fourier coefficients: f̂(S)_p = ⟨f, φ_S^(p)⟩_p
- p-noise operator T_{1-2ε}: smoothing towards the p-biased mean

Applications:
- Analysis of Boolean functions under non-uniform input distributions
- Sharp threshold phenomena (p → 0)
- Monotone function analysis
- Influence under biased distributions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Core p-biased analysis
    "p_biased_fourier_coefficients",
    "p_biased_fourier_coefficient",
    "p_biased_influence",
    "p_biased_total_influence",
    "p_biased_noise_stability",
    "p_biased_expectation",
    "p_biased_variance",
    "biased_measure_mass",
    # Sensitivity under μ_p (from Tal's library)
    "p_biased_sensitivity",
    "p_biased_average_sensitivity",
    "p_biased_total_influence_fourier",
    # Utility functions
    "parity_biased_coefficient",
    # Analyzer class
    "PBiasedAnalyzer",
]


def _p_biased_basis(p: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute p-biased basis transformation parameters.

    Returns:
        (mu, sigma) where:
        - mu = E[x] = 1 - 2p (mean of single coordinate)
        - sigma = 2*sqrt(p*(1-p)) (std of single coordinate)
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0,1), got {p}")

    mu = 1.0 - 2.0 * p
    sigma = 2.0 * np.sqrt(p * (1.0 - p))

    return mu, sigma


def biased_measure_mass(p: float, n: int, subset_mask: int) -> float:
    """
    Compute μ_p(x : x has 1s exactly at positions in subset_mask).

    Args:
        p: Bias parameter (Pr[x_i = -1] = p)
        n: Number of variables
        subset_mask: Bitmask of positions that should be -1

    Returns:
        Probability mass under the p-biased measure
    """
    k = bin(subset_mask).count("1")  # Number of -1 coordinates
    return (p**k) * ((1.0 - p) ** (n - k))


def p_biased_expectation(f: "BooleanFunction", p: float = 0.5) -> float:
    """
    Compute E_{μ_p}[f] - the expectation of f under p-biased measure.

    Args:
        f: BooleanFunction (in ±1 convention, or converted)
        p: Bias parameter

    Returns:
        Expected value of f under p-biased distribution
    """
    n = f.n_vars or 0
    if n == 0:
        return float(f.evaluate(0))

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    # Convert {0,1} to {-1,+1} if needed (assuming truth table is in {0,1})
    truth_table_pm = 1.0 - 2.0 * truth_table

    size = 1 << n
    total = 0.0

    for x in range(size):
        # Count bits set (which become -1 in the ±1 convention)
        k = bin(x).count("1")
        prob = (p**k) * ((1.0 - p) ** (n - k))
        total += truth_table_pm[x] * prob

    return total


def p_biased_variance(f: "BooleanFunction", p: float = 0.5) -> float:
    """
    Compute Var_{μ_p}[f] - the variance under p-biased measure.

    Args:
        f: BooleanFunction
        p: Bias parameter

    Returns:
        Variance of f under p-biased distribution
    """
    n = f.n_vars or 0
    if n == 0:
        return 0.0

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    truth_table_pm = 1.0 - 2.0 * truth_table

    size = 1 << n

    # Compute E[f] and E[f^2]
    ef = 0.0
    ef2 = 0.0

    for x in range(size):
        k = bin(x).count("1")
        prob = (p**k) * ((1.0 - p) ** (n - k))
        fx = truth_table_pm[x]
        ef += fx * prob
        ef2 += (fx**2) * prob

    return ef2 - ef**2


def p_biased_fourier_coefficients(f: "BooleanFunction", p: float = 0.5) -> Dict[int, float]:
    """
    Compute the p-biased Fourier coefficients of f.

    The p-biased Fourier expansion is:
        f(x) = Σ_S f̂(S)_p φ_S^(p)(x)

    where φ_S^(p)(x) = ∏_{i∈S} φ^(p)(x_i) and φ^(p)(x_i) = (x_i - μ)/σ.

    Args:
        f: BooleanFunction to analyze
        p: Bias parameter (default 0.5 = uniform)

    Returns:
        Dictionary mapping subset masks to p-biased Fourier coefficients
    """
    n = f.n_vars or 0
    if n == 0:
        return {0: float(f.evaluate(0))}

    mu, sigma = _p_biased_basis(p, n)

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    truth_table_pm = 1.0 - 2.0 * truth_table

    size = 1 << n
    coefficients = {}

    for S in range(size):
        # Compute f̂(S)_p = E_{μ_p}[f(x) φ_S^(p)(x)]
        coeff = 0.0

        for x in range(size):
            # Compute probability mass of x
            k = bin(x).count("1")
            prob = (p**k) * ((1.0 - p) ** (n - k))

            # Compute φ_S^(p)(x) = ∏_{i∈S} (x_i - μ)/σ
            phi = 1.0
            for i in range(n):
                if (S >> i) & 1:  # i is in S
                    x_i = -1.0 if (x >> i) & 1 else 1.0
                    phi *= (x_i - mu) / sigma

            coeff += truth_table_pm[x] * phi * prob

        if abs(coeff) > 1e-10:  # Store non-negligible coefficients
            coefficients[S] = coeff

    return coefficients


def p_biased_influence(f: "BooleanFunction", i: int, p: float = 0.5) -> float:
    """
    Compute the p-biased influence of variable i on f.

    The p-biased influence is:
        Inf_i^(p)[f] = E_{μ_p}[(D_i f)^2] = Σ_{S∋i} f̂(S)_p^2 / (p(1-p))

    where D_i f(x) = (f(x^{(i→+1)}) - f(x^{(i→-1)})) / 2.

    Args:
        f: BooleanFunction to analyze
        i: Variable index
        p: Bias parameter

    Returns:
        p-biased influence of variable i
    """
    if i < 0 or i >= (f.n_vars or 0):
        raise ValueError(f"Variable index {i} out of range")

    n = f.n_vars or 0
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    truth_table_pm = 1.0 - 2.0 * truth_table

    size = 1 << n

    # Compute E_{μ_p}[(f(x^{i=+1}) - f(x^{i=-1}))^2] / 4
    total = 0.0

    for x in range(size):
        # Skip if bit i is set (we'll handle pairs)
        if (x >> i) & 1:
            continue

        x_with_i = x | (1 << i)

        # f at x (bit i is 0, meaning x_i = +1)
        # f at x_with_i (bit i is 1, meaning x_i = -1)
        f_plus = truth_table_pm[x]
        f_minus = truth_table_pm[x_with_i]

        # Probability of the rest of the coordinates
        k_rest = bin(x).count("1")  # Count of -1s excluding position i
        prob_rest = (p**k_rest) * ((1.0 - p) ** (n - 1 - k_rest))

        # The derivative squared
        diff_sq = ((f_plus - f_minus) / 2.0) ** 2

        total += diff_sq * prob_rest

    return total


def p_biased_total_influence(f: "BooleanFunction", p: float = 0.5) -> float:
    """
    Compute the total p-biased influence.

    I^(p)[f] = Σ_i Inf_i^(p)[f]

    Args:
        f: BooleanFunction to analyze
        p: Bias parameter

    Returns:
        Total p-biased influence
    """
    n = f.n_vars or 0
    return sum(p_biased_influence(f, i, p) for i in range(n))


def p_biased_noise_stability(f: "BooleanFunction", rho: float, p: float = 0.5) -> float:
    """
    Compute the p-biased noise stability at correlation rho.

    Stab_ρ^(p)[f] = E[f(x)f(y)] where (x,y) are ρ-correlated under μ_p

    Args:
        f: BooleanFunction to analyze
        rho: Noise correlation parameter in [-1, 1]
        p: Bias parameter

    Returns:
        p-biased noise stability
    """
    n = f.n_vars or 0
    if n == 0:
        return float(f.evaluate(0)) ** 2

    # Use Fourier formula: Stab_ρ^(p)[f] = Σ_S ρ^|S| f̂(S)_p^2
    coeffs = p_biased_fourier_coefficients(f, p)

    stability = 0.0
    for S, coeff in coeffs.items():
        k = bin(S).count("1")
        stability += (rho**k) * (coeff**2)

    return stability


# ============================================================================
# Functions from Avishay Tal's library
# ============================================================================


def p_biased_fourier_coefficient(f: "BooleanFunction", p: float, S: int) -> float:
    """
    Compute single p-biased Fourier coefficient using Tal's formula.

    This is an alternative (and often faster) formula for computing
    individual p-biased Fourier coefficients. Uses the explicit basis:

        φ_S^(p)(x) = ∏_{i∈S} φ^(p)(x_i)

    where φ^(p)(x_i) = -√(q/p) if x_i = -1, else √(p/q)
    (with q = 1-p)

    This formula is from Tal's FourierCoefMuP function.

    Args:
        f: BooleanFunction to analyze
        p: Bias parameter in (0, 1)
        S: Subset mask (integer with bits indicating which variables)

    Returns:
        The p-biased Fourier coefficient f̂(S)_p

    References:
        - Tal's BooleanFunc.py: FourierCoefMuP
        - O'Donnell Chapter 8
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0,1), got {p}")

    n = f.n_vars or 0
    if n == 0:
        return float(f.evaluate(0)) if S == 0 else 0.0

    q = 1.0 - p
    sqrt_q_over_p = np.sqrt(q / p)
    sqrt_p_over_q = np.sqrt(p / q)

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)

    val = 0.0
    size = 1 << n
    popcnt_S = bin(S).count("1")

    for x in range(size):
        # Probability of x under μ_p
        popcnt_x = bin(x).count("1")
        pr = (p**popcnt_x) * (q ** (n - popcnt_x))

        # Compute φ_S^(p)(x):
        # For each i in S: if x_i = -1 (bit set), use -√(q/p)
        #                  if x_i = +1 (bit not set), use √(p/q)
        overlap = bin(x & S).count("1")  # Number of bits set in both x and S
        phi = ((-sqrt_q_over_p) ** overlap) * (sqrt_p_over_q ** (popcnt_S - overlap))

        # f in ±1 convention: 0 → +1, 1 → -1
        f_x = -1.0 if truth_table[x] else 1.0

        val += pr * phi * f_x

    return val


def p_biased_sensitivity(f: "BooleanFunction", x: int, p: float = 0.5) -> int:
    """
    Compute the sensitivity of f at input x.

    This is the same as standard sensitivity (number of sensitive coordinates),
    independent of p. Included for API consistency with p_biased_average_sensitivity.

    Args:
        f: BooleanFunction to analyze
        x: Input point (integer)
        p: Bias parameter (not used, for API consistency)

    Returns:
        Number of coordinates i where f(x) ≠ f(x^i)
    """
    n = f.n_vars or 0
    if n == 0:
        return 0

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)
    f_x = truth_table[x]

    sens = 0
    for i in range(n):
        x_flip = x ^ (1 << i)
        if truth_table[x_flip] != f_x:
            sens += 1

    return sens


def p_biased_average_sensitivity(f: "BooleanFunction", p: float = 0.5) -> float:
    """
    Compute the average sensitivity under p-biased distribution μ_p.

    This computes:
        as_p(f) = E_{x ~ μ_p}[s(f, x)]

    where s(f, x) is the sensitivity at input x.

    This is Tal's `asMuP` function. Note that by Poincaré's inequality,
    this equals the p-biased total influence for Boolean functions.

    Args:
        f: BooleanFunction to analyze
        p: Bias parameter

    Returns:
        Average sensitivity under μ_p

    References:
        - Tal's BooleanFunc.py: asMuP
        - O'Donnell Proposition 8.28
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0,1), got {p}")

    n = f.n_vars or 0
    if n == 0:
        return 0.0

    q = 1.0 - p
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)

    total = 0.0
    size = 1 << n

    for x in range(size):
        # Probability of x under μ_p
        popcnt_x = bin(x).count("1")
        pr = (p**popcnt_x) * (q ** (n - popcnt_x))

        # Sensitivity at x
        sens = p_biased_sensitivity(f, x, p)

        total += pr * sens

    return total


def p_biased_total_influence_fourier(f: "BooleanFunction", p: float = 0.5) -> float:
    """
    Compute total p-biased influence via Fourier coefficients.

    The p-biased total influence via Fourier is:
        I^(p)[f] = (1 / (4p(1-p))) × Σ_S |S| · f̂(S)_p²

    The normalization factor 4p(1-p) is the variance of a single ±1 coordinate
    under the p-biased distribution. At p=0.5, this factor equals 1.

    This should equal the average sensitivity under μ_p (Poincaré inequality).

    Note: Tal's original `asFourierMuP` function computes Σ_S |S| f̂(S)_p²
    WITHOUT the normalization, which only equals the total influence at p=0.5.

    Args:
        f: BooleanFunction to analyze
        p: Bias parameter

    Returns:
        Total p-biased influence (via Fourier)

    References:
        - O'Donnell Theorem 8.32 (p-biased Poincaré inequality)
        - O'Donnell Proposition 8.28
    """
    if not (0 < p < 1):
        raise ValueError(f"p must be in (0,1), got {p}")

    n = f.n_vars or 0
    if n == 0:
        return 0.0

    size = 1 << n
    total = 0.0

    for S in range(size):
        coeff = p_biased_fourier_coefficient(f, p, S)
        popcnt_S = bin(S).count("1")
        total += coeff**2 * popcnt_S

    # Normalization factor: 4p(1-p) is the variance of a single coordinate
    # At p=0.5, this is 4 * 0.5 * 0.5 = 1, so no change
    normalization = 4.0 * p * (1.0 - p)

    return total / normalization


def parity_biased_coefficient(n: int, k: int, i: int) -> float:
    """
    Compute the p-biased Fourier coefficient of the parity function.

    For the parity function PAR_n(x) = x_1 ⊕ x_2 ⊕ ... ⊕ x_n (XOR of all bits),
    this computes a specific coefficient related to the bias.

    The formula uses Krawchouk-like recurrence:
        S = Σ_{j=0}^{i} (-1)^j * C(n-k, j) * C(k, i-j)
        result = S / C(n, i)

    where k relates to the bias via k = p*n (expected number of 1s).

    This is Tal's `parity_biased` function.

    Args:
        n: Number of variables
        k: Bias-related parameter (typically floor(p*n) or similar)
        i: Coefficient index

    Returns:
        The biased parity coefficient

    References:
        - Tal's BooleanFunc.py: parity_biased
        - Krawchouk polynomials in coding theory
    """
    from ..utils.math import over

    if i < 0 or i > n:
        return 0.0

    S = 0.0
    for j in range(i + 1):
        S += ((-1) ** j) * over(n - k, j) * over(k, i - j)

    denom = over(n, i)
    return S / float(denom) if denom > 0 else 0.0


class PBiasedAnalyzer:
    """
    Comprehensive p-biased analysis for Boolean functions.

    This class provides caching and convenient methods for analyzing
    Boolean functions under p-biased distributions.
    """

    def __init__(self, f: "BooleanFunction", p: float = 0.5):
        """
        Initialize p-biased analyzer.

        Args:
            f: BooleanFunction to analyze
            p: Bias parameter
        """
        self.function = f
        self.p = p
        self._coefficients: Optional[Dict[int, float]] = None

    @property
    def coefficients(self) -> Dict[int, float]:
        """Get cached p-biased Fourier coefficients."""
        if self._coefficients is None:
            self._coefficients = p_biased_fourier_coefficients(self.function, self.p)
        return self._coefficients

    def expectation(self) -> float:
        """Get E[f] under p-biased measure."""
        return p_biased_expectation(self.function, self.p)

    def variance(self) -> float:
        """Get Var[f] under p-biased measure."""
        return p_biased_variance(self.function, self.p)

    def influence(self, i: int) -> float:
        """Get p-biased influence of variable i."""
        return p_biased_influence(self.function, i, self.p)

    def influences(self) -> List[float]:
        """Get p-biased influences of all variables."""
        n = self.function.n_vars or 0
        return [self.influence(i) for i in range(n)]

    def total_influence(self) -> float:
        """Get total p-biased influence."""
        return p_biased_total_influence(self.function, self.p)

    def noise_stability(self, rho: float) -> float:
        """Get p-biased noise stability at correlation rho."""
        return p_biased_noise_stability(self.function, rho, self.p)

    def spectral_norm(self, level: int) -> float:
        """Get L2 norm of degree-level Fourier coefficients."""
        total = 0.0
        for S, coeff in self.coefficients.items():
            if bin(S).count("1") == level:
                total += coeff**2
        return np.sqrt(total)

    def max_influence(self) -> Tuple[int, float]:
        """Find variable with maximum p-biased influence."""
        n = self.function.n_vars or 0
        if n == 0:
            return (0, 0.0)

        influences = self.influences()
        max_idx = int(np.argmax(influences))
        return (max_idx, influences[max_idx])

    def average_sensitivity(self) -> float:
        """
        Compute average sensitivity under μ_p distribution.

        By Poincaré's inequality, this equals total_influence() for Boolean functions.
        """
        return p_biased_average_sensitivity(self.function, self.p)

    def total_influence_fourier(self) -> float:
        """
        Compute total influence via Fourier formula.

        This should equal average_sensitivity() and total_influence(),
        providing a cross-validation check.
        """
        return p_biased_total_influence_fourier(self.function, self.p)

    def fourier_coefficient(self, S: int) -> float:
        """
        Compute single Fourier coefficient using Tal's efficient formula.

        Args:
            S: Subset mask

        Returns:
            The p-biased Fourier coefficient f̂(S)_p
        """
        return p_biased_fourier_coefficient(self.function, self.p, S)

    def validate(self, tol: float = 1e-6) -> Dict[str, bool]:
        """
        Cross-validate p-biased computations.

        Checks that mathematically equivalent quantities match:
        - total_influence() ≈ average_sensitivity() ≈ total_influence_fourier()

        Args:
            tol: Tolerance for floating point comparison

        Returns:
            Dictionary of validation checks and their results
        """
        ti = self.total_influence()
        as_p = self.average_sensitivity()
        ti_f = self.total_influence_fourier()

        return {
            "total_influence ≈ average_sensitivity": abs(ti - as_p) < tol,
            "total_influence ≈ fourier_formula": abs(ti - ti_f) < tol,
            "average_sensitivity ≈ fourier_formula": abs(as_p - ti_f) < tol,
        }

    def summary(self) -> str:
        """Get human-readable summary of p-biased analysis."""
        lines = [
            f"P-biased Analysis (p={self.p:.4f})",
            f"  Variables: {self.function.n_vars}",
            f"  Expectation: {self.expectation():.6f}",
            f"  Variance: {self.variance():.6f}",
            f"  Total Influence: {self.total_influence():.6f}",
            f"  Average Sensitivity (μ_p): {self.average_sensitivity():.6f}",
            f"  Non-zero Fourier coefficients: {len(self.coefficients)}",
        ]
        return "\n".join(lines)
