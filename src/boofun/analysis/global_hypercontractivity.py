"""
Global Hypercontractivity Module

Implements concepts from Keevash, Lifshitz, Long & Minzer's
"Global hypercontractivity and its applications" (arXiv:1906.05039).

This module provides tools for analyzing Boolean functions under p-biased
measures, particularly for "global" functions that don't depend too much
on any small set of coordinates.

Key concepts:
- Generalized influences: I_S(f) for sets S ⊆ [n]
- Global functions: Functions with small generalized influences
- p-biased Fourier analysis
- Sharp threshold phenomena

Reference:
    Keevash, P., Lifshitz, N., Long, E., & Minzer, D. (2019).
    Global hypercontractivity and its applications.
    arXiv:1906.05039
"""

from itertools import combinations
from typing import TYPE_CHECKING, Dict, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction


def sigma(p: float) -> float:
    """
    Compute σ(p) = √(p(1-p)), the standard deviation of a p-biased bit.

    Args:
        p: Bias parameter in (0, 1)

    Returns:
        Standard deviation σ(p)
    """
    if p <= 0 or p >= 1:
        return 0.0
    return np.sqrt(p * (1 - p))


def lambda_p(p: float) -> float:
    """
    Compute λ(p) = E[(χ_i^p)^4], the 4th moment of a p-biased character.

    This quantity controls the failure of standard hypercontractivity
    for small p. For p = 1/2, λ = 3. As p → 0 or p → 1, λ → ∞.

    From Section II.1 of the paper:
    λ = σ^{-2}((1-p)^3 + p^3)

    Args:
        p: Bias parameter in (0, 1)

    Returns:
        4th moment λ(p), or infinity if p is at boundary
    """
    s = sigma(p)
    if s == 0:
        return float("inf")
    return s ** (-2) * ((1 - p) ** 3 + p**3)


def p_biased_character(x: np.ndarray, S: Set[int], p: float) -> float:
    """
    Compute the p-biased character χ_S^p(x).

    χ_S^p(x) = Π_{i∈S} (x_i - p)/σ

    where σ = √(p(1-p)).

    Args:
        x: Binary input vector
        S: Set of variable indices
        p: Bias parameter

    Returns:
        Value of χ_S^p(x)
    """
    s = sigma(p)
    if s == 0:
        return 0.0

    result = 1.0
    for i in S:
        result *= (x[i] - p) / s
    return result


def generalized_influence(f: "BooleanFunction", S: Set[int], p: float = 0.5) -> float:
    """
    Compute the generalized S-influence I_S(f) under μ_p.

    From Definition I.1.2 and equation (2) in the paper:
    I_S(f) = σ^{-2|S|} ||D_S(f)||_2^2 = σ^{-2|S|} Σ_{T⊇S} f̂(T)^2

    Args:
        f: Boolean function to analyze
        S: Set of variable indices
        p: Bias parameter (default 0.5 for uniform measure)

    Returns:
        Generalized S-influence
    """
    from . import SpectralAnalyzer

    n = f.n_vars
    s = sigma(p)

    # Get Fourier coefficients (under uniform for simplicity)
    analyzer = SpectralAnalyzer(f)
    fourier = analyzer.fourier_expansion()

    # Sum over all T ⊇ S
    total = 0.0
    for T_idx in range(2**n):
        T = set(i for i in range(n) if (T_idx >> i) & 1)
        if S.issubset(T):
            total += fourier[T_idx] ** 2

    if len(S) == 0:
        return total  # I_∅(f) = ||f||_2^2

    return s ** (-2 * len(S)) * total


def is_alpha_global(
    f: "BooleanFunction", alpha: float, max_set_size: int = 3, p: float = 0.5
) -> Tuple[bool, Dict]:
    """
    Check if f has α-small generalized influences (Definition I.1.2).

    A function has α-small generalized influences if:
    I_S(f) ≤ α·E[f^2] for all S ⊆ [n]

    Args:
        f: Boolean function to test
        alpha: Threshold for generalized influences
        max_set_size: Maximum |S| to check (for efficiency)
        p: Bias parameter

    Returns:
        Tuple of (is_global, details_dict) where details_dict contains:
        - max_generalized_influence: Largest I_S found
        - worst_set: The set S achieving the maximum
        - threshold: α·E[f^2]
        - all_influences: Dictionary of all computed I_S values
    """
    n = f.n_vars
    ef2 = 1.0  # E[f^2] = 1 for Boolean functions in ±1 representation

    max_influence = 0.0
    worst_set = set()
    all_influences = {}

    # Check all sets up to max_set_size
    for k in range(1, min(max_set_size + 1, n + 1)):
        for S_tuple in combinations(range(n), k):
            S = set(S_tuple)
            inf_S = generalized_influence(f, S, p)
            all_influences[S_tuple] = inf_S

            if inf_S > max_influence:
                max_influence = inf_S
                worst_set = S

    is_global = max_influence <= alpha * ef2

    return is_global, {
        "max_generalized_influence": max_influence,
        "worst_set": worst_set,
        "threshold": alpha * ef2,
        "all_influences": all_influences,
        "alpha": alpha,
        "p": p,
    }


def p_biased_expectation(f: "BooleanFunction", p: float, samples: int = 10000) -> float:
    """
    Estimate E_μp[f(x)] via Monte Carlo sampling.

    Args:
        f: Boolean function
        p: Bias parameter
        samples: Number of Monte Carlo samples

    Returns:
        Estimated expectation under μ_p
    """
    n = f.n_vars
    total = 0.0

    for _ in range(samples):
        x = (np.random.random(n) < p).astype(int)
        total += f.evaluate(x)

    return total / samples


def p_biased_influence(f: "BooleanFunction", i: int, p: float, samples: int = 5000) -> float:
    """
    Estimate Inf_i^p[f] under the p-biased measure.

    Inf_i^p[f] = Pr_{x~μp}[f(x) ≠ f(x^i)]

    where x^i differs from x only in coordinate i.

    Args:
        f: Boolean function
        i: Variable index
        p: Bias parameter
        samples: Number of Monte Carlo samples

    Returns:
        Estimated p-biased influence of variable i
    """
    n = f.n_vars
    count = 0

    for _ in range(samples):
        x = (np.random.random(n) < p).astype(int)
        x0 = x.copy()
        x0[i] = 0
        x1 = x.copy()
        x1[i] = 1

        # Use bit_strings=True to interpret arrays as bit vectors
        if f.evaluate(x0, bit_strings=True) != f.evaluate(x1, bit_strings=True):
            count += 1

    return count / samples


def p_biased_total_influence(f: "BooleanFunction", p: float, samples: int = 3000) -> float:
    """
    Estimate I^p[f] = Σ_i Inf_i^p[f] under the p-biased measure.

    From equation (1) in the paper:
    I(f) = (p(1-p))^{-1} Σ_S |S| f̂(S)^2

    Args:
        f: Boolean function
        p: Bias parameter
        samples: Number of Monte Carlo samples per variable

    Returns:
        Estimated total influence under μ_p
    """
    return sum(p_biased_influence(f, i, p, samples) for i in range(f.n_vars))


def noise_stability_p_biased(
    f: "BooleanFunction", rho: float, p: float, samples: int = 5000
) -> float:
    """
    Estimate noise stability S_ρ(f) under the p-biased measure.

    S_ρ(f) = E_{x~μp}[f(x)·(T_ρf)(x)]

    where (T_ρf)(x) = E_{y~N_ρ(x)}[f(y)] and y is obtained by
    keeping each bit with probability ρ or resampling from μp.

    Args:
        f: Boolean function
        rho: Noise correlation parameter
        p: Bias parameter
        samples: Number of Monte Carlo samples

    Returns:
        Estimated noise stability
    """
    n = f.n_vars
    total = 0.0

    for _ in range(samples):
        # Sample x from μp
        x = (np.random.random(n) < p).astype(int)

        # Sample y from N_ρ(x)
        y = x.copy()
        for i in range(n):
            if np.random.random() > rho:
                y[i] = int(np.random.random() < p)

        # Compute contribution
        f_x = 2 * f.evaluate(x) - 1  # Convert to ±1
        f_y = 2 * f.evaluate(y) - 1
        total += f_x * f_y

    return total / samples


def threshold_curve(f: "BooleanFunction", p_range: np.ndarray, samples: int = 2000) -> np.ndarray:
    """
    Compute μ_p(f) for each p in the range.

    This is useful for visualizing sharp threshold phenomena.

    Args:
        f: Boolean function
        p_range: Array of p values to evaluate
        samples: Number of samples per p value

    Returns:
        Array of μ_p(f) values
    """
    return np.array([p_biased_expectation(f, p, samples) for p in p_range])


def find_critical_p(f: "BooleanFunction", samples: int = 3000, tolerance: float = 0.01) -> float:
    """
    Find the critical probability p_c where μ_p(f) ≈ 0.5.

    For monotone functions, this is the "threshold" of the function.

    Args:
        f: Boolean function (should be monotone for meaningful result)
        samples: Number of samples for expectation estimation
        tolerance: Tolerance for binary search

    Returns:
        Critical probability p_c
    """
    # Binary search for p where μ_p(f) ≈ 0.5
    lo, hi = 0.001, 0.999

    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        mu = p_biased_expectation(f, mid, samples)

        if mu < 0.5:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


def hypercontractivity_bound(f: "BooleanFunction", p: float = 0.5) -> Dict:
    """
    Compute the hypercontractivity bound for a function.

    From Theorem I.1.3: If f has α-small generalized influences, then
    ||T_{1/5}f||_4 ≤ α^{1/4} ||f||_2

    Args:
        f: Boolean function
        p: Bias parameter

    Returns:
        Dictionary with:
        - alpha: Maximum generalized influence / E[f^2]
        - bound: α^{1/4} (the hypercontractive bound)
        - is_global: Whether the function is global
        - details: Full globality check results
    """
    is_global, details = is_alpha_global(f, alpha=1.0, max_set_size=3, p=p)

    # α is the ratio of max I_S to E[f^2]
    alpha = details["max_generalized_influence"]
    bound = alpha ** (1 / 4)

    return {
        "alpha": alpha,
        "bound": bound,
        "is_global": is_global,
        "worst_set": details["worst_set"],
        "details": details,
    }


class GlobalHypercontractivityAnalyzer:
    """
    Analyzer for global hypercontractivity properties.

    This class provides comprehensive analysis of Boolean functions
    under p-biased measures, following the framework of Keevash et al.
    """

    def __init__(self, f: "BooleanFunction", p: float = 0.5):
        """
        Initialize the analyzer.

        Args:
            f: Boolean function to analyze
            p: Default bias parameter
        """
        self.f = f
        self.p = p
        self.n_vars = f.n_vars

        # Cache for expensive computations
        self._globality_cache = {}
        self._influence_cache = {}

    def sigma(self) -> float:
        """Return σ(p) for the current bias."""
        return sigma(self.p)

    def lambda_p(self) -> float:
        """Return λ(p) for the current bias."""
        return lambda_p(self.p)

    def is_global(self, alpha: float = 0.5, max_set_size: int = 3) -> Tuple[bool, Dict]:
        """
        Check if the function is α-global.

        Args:
            alpha: Threshold parameter
            max_set_size: Maximum set size to check

        Returns:
            Tuple of (is_global, details)
        """
        cache_key = (alpha, max_set_size, self.p)
        if cache_key not in self._globality_cache:
            self._globality_cache[cache_key] = is_alpha_global(self.f, alpha, max_set_size, self.p)
        return self._globality_cache[cache_key]

    def generalized_influence(self, S: Set[int]) -> float:
        """
        Compute I_S(f).

        Args:
            S: Set of variable indices

        Returns:
            Generalized S-influence
        """
        S_key = frozenset(S)
        if S_key not in self._influence_cache:
            self._influence_cache[S_key] = generalized_influence(self.f, S, self.p)
        return self._influence_cache[S_key]

    def expectation(self, samples: int = 5000) -> float:
        """Estimate E_μp[f]."""
        return p_biased_expectation(self.f, self.p, samples)

    def total_influence(self, samples: int = 3000) -> float:
        """Estimate I^p[f]."""
        return p_biased_total_influence(self.f, self.p, samples)

    def noise_stability(self, rho: float, samples: int = 5000) -> float:
        """Estimate S_ρ(f) under μp."""
        return noise_stability_p_biased(self.f, rho, self.p, samples)

    def hypercontractive_bound(self) -> Dict:
        """Compute the hypercontractivity bound."""
        return hypercontractivity_bound(self.f, self.p)

    def threshold_curve(self, p_range: np.ndarray = None, samples: int = 1000) -> np.ndarray:
        """
        Compute the threshold curve μ_p(f) over a range of p.

        Args:
            p_range: Array of p values (default: 0.01 to 0.99)
            samples: Monte Carlo samples per point

        Returns:
            Array of μ_p(f) values
        """
        if p_range is None:
            p_range = np.linspace(0.01, 0.99, 50)
        return threshold_curve(self.f, p_range, samples)

    def summary(self, samples: int = 1000) -> Dict:
        """
        Comprehensive summary of the function's global hypercontractivity properties.

        Args:
            samples: Monte Carlo samples for numerical estimates

        Returns:
            Dictionary with all key properties
        """
        is_global, details = self.is_global(alpha=0.5)
        hc_bound = self.hypercontractive_bound()

        return {
            "n_vars": self.n_vars,
            "bias_p": self.p,
            "sigma_p": self.sigma(),
            "lambda_p": self.lambda_p(),
            "is_global_alpha_0.5": is_global,
            "max_generalized_influence": details["max_generalized_influence"],
            "worst_set": details["worst_set"],
            "hypercontractive_bound": hc_bound["bound"],
            "expectation_mu_p": self.expectation(samples),
            "total_influence": self.total_influence(samples),
            "noise_stability_0.9": self.noise_stability(0.9, samples),
        }


# Module exports
__all__ = [
    "sigma",
    "lambda_p",
    "p_biased_character",
    "generalized_influence",
    "is_alpha_global",
    "p_biased_expectation",
    "p_biased_influence",
    "p_biased_total_influence",
    "noise_stability_p_biased",
    "threshold_curve",
    "find_critical_p",
    "hypercontractivity_bound",
    "GlobalHypercontractivityAnalyzer",
]
