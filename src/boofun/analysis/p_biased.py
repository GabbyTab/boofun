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

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "p_biased_fourier_coefficients",
    "p_biased_influence",
    "p_biased_total_influence",
    "p_biased_noise_stability",
    "p_biased_expectation",
    "p_biased_variance",
    "biased_measure_mass",
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
    return (p ** k) * ((1.0 - p) ** (n - k))


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
        prob = (p ** k) * ((1.0 - p) ** (n - k))
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
        prob = (p ** k) * ((1.0 - p) ** (n - k))
        fx = truth_table_pm[x]
        ef += fx * prob
        ef2 += (fx ** 2) * prob
    
    return ef2 - ef ** 2


def p_biased_fourier_coefficients(
    f: "BooleanFunction", 
    p: float = 0.5
) -> Dict[int, float]:
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
            prob = (p ** k) * ((1.0 - p) ** (n - k))
            
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
        prob_rest = (p ** k_rest) * ((1.0 - p) ** (n - 1 - k_rest))
        
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
        stability += (rho ** k) * (coeff ** 2)
    
    return stability


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
                total += coeff ** 2
        return np.sqrt(total)
    
    def max_influence(self) -> Tuple[int, float]:
        """Find variable with maximum p-biased influence."""
        n = self.function.n_vars or 0
        if n == 0:
            return (0, 0.0)
        
        influences = self.influences()
        max_idx = int(np.argmax(influences))
        return (max_idx, influences[max_idx])
    
    def summary(self) -> str:
        """Get human-readable summary of p-biased analysis."""
        lines = [
            f"P-biased Analysis (p={self.p:.4f})",
            f"  Variables: {self.function.n_vars}",
            f"  Expectation: {self.expectation():.6f}",
            f"  Variance: {self.variance():.6f}",
            f"  Total Influence: {self.total_influence():.6f}",
            f"  Non-zero Fourier coefficients: {len(self.coefficients)}",
        ]
        return "\n".join(lines)
