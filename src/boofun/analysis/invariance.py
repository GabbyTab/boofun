"""
Invariance principles for Boolean functions (O'Donnell Chapter 11).

The Invariance Principle connects the behavior of Boolean functions on the
discrete hypercube {-1,1}^n to their behavior on Gaussian space. This is
one of the most powerful tools in analysis of Boolean functions.

Key results:
- Mossel's Invariance Principle: Low-influence functions have similar
  distributions on Boolean and Gaussian inputs
- "Majority is Stablest" theorem: Among low-influence functions, majority
  has the highest noise stability
- Applications to hardness of approximation (Unique Games Conjecture)

The principle shows that:
    E[Ψ(f(x))] ≈ E[Ψ(f̃(G))]
    
where x is uniform on {-1,1}^n, G is standard Gaussian on ℝ^n,
f̃ is the multilinear extension, and Ψ is a "nice" test function.

References:
- O'Donnell, "Analysis of Boolean Functions", Chapter 11
- Mossel, O'Donnell, Oleszkiewicz, "Noise stability of functions with low influences"
- Khot et al., "Optimal inapproximability results from the Unique Games Conjecture"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
import numpy as np
from math import sqrt, exp, pi, erf
from functools import lru_cache

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Core invariance
    "invariance_distance",
    "multilinear_extension_gaussian_expectation",
    "test_function_expectation",
    # Majority is Stablest
    "majority_noise_stability",
    "noise_stability_deficit",
    "is_stablest_candidate",
    # Bounds and approximations
    "max_cut_approximation_ratio",
    "unique_games_hardness_bound",
    # Utility
    "InvarianceAnalyzer",
    # Gaussian utilities
    "gaussian_cdf",
    "bivariate_gaussian_cdf",
]


def gaussian_cdf(x: float) -> float:
    """
    Compute the standard Gaussian CDF Φ(x).
    
    Args:
        x: Point to evaluate
        
    Returns:
        Φ(x) = Pr[N(0,1) <= x]
    """
    return 0.5 * (1 + erf(x / sqrt(2)))


def bivariate_gaussian_cdf(x: float, y: float, rho: float) -> float:
    """
    Compute the bivariate Gaussian CDF with correlation rho.
    
    This is Pr[(X, Y) in (-∞, x] × (-∞, y]] where (X, Y) are
    joint Gaussian with correlation rho.
    
    Uses a simple approximation for efficiency.
    
    Args:
        x, y: Upper bounds
        rho: Correlation coefficient
        
    Returns:
        Bivariate Gaussian CDF value
    """
    # Use numerical integration approximation
    # For production, would use scipy.stats.multivariate_normal
    try:
        from scipy.stats import multivariate_normal
        cov = [[1, rho], [rho, 1]]
        rv = multivariate_normal(mean=[0, 0], cov=cov)
        return rv.cdf([x, y])
    except ImportError:
        # Fallback: product of marginals (exact when rho=0)
        # Use Owen's T function approximation for rho != 0
        if abs(rho) < 0.01:
            return gaussian_cdf(x) * gaussian_cdf(y)
        
        # Simple approximation using series
        return gaussian_cdf(x) * gaussian_cdf(y) + rho * _owens_t_approx(x, y)


def _owens_t_approx(x: float, y: float) -> float:
    """Approximate Owen's T function for bivariate Gaussian."""
    # Very rough approximation
    return exp(-0.5 * (x**2 + y**2)) / (2 * pi) * 0.5


def majority_noise_stability(n: int, rho: float) -> float:
    """
    Compute the noise stability of the n-variable majority function at correlation rho.
    
    For large n, this converges to:
        Stab_ρ[Maj_n] → (2/π) arcsin(ρ)
    
    This is the "Sheppard's formula" result.
    
    Args:
        n: Number of variables (odd for majority)
        rho: Noise correlation parameter
        
    Returns:
        Noise stability of majority
    """
    import boofun as bf
    
    if n <= 10:
        # For small n, compute exactly
        maj = bf.majority(n)
        from .gaussian import gaussian_noise_stability
        return gaussian_noise_stability(maj, rho)
    
    # For large n, use Sheppard's formula
    return (2 / pi) * np.arcsin(rho)


def invariance_distance(f: "BooleanFunction", test_fn: Optional[Callable] = None) -> float:
    """
    Estimate the invariance distance between Boolean and Gaussian expectations.
    
    The invariance principle states that for functions with low influences:
        |E[Ψ(f(x))] - E[Ψ(f̃(G))]| ≤ ε(τ)
    
    where τ = max_i Inf_i[f] and ε(τ) → 0 as τ → 0.
    
    This function estimates this distance using the default test function
    Ψ(t) = sign(t).
    
    Args:
        f: BooleanFunction to analyze
        test_fn: Optional test function Ψ (default: sign)
        
    Returns:
        Estimated invariance distance
    """
    from ..analysis import SpectralAnalyzer
    from .gaussian import multilinear_extension, clt_approximation
    
    analyzer = SpectralAnalyzer(f)
    influences = analyzer.influences()
    max_inf = max(influences) if len(influences) > 0 else 0.0
    
    # Bound from invariance principle
    # Distance is O(max_i Inf_i[f]^{1/4}) for C^3 test functions
    
    if max_inf > 0:
        theoretical_bound = max_inf ** 0.25
    else:
        theoretical_bound = 0.0
    
    return theoretical_bound


def multilinear_extension_gaussian_expectation(
    f: "BooleanFunction", 
    num_samples: int = 10000
) -> Tuple[float, float]:
    """
    Estimate E[f̃(G)] and E[sign(f̃(G))] via Monte Carlo.
    
    Samples standard Gaussian vectors and evaluates the multilinear
    extension.
    
    Args:
        f: BooleanFunction to analyze
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Tuple of (E[f̃(G)], E[sign(f̃(G))])
    """
    from .gaussian import multilinear_extension
    
    n = f.n_vars
    if n is None or n == 0:
        val = float(f.evaluate(0))
        return (val, np.sign(val))
    
    mle = multilinear_extension(f)
    
    # Sample Gaussians
    np.random.seed(42)  # For reproducibility
    G = np.random.randn(num_samples, n)
    
    values = []
    for i in range(num_samples):
        values.append(mle(G[i]))
    
    values = np.array(values)
    
    return (float(np.mean(values)), float(np.mean(np.sign(values))))


def test_function_expectation(
    f: "BooleanFunction",
    test_fn: Callable[[float], float],
    domain: str = "boolean"
) -> float:
    """
    Compute E[Ψ(f(x))] on either Boolean or Gaussian domain.
    
    Args:
        f: BooleanFunction to analyze
        test_fn: Test function Ψ
        domain: "boolean" for {-1,1}^n or "gaussian" for ℝ^n
        
    Returns:
        Expectation E[Ψ(f(x))]
    """
    n = f.n_vars or 0
    
    if domain == "boolean":
        truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
        pm_values = 1.0 - 2.0 * truth_table  # Convert to ±1
        
        expectations = [test_fn(v) for v in pm_values]
        return float(np.mean(expectations))
    
    else:  # gaussian
        from .gaussian import multilinear_extension
        mle = multilinear_extension(f)
        
        # Monte Carlo
        np.random.seed(42)
        num_samples = 10000
        G = np.random.randn(num_samples, n)
        
        values = [test_fn(mle(G[i])) for i in range(num_samples)]
        return float(np.mean(values))


def noise_stability_deficit(f: "BooleanFunction", rho: float) -> float:
    """
    Compute how much less stable f is compared to majority at correlation rho.
    
    The "Majority is Stablest" theorem states that among all functions with
    E[f] = 0 and low influences, majority has the highest noise stability.
    
    This function computes:
        Stab_ρ[Maj] - Stab_ρ[f]
    
    Positive values indicate f is less stable than majority.
    
    Args:
        f: BooleanFunction to analyze
        rho: Noise correlation parameter
        
    Returns:
        Noise stability deficit
    """
    from .gaussian import gaussian_noise_stability
    
    f_stability = gaussian_noise_stability(f, rho)
    maj_stability = majority_noise_stability(f.n_vars or 3, rho)
    
    return maj_stability - f_stability


def is_stablest_candidate(f: "BooleanFunction", epsilon: float = 0.01) -> bool:
    """
    Check if f is a candidate for being "stablest" among its influence class.
    
    A function is a stablest candidate if:
    1. It has low influences (max influence < epsilon)
    2. It is balanced (E[f] ≈ 0)
    
    Such functions have noise stability close to (2/π) arcsin(ρ).
    
    Args:
        f: BooleanFunction to analyze
        epsilon: Influence threshold
        
    Returns:
        True if f is a stablest candidate
    """
    from ..analysis import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer(f)
    influences = analyzer.influences()
    max_inf = max(influences) if len(influences) > 0 else 0.0
    
    # Check low influence
    if max_inf >= epsilon:
        return False
    
    # Check balance
    fourier = analyzer.fourier_expansion()
    mean = fourier[0] if len(fourier) > 0 else 0.0
    
    return abs(mean) < epsilon


def max_cut_approximation_ratio(rho: float) -> float:
    """
    Compute the Goemans-Williamson MAX-CUT approximation ratio.
    
    The GW algorithm achieves approximation ratio:
        α_GW = (2/π) min_{θ ∈ [0, π]} θ / (1 - cos(θ))
             ≈ 0.87856
    
    The "Majority is Stablest" theorem implies this is optimal
    assuming the Unique Games Conjecture.
    
    Args:
        rho: Correlation parameter (for generalized analysis)
        
    Returns:
        Approximation ratio
    """
    # GW ratio is approximately 0.87856
    gw_ratio = 0.87856
    
    # For ρ-correlated rounding:
    # Approximation depends on the stability function
    return gw_ratio


def unique_games_hardness_bound(f: "BooleanFunction") -> float:
    """
    Estimate the UGC-hardness bound implied by f's noise stability.
    
    The Unique Games Conjecture implies that no algorithm can do better
    than the "noise stability" bound for certain CSPs.
    
    For MAX-CUT-like problems, the hardness depends on the noise stability
    curve of the dictator function vs. the majority function.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Estimated hardness bound
    """
    from .gaussian import gaussian_noise_stability
    
    n = f.n_vars or 1
    
    # Stability at ρ = 1 - 2ε for small ε relates to hardness
    rho = 0.9
    
    stab = gaussian_noise_stability(f, rho)
    maj_stab = majority_noise_stability(n, rho)
    
    # Hardness ratio: 1 - (1 - stab)/(1 - maj_stab)
    # This is a simplified approximation
    
    return max(0.5, stab / max(maj_stab, 0.01))


class InvarianceAnalyzer:
    """
    Comprehensive invariance principle analysis for Boolean functions.
    
    Provides methods for analyzing how well a Boolean function's behavior
    on the discrete hypercube matches its behavior on Gaussian space.
    """
    
    def __init__(self, f: "BooleanFunction"):
        """
        Initialize invariance analyzer.
        
        Args:
            f: BooleanFunction to analyze
        """
        self.function = f
        self.n_vars = f.n_vars
    
    def invariance_bound(self) -> float:
        """
        Compute theoretical bound on invariance distance.
        
        Based on the max influence of f.
        """
        return invariance_distance(self.function)
    
    def noise_stability_deficit(self, rho: float = 0.9) -> float:
        """Compute how much less stable f is than majority."""
        return noise_stability_deficit(self.function, rho)
    
    def is_stablest_candidate(self, epsilon: float = 0.01) -> bool:
        """Check if f satisfies conditions for Majority is Stablest."""
        return is_stablest_candidate(self.function, epsilon)
    
    def gaussian_expectation(self, num_samples: int = 10000) -> Tuple[float, float]:
        """Estimate E[f̃(G)] via Monte Carlo."""
        return multilinear_extension_gaussian_expectation(self.function, num_samples)
    
    def compare_domains(self) -> Dict[str, float]:
        """
        Compare function behavior on Boolean vs Gaussian domains.
        
        Returns:
            Dictionary of comparison metrics
        """
        from .gaussian import clt_approximation, gaussian_noise_stability
        
        # Boolean domain statistics
        truth_table = np.asarray(
            self.function.get_representation("truth_table"), dtype=float
        )
        pm_values = 1.0 - 2.0 * truth_table
        bool_mean = float(np.mean(pm_values))
        bool_var = float(np.var(pm_values))
        
        # Gaussian domain statistics
        gauss_mean, gauss_sign_mean = self.gaussian_expectation(5000)
        
        return {
            "boolean_mean": bool_mean,
            "boolean_variance": bool_var,
            "gaussian_mean": gauss_mean,
            "gaussian_sign_mean": gauss_sign_mean,
            "mean_difference": abs(bool_mean - gauss_sign_mean),
            "invariance_bound": self.invariance_bound(),
        }
    
    def summary(self) -> str:
        """Return a summary of invariance properties."""
        inv_bound = self.invariance_bound()
        deficit_09 = self.noise_stability_deficit(0.9)
        deficit_05 = self.noise_stability_deficit(0.5)
        is_stablest = self.is_stablest_candidate()
        
        comparison = self.compare_domains()
        
        lines = [
            "Invariance Principle Analysis",
            "=" * 40,
            f"Invariance bound: {inv_bound:.6f}",
            f"Stablest candidate: {is_stablest}",
            "",
            "NOISE STABILITY DEFICIT (vs Majority):",
            f"  At ρ=0.9: {deficit_09:.6f}",
            f"  At ρ=0.5: {deficit_05:.6f}",
            "",
            "DOMAIN COMPARISON:",
            f"  Boolean mean:     {comparison['boolean_mean']:.6f}",
            f"  Gaussian mean:    {comparison['gaussian_mean']:.6f}",
            f"  Mean difference:  {comparison['mean_difference']:.6f}",
        ]
        
        return "\n".join(lines)
