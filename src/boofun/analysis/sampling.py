"""
Boolean functions as random variables: sampling and estimation.

This module provides a probabilistic view of Boolean functions, enabling:
- Sampling from the hypercube under uniform and p-biased distributions
- Spectral sampling (sampling subsets proportional to Fourier weight)
- Monte Carlo estimation of Fourier coefficients and influences
- Statistical analysis (expectation, variance, covariance)

This aligns with O'Donnell's "Analysis of Boolean Functions" Chapters 1-3,
which treat Boolean functions as random variables on the discrete cube.

Key concepts:
- f: {-1,+1}^n → R as a random variable with E[f] = f̂(∅)
- Fourier weight W^k[f] = Σ_{|S|=k} f̂(S)² forms a probability distribution
- Spectral sampling: draw S with Pr[S] ∝ f̂(S)²
- Monte Carlo: estimate f̂(S) = E[f(x)χ_S(x)] via sampling

References:
- O'Donnell Chapter 1: Fourier expansion and basic identities
- O'Donnell Chapter 2: Influences and their estimation
- O'Donnell Chapter 3: Learning and sampling algorithms
- Goldreich-Levin algorithm for finding heavy Fourier coefficients
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Sampling functions
    "sample_uniform",
    "sample_biased",
    "sample_spectral",
    "sample_input_output_pairs",
    # Estimation functions
    "estimate_fourier_coefficient",
    "estimate_influence",
    "estimate_expectation",
    "estimate_variance",
    "estimate_total_influence",
    # Random variable view
    "RandomVariableView",
    "SpectralDistribution",
]


def sample_uniform(
    n: int,
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample uniformly from {0,1}^n.
    
    Args:
        n: Number of variables
        n_samples: Number of samples to draw
        rng: Random number generator (default: numpy default)
        
    Returns:
        Array of shape (n_samples,) with integer inputs in [0, 2^n)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    return rng.integers(0, 1 << n, size=n_samples, dtype=np.int64)


def sample_biased(
    n: int,
    p: float,
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample from the p-biased distribution μ_p on {0,1}^n.
    
    Each coordinate is independently 1 with probability p, 0 with probability 1-p.
    
    Args:
        n: Number of variables
        p: Bias parameter (Pr[bit = 1])
        n_samples: Number of samples
        rng: Random number generator
        
    Returns:
        Array of shape (n_samples,) with integer inputs
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if not (0 <= p <= 1):
        raise ValueError(f"p must be in [0,1], got {p}")
    
    # Generate each bit independently
    bits = rng.random((n_samples, n)) < p
    
    # Convert bit arrays to integers
    powers = 1 << np.arange(n)
    return np.sum(bits * powers, axis=1).astype(np.int64)


def sample_input_output_pairs(
    f: "BooleanFunction",
    n_samples: int,
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample (input, output) pairs from a Boolean function.
    
    Draws inputs from the p-biased distribution and evaluates f.
    
    Args:
        f: BooleanFunction to sample from
        n_samples: Number of samples
        p: Bias parameter (0.5 = uniform)
        rng: Random number generator
        
    Returns:
        Tuple of (inputs, outputs) where:
        - inputs: array of shape (n_samples,) with integer inputs
        - outputs: array of shape (n_samples,) with function values in {0,1}
    """
    n = f.n_vars or 0
    if n == 0:
        val = int(f.evaluate(0))
        return np.zeros(n_samples, dtype=np.int64), np.full(n_samples, val, dtype=np.int8)
    
    if p == 0.5:
        inputs = sample_uniform(n, n_samples, rng)
    else:
        inputs = sample_biased(n, p, n_samples, rng)
    
    # Evaluate function at each input
    outputs = np.array([int(f.evaluate(int(x))) for x in inputs], dtype=np.int8)
    
    return inputs, outputs


def sample_spectral(
    f: "BooleanFunction",
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Sample subsets S with probability proportional to f̂(S)².
    
    This is "spectral sampling" - drawing from the Fourier weight distribution.
    By Parseval's identity, this is a valid probability distribution for
    non-constant functions (sum of weights = 1 for ±1-valued functions).
    
    Useful for:
    - Finding heavy Fourier coefficients
    - Learning algorithms (Goldreich-Levin)
    - Analyzing spectral structure
    
    Args:
        f: BooleanFunction to sample from
        n_samples: Number of samples
        rng: Random number generator
        
    Returns:
        Array of shape (n_samples,) with subset masks (integers)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = f.n_vars or 0
    if n == 0:
        return np.zeros(n_samples, dtype=np.int64)
    
    # Compute Fourier coefficients
    from . import SpectralAnalyzer
    analyzer = SpectralAnalyzer(f)
    coeffs = analyzer.fourier_expansion()
    
    # Compute weights (squared coefficients)
    weights = coeffs ** 2
    total_weight = np.sum(weights)
    
    if total_weight < 1e-15:
        # Constant function - all weight on empty set
        return np.zeros(n_samples, dtype=np.int64)
    
    # Normalize to probability distribution
    probs = weights / total_weight
    
    # Sample from the distribution
    subsets = rng.choice(len(probs), size=n_samples, p=probs)
    
    return subsets.astype(np.int64)


def estimate_fourier_coefficient(
    f: "BooleanFunction",
    S: int,
    n_samples: int,
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    return_confidence: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Estimate Fourier coefficient f̂(S) via Monte Carlo sampling.
    
    Uses the identity f̂(S) = E_x[f(x) χ_S(x)] where χ_S(x) = (-1)^{⟨x,S⟩}.
    
    For p-biased sampling, uses the adjusted formula from O'Donnell Chapter 8.
    
    Args:
        f: BooleanFunction to analyze
        S: Subset mask (integer)
        n_samples: Number of samples
        p: Bias parameter (0.5 = uniform)
        rng: Random number generator
        return_confidence: If True, return (estimate, std_error)
        
    Returns:
        Estimated f̂(S), or (estimate, std_error) if return_confidence=True
        
    Note:
        Error scales as O(1/√n_samples). For 1% relative error, need ~10000 samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = f.n_vars or 0
    if n == 0:
        val = 1.0 - 2.0 * float(f.evaluate(0))  # Convert to ±1
        result = val if S == 0 else 0.0
        return (result, 0.0) if return_confidence else result
    
    # Sample inputs
    inputs, outputs = sample_input_output_pairs(f, n_samples, p, rng)
    
    # Convert outputs to ±1 (O'Donnell convention: 0→+1, 1→-1)
    f_vals = 1.0 - 2.0 * outputs.astype(float)
    
    # Compute χ_S(x) = (-1)^{popcount(x & S)}
    inner_products = np.array([bin(int(x) & S).count("1") for x in inputs])
    chi_vals = 1.0 - 2.0 * (inner_products % 2)
    
    # Estimate: E[f(x) χ_S(x)]
    products = f_vals * chi_vals
    estimate = np.mean(products)
    
    if return_confidence:
        std_error = np.std(products) / np.sqrt(n_samples)
        return (float(estimate), float(std_error))
    
    return float(estimate)


def estimate_influence(
    f: "BooleanFunction",
    i: int,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
    return_confidence: bool = False
) -> Union[float, Tuple[float, float]]:
    """
    Estimate influence of variable i via Monte Carlo sampling.
    
    Uses the identity Inf_i[f] = Pr_x[f(x) ≠ f(x^{(i)})].
    
    Args:
        f: BooleanFunction to analyze
        i: Variable index (0-indexed)
        n_samples: Number of samples
        rng: Random number generator
        return_confidence: If True, return (estimate, std_error)
        
    Returns:
        Estimated Inf_i[f], or (estimate, std_error) if return_confidence=True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = f.n_vars or 0
    if n == 0 or i < 0 or i >= n:
        return (0.0, 0.0) if return_confidence else 0.0
    
    # Sample inputs
    inputs = sample_uniform(n, n_samples, rng)
    
    # Compute f(x) and f(x^{(i)})
    flipped = inputs ^ (1 << i)  # Flip bit i
    
    f_x = np.array([int(f.evaluate(int(x))) for x in inputs])
    f_flipped = np.array([int(f.evaluate(int(x))) for x in flipped])
    
    # Influence = fraction where f(x) ≠ f(x^{(i)})
    disagreements = (f_x != f_flipped).astype(float)
    estimate = np.mean(disagreements)
    
    if return_confidence:
        std_error = np.std(disagreements) / np.sqrt(n_samples)
        return (float(estimate), float(std_error))
    
    return float(estimate)


def estimate_expectation(
    f: "BooleanFunction",
    n_samples: int,
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Estimate E[f] under the (p-biased) distribution.
    
    For uniform distribution, E[f] = f̂(∅) = bias of f in ±1 representation.
    
    Args:
        f: BooleanFunction (treated as ±1-valued)
        n_samples: Number of samples
        p: Bias parameter
        rng: Random number generator
        
    Returns:
        Estimated E[f]
    """
    _, outputs = sample_input_output_pairs(f, n_samples, p, rng)
    
    # Convert to ±1
    pm_outputs = 1.0 - 2.0 * outputs.astype(float)
    
    return float(np.mean(pm_outputs))


def estimate_variance(
    f: "BooleanFunction",
    n_samples: int,
    p: float = 0.5,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Estimate Var[f] under the (p-biased) distribution.
    
    For ±1-valued functions, Var[f] = 1 - E[f]² = Σ_{S≠∅} f̂(S)².
    
    Args:
        f: BooleanFunction
        n_samples: Number of samples
        p: Bias parameter
        rng: Random number generator
        
    Returns:
        Estimated Var[f]
    """
    _, outputs = sample_input_output_pairs(f, n_samples, p, rng)
    pm_outputs = 1.0 - 2.0 * outputs.astype(float)
    
    return float(np.var(pm_outputs))


def estimate_total_influence(
    f: "BooleanFunction",
    n_samples: int,
    rng: Optional[np.random.Generator] = None
) -> float:
    """
    Estimate total influence I[f] = Σ_i Inf_i[f] via sampling.
    
    Uses the identity I[f] = E[s(f,x)] where s(f,x) is the sensitivity at x.
    
    Args:
        f: BooleanFunction
        n_samples: Number of samples
        rng: Random number generator
        
    Returns:
        Estimated total influence
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = f.n_vars or 0
    if n == 0:
        return 0.0
    
    # Sample inputs
    inputs = sample_uniform(n, n_samples, rng)
    
    # Compute sensitivity at each input
    sensitivities = []
    for x in inputs:
        x = int(x)
        f_x = int(f.evaluate(x))
        sens = 0
        for i in range(n):
            f_flipped = int(f.evaluate(x ^ (1 << i)))
            if f_x != f_flipped:
                sens += 1
        sensitivities.append(sens)
    
    return float(np.mean(sensitivities))


@dataclass
class SpectralDistribution:
    """
    Represents the spectral (Fourier weight) distribution of a Boolean function.
    
    The spectral distribution has:
    - Support: all subsets S ⊆ [n]
    - Probabilities: Pr[S] = f̂(S)² / Σ_T f̂(T)²
    
    For ±1-valued functions, Σ_S f̂(S)² = 1 by Parseval.
    
    Attributes:
        weights: Array of f̂(S)² values indexed by subset
        probabilities: Normalized probabilities
        n_vars: Number of variables
    """
    weights: np.ndarray
    probabilities: np.ndarray
    n_vars: int
    
    @classmethod
    def from_function(cls, f: "BooleanFunction") -> "SpectralDistribution":
        """Create spectral distribution from a Boolean function."""
        from . import SpectralAnalyzer
        
        n = f.n_vars or 0
        analyzer = SpectralAnalyzer(f)
        coeffs = analyzer.fourier_expansion()
        
        weights = coeffs ** 2
        total = np.sum(weights)
        probs = weights / total if total > 1e-15 else weights
        
        return cls(weights=weights, probabilities=probs, n_vars=n)
    
    def sample(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample subsets from the spectral distribution."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.choice(len(self.probabilities), size=n_samples, p=self.probabilities)
    
    def weight_at_degree(self, k: int) -> float:
        """Total weight at degree k: W^k[f] = Σ_{|S|=k} f̂(S)²."""
        total = 0.0
        for s in range(len(self.weights)):
            if bin(s).count("1") == k:
                total += self.weights[s]
        return total
    
    def entropy(self) -> float:
        """Shannon entropy of the spectral distribution."""
        # H = -Σ p(S) log p(S)
        nonzero = self.probabilities > 1e-15
        return float(-np.sum(self.probabilities[nonzero] * np.log2(self.probabilities[nonzero])))
    
    def effective_support_size(self, threshold: float = 0.01) -> int:
        """Count subsets with probability > threshold."""
        return int(np.sum(self.probabilities > threshold))


class RandomVariableView:
    """
    View a Boolean function as a random variable on the hypercube.
    
    This class provides a probabilistic interface to Boolean functions,
    supporting operations like expectation, variance, and sampling.
    
    In O'Donnell's framework:
    - f: {-1,+1}^n → R is a random variable
    - E[f] = f̂(∅)
    - Var[f] = Σ_{S≠∅} f̂(S)²
    - f = Σ_S f̂(S) χ_S is an orthonormal decomposition
    
    Attributes:
        function: The underlying BooleanFunction
        p: Bias parameter for sampling (default 0.5 = uniform)
    """
    
    def __init__(self, f: "BooleanFunction", p: float = 0.5):
        """
        Initialize random variable view.
        
        Args:
            f: BooleanFunction to view as random variable
            p: Bias parameter (default 0.5 = uniform distribution)
        """
        self.function = f
        self.p = p
        self._spectral: Optional[SpectralDistribution] = None
        self._rng = np.random.default_rng()
    
    def seed(self, seed: int) -> "RandomVariableView":
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        return self
    
    @property
    def n_vars(self) -> int:
        """Number of variables."""
        return self.function.n_vars or 0
    
    @property
    def spectral_distribution(self) -> SpectralDistribution:
        """Get the spectral distribution (cached)."""
        if self._spectral is None:
            self._spectral = SpectralDistribution.from_function(self.function)
        return self._spectral
    
    # Exact computations (for small n)
    
    def expectation(self) -> float:
        """Compute exact E[f] = f̂(∅)."""
        from .p_biased import p_biased_expectation
        return p_biased_expectation(self.function, self.p)
    
    def variance(self) -> float:
        """Compute exact Var[f]."""
        from .p_biased import p_biased_variance
        return p_biased_variance(self.function, self.p)
    
    def fourier_coefficient(self, S: int) -> float:
        """Get exact Fourier coefficient f̂(S)."""
        if self.p == 0.5:
            from . import SpectralAnalyzer
            analyzer = SpectralAnalyzer(self.function)
            coeffs = analyzer.fourier_expansion()
            return float(coeffs[S]) if S < len(coeffs) else 0.0
        else:
            from .p_biased import p_biased_fourier_coefficient
            return p_biased_fourier_coefficient(self.function, self.p, S)
    
    def influence(self, i: int) -> float:
        """Get exact influence of variable i."""
        if self.p == 0.5:
            from . import SpectralAnalyzer
            analyzer = SpectralAnalyzer(self.function)
            return float(analyzer.influences()[i])
        else:
            from .p_biased import p_biased_influence
            return p_biased_influence(self.function, i, self.p)
    
    def total_influence(self) -> float:
        """Get exact total influence."""
        if self.p == 0.5:
            from . import SpectralAnalyzer
            analyzer = SpectralAnalyzer(self.function)
            return float(analyzer.total_influence())
        else:
            from .p_biased import p_biased_total_influence
            return p_biased_total_influence(self.function, self.p)
    
    # Sampling methods
    
    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (input, output) pairs.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Tuple of (inputs, outputs) arrays
        """
        return sample_input_output_pairs(self.function, n_samples, self.p, self._rng)
    
    def sample_inputs(self, n_samples: int) -> np.ndarray:
        """Sample inputs from the (p-biased) distribution."""
        if self.p == 0.5:
            return sample_uniform(self.n_vars, n_samples, self._rng)
        else:
            return sample_biased(self.n_vars, self.p, n_samples, self._rng)
    
    def sample_spectral(self, n_samples: int) -> np.ndarray:
        """Sample subsets from the Fourier weight distribution."""
        return self.spectral_distribution.sample(n_samples, self._rng)
    
    # Estimation methods
    
    def estimate_expectation(self, n_samples: int) -> float:
        """Estimate E[f] via sampling."""
        return estimate_expectation(self.function, n_samples, self.p, self._rng)
    
    def estimate_variance(self, n_samples: int) -> float:
        """Estimate Var[f] via sampling."""
        return estimate_variance(self.function, n_samples, self.p, self._rng)
    
    def estimate_fourier_coefficient(
        self, S: int, n_samples: int, return_confidence: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """Estimate f̂(S) via sampling."""
        return estimate_fourier_coefficient(
            self.function, S, n_samples, self.p, self._rng, return_confidence
        )
    
    def estimate_influence(
        self, i: int, n_samples: int, return_confidence: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """Estimate Inf_i[f] via sampling."""
        return estimate_influence(
            self.function, i, n_samples, self._rng, return_confidence
        )
    
    def estimate_total_influence(self, n_samples: int) -> float:
        """Estimate I[f] via sampling."""
        return estimate_total_influence(self.function, n_samples, self._rng)
    
    # Comparison with exact values
    
    def validate_estimates(
        self, n_samples: int = 10000, tolerance: float = 0.1
    ) -> Dict[str, bool]:
        """
        Validate that estimates are close to exact values.
        
        Args:
            n_samples: Number of samples for estimation
            tolerance: Maximum allowed relative error
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Expectation
        exact_E = self.expectation()
        est_E = self.estimate_expectation(n_samples)
        results["expectation"] = abs(exact_E - est_E) < tolerance * max(abs(exact_E), 0.1)
        
        # Variance
        exact_Var = self.variance()
        est_Var = self.estimate_variance(n_samples)
        results["variance"] = abs(exact_Var - est_Var) < tolerance * max(exact_Var, 0.1)
        
        # Total influence
        exact_I = self.total_influence()
        est_I = self.estimate_total_influence(n_samples)
        results["total_influence"] = abs(exact_I - est_I) < tolerance * max(exact_I, 0.1)
        
        # f̂(∅) = expectation
        est_f_empty = self.estimate_fourier_coefficient(0, n_samples)
        results["fourier_empty"] = abs(exact_E - est_f_empty) < tolerance * max(abs(exact_E), 0.1)
        
        return results
    
    def summary(self) -> str:
        """Return human-readable summary."""
        sd = self.spectral_distribution
        lines = [
            f"RandomVariableView (n={self.n_vars}, p={self.p})",
            f"  E[f] = {self.expectation():.6f}",
            f"  Var[f] = {self.variance():.6f}",
            f"  I[f] = {self.total_influence():.6f}",
            f"  Spectral entropy = {sd.entropy():.4f} bits",
            f"  Effective support = {sd.effective_support_size()} subsets",
            "",
            "  Weight by degree:",
        ]
        for k in range(min(self.n_vars + 1, 6)):
            w = sd.weight_at_degree(k)
            lines.append(f"    W^{k}[f] = {w:.6f}")
        if self.n_vars >= 6:
            lines.append("    ...")
        
        return "\n".join(lines)
