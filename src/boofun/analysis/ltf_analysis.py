"""
Linear Threshold Function (LTF) Analysis Module.

Based on O'Donnell Chapter 5: Threshold Functions and their Analysis.

Key Concepts:
- LTF: f(x) = sign(w₁x₁ + ... + wₙxₙ - θ)
- Geometrically: a hyperplane cutting the Boolean hypercube
- Chow parameters: (E[f], Inf₁[f], ..., Infₙ[f]) uniquely identify LTFs
- CLT applies: weighted sum approaches Gaussian for large n

Theory:
- NOT all Boolean functions are LTFs (e.g., XOR/PARITY is NOT an LTF)
- AND, OR, Majority, Threshold-k are all LTFs
- LTFs are exactly the "halfspace" functions
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

try:
    from scipy.special import erfc
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class LTFAnalysis:
    """
    Complete analysis of a Linear Threshold Function.
    
    Attributes:
        is_ltf: Whether the function is an LTF
        weights: Weight vector (normalized)
        threshold: Threshold value
        chow_parameters: (E[f], Inf₁[f], ..., Infₙ[f])
        critical_index: Index where cumulative influence reaches 1/2
        regularity: Measure of weight concentration
    """
    is_ltf: bool
    weights: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    chow_parameters: Optional[np.ndarray] = None
    critical_index: Optional[int] = None
    regularity: Optional[float] = None
    gaussian_noise_stability: Optional[float] = None
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "LTF Analysis",
            "=" * 40,
            f"Is LTF: {self.is_ltf}",
        ]
        
        if self.is_ltf and self.weights is not None:
            lines.append(f"Weights: {np.round(self.weights, 4)}")
            lines.append(f"Threshold: {self.threshold:.4f}")
            lines.append(f"Chow params: {np.round(self.chow_parameters, 4)}")
            lines.append(f"Critical index: {self.critical_index}")
            lines.append(f"Regularity τ: {self.regularity:.4f}")
            if self.gaussian_noise_stability is not None:
                lines.append(f"Gaussian noise stability: {self.gaussian_noise_stability:.4f}")
        
        return "\n".join(lines)


def chow_parameters(f: "BooleanFunction") -> np.ndarray:
    """
    Compute Chow parameters of a Boolean function.
    
    The Chow parameters are: (E[f], Inf₁[f], ..., Infₙ[f])
    
    Key theorem (Chow, 1961): Two LTFs are identical if and only if
    they have the same Chow parameters.
    
    Args:
        f: Boolean function
        
    Returns:
        Array of length n+1 containing Chow parameters
    """
    from ..analysis import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer(f)
    fourier = analyzer.fourier_expansion()
    influences = analyzer.influences()
    
    # Chow parameters: (f̂(∅), f̂({1}), ..., f̂({n}))
    # Note: influences in ±1 convention equal 2*f̂({i})²
    # But the signed version is just f̂({i})
    
    n = f.n_vars
    chow = np.zeros(n + 1)
    chow[0] = fourier[0]  # E[f] = f̂(∅)
    
    # Extract degree-1 coefficients
    for i in range(n):
        # Index of singleton {i} in Walsh ordering
        idx = 1 << (n - 1 - i)
        chow[i + 1] = fourier[idx]
    
    return chow


def is_ltf(f: "BooleanFunction") -> bool:
    """
    Test if a Boolean function is a Linear Threshold Function.
    
    Uses linear programming to check if the function is linearly separable.
    
    Args:
        f: Boolean function to test
        
    Returns:
        True if f is an LTF
    """
    from ..core.representations.ltf import LTFRepresentation
    
    # Get truth table
    tt = list(f.get_representation("truth_table"))
    n = f.n_vars
    
    try:
        repr_obj = LTFRepresentation()
        repr_obj._find_ltf_parameters(tt, n)
        return True
    except ValueError:
        return False


def find_ltf_weights(f: "BooleanFunction") -> Tuple[np.ndarray, float]:
    """
    Find LTF weights and threshold for a Boolean function.
    
    Args:
        f: Boolean function (must be an LTF)
        
    Returns:
        Tuple of (weights array, threshold)
        
    Raises:
        ValueError: If f is not an LTF
    """
    from ..core.representations.ltf import LTFRepresentation
    
    tt = list(f.get_representation("truth_table"))
    n = f.n_vars
    
    repr_obj = LTFRepresentation()
    params = repr_obj._find_ltf_parameters(tt, n)
    
    return params.weights, params.threshold


def normalize_ltf_weights(weights: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
    """
    Normalize LTF weights to have unit L2 norm.
    
    Standard form: Σ wᵢ² = 1
    
    Args:
        weights: Original weight vector
        threshold: Original threshold
        
    Returns:
        Tuple of (normalized weights, normalized threshold)
    """
    norm = np.linalg.norm(weights)
    if norm < 1e-10:
        return weights, threshold
    
    return weights / norm, threshold / norm


def critical_index(weights: np.ndarray) -> int:
    """
    Compute the critical index of an LTF.
    
    The critical index k* is the smallest k such that:
    Σᵢ₌₁ᵏ wᵢ² ≥ 1/2 · Σᵢ₌₁ⁿ wᵢ²
    
    This measures where "half the influence" is concentrated.
    
    Args:
        weights: LTF weight vector (not necessarily normalized)
        
    Returns:
        Critical index (1-indexed)
    """
    # Sort by absolute value (descending)
    sorted_sq = np.sort(weights**2)[::-1]
    total = np.sum(sorted_sq)
    
    if total < 1e-10:
        return len(weights)
    
    cumsum = np.cumsum(sorted_sq)
    idx = np.searchsorted(cumsum, total / 2)
    
    return min(idx + 1, len(weights))


def regularity(weights: np.ndarray) -> float:
    """
    Compute the regularity parameter τ of an LTF.
    
    τ = max|wᵢ| / ||w||₂
    
    Small τ means "regular" LTF (no single coordinate dominates).
    Large τ means closer to a dictator.
    
    Args:
        weights: LTF weight vector
        
    Returns:
        Regularity parameter τ ∈ [0, 1]
    """
    norm = np.linalg.norm(weights)
    if norm < 1e-10:
        return 0.0
    
    return np.max(np.abs(weights)) / norm


def ltf_influence_from_weights(weights: np.ndarray) -> np.ndarray:
    """
    Compute influences of an LTF from its weights.
    
    For a normalized LTF with Σwᵢ² = 1:
    Inf_i[f] ≈ (2/π) · wᵢ² · (1 / σ)
    
    where σ = √(Σwⱼ²) = 1 for normalized weights.
    
    This is the CLT/Berry-Esseen approximation.
    
    Args:
        weights: LTF weight vector
        
    Returns:
        Array of influences
    """
    norm_sq = np.sum(weights**2)
    if norm_sq < 1e-10:
        return np.zeros_like(weights)
    
    # Approximate influence using Gaussian formula
    # Inf_i ≈ (2/π) * w_i² / Σw_j²
    influences = (2 / np.pi) * (weights**2) / norm_sq
    
    return influences


def ltf_noise_stability_gaussian(rho: float) -> float:
    """
    Gaussian noise stability for regular LTFs (Sheppard's formula).
    
    For the majority function and regular LTFs:
    Stab_ρ[f] ≈ (1/2) + (1/π) · arcsin(ρ)
    
    This is exact in the limit as n → ∞ for regular LTFs.
    
    Args:
        rho: Noise parameter ρ ∈ [0, 1]
        
    Returns:
        Noise stability estimate
    """
    return 0.5 + (1 / np.pi) * np.arcsin(rho)


def ltf_total_influence_estimate(n: int, regularity_tau: float = 0.0) -> float:
    """
    Estimate total influence of an LTF.
    
    For regular LTFs (small τ):
    I[f] ≈ √(2/π) · √n ≈ 0.798 · √n
    
    Args:
        n: Number of variables
        regularity_tau: Regularity parameter (0 for perfectly regular)
        
    Returns:
        Estimated total influence
    """
    # Regular case
    regular_estimate = np.sqrt(2 / np.pi) * np.sqrt(n)
    
    # Adjust for non-regularity (more concentrated → lower total influence)
    if regularity_tau > 0:
        # Interpolate toward dictator (I[f] = 1) as τ → 1
        return regular_estimate * (1 - regularity_tau) + 1 * regularity_tau
    
    return regular_estimate


def analyze_ltf(f: "BooleanFunction") -> LTFAnalysis:
    """
    Perform complete LTF analysis of a Boolean function.
    
    Args:
        f: Boolean function
        
    Returns:
        LTFAnalysis object with all computed properties
    """
    from ..analysis import SpectralAnalyzer
    
    # First check if it's an LTF
    is_ltf_flag = is_ltf(f)
    
    if not is_ltf_flag:
        return LTFAnalysis(is_ltf=False)
    
    # Get weights and threshold
    try:
        weights, threshold = find_ltf_weights(f)
    except ValueError:
        return LTFAnalysis(is_ltf=False)
    
    # Normalize weights
    weights_norm, threshold_norm = normalize_ltf_weights(weights, threshold)
    
    # Compute Chow parameters
    chow = chow_parameters(f)
    
    # Compute critical index
    crit_idx = critical_index(weights)
    
    # Compute regularity
    reg = regularity(weights)
    
    # Estimate Gaussian noise stability
    if reg < 0.5:  # Only meaningful for somewhat regular LTFs
        gaussian_stab = ltf_noise_stability_gaussian(0.5)
    else:
        gaussian_stab = None
    
    return LTFAnalysis(
        is_ltf=True,
        weights=weights,
        threshold=threshold,
        chow_parameters=chow,
        critical_index=crit_idx,
        regularity=reg,
        gaussian_noise_stability=gaussian_stab
    )


# ============================================================
# LTF Constructors
# ============================================================

def create_weighted_majority(weights: List[float], threshold: Optional[float] = None) -> "BooleanFunction":
    """
    Create a weighted majority (LTF) function.
    
    f(x) = sign(w₁x₁ + ... + wₙxₙ - θ)
    
    If threshold is not specified, uses θ = 0 (symmetric around 0 in ±1 space).
    
    Args:
        weights: Weight for each variable
        threshold: Threshold value (default: 0)
        
    Returns:
        BooleanFunction representing the weighted majority
        
    Example:
        # Nassau County voting system
        nassau = create_weighted_majority([31, 31, 28, 21, 2, 2])
    """
    import boofun as bf
    
    n = len(weights)
    weights = np.array(weights)
    
    if threshold is None:
        threshold = 0.0
    
    # Build truth table
    truth_table = []
    for i in range(2**n):
        x = np.array([int(b) for b in format(i, f'0{n}b')])
        # Convert to ±1: x_pm = 1 - 2*x, but simpler to compute directly
        x_pm = 1 - 2 * x
        val = 1 if np.dot(weights, x_pm) >= threshold else 0
        truth_table.append(val)
    
    return bf.create(truth_table)


def create_threshold_function(n: int, k: int) -> "BooleanFunction":
    """
    Create a k-threshold function.
    
    f(x) = 1 if Σxᵢ ≥ k, else 0
    
    Args:
        n: Number of variables
        k: Threshold (1 ≤ k ≤ n)
        
    Returns:
        BooleanFunction for threshold-k
        
    Examples:
        AND = threshold(n, n)
        OR = threshold(n, 1)
        MAJORITY = threshold(n, (n+1)/2)
    """
    import boofun as bf
    
    if k <= 0 or k > n:
        raise ValueError(f"Threshold k={k} must be in range [1, {n}]")
    
    truth_table = []
    for i in range(2**n):
        x = [int(b) for b in format(i, f'0{n}b')]
        val = 1 if sum(x) >= k else 0
        truth_table.append(val)
    
    return bf.create(truth_table)


# ============================================================
# Tests for LTF Properties
# ============================================================

def is_symmetric_ltf(f: "BooleanFunction") -> bool:
    """
    Check if function is a symmetric LTF (all weights equal).
    
    Symmetric LTFs are exactly the threshold functions.
    
    Args:
        f: Boolean function
        
    Returns:
        True if f is a symmetric LTF
    """
    if not is_ltf(f):
        return False
    
    weights, _ = find_ltf_weights(f)
    
    # Check if all weights are equal (up to sign)
    abs_weights = np.abs(weights)
    return np.allclose(abs_weights, abs_weights[0], rtol=1e-5)


def is_regular_ltf(f: "BooleanFunction", tau_threshold: float = 0.5) -> bool:
    """
    Check if function is a regular LTF (low regularity τ).
    
    Regular LTFs have no dominant variable.
    
    Args:
        f: Boolean function
        tau_threshold: Maximum regularity to be considered "regular"
        
    Returns:
        True if f is a regular LTF
    """
    if not is_ltf(f):
        return False
    
    weights, _ = find_ltf_weights(f)
    tau = regularity(weights)
    
    return tau <= tau_threshold


def dummy_voters(f: "BooleanFunction") -> List[int]:
    """
    Find "dummy voters" in a weighted voting system.
    
    A dummy voter has zero influence - their vote never matters.
    
    Args:
        f: Boolean function (should be an LTF)
        
    Returns:
        List of variable indices with zero influence
    """
    from ..analysis import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer(f)
    influences = analyzer.influences()
    
    # Find variables with negligible influence
    return [i for i, inf in enumerate(influences) if inf < 1e-10]


# ============================================================
# Chow Parameters Matching
# ============================================================

def chow_distance(f: "BooleanFunction", g: "BooleanFunction") -> float:
    """
    Compute distance between Chow parameters of two functions.
    
    If both are LTFs, distance = 0 implies f ≡ g.
    
    Args:
        f, g: Boolean functions
        
    Returns:
        L2 distance between Chow parameter vectors
    """
    chow_f = chow_parameters(f)
    chow_g = chow_parameters(g)
    
    return np.linalg.norm(chow_f - chow_g)


def find_closest_ltf(f: "BooleanFunction") -> Tuple["BooleanFunction", float]:
    """
    Find the LTF closest to a given function (by Chow parameters).
    
    Uses Chow parameters to find the best LTF approximation.
    This is a fundamental result: the LTF with matching Chow parameters
    is the closest LTF in many senses.
    
    Args:
        f: Boolean function (may or may not be an LTF)
        
    Returns:
        Tuple of (closest LTF, distance)
    """
    chow = chow_parameters(f)
    n = f.n_vars
    
    # The degree-1 Chow parameters suggest weights
    # Weight_i ∝ Chow_{i+1}
    weights = chow[1:]  # Skip E[f]
    
    # Normalize
    if np.linalg.norm(weights) > 1e-10:
        weights = weights / np.linalg.norm(weights)
    
    # Threshold from E[f]
    # E[f] = 2*Φ(-θ/σ) - 1 for Gaussian
    # Simplification: θ ≈ -Φ⁻¹((1 + E[f])/2) for normalized weights
    expected = chow[0]
    if SCIPY_AVAILABLE:
        threshold = -norm.ppf((1 + expected) / 2)
    else:
        # Simple approximation
        threshold = -expected * np.sqrt(np.pi / 2)
    
    # Create the LTF
    approx_ltf = create_weighted_majority(weights.tolist(), threshold)
    
    # Compute distance
    dist = chow_distance(f, approx_ltf)
    
    return approx_ltf, dist


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    # Analysis dataclass
    'LTFAnalysis',
    
    # Core functions
    'chow_parameters',
    'is_ltf',
    'find_ltf_weights',
    'normalize_ltf_weights',
    'analyze_ltf',
    
    # LTF properties
    'critical_index',
    'regularity',
    'ltf_influence_from_weights',
    'ltf_noise_stability_gaussian',
    'ltf_total_influence_estimate',
    
    # Constructors
    'create_weighted_majority',
    'create_threshold_function',
    
    # Tests
    'is_symmetric_ltf',
    'is_regular_ltf',
    'dummy_voters',
    
    # Chow parameters
    'chow_distance',
    'find_closest_ltf',
]
