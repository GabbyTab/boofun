"""
Hypercontractivity tools for Boolean function analysis.

Hypercontractivity is a key technique from Chapter 9 of O'Donnell's book,
providing powerful bounds on the behavior of low-degree polynomials.

Key results:
- Bonami's Lemma: Bounds on Lq norms of low-degree polynomials
- KKL Theorem: At least one variable has influence Ω(log n / n)
- Friedgut's Junta Theorem: Low total influence implies close to junta

Mathematical Background:
    The noise operator T_ρ acts on functions by:
        T_ρ f(x) = E_y[f(y)] where y is ρ-correlated with x
    
    In Fourier: (T_ρ f)^(S) = ρ^|S| f̂(S)
    
    Bonami's Lemma: For q > 2, ‖T_ρ f‖_q ≤ ‖f‖_2 when ρ ≤ 1/√(q-1)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
from math import sqrt, log, exp

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Core hypercontractivity
    "noise_operator",
    "lq_norm",
    "bonami_lemma_bound",
    # KKL-type theorems
    "kkl_lower_bound",
    "max_influence_bound",
    # Junta theorems
    "friedgut_junta_bound",
    "junta_approximation_error",
    # Utilities
    "level_d_inequality",
    "hypercontractive_inequality",
]


def noise_operator(f: "BooleanFunction", rho: float) -> np.ndarray:
    """
    Apply the noise operator T_ρ to a Boolean function.
    
    The noise operator is defined as:
        (T_ρ f)(x) = E_y[f(y)]
    where y is obtained by independently keeping each bit of x with
    probability (1+ρ)/2 and flipping it with probability (1-ρ)/2.
    
    In Fourier domain: (T_ρ f)^(S) = ρ^|S| · f̂(S)
    
    Args:
        f: BooleanFunction to apply noise to
        rho: Correlation parameter in [-1, 1]
        
    Returns:
        Array of (T_ρ f)(x) values for all x
        
    Note:
        Returns real values in [-1, 1], not Boolean values.
    """
    from . import SpectralAnalyzer
    
    n = f.n_vars or 0
    size = 1 << n
    
    # Get Fourier coefficients
    analyzer = SpectralAnalyzer(f)
    coeffs = analyzer.fourier_expansion()
    
    # Apply noise in Fourier domain: multiply by ρ^|S|
    noisy_coeffs = np.zeros_like(coeffs)
    for s in range(size):
        degree = bin(s).count("1")
        noisy_coeffs[s] = coeffs[s] * (rho ** degree)
    
    # Inverse transform: compute T_ρ f(x) for all x
    result = np.zeros(size)
    for x in range(size):
        total = 0.0
        for s in range(size):
            # χ_S(x) = (-1)^{|x ∩ S|}
            chi_val = 1 - 2 * (bin(x & s).count("1") % 2)
            total += noisy_coeffs[s] * chi_val
        result[x] = total
    
    return result


def lq_norm(f: "BooleanFunction", q: float) -> float:
    """
    Compute the L_q norm of f: ‖f‖_q = (E[|f(x)|^q])^{1/q}.
    
    Args:
        f: BooleanFunction to analyze
        q: Norm parameter (q ≥ 1)
        
    Returns:
        L_q norm of f
    """
    if q < 1:
        raise ValueError("q must be at least 1")
    
    n = f.n_vars or 0
    size = 1 << n
    
    # Get ±1 values
    tt = np.asarray(f.get_representation("truth_table"), dtype=float)
    pm_values = 1.0 - 2.0 * tt
    
    if q == np.inf:
        return float(np.max(np.abs(pm_values)))
    
    # E[|f|^q]^{1/q}
    return float(np.mean(np.abs(pm_values) ** q) ** (1.0 / q))


def bonami_lemma_bound(f: "BooleanFunction", q: float, rho: float) -> Tuple[float, float]:
    """
    Verify Bonami's Lemma: ‖T_ρ f‖_q ≤ ‖f‖_2.
    
    Bonami's Lemma (O'Donnell Theorem 9.22):
    For any f: {±1}^n → ℝ, if 1 ≤ q ≤ 2 or q > 2 with ρ ≤ 1/√(q-1), then:
        ‖T_ρ f‖_q ≤ ‖f‖_2
    
    Args:
        f: BooleanFunction to analyze
        q: Norm parameter
        rho: Noise parameter
        
    Returns:
        Tuple of (‖T_ρ f‖_q, ‖f‖_2) for verification
        
    Note:
        For Boolean functions, ‖f‖_2 = 1 always.
    """
    n = f.n_vars or 0
    size = 1 << n
    
    # Compute T_ρ f values
    noisy_values = noise_operator(f, rho)
    
    # Compute L_q norm of T_ρ f
    if q == np.inf:
        lq_noisy = float(np.max(np.abs(noisy_values)))
    else:
        lq_noisy = float(np.mean(np.abs(noisy_values) ** q) ** (1.0 / q))
    
    # L_2 norm of f (= 1 for Boolean functions)
    l2_f = lq_norm(f, 2)
    
    return (lq_noisy, l2_f)


def kkl_lower_bound(total_influence: float, n: int) -> float:
    """
    KKL theorem lower bound on maximum influence.
    
    KKL Theorem (O'Donnell Theorem 9.24):
    For any Boolean f: {±1}^n → {±1} that is not constant:
        max_i Inf_i[f] ≥ Ω(log n / n) · Var[f]
    
    More precisely, if I[f] is the total influence:
        max_i Inf_i[f] ≥ Var[f] · Ω(log n) / n
    
    For balanced functions (Var[f] = 1):
        max_i Inf_i[f] ≥ c · log n / n for some constant c
    
    Args:
        total_influence: Total influence I[f]
        n: Number of variables
        
    Returns:
        Lower bound on maximum influence
        
    Note:
        The constant c ≈ 0.57 in the precise statement.
    """
    if n <= 1:
        return 0.0
    
    # KKL bound: max_i Inf_i ≥ c * log(n) / n where c ≈ 0.57
    # Using the relation with total influence
    c = 0.5  # Conservative constant
    return c * log(n) / n


def max_influence_bound(f: "BooleanFunction") -> Tuple[float, float, float]:
    """
    Compute max influence and compare with KKL lower bound.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Tuple of (max_influence, kkl_bound, total_influence)
    """
    from . import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer(f)
    influences = analyzer.influences()
    total = analyzer.total_influence()
    
    max_inf = float(np.max(influences))
    n = f.n_vars or 1
    
    kkl_bound = kkl_lower_bound(total, n)
    
    return (max_inf, kkl_bound, total)


def friedgut_junta_bound(total_influence: float, epsilon: float) -> int:
    """
    Friedgut's Junta Theorem bound.
    
    Friedgut's Theorem (O'Donnell Theorem 9.40):
    If f: {±1}^n → {±1} has total influence I[f], then f is
    ε-close to a k-junta where k ≤ 2^{O(I[f]/ε)}.
    
    Args:
        total_influence: Total influence I[f]
        epsilon: Distance threshold
        
    Returns:
        Upper bound on junta size k
    """
    if epsilon <= 0 or total_influence < 0:
        return int(1e9)  # Return large number for edge cases
    
    # k ≤ 2^{O(I[f]/ε)}
    # Using a reasonable constant factor
    exponent = 2 * total_influence / epsilon
    
    # Cap to prevent overflow
    max_exp = 30
    if exponent > max_exp:
        return 2 ** max_exp
    
    return int(2 ** exponent)


def junta_approximation_error(f: "BooleanFunction", junta_vars: List[int]) -> float:
    """
    Compute the error of approximating f by its projection onto junta variables.
    
    The projection of f onto variables J is:
        f_J(x) = E[f | x_J] = averaging f over non-J coordinates
    
    The error is dist(f, f_J) = Pr[f(x) ≠ f_J(x)].
    
    Args:
        f: BooleanFunction to analyze
        junta_vars: List of variable indices to keep
        
    Returns:
        Approximation error (probability of disagreement)
        
    Note:
        This is related to the sum of influences of non-junta variables:
        dist(f, f_J) ≤ Σ_{i ∉ J} Inf_i[f]
    """
    from . import SpectralAnalyzer
    
    n = f.n_vars or 0
    size = 1 << n
    
    # Compute projection: average over non-junta variables
    non_junta = [i for i in range(n) if i not in junta_vars]
    num_non_junta = len(non_junta)
    
    if num_non_junta == 0:
        return 0.0  # f is already a junta on these variables
    
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    pm_values = 1.0 - 2.0 * truth_table
    
    # Compute projection values
    projection = np.zeros(size)
    
    for x in range(size):
        # Average over all assignments to non-junta variables
        total = 0.0
        count = 0
        
        # Generate all assignments that agree with x on junta_vars
        for y in range(1 << num_non_junta):
            # Build input that matches x on junta_vars
            x_prime = x
            for j, var in enumerate(non_junta):
                # Set bit var in x_prime to bit j of y
                if (y >> j) & 1:
                    x_prime |= (1 << (n - 1 - var))
                else:
                    x_prime &= ~(1 << (n - 1 - var))
            
            total += pm_values[x_prime]
            count += 1
        
        projection[x] = total / count
    
    # Compute disagreement probability
    # Using threshold: projection gives real values, threshold to ±1
    projected_bool = (projection <= 0).astype(int)
    original_bool = truth_table.astype(int)
    
    disagreements = np.sum(projected_bool != original_bool)
    return disagreements / size


def level_d_inequality(f: "BooleanFunction", d: int, q: float = 4) -> Tuple[float, float]:
    """
    Level-d inequality (O'Donnell Lemma 9.23).
    
    For the degree-d part of f, denoted f^{=d}:
        ‖f^{=d}‖_q ≤ (q-1)^{d/2} · ‖f^{=d}‖_2
    
    This is a consequence of Bonami's lemma.
    
    Args:
        f: BooleanFunction to analyze
        d: Degree level
        q: Norm parameter (q > 2)
        
    Returns:
        Tuple of (‖f^{=d}‖_q, (q-1)^{d/2} · ‖f^{=d}‖_2)
    """
    from . import SpectralAnalyzer
    
    n = f.n_vars or 0
    size = 1 << n
    
    analyzer = SpectralAnalyzer(f)
    coeffs = analyzer.fourier_expansion()
    
    # Extract degree-d coefficients
    degree_d_coeffs = np.zeros(size)
    for s in range(size):
        if bin(s).count("1") == d:
            degree_d_coeffs[s] = coeffs[s]
    
    # Compute L_2 norm of degree-d part: √(Σ f̂(S)^2 for |S|=d)
    l2_degree_d = sqrt(np.sum(degree_d_coeffs ** 2))
    
    # Compute L_q norm by reconstructing degree-d function
    degree_d_values = np.zeros(size)
    for x in range(size):
        total = 0.0
        for s in range(size):
            if bin(s).count("1") == d:
                chi_val = 1 - 2 * (bin(x & s).count("1") % 2)
                total += degree_d_coeffs[s] * chi_val
        degree_d_values[x] = total
    
    lq_degree_d = float(np.mean(np.abs(degree_d_values) ** q) ** (1.0 / q))
    
    # Bonami bound: (q-1)^{d/2} · L_2
    bonami_bound = ((q - 1) ** (d / 2)) * l2_degree_d
    
    return (lq_degree_d, bonami_bound)


def hypercontractive_inequality(
    f: "BooleanFunction", 
    rho: float,
    p: float = 2,
    q: float = 4
) -> Tuple[float, float, bool]:
    """
    Verify the hypercontractive inequality.
    
    Hypercontractive Inequality (O'Donnell Theorem 9.21):
    For 1 < p ≤ q < ∞ and ρ = √((p-1)/(q-1)):
        ‖T_ρ f‖_q ≤ ‖f‖_p
    
    Args:
        f: BooleanFunction to analyze
        rho: Noise parameter
        p: Input norm parameter
        q: Output norm parameter
        
    Returns:
        Tuple of (‖T_ρ f‖_q, ‖f‖_p, inequality_satisfied)
    """
    # Compute T_ρ f
    noisy_values = noise_operator(f, rho)
    
    # L_q norm of T_ρ f
    lq_noisy = float(np.mean(np.abs(noisy_values) ** q) ** (1.0 / q))
    
    # L_p norm of f
    lp_f = lq_norm(f, p)
    
    # Check inequality
    satisfied = lq_noisy <= lp_f + 1e-10  # Small tolerance for numerical errors
    
    return (lq_noisy, lp_f, satisfied)
