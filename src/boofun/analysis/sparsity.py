"""
Fourier sparsity analysis for Boolean functions.

This module provides functions for analyzing the Fourier sparsity of Boolean
functions - the number and structure of non-zero Fourier coefficients.

Fourier sparsity is important for:
- Learning algorithms (sparse functions are easier to learn)
- Circuit complexity (sparsity bounds imply circuit bounds)
- Approximate degree bounds

Key concepts:
- sparsity(f): Number of non-zero Fourier coefficients |{S : f̂(S) ≠ 0}|
- granularity(f): GCD structure of Fourier coefficients  
- sparsity_up_to_constants(f): Sparsity ignoring constant multiples

Bounds:
- If deg(f) = d, then sparsity(f) ≤ 4^d (Nisan-Szegedy style)
- If sparsity(f) = k, then deg(f) ≤ k

References:
- O'Donnell HW1 Problem 7
- Tal's BooleanFunc.py (sparsity, sparsity_upto_constants)
- Gopalan et al., "Sparsity and Fourier Analysis" (2011)
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "fourier_sparsity",
    "fourier_sparsity_up_to_constants",
    "granularity",
    "fourier_support",
    "sparsity_by_degree",
    "effective_sparsity",
]


def _get_fourier_coefficients(f: "BooleanFunction") -> np.ndarray:
    """Get Fourier coefficients, computing if necessary."""
    from . import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer(f)
    return analyzer.fourier_expansion()


def fourier_sparsity(f: "BooleanFunction", threshold: float = 1e-10) -> int:
    """
    Count the number of non-zero Fourier coefficients.
    
    The Fourier sparsity is:
        sparsity(f) = |{S : |f̂(S)| > threshold}|
        
    For any Boolean function f of degree d:
        sparsity(f) ≤ 4^d  (Nisan-Szegedy style bound)
        
    Args:
        f: BooleanFunction to analyze
        threshold: Minimum magnitude to count as non-zero
        
    Returns:
        Number of non-zero Fourier coefficients
        
    References:
        - Tal's BooleanFunc.py: sparsity
        - O'Donnell HW1 Problem 7
    """
    coeffs = _get_fourier_coefficients(f)
    return int(np.sum(np.abs(coeffs) > threshold))


def fourier_sparsity_up_to_constants(
    f: "BooleanFunction", threshold: float = 1e-10
) -> int:
    """
    Compute Fourier sparsity ignoring constant multiples.
    
    This counts the number of distinct non-trivial Fourier coefficient values,
    where "trivial" means 0 or ±f̂(∅). This measure captures the "essential"
    complexity of the Fourier spectrum.
    
    Specifically, counts:
        |{S : f̂(S) ≠ 0 and f̂(S) ∉ {0, f̂(∅), -f̂(∅)}}|
        
    This is Tal's `sparsity_upto_constants` function.
    
    Args:
        f: BooleanFunction to analyze
        threshold: Minimum magnitude for non-zero
        
    Returns:
        Effective sparsity ignoring constants
        
    References:
        - Tal's BooleanFunc.py: sparsity_upto_constants
    """
    coeffs = _get_fourier_coefficients(f)
    n = f.n_vars or 0
    size = 1 << n
    
    if size == 0:
        return 0
    
    # Get f̂(∅)
    f_hat_empty = coeffs[0]
    trivial_values = {0.0, f_hat_empty, -f_hat_empty}
    
    # Count non-trivial, non-zero coefficients
    count = 0
    for s in range(size):
        coeff = coeffs[s]
        if abs(coeff) > threshold:
            # Check if it's trivial
            is_trivial = any(abs(coeff - v) < threshold for v in trivial_values)
            if not is_trivial:
                count += 1
    
    return size - count  # Return count of non-trivial


def granularity(f: "BooleanFunction", threshold: float = 1e-10) -> Dict[float, int]:
    """
    Analyze the granularity structure of Fourier coefficients.
    
    Returns a count of how many times each distinct coefficient value appears.
    This reveals patterns in the Fourier spectrum, e.g., all coefficients
    being powers of 1/2.
    
    Args:
        f: BooleanFunction to analyze
        threshold: Precision for rounding coefficients
        
    Returns:
        Dictionary mapping coefficient values to their counts
    """
    coeffs = _get_fourier_coefficients(f)
    
    # Round to threshold precision for grouping
    rounded = np.round(coeffs / threshold) * threshold
    
    counter: Counter = Counter(rounded)
    return dict(counter)


def fourier_support(f: "BooleanFunction", threshold: float = 1e-10) -> List[int]:
    """
    Return the support of the Fourier spectrum.
    
    The support is the set of subsets S where f̂(S) ≠ 0.
    
    Args:
        f: BooleanFunction to analyze
        threshold: Minimum magnitude for non-zero
        
    Returns:
        List of subset masks where f̂(S) ≠ 0, sorted by |S| then by value
    """
    coeffs = _get_fourier_coefficients(f)
    
    support = []
    for s, coeff in enumerate(coeffs):
        if abs(coeff) > threshold:
            support.append(s)
    
    # Sort by degree (number of bits), then by value
    support.sort(key=lambda s: (bin(s).count("1"), s))
    
    return support


def sparsity_by_degree(f: "BooleanFunction", threshold: float = 1e-10) -> Dict[int, int]:
    """
    Count non-zero Fourier coefficients at each degree.
    
    Returns a dictionary where result[d] = |{S : |S| = d and f̂(S) ≠ 0}|.
    
    Args:
        f: BooleanFunction to analyze
        threshold: Minimum magnitude for non-zero
        
    Returns:
        Dictionary mapping degree to count of non-zero coefficients at that degree
    """
    coeffs = _get_fourier_coefficients(f)
    n = f.n_vars or 0
    
    counts: Dict[int, int] = {}
    
    for s, coeff in enumerate(coeffs):
        if abs(coeff) > threshold:
            degree = bin(s).count("1")
            counts[degree] = counts.get(degree, 0) + 1
    
    return counts


def effective_sparsity(
    f: "BooleanFunction",
    weight_threshold: float = 0.01
) -> Tuple[int, float]:
    """
    Compute effective sparsity based on Fourier weight concentration.
    
    Returns the number of coefficients needed to capture (1 - weight_threshold)
    of the total Fourier weight.
    
    This is useful because many functions have most of their weight
    concentrated on a small number of coefficients.
    
    Args:
        f: BooleanFunction to analyze
        weight_threshold: Fraction of weight to allow outside the effective support
        
    Returns:
        Tuple of (effective_sparsity, weight_captured)
        - effective_sparsity: minimum k such that top-k coefficients have weight ≥ 1 - threshold
        - weight_captured: actual fraction of weight captured
    """
    coeffs = _get_fourier_coefficients(f)
    
    # Sort by squared magnitude (descending)
    weights = coeffs ** 2
    sorted_indices = np.argsort(weights)[::-1]
    
    total_weight = np.sum(weights)
    if total_weight < 1e-15:
        return (0, 0.0)
    
    target_weight = total_weight * (1.0 - weight_threshold)
    
    cumulative = 0.0
    for k, idx in enumerate(sorted_indices):
        cumulative += weights[idx]
        if cumulative >= target_weight:
            return (k + 1, cumulative / total_weight)
    
    return (len(coeffs), 1.0)
