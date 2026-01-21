"""
Symmetry analysis for Boolean functions.

This module provides tools for analyzing symmetric properties of Boolean functions,
including symmetrization, symmetric degree, and transformations to monotone form.

Key concepts:
- A function is **symmetric** if f(x) depends only on |x| (Hamming weight)
- **Symmetrization**: Transform any function to its symmetric version
- **Symmetric degree**: Maximum Hamming weight with nonzero count
- **Shift to monotone**: Find a shift that makes the function monotone (if possible)

References:
- O'Donnell Chapter 4: Influences and random restrictions
- Krawchouk polynomials for symmetric function analysis
- Tal's PhD library: symmetric function utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "symmetrize",
    "symmetrize_profile",
    "is_symmetric",
    "degree_sym",
    "sens_sym",
    "sens_sym_by_weight",
    "shift_function",
    "find_monotone_shift",
    "symmetric_representation",
]


def symmetrize(f: "BooleanFunction") -> np.ndarray:
    """
    Return counts of true outputs grouped by Hamming weight.
    
    For a function f: {0,1}^n → {0,1}, returns an array counts[k] where
    counts[k] = |{x : |x|=k and f(x)=1}|.
    
    Args:
        f: BooleanFunction to symmetrize
        
    Returns:
        Array of length n+1 with counts by Hamming weight
        
    Example:
        >>> f = bf.AND(3)  # Returns 1 only on 111
        >>> symmetrize(f)
        array([0, 0, 0, 1])  # Only weight-3 has a 1
    """
    n = f.n_vars or 0
    counts = np.zeros(n + 1, dtype=int)
    for x in range(1 << n):
        if bool(f.evaluate(x)):
            counts[bin(x).count("1")] += 1
    return counts


def symmetrize_profile(f: "BooleanFunction") -> Dict[int, Tuple[int, int]]:
    """
    Return detailed profile of function values by Hamming weight.
    
    For each weight k, returns (num_zeros, num_ones) among inputs of weight k.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Dictionary {k: (num_zeros, num_ones)} for k = 0..n
        
    Example:
        >>> f = bf.majority(3)
        >>> symmetrize_profile(f)
        {0: (1, 0), 1: (3, 0), 2: (0, 3), 3: (0, 1)}
    """
    n = f.n_vars or 0
    profile = {}
    
    for k in range(n + 1):
        num_zeros = 0
        num_ones = 0
        for x in range(1 << n):
            if bin(x).count("1") == k:
                if bool(f.evaluate(x)):
                    num_ones += 1
                else:
                    num_zeros += 1
        profile[k] = (num_zeros, num_ones)
    
    return profile


def is_symmetric(f: "BooleanFunction") -> bool:
    """
    Check if a function is symmetric (value depends only on Hamming weight).
    
    A function is symmetric if f(x) = f(y) whenever |x| = |y|.
    
    Args:
        f: BooleanFunction to check
        
    Returns:
        True if f is symmetric
        
    Example:
        >>> is_symmetric(bf.AND(3))  # True: f(x)=1 iff |x|=3
        True
        >>> is_symmetric(bf.create([0,1,0,0]))  # False: dictator is not symmetric
        False
    """
    n = f.n_vars or 0
    
    # For each weight, check that all inputs have the same output
    for k in range(n + 1):
        values = []
        for x in range(1 << n):
            if bin(x).count("1") == k:
                values.append(bool(f.evaluate(x)))
        
        if len(set(values)) > 1:
            return False
    
    return True


def degree_sym(f: "BooleanFunction") -> int:
    """
    Symmetric degree: largest Hamming weight with nonzero output count.
    
    For symmetric functions, this equals the degree of the representing polynomial.
    For non-symmetric functions, this is an upper bound.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Maximum k such that there exists x with |x|=k and f(x)=1
        
    Note:
        This is O(2^n) but fast in practice for small n.
    """
    counts = symmetrize(f)
    nz = np.nonzero(counts)[0]
    return int(nz.max()) if nz.size else 0


def sens_sym(f: "BooleanFunction") -> float:
    """
    Symmetric sensitivity proxy: mean Hamming weight of true inputs.
    
    For symmetric functions, this gives insight into where the function
    transitions from 0 to 1.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Average |x| among inputs x with f(x)=1
    """
    counts = symmetrize(f)
    total = counts.sum()
    if total == 0:
        return 0.0
    weights = np.arange(len(counts))
    return float(np.dot(weights, counts) / total)


def sens_sym_by_weight(f: "BooleanFunction") -> np.ndarray:
    """
    Compute sensitivity for each Hamming weight class.
    
    For weight k, computes the average sensitivity among inputs of weight k.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Array sens[k] = average sensitivity at Hamming weight k
        
    Note:
        From Tal's library: useful for understanding sensitivity structure
        of symmetric and near-symmetric functions.
    """
    n = f.n_vars or 0
    if n == 0:
        return np.array([0.0])
    
    sensitivities = np.zeros(n + 1)
    counts = np.zeros(n + 1, dtype=int)
    
    for x in range(1 << n):
        k = bin(x).count("1")
        f_x = bool(f.evaluate(x))
        
        # Count sensitive bits
        sens = 0
        for i in range(n):
            f_flipped = bool(f.evaluate(x ^ (1 << i)))
            if f_x != f_flipped:
                sens += 1
        
        sensitivities[k] += sens
        counts[k] += 1
    
    # Average by count
    with np.errstate(divide='ignore', invalid='ignore'):
        result = sensitivities / counts
        result[counts == 0] = 0.0
    
    return result


def shift_function(f: "BooleanFunction", shift: int) -> "BooleanFunction":
    """
    Shift a Boolean function by XORing all inputs with a constant.
    
    Returns g where g(x) = f(x ⊕ shift).
    
    Args:
        f: BooleanFunction to shift
        shift: Integer mask to XOR with inputs
        
    Returns:
        Shifted BooleanFunction
        
    Note:
        From Tal's library: useful for finding monotone shifts.
    """
    import boofun as bf
    
    n = f.n_vars or 0
    if n == 0:
        return f
    
    # Build new truth table
    new_tt = []
    for x in range(1 << n):
        shifted_x = x ^ shift
        new_tt.append(int(f.evaluate(shifted_x)))
    
    return bf.create(new_tt)


def _is_monotone_increasing(f: "BooleanFunction") -> bool:
    """Check if f is monotone increasing (x ≤ y implies f(x) ≤ f(y))."""
    n = f.n_vars or 0
    
    for x in range(1 << n):
        f_x = bool(f.evaluate(x))
        if f_x:
            continue
        
        # Check that no superset has f=1 (would violate monotonicity)
        for y in range(1 << n):
            if (x & y) == x and x != y:  # y contains all bits of x
                if bool(f.evaluate(y)):
                    # f(x)=0 but f(y)=1 where x ⊆ y: not monotone
                    pass  # This alone doesn't violate, need reverse check
        
    # Better check: for each pair x ≤ y (bitwise), verify f(x) ≤ f(y)
    for x in range(1 << n):
        for y in range(1 << n):
            if (x & y) == x:  # x ≤ y (bitwise)
                if bool(f.evaluate(x)) and not bool(f.evaluate(y)):
                    return False
    
    return True


def find_monotone_shift(f: "BooleanFunction") -> Optional[int]:
    """
    Find a shift that makes the function monotone, if one exists.
    
    A function g is monotone if x ≤ y (bitwise) implies g(x) ≤ g(y).
    This searches for a shift value s such that g(x) = f(x ⊕ s) is monotone.
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        Shift value that makes f monotone, or None if no such shift exists
        
    Note:
        This is exponential in n (tries all 2^n shifts) but useful for small n.
        From Tal's library: shift_to_mono concept.
    """
    n = f.n_vars or 0
    if n == 0:
        return 0
    
    for shift in range(1 << n):
        g = shift_function(f, shift)
        if _is_monotone_increasing(g):
            return shift
    
    return None


def symmetric_representation(f: "BooleanFunction") -> List[int]:
    """
    Get the symmetric representation of a function.
    
    Returns a list spec[k] ∈ {0, 1, -1} where:
    - spec[k] = 1 if all weight-k inputs map to 1
    - spec[k] = 0 if all weight-k inputs map to 0
    - spec[k] = -1 if weight-k inputs have mixed outputs (not symmetric)
    
    Args:
        f: BooleanFunction to analyze
        
    Returns:
        List of length n+1 with symmetric specification
        
    Example:
        >>> symmetric_representation(bf.majority(3))
        [0, 0, 1, 1]  # weights 0,1 → 0, weights 2,3 → 1
    """
    profile = symmetrize_profile(f)
    n = f.n_vars or 0
    
    spec = []
    for k in range(n + 1):
        zeros, ones = profile[k]
        if zeros > 0 and ones > 0:
            spec.append(-1)  # Mixed
        elif ones > 0:
            spec.append(1)
        else:
            spec.append(0)
    
    return spec
