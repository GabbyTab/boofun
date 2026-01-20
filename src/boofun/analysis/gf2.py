"""
GF(2) analysis for Boolean functions (XOR/Algebraic Normal Form).

This module provides analysis tools based on the polynomial representation
of Boolean functions over the field GF(2) = {0, 1}.

Every Boolean function f: {0,1}^n -> {0,1} can be uniquely represented as:
    f(x) = ⊕_{S ⊆ [n]} c_S * ∏_{i∈S} x_i

where c_S ∈ {0,1} are the GF(2) Fourier coefficients (ANF coefficients).

Key concepts:
- GF(2) degree (algebraic degree): max |S| where c_S ≠ 0
- Real degree: max |S| where f̂(S) ≠ 0 (Walsh-Hadamard)
- GF(2) Fourier coefficients: obtained via Möbius transform

Relationships (from O'Donnell Chapter 1):
- GF(2) degree ≤ n for any function on n variables
- Linear functions have GF(2) degree ≤ 1
- The XOR function x₁ ⊕ x₂ ⊕ ... ⊕ xₙ has GF(2) degree 1
- Majority function has GF(2) degree n
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Set

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "gf2_fourier_transform",
    "gf2_degree",
    "gf2_monomials",
    "gf2_to_string",
    "is_linear_over_gf2",
    "correlation_with_parity",
]


def gf2_fourier_transform(f: "BooleanFunction") -> np.ndarray:
    """
    Compute the GF(2) Fourier transform (Möbius transform) of f.

    The result is an array where result[S] = 1 if the monomial corresponding
    to subset S appears in the ANF of f, and 0 otherwise.

    Args:
        f: BooleanFunction to analyze

    Returns:
        numpy array of length 2^n where result[S] ∈ {0,1} is the coefficient
        of monomial S in the ANF representation.

    Note:
        The index S encodes a subset: bit i is set iff variable i is in the monomial.
        E.g., S=3 (binary 11) represents the monomial x₀*x₁.

    Example:
        >>> xor = bf.create([0, 1, 1, 0])
        >>> coeffs = gf2_fourier_transform(xor)
        >>> # coeffs[0]=0 (no constant), coeffs[1]=1 (x0), coeffs[2]=1 (x1), coeffs[3]=0 (no x0*x1)
    """
    n = f.n_vars or 0
    if n == 0:
        return np.array([0])

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)
    size = len(truth_table)

    # Copy for in-place Möbius transform
    coeffs = truth_table.copy()

    # In-place Möbius transform over GF(2)
    # This computes: c_S = ⊕_{T⊆S} f(T)
    for i in range(n):
        step = 1 << i
        for mask in range(size):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]

    return coeffs


def gf2_degree(f: "BooleanFunction") -> int:
    """
    Compute the GF(2) degree (algebraic degree) of f.

    The GF(2) degree is the size of the largest monomial in the ANF:
        deg_2(f) = max{|S| : c_S ≠ 0}

    This is different from the "real" Fourier degree which uses Walsh coefficients.

    Args:
        f: BooleanFunction to analyze

    Returns:
        The algebraic degree of f over GF(2)

    Note:
        - Constant functions have degree 0
        - Linear functions have degree ≤ 1
        - XOR(x₁,...,xₙ) has degree 1
        - AND(x₁,...,xₙ) has degree n
        - Majority_n has degree n
    """
    coeffs = gf2_fourier_transform(f)

    max_deg = -1
    for mask, coeff in enumerate(coeffs):
        if coeff != 0:
            deg = bin(mask).count("1")
            max_deg = max(max_deg, deg)

    return max_deg if max_deg >= 0 else 0


def gf2_monomials(f: "BooleanFunction") -> List[Set[int]]:
    """
    Get all monomials with non-zero coefficients in the ANF of f.

    Args:
        f: BooleanFunction to analyze

    Returns:
        List of sets, where each set contains the variable indices in a monomial.
        E.g., [{0}, {1}, {0,2}] represents x₀ ⊕ x₁ ⊕ x₀*x₂
    """
    coeffs = gf2_fourier_transform(f)
    n = f.n_vars or 0

    monomials = []
    for mask, coeff in enumerate(coeffs):
        if coeff != 0:
            monomial = {i for i in range(n) if (mask >> i) & 1}
            monomials.append(monomial)

    return monomials


def gf2_to_string(f: "BooleanFunction", var_prefix: str = "x") -> str:
    """
    Get a string representation of the ANF of f.

    Args:
        f: BooleanFunction to analyze
        var_prefix: Prefix for variable names (default: "x")

    Returns:
        String like "1 ⊕ x0 ⊕ x1 ⊕ x0*x1"
    """
    monomials = gf2_monomials(f)

    if not monomials:
        return "0"

    terms = []
    for monomial in sorted(monomials, key=lambda s: (len(s), sorted(s))):
        if len(monomial) == 0:
            terms.append("1")
        else:
            vars_str = "*".join(f"{var_prefix}{i}" for i in sorted(monomial))
            terms.append(vars_str)

    return " ⊕ ".join(terms)


def is_linear_over_gf2(f: "BooleanFunction") -> bool:
    """
    Check if f is linear over GF(2) (i.e., degree ≤ 1).

    A function is linear over GF(2) if it can be written as:
        f(x) = c ⊕ a₁x₁ ⊕ a₂x₂ ⊕ ... ⊕ aₙxₙ

    for some constants c, a₁, ..., aₙ ∈ {0,1}.

    Note: This is different from being linear over ℝ (which would require
    all variables to have the same influence and specific Fourier structure).
    """
    return gf2_degree(f) <= 1


def correlation_with_parity(f: "BooleanFunction", subset: Set[int]) -> float:
    """
    Compute the correlation of f with the parity function on a subset.

    The correlation is:
        corr(f, χ_S) = E[(-1)^(f(x) ⊕ ⊕_{i∈S} x_i)]

    This equals the (normalized) Walsh-Hadamard coefficient f̂(S).

    Args:
        f: BooleanFunction to analyze
        subset: Set of variable indices for the parity function

    Returns:
        Correlation in [-1, 1]
    """
    n = f.n_vars or 0
    if n == 0:
        return 1.0

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)
    size = len(truth_table)

    # Compute subset mask
    subset_mask = sum(1 << i for i in subset)

    # Compute correlation
    total = 0
    for x in range(size):
        f_x = truth_table[x]
        parity_x = bin(x & subset_mask).count("1") % 2
        # (-1)^(f(x) XOR parity_x)
        total += (-1) ** (f_x ^ parity_x)

    return total / size


def variable_degree(f: "BooleanFunction", var: int) -> int:
    """
    Compute the degree of variable var in the ANF of f.

    This is the maximum degree of any monomial containing var.

    Args:
        f: BooleanFunction to analyze
        var: Variable index

    Returns:
        Maximum degree of monomials containing var (0 if var is irrelevant)
    """
    coeffs = gf2_fourier_transform(f)
    f.n_vars or 0

    max_deg = 0
    for mask, coeff in enumerate(coeffs):
        if coeff != 0 and (mask >> var) & 1:
            deg = bin(mask).count("1")
            max_deg = max(max_deg, deg)

    return max_deg


def connected_variables(f: "BooleanFunction", var_set: Set[int]) -> bool:
    """
    Check if all variables in var_set appear together in some monomial.

    Args:
        f: BooleanFunction to analyze
        var_set: Set of variable indices to check

    Returns:
        True if there exists a monomial containing all variables in var_set
    """
    coeffs = gf2_fourier_transform(f)

    # Compute mask for var_set
    var_mask = sum(1 << i for i in var_set)

    for mask, coeff in enumerate(coeffs):
        if coeff != 0:
            # Check if var_mask is a subset of mask
            if (mask & var_mask) == var_mask:
                return True

    return False


def fourier_weight_by_degree(f: "BooleanFunction") -> List[int]:
    """
    Compute the GF(2) Fourier weight at each degree level.

    Returns:
        List where result[d] = number of degree-d monomials with non-zero coefficient
    """
    coeffs = gf2_fourier_transform(f)
    n = f.n_vars or 0

    weights = [0] * (n + 1)
    for mask, coeff in enumerate(coeffs):
        if coeff != 0:
            deg = bin(mask).count("1")
            weights[deg] += 1

    return weights
