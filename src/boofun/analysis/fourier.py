"""
Fourier analysis utilities for Boolean functions.

This module provides core Fourier analysis tools as described in Chapter 1
of O'Donnell's "Analysis of Boolean Functions", including:

- Parseval's identity: ‖f‖² = Σ_S f̂(S)²
- Plancherel's theorem: ⟨f,g⟩ = Σ_S f̂(S)ĝ(S)
- Convolution: (f * g)(x) = E_y[f(y)g(x⊕y)]
- Function transformations (negation, odd/even parts)
- Restriction operators

Mathematical Background (O'Donnell Chapter 1):
    Every function f: {-1,1}^n → ℝ has a unique Fourier expansion:
        f(x) = Σ_{S⊆[n]} f̂(S) χ_S(x)

    where χ_S(x) = ∏_{i∈S} x_i are the Fourier characters.

    Key identities:
    - Parseval: E[f(x)²] = Σ_S f̂(S)² (L2 norm preservation)
    - Plancherel: E[f(x)g(x)] = Σ_S f̂(S)ĝ(S) (inner product)
    - Convolution: (f*g)^(S) = f̂(S) · ĝ(S) (pointwise in Fourier domain)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Fundamental identities
    "parseval_verify",
    "plancherel_inner_product",
    "convolution",
    "convolution_values",
    # Function transformations (from HW1)
    "negate_inputs",
    "odd_part",
    "even_part",
    "tensor_product",
    "restriction",
    # Fourier utilities
    "fourier_degree",
    "spectral_norm",
    "fourier_sparsity",
    "dominant_coefficients",
    # From Tal's library
    "correlation",
    "truncate_to_degree",
    "annealed_influence",
    "fourier_weight_distribution",
    "min_fourier_coefficient_size",
    # Examples from O'Donnell
    "compute_mux3_fourier",
    "compute_nae3_fourier",
    "compute_and_fourier",
]


def _get_fourier_coefficients(f: "BooleanFunction") -> np.ndarray:
    """Get Fourier coefficients, computing if necessary."""
    from . import SpectralAnalyzer

    analyzer = SpectralAnalyzer(f)
    return analyzer.fourier_expansion()


def parseval_verify(f: "BooleanFunction", tolerance: float = 1e-10) -> Tuple[bool, float, float]:
    """
    Verify Parseval's identity for a Boolean function.

    Parseval's identity states that for f: {-1,1}^n → ℝ:
        E[f(x)²] = Σ_S f̂(S)²

    For Boolean functions f: {-1,1}^n → {-1,1}, this gives:
        1 = Σ_S f̂(S)²  (since f(x)² = 1 always)

    Args:
        f: BooleanFunction to verify
        tolerance: Maximum allowed deviation from expected value

    Returns:
        Tuple of (passes, lhs_value, rhs_value) where:
        - passes: True if |lhs - rhs| < tolerance
        - lhs_value: E[f(x)²] computed directly
        - rhs_value: Σ_S f̂(S)² from Fourier coefficients

    Example:
        >>> xor = bf.create([0, 1, 1, 0])
        >>> passes, lhs, rhs = parseval_verify(xor)
        >>> passes  # Should be True
        True
    """
    n = f.n_vars or 0
    if n == 0:
        val = float(f.evaluate(0))
        pm_val = 1.0 - 2.0 * val  # Convert to ±1
        return (True, pm_val**2, pm_val**2)

    # Compute LHS: E[f(x)²]
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    pm_values = 1.0 - 2.0 * truth_table  # Convert {0,1} to {+1,-1}
    lhs = np.mean(pm_values**2)

    # Compute RHS: Σ_S f̂(S)²
    fourier_coeffs = _get_fourier_coefficients(f)
    rhs = np.sum(fourier_coeffs**2)

    passes = abs(lhs - rhs) < tolerance
    return (passes, float(lhs), float(rhs))


def plancherel_inner_product(f: "BooleanFunction", g: "BooleanFunction") -> float:
    """
    Compute inner product using Plancherel's theorem.

    Plancherel's theorem (generalization of Parseval):
        ⟨f, g⟩ = E[f(x)g(x)] = Σ_S f̂(S)ĝ(S)

    Args:
        f, g: BooleanFunctions to compute inner product of

    Returns:
        Inner product ⟨f, g⟩

    Raises:
        ValueError: If f and g have different number of variables
    """
    if f.n_vars != g.n_vars:
        raise ValueError(f"Functions must have same number of variables: {f.n_vars} vs {g.n_vars}")

    f_coeffs = _get_fourier_coefficients(f)
    g_coeffs = _get_fourier_coefficients(g)

    return float(np.sum(f_coeffs * g_coeffs))


def convolution(f: "BooleanFunction", g: "BooleanFunction") -> np.ndarray:
    """
    Compute the Fourier coefficients of the convolution of two Boolean functions.

    The convolution is defined as:
        (f * g)(x) = E_y[f(y)g(x ⊕ y)]

    By the Convolution Theorem, in the Fourier domain this becomes pointwise
    multiplication:
        (f * g)^(S) = f̂(S) · ĝ(S)

    Args:
        f, g: BooleanFunctions to convolve

    Returns:
        numpy array of Fourier coefficients (f * g)^(S) indexed by S

    Raises:
        ValueError: If f and g have different number of variables

    Note:
        The convolution of two Boolean functions is generally NOT a Boolean
        function - it's a real-valued function with values in [-1, 1].
        This function returns the exact Fourier coefficients without any
        loss of information.

    Example:
        >>> f = bf.majority(3)
        >>> g = bf.parity(3)
        >>> conv_coeffs = convolution(f, g)
        >>> # Verify: (f*g)^(S) = f̂(S) · ĝ(S)
        >>> assert np.allclose(conv_coeffs, f.fourier() * g.fourier())
    """
    if f.n_vars != g.n_vars:
        raise ValueError(f"Functions must have same number of variables: {f.n_vars} vs {g.n_vars}")

    # The Convolution Theorem: (f * g)^(S) = f̂(S) · ĝ(S)
    # This IS the convolution in Fourier space - just pointwise multiplication!
    f_coeffs = _get_fourier_coefficients(f)
    g_coeffs = _get_fourier_coefficients(g)

    return f_coeffs * g_coeffs


def convolution_values(f: "BooleanFunction", g: "BooleanFunction") -> np.ndarray:
    """
    Compute the time-domain values of the convolution of two Boolean functions.

    The convolution is defined as:
        (f * g)(x) = E_y[f(y)g(x ⊕ y)]

    This function computes (f * g)(x) for all x via inverse Fourier transform.

    Args:
        f, g: BooleanFunctions to convolve

    Returns:
        numpy array of real values (f * g)(x) indexed by x

    Note:
        The values are in [-1, 1], NOT {0, 1} or {-1, 1}.
        This is the correlation between f and g shifted by x.
    """
    conv_coeffs = convolution(f, g)
    n = f.n_vars or 0
    size = 1 << n

    # Inverse Fourier transform: (f*g)(x) = Σ_S (f*g)^(S) · χ_S(x)
    values = np.zeros(size)
    for x in range(size):
        total = 0.0
        for s in range(size):
            # χ_S(x) = (-1)^{popcount(x & S)}
            chi_val = 1 - 2 * (bin(x & s).count("1") % 2)
            total += conv_coeffs[s] * chi_val
        values[x] = total

    return values


def negate_inputs(f: "BooleanFunction") -> "BooleanFunction":
    """
    Compute g(x) = f(-x) where -x flips all bits.

    From HW1 Problem 1: If f(x) = Σ_S f̂(S) χ_S(x), then
        g(x) = f(-x) = Σ_S (-1)^|S| f̂(S) χ_S(x)

    This flips the sign of odd-degree Fourier coefficients.

    Args:
        f: BooleanFunction to transform

    Returns:
        New BooleanFunction g where g(x) = f(-x)
    """
    from ..core.base import BooleanFunction
    from ..core.factory import BooleanFunctionFactory

    n = f.n_vars or 0
    size = 1 << n

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    # f(-x) means flipping all bits: x → x ⊕ (2^n - 1)
    flip_mask = size - 1
    new_tt = np.array([truth_table[x ^ flip_mask] for x in range(size)], dtype=bool)

    return BooleanFunctionFactory.from_truth_table(BooleanFunction, new_tt, n=n)


def odd_part(f: "BooleanFunction") -> np.ndarray:
    """
    Compute the odd part of f: f^odd(x) = (f(x) - f(-x)) / 2.

    From HW1 Problem 1: The odd part contains only odd-degree Fourier coefficients.
        f^odd(x) = Σ_{|S| odd} f̂(S) χ_S(x)

    Args:
        f: BooleanFunction to analyze

    Returns:
        Array of function values for f^odd (real-valued, not Boolean)

    Note:
        The result is real-valued, not necessarily Boolean.
    """
    n = f.n_vars or 0
    size = 1 << n

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    pm_values = 1.0 - 2.0 * truth_table

    flip_mask = size - 1
    pm_negated = np.array([pm_values[x ^ flip_mask] for x in range(size)])

    return (pm_values - pm_negated) / 2.0


def even_part(f: "BooleanFunction") -> np.ndarray:
    """
    Compute the even part of f: f^even(x) = (f(x) + f(-x)) / 2.

    From HW1 Problem 1: The even part contains only even-degree Fourier coefficients.
        f^even(x) = Σ_{|S| even} f̂(S) χ_S(x)

    Args:
        f: BooleanFunction to analyze

    Returns:
        Array of function values for f^even (real-valued)
    """
    n = f.n_vars or 0
    size = 1 << n

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)
    pm_values = 1.0 - 2.0 * truth_table

    flip_mask = size - 1
    pm_negated = np.array([pm_values[x ^ flip_mask] for x in range(size)])

    return (pm_values + pm_negated) / 2.0


def tensor_product(f: "BooleanFunction", g: "BooleanFunction") -> "BooleanFunction":
    """
    Compute tensor product: h(x₁, x₂) = f(x₁) · g(x₂).

    From HW1 Problem 1: If f: {-1,1}^n → ℝ and g: {-1,1}^m → ℝ, then
        h: {-1,1}^{n+m} → ℝ with h(x₁, x₂) = f(x₁) · g(x₂)

    has Fourier expansion:
        ĥ(S₁ ∪ S₂) = f̂(S₁) · ĝ(S₂)

    where S₁ ⊆ [n] and S₂ ⊆ [m] (shifted to [n+1, n+m]).

    Args:
        f: First BooleanFunction on n variables
        g: Second BooleanFunction on m variables

    Returns:
        Tensor product on n+m variables
    """
    from ..core.base import BooleanFunction
    from ..core.factory import BooleanFunctionFactory

    n = f.n_vars or 0
    m = g.n_vars or 0

    f_tt = np.asarray(f.get_representation("truth_table"), dtype=float)
    g_tt = np.asarray(g.get_representation("truth_table"), dtype=float)

    # Convert to ±1
    f_pm = 1.0 - 2.0 * f_tt
    g_pm = 1.0 - 2.0 * g_tt

    # Build tensor product truth table
    new_size = (1 << n) * (1 << m)
    new_tt = np.zeros(new_size, dtype=bool)

    for x1 in range(1 << n):
        for x2 in range(1 << m):
            # Combined index: x1 in high bits, x2 in low bits
            idx = (x1 << m) | x2
            # Product of ±1 values
            product = f_pm[x1] * g_pm[x2]
            # Convert back to {0,1}: +1 → 0, -1 → 1
            new_tt[idx] = product < 0

    return BooleanFunctionFactory.from_truth_table(BooleanFunction, new_tt, n=n + m)


def restriction(f: "BooleanFunction", fixed_vars: Dict[int, int]) -> "BooleanFunction":
    """
    Compute restriction of f by fixing some variables.

    From HW1 Problem 1: If we fix the last (n-k) variables to -1:
        g(x₁,...,x_k) = f(x₁,...,x_k, -1, -1, ..., -1)

    The Fourier coefficients satisfy:
        ĝ(T) = Σ_{S⊇T, S⊆[k]} f̂(S ∪ {fixed vars where fixed = -1})

    Args:
        f: BooleanFunction to restrict
        fixed_vars: Dictionary mapping variable indices to fixed values (0 or 1)

    Returns:
        Restricted BooleanFunction on remaining variables

    Example:
        >>> f = bf.create([0, 1, 1, 0])  # XOR
        >>> g = restriction(f, {0: 1})   # Fix x0 = 1
        >>> # g is now NOT(x1) on 1 variable
    """
    # Use the fix method from BooleanFunction
    result = f
    # Sort by index descending to fix from right to left
    for var, val in sorted(fixed_vars.items(), reverse=True):
        result = result.fix(var, val)
    return result


def fourier_degree(f: "BooleanFunction") -> int:
    """
    Compute the Fourier degree (real degree) of f.

    The Fourier degree is the maximum |S| such that f̂(S) ≠ 0.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Maximum degree of non-zero Fourier coefficient

    Note:
        This is different from the GF(2) degree (algebraic degree).
        For f(x) = x₁x₂ (AND), both degrees are 2.
        For f(x) = x₁ ⊕ x₂ (XOR), real degree is 2 but GF(2) degree is 1.
    """
    coeffs = _get_fourier_coefficients(f)

    max_degree = 0
    for s, coeff in enumerate(coeffs):
        if abs(coeff) > 1e-10:
            degree = bin(s).count("1")
            max_degree = max(max_degree, degree)

    return max_degree


def spectral_norm(f: "BooleanFunction", p: int = 2) -> float:
    """
    Compute the L_p spectral norm of f.

    Args:
        f: BooleanFunction to analyze
        p: Norm parameter (1, 2, or inf)

    Returns:
        ‖f̂‖_p = (Σ_S |f̂(S)|^p)^{1/p}
    """
    coeffs = _get_fourier_coefficients(f)

    if p == 1:
        return float(np.sum(np.abs(coeffs)))
    elif p == 2:
        return float(np.sqrt(np.sum(coeffs**2)))
    elif p == np.inf:
        return float(np.max(np.abs(coeffs)))
    else:
        return float(np.sum(np.abs(coeffs) ** p) ** (1.0 / p))


def fourier_sparsity(f: "BooleanFunction", threshold: float = 1e-10) -> int:
    """
    Count the number of non-zero Fourier coefficients.

    From HW1 Problem 7: Functions of degree k have at most 4^k non-zero
    Fourier coefficients.

    Args:
        f: BooleanFunction to analyze
        threshold: Minimum absolute value to count as non-zero

    Returns:
        Number of Fourier coefficients with |f̂(S)| > threshold
    """
    coeffs = _get_fourier_coefficients(f)
    return int(np.sum(np.abs(coeffs) > threshold))


def dominant_coefficients(
    f: "BooleanFunction", top_k: int = 10, threshold: float = 0.01
) -> List[Tuple[int, float]]:
    """
    Find the dominant (largest magnitude) Fourier coefficients.

    Args:
        f: BooleanFunction to analyze
        top_k: Maximum number of coefficients to return
        threshold: Minimum magnitude to include

    Returns:
        List of (subset_mask, coefficient) pairs, sorted by magnitude
    """
    coeffs = _get_fourier_coefficients(f)

    # Get indices sorted by absolute value
    sorted_indices = np.argsort(np.abs(coeffs))[::-1]

    result = []
    for idx in sorted_indices:
        coeff = coeffs[idx]
        if abs(coeff) < threshold:
            break
        result.append((int(idx), float(coeff)))
        if len(result) >= top_k:
            break

    return result


# =============================================================================
# Functions from Avishay Tal's library
# =============================================================================


def correlation(f: "BooleanFunction", g: "BooleanFunction") -> float:
    """
    Compute the correlation between two Boolean functions.

    The correlation is defined as:
        Corr(f, g) = E[f(x)g(x)] = ⟨f, g⟩

    where the expectation is over the uniform distribution on {-1,+1}^n.

    For ±1-valued functions, Corr(f,f) = 1, and Corr(f,g) ∈ [-1,1].

    This is mathematically equivalent to plancherel_inner_product but
    provided as a more intuitive interface from Tal's library.

    Args:
        f: First BooleanFunction
        g: Second BooleanFunction

    Returns:
        Correlation ⟨f, g⟩ in [-1, 1]

    Raises:
        ValueError: If f and g have different number of variables

    References:
        - Tal's BooleanFunc.py: correlation
        - O'Donnell Definition 1.6
    """
    if f.n_vars != g.n_vars:
        raise ValueError(f"Functions must have same n_vars: {f.n_vars} vs {g.n_vars}")

    n = f.n_vars or 0
    if n == 0:
        f_val = 1.0 - 2.0 * float(f.evaluate(0))
        g_val = 1.0 - 2.0 * float(g.evaluate(0))
        return f_val * g_val

    f_tt = np.asarray(f.get_representation("truth_table"), dtype=float)
    g_tt = np.asarray(g.get_representation("truth_table"), dtype=float)

    # Convert to ±1
    f_pm = 1.0 - 2.0 * f_tt
    g_pm = 1.0 - 2.0 * g_tt

    return float(np.mean(f_pm * g_pm))


def truncate_to_degree(f: "BooleanFunction", d: int) -> np.ndarray:
    """
    Truncate the Fourier expansion of f to degree at most d.

    Returns the function:
        f^{≤d}(x) = Σ_{|S| ≤ d} f̂(S) χ_S(x)

    This is useful for approximating functions by their low-degree parts.
    The truncated function is generally not Boolean-valued.

    Args:
        f: BooleanFunction to truncate
        d: Maximum degree to keep

    Returns:
        Array of function values for f^{≤d} (real-valued, not necessarily Boolean)

    References:
        - Tal's BooleanFunc.py: truncated_degree_d
        - O'Donnell Definition 1.14 (degree-d part)
    """
    n = f.n_vars or 0
    if n == 0:
        val = float(f.evaluate(0))
        return np.array([1.0 - 2.0 * val])

    size = 1 << n
    coeffs = _get_fourier_coefficients(f)

    # Zero out coefficients with |S| > d
    truncated_coeffs = np.zeros(size)
    for s in range(size):
        if bin(s).count("1") <= d:
            truncated_coeffs[s] = coeffs[s]

    # Inverse Fourier transform to get function values
    result = np.zeros(size)
    for x in range(size):
        total = 0.0
        for s in range(size):
            # χ_S(x) = (-1)^{|x ∩ S|}
            chi_val = 1 - 2 * (bin(x & s).count("1") % 2)
            total += truncated_coeffs[s] * chi_val
        result[x] = total

    return result


def annealed_influence(f: "BooleanFunction", i: int, rho: float) -> float:
    """
    Compute the annealed (noisy) influence of variable i.

    The annealed influence at correlation ρ is:
        Inf_i^{(ρ)}[f] = Σ_{S∋i} ρ^{|S|-1} f̂(S)²

    This interpolates between:
    - ρ = 1: Standard influence Inf_i[f] = Σ_{S∋i} f̂(S)²
    - ρ → 0: Only degree-1 contributions

    The annealed influence measures how sensitive f is to variable i
    after noise has been applied.

    Args:
        f: BooleanFunction to analyze
        i: Variable index (0-indexed)
        rho: Noise correlation parameter in (0, 1]

    Returns:
        Annealed influence of variable i

    References:
        - Tal's BooleanFunc.py: ann_influence
        - O'Donnell Chapter 2 (noise sensitivity)
    """
    n = f.n_vars or 0
    if n == 0:
        return 0.0

    if i < 0 or i >= n:
        raise ValueError(f"Variable index {i} out of range [0, {n})")

    if not (0 < rho <= 1):
        raise ValueError(f"rho must be in (0, 1], got {rho}")

    coeffs = _get_fourier_coefficients(f)
    size = 1 << n

    total = 0.0
    for s in range(size):
        # Check if i is in S
        if (s >> i) & 1:
            subset_size = bin(s).count("1")
            # ρ^{|S|-1} f̂(S)²
            total += (rho ** (subset_size - 1)) * (coeffs[s] ** 2)

    return total


def fourier_weight_distribution(f: "BooleanFunction") -> Dict[int, float]:
    """
    Compute the distribution of Fourier weight by degree.

    Returns W^k[f] = Σ_{|S|=k} f̂(S)² for each k.

    By Parseval's identity: Σ_k W^k[f] = 1 for ±1-valued functions.

    This is useful for understanding the "spectral profile" of a function:
    - Low-degree weight → function is "simple"
    - High-degree weight → function is "complex"

    Args:
        f: BooleanFunction to analyze

    Returns:
        Dictionary mapping degree k to W^k[f]

    References:
        - Tal's BooleanFunc.py: fourier_weights
        - O'Donnell Definition 1.14 (spectral sample)
    """
    n = f.n_vars or 0
    if n == 0:
        return {0: 1.0}

    coeffs = _get_fourier_coefficients(f)
    size = 1 << n

    weights: Dict[int, float] = {}

    for s in range(size):
        degree = bin(s).count("1")
        weight = coeffs[s] ** 2
        weights[degree] = weights.get(degree, 0.0) + weight

    return weights


def min_fourier_coefficient_size(f: "BooleanFunction", threshold: float = 1e-10) -> int:
    """
    Find the minimum subset size |S| with non-zero f̂(S).

    Returns min { |S| : f̂(S) ≠ 0 }.

    This is 0 if f has non-zero bias (f̂(∅) ≠ 0), otherwise it's
    the minimum degree of a non-zero Fourier coefficient.

    Args:
        f: BooleanFunction to analyze
        threshold: Minimum magnitude to count as non-zero

    Returns:
        Minimum subset size with non-zero coefficient

    References:
        - Tal's BooleanFunc.py: min_size_fourier_coef
    """
    n = f.n_vars or 0
    if n == 0:
        return 0

    coeffs = _get_fourier_coefficients(f)

    min_size = n + 1  # Sentinel
    for s, coeff in enumerate(coeffs):
        if abs(coeff) > threshold:
            size = bin(s).count("1")
            min_size = min(min_size, size)
            if min_size == 0:
                break  # Can't get smaller than 0

    return min_size if min_size <= n else 0


# =============================================================================
# Example functions from O'Donnell book and HW1
# =============================================================================


def compute_mux3_fourier() -> Dict[int, float]:
    """
    Compute Fourier expansion of MUX₃: {-1,1}³ → {-1,1}.

    From HW1 Problem 2a:
    MUX₃(x₁, x₂, x₃) outputs x₂ if x₁ = 1, and x₃ if x₁ = -1.

    Returns:
        Dictionary mapping subset masks to Fourier coefficients

    Mathematical derivation:
        MUX₃(x) = (1+x₁)/2 · x₂ + (1-x₁)/2 · x₃
                = x₂/2 + x₁x₂/2 + x₃/2 - x₁x₃/2

        So: f̂({2}) = 1/2, f̂({1,2}) = 1/2, f̂({3}) = 1/2, f̂({1,3}) = -1/2
    """
    import boofun as bf

    # MUX₃: output x₂ if x₁=+1 (i.e., x₁=0 in {0,1}), else x₃
    # In {0,1}: MUX(x₁,x₂,x₃) = (1-x₁)x₂ + x₁x₃
    # Truth table: inputs ordered as (x₁,x₂,x₃) in MSB order
    # 000→x₂=0, 001→x₂=0, 010→x₂=1, 011→x₂=1, 100→x₃=0, 101→x₃=1, 110→x₃=0, 111→x₃=1
    tt = [0, 0, 1, 1, 0, 1, 0, 1]
    mux3 = bf.create(tt)

    coeffs = _get_fourier_coefficients(mux3)

    result = {}
    for s in range(len(coeffs)):
        if abs(coeffs[s]) > 1e-10:
            result[s] = float(coeffs[s])

    return result


def compute_nae3_fourier() -> Dict[int, float]:
    """
    Compute Fourier expansion of NAE₃: {-1,1}³ → {0,1}.

    From HW1 Problem 2b:
    NAE₃(x₁, x₂, x₃) = 1 iff not all bits are equal.

    Returns:
        Dictionary mapping subset masks to Fourier coefficients
    """
    import boofun as bf

    # NAE₃: output 1 if not all equal
    # 000→0, 001→1, 010→1, 011→1, 100→1, 101→1, 110→1, 111→0
    tt = [0, 1, 1, 1, 1, 1, 1, 0]
    nae3 = bf.create(tt)

    coeffs = _get_fourier_coefficients(nae3)

    result = {}
    for s in range(len(coeffs)):
        if abs(coeffs[s]) > 1e-10:
            result[s] = float(coeffs[s])

    return result


def compute_and_fourier(n: int) -> Dict[int, float]:
    """
    Compute Fourier expansion of AND_n: {-1,1}^n → {-1,1}.

    From HW1 Problem 2c:
    AND_n(x) = 1 iff all x_i = 1.

    The Fourier expansion is:
        AND_n(x) = (1/2^n) Σ_{S⊆[n]} (-1)^{n-|S|} ∏_{i∈S} x_i

    So f̂(S) = (-1)^{n-|S|} / 2^n for all S.

    Returns:
        Dictionary mapping subset masks to Fourier coefficients
    """
    import boofun as bf

    # AND_n: all 0s except last position (all 1s → output 1)
    size = 1 << n
    tt = [0] * size
    tt[size - 1] = 1  # Only 111...1 → 1
    and_n = bf.create(tt)

    coeffs = _get_fourier_coefficients(and_n)

    result = {}
    for s in range(len(coeffs)):
        if abs(coeffs[s]) > 1e-10:
            result[s] = float(coeffs[s])

    return result
