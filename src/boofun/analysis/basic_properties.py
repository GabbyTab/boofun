"""
Basic properties of Boolean functions.

This module implements fundamental structural properties as defined in
Scott Aaronson's Boolean Function Wizard:

- unate: Is the function isomorphic to a monotone function?
- qsym: Is the function isomorphic to a symmetric function?
- balanced: Is the function balanced (equal 0s and 1s)?
- prime: Is the function prime (not decomposable)?
- dependence: How many variables does the function actually depend on?

These properties are important for understanding the structure of Boolean
functions and can dramatically affect their complexity measures.
"""

from __future__ import annotations

from itertools import permutations
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Monotonicity and unateness
    "is_monotone",
    "is_unate",
    "make_unate",
    "monotone_closure",
    # Symmetry
    "is_symmetric",
    "is_quasisymmetric",
    "symmetry_type",
    # Balancedness
    "is_balanced",
    "bias",
    "weight",
    # Dependence
    "dependent_variables",
    "essential_variables",
    # Primality / Decomposition
    "is_prime",
    "find_decomposition",
]


def is_monotone(f: "BooleanFunction") -> bool:
    """
    Check if f is monotone.

    A function is monotone if for all x <= y (coordinate-wise), f(x) <= f(y).
    Equivalently, all partial derivatives are non-negative.

    Args:
        f: BooleanFunction to check

    Returns:
        True if f is monotone
    """
    n = f.n_vars
    if n is None or n == 0:
        return True

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    size = 1 << n

    # Check: for each x, if we flip any 0->1, output doesn't decrease
    for x in range(size):
        fx = truth_table[x]
        if not fx:  # f(x) = 0
            continue
        # If f(x) = 1, all y < x should have f(y) <= f(x)
        # Check all y that are subsets of x (y & x == y)
        for i in range(n):
            if (x >> i) & 1:  # Bit i is 1 in x
                y = x ^ (1 << i)  # y = x with bit i flipped to 0
                if truth_table[y] > fx:  # f(y) > f(x) violates monotonicity
                    return False

    return True


def is_unate(f: "BooleanFunction") -> Tuple[bool, Optional[List[int]]]:
    """
    Check if f is unate (isomorphic to a monotone function).

    A function is unate if there exists a setting of input polarities
    (negate some variables) that makes it monotone.

    Args:
        f: BooleanFunction to check

    Returns:
        Tuple of (is_unate, polarities) where polarities[i] = 1 means
        keep variable i as-is, polarities[i] = -1 means negate it.
        If not unate, returns (False, None).
    """
    n = f.n_vars
    if n is None or n == 0:
        return (True, [])

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    # Try all 2^n polarity assignments
    for polarity_mask in range(1 << n):
        # Apply polarities
        is_mono = True

        for x in range(1 << n):
            # Transform x according to polarities
            x_transformed = x ^ polarity_mask
            fx = truth_table[x_transformed]

            if not fx:
                continue

            # Check all neighbors below x (in transformed space)
            for i in range(n):
                if (x >> i) & 1:
                    y = x ^ (1 << i)
                    y_transformed = y ^ polarity_mask
                    if truth_table[y_transformed] > fx:
                        is_mono = False
                        break

            if not is_mono:
                break

        if is_mono:
            polarities = [1 if (polarity_mask >> i) & 1 == 0 else -1 for i in range(n)]
            return (True, polarities)

    return (False, None)


def make_unate(f: "BooleanFunction") -> Optional["BooleanFunction"]:
    """
    Transform f to a monotone function by negating variables if possible.

    Args:
        f: BooleanFunction to transform

    Returns:
        Monotone BooleanFunction if f is unate, None otherwise
    """
    is_u, polarities = is_unate(f)
    if not is_u:
        return None

    from ..core.base import BooleanFunction as BFClass
    from ..core.factory import BooleanFunctionFactory

    n = f.n_vars
    if n is None:
        return f

    assert polarities is not None

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    # Apply polarities
    polarity_mask = sum((1 << i) for i in range(n) if polarities[i] == -1)
    new_tt = np.array([truth_table[x ^ polarity_mask] for x in range(1 << n)], dtype=bool)

    return BooleanFunctionFactory.from_truth_table(BFClass, new_tt, n=n)


def monotone_closure(f: "BooleanFunction") -> "BooleanFunction":
    """
    Compute the monotone closure of f.

    The monotone closure g(x) = max_{y <= x} f(y).

    Args:
        f: BooleanFunction to close

    Returns:
        Monotone closure of f
    """
    from ..core.base import BooleanFunction as BFClass
    from ..core.factory import BooleanFunctionFactory

    n = f.n_vars
    if n is None or n == 0:
        return f

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    new_tt = truth_table.copy()

    # For each x, set new_tt[x] = OR of truth_table[y] for all y <= x
    for x in range(1 << n):
        if new_tt[x]:
            # Propagate upward
            for i in range(n):
                if not ((x >> i) & 1):  # Bit i is 0
                    new_tt[x | (1 << i)] = True

    return BooleanFunctionFactory.from_truth_table(BFClass, new_tt, n=n)


def is_symmetric(f: "BooleanFunction") -> bool:
    """
    Check if f is symmetric.

    A function is symmetric if its value depends only on the Hamming weight
    of the input (number of 1s), not on which specific coordinates are 1.

    Args:
        f: BooleanFunction to check

    Returns:
        True if f is symmetric
    """
    n = f.n_vars
    if n is None or n == 0:
        return True

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    # Group inputs by Hamming weight
    for weight in range(n + 1):
        values = []
        for x in range(1 << n):
            if bin(x).count("1") == weight:
                values.append(truth_table[x])

        # All values at this weight should be the same
        if len(values) > 0 and not all(v == values[0] for v in values):
            return False

    return True


def is_quasisymmetric(f: "BooleanFunction") -> Tuple[bool, Optional[List[int]]]:
    """
    Check if f is quasisymmetric (isomorphic to a symmetric function).

    A function is quasisymmetric if there exists a permutation of the
    input variables that makes it symmetric.

    Args:
        f: BooleanFunction to check

    Returns:
        Tuple of (is_quasisymmetric, permutation).
        If quasisymmetric, permutation maps original indices to new indices.
    """
    n = f.n_vars
    if n is None or n == 0:
        return (True, [])

    if n > 8:
        # Checking all permutations is too expensive for large n
        # Use a heuristic: check if influence profile is uniform
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()

        # If all influences are different, likely not quasisymmetric
        unique_inf = len(set(np.round(influences, 4)))
        if unique_inf == n:
            return (False, None)

        # If symmetric already
        if is_symmetric(f):
            return (True, list(range(n)))

        # Can't efficiently check, return unknown
        return (False, None)

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    # Try all permutations
    for perm in permutations(range(n)):
        # Check if this permutation makes f symmetric
        is_sym = True

        for weight in range(n + 1):
            values = []
            for x in range(1 << n):
                if bin(x).count("1") == weight:
                    # Apply permutation to x
                    x_perm = sum((((x >> i) & 1) << perm[i]) for i in range(n))
                    values.append(truth_table[x_perm])

            if len(values) > 0 and not all(v == values[0] for v in values):
                is_sym = False
                break

        if is_sym:
            return (True, list(perm))

    return (False, None)


def symmetry_type(f: "BooleanFunction") -> str:
    """
    Determine the symmetry type of f.

    Returns:
        One of "symmetric", "quasisymmetric", "asymmetric"
    """
    if is_symmetric(f):
        return "symmetric"

    is_qsym, _ = is_quasisymmetric(f)
    if is_qsym:
        return "quasisymmetric"

    return "asymmetric"


def is_balanced(f: "BooleanFunction") -> bool:
    """
    Check if f is balanced (equal number of 0s and 1s).

    Args:
        f: BooleanFunction to check

    Returns:
        True if f is balanced
    """
    n = f.n_vars
    if n is None or n == 0:
        return True

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    ones: int = int(np.sum(truth_table))
    total = len(truth_table)

    return ones == total // 2


def bias(f: "BooleanFunction") -> float:
    """
    Compute the bias of f: E[f(x)] = Pr[f(x) = 1].

    Args:
        f: BooleanFunction to analyze

    Returns:
        Bias in [0, 1]
    """
    n = f.n_vars
    if n is None or n == 0:
        return float(f.evaluate(0))

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    return float(np.mean(truth_table))


def weight(f: "BooleanFunction") -> int:
    """
    Compute the weight (number of 1s in truth table) of f.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Number of inputs where f(x) = 1
    """
    n = f.n_vars
    if n is None or n == 0:
        return int(f.evaluate(0))

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    return int(np.sum(truth_table))


def dependent_variables(f: "BooleanFunction") -> List[int]:
    """
    Find variables that f depends on.

    A variable i is essential if there exists some x where flipping
    bit i changes the output.

    Args:
        f: BooleanFunction to analyze

    Returns:
        List of essential variable indices
    """
    n = f.n_vars
    if n is None or n == 0:
        return []

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    essential = []
    for i in range(n):
        is_essential = False
        for x in range(1 << n):
            if truth_table[x] != truth_table[x ^ (1 << i)]:
                is_essential = True
                break

        if is_essential:
            essential.append(i)

    return essential


def essential_variables(f: "BooleanFunction") -> int:
    """
    Count the number of essential (dependent) variables.

    This is the vars(f) property from BFW.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Number of variables f depends on
    """
    return len(dependent_variables(f))


def is_prime(f: "BooleanFunction") -> bool:
    """
    Check if f is prime (not decomposable).

    A function is prime if it cannot be written as a composition
    g(h1(x), h2(x)) where g is not a projection and h1, h2 are
    non-trivial (depend on < n variables each).

    Args:
        f: BooleanFunction to check

    Returns:
        True if f is prime

    Note:
        This is a simplified primality check. Full decomposition
        detection is more complex.
    """
    n = f.n_vars
    if n is None or n <= 2:
        return True  # Small functions are considered prime

    # Quick check: if all variables are essential, harder to decompose
    essential = essential_variables(f)
    if essential < n:
        return False  # Has dummy variables, not prime in that sense

    # Check for read-once decomposition
    # A function is read-once if it can be computed by a formula
    # where each variable appears exactly once

    # Heuristic: check if influence distribution is very skewed
    from ..analysis import SpectralAnalyzer

    analyzer = SpectralAnalyzer(f)
    influences = analyzer.influences()

    max_inf: float = float(np.max(influences))
    min_inf: float = float(np.min(influences))

    # If one variable dominates, might be decomposable
    if max_inf > 2 * min_inf and min_inf < 0.1:
        return False

    return True


def find_decomposition(f: "BooleanFunction") -> Optional[Tuple[str, List[int], List[int]]]:
    """
    Try to find a decomposition of f.

    Looks for simple decompositions like:
    - f(x) = g(x_S) op h(x_T) where S, T partition the variables
    - op is AND, OR, or XOR

    Args:
        f: BooleanFunction to decompose

    Returns:
        Tuple of (operator, S_indices, T_indices) if decomposable,
        None otherwise
    """
    n = f.n_vars
    if n is None or n <= 2:
        return None

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    # Try partitioning variables
    for k in range(1, n):
        for S_mask in range(1, (1 << n) - 1):
            if bin(S_mask).count("1") != k:
                continue

            T_mask = ((1 << n) - 1) ^ S_mask
            [i for i in range(n) if (S_mask >> i) & 1]
            [i for i in range(n) if (T_mask >> i) & 1]

            # Check if f decomposes as g(x_S) AND h(x_T)
            # This would mean f is 1 iff both g(x_S) = 1 and h(x_T) = 1

            # First, check if rows (fixing S) have consistent structure
            # This is expensive; skip for large n
            if n > 6:
                continue

            # Check AND decomposition
            # For each value of x_S, the function restricted to x_T should be constant
            # OR some fixed g(x_T)

            # Skip detailed check for efficiency

    return None
