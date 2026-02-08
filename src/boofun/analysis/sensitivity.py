"""
Sensitivity analysis for Boolean functions.

This module provides sensitivity-related functions for analyzing Boolean functions,
including pointwise sensitivity, average sensitivity, and higher moments.

Sensitivity measures how "locally" a function changes: s(f, x) counts how many
single-bit flips of x change the output of f. This is fundamental to understanding
the complexity and structure of Boolean functions.

Key concepts:
- s(f, x): Sensitivity at input x = |{i : f(x) ≠ f(x^i)}|
- s(f): Maximum sensitivity = max_x s(f, x)
- as(f): Average sensitivity = E_x[s(f, x)] = total influence I[f]
- as_t(f): t-th moment = E_x[s(f, x)^t]

By the Poincaré inequality, average sensitivity equals total influence:
    E_x[s(f, x)] = Σ_i Inf_i[f]

References:
- O'Donnell Chapter 2 (Influences)
- Tal's BooleanFunc.py (sensitivity, average_sensitivity_moment)
- Huang's Sensitivity Theorem (2019): s(f) ≥ √deg(f)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.base import BooleanFunction

__all__ = [
    # Basic sensitivity
    "sensitivity_at",
    "sensitive_coordinates",
    "sensitivity_profile",
    # Aggregate measures
    "max_sensitivity",
    "min_sensitivity",
    "average_sensitivity",
    "total_influence_via_sensitivity",
    # Moments (from Tal's library)
    "average_sensitivity_moment",
    "sensitivity_histogram",
    # Utility
    "arg_max_sensitivity",
    "arg_min_sensitivity",
]


def sensitivity_at(f: BooleanFunction, x: int) -> int:
    """
    Return the sensitivity of f at input index x.

    The sensitivity s(f, x) is the number of coordinates i such that
    f(x) ≠ f(x^i), where x^i is x with the i-th bit flipped.

    Args:
        f: BooleanFunction to analyze
        x: Input point as integer

    Returns:
        Number of sensitive coordinates at x
    """
    n = f.n_vars or 0
    base = bool(f.evaluate(int(x)))
    count = 0
    for i in range(n):
        if bool(f.evaluate(int(x) ^ (1 << i))) != base:
            count += 1
    return count


def sensitive_coordinates(f: BooleanFunction, x: int) -> List[int]:
    """
    Return the list of sensitive coordinates at input x.

    A coordinate i is sensitive at x if f(x) ≠ f(x^i).

    This is Tal's `sens_coor` function.

    Args:
        f: BooleanFunction to analyze
        x: Input point as integer

    Returns:
        List of variable indices i where f(x) ≠ f(x^i)

    References:
        - Tal's BooleanFunc.py: sens_coor
    """
    n = f.n_vars or 0
    base = bool(f.evaluate(int(x)))
    sensitive = []
    for i in range(n):
        if bool(f.evaluate(int(x) ^ (1 << i))) != base:
            sensitive.append(i)
    return sensitive


def sensitivity_profile(f: BooleanFunction) -> np.ndarray:
    """
    Return per-input sensitivities as a NumPy array.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Array where profile[x] = s(f, x) for each input x
    """
    n = f.n_vars or 0
    size = 1 << n
    return np.array([sensitivity_at(f, i) for i in range(size)], dtype=int)


def max_sensitivity(f: BooleanFunction, output_value: Optional[int] = None) -> int:
    """
    Compute the maximum sensitivity s(f).

    The maximum sensitivity is s(f) = max_x s(f, x).

    Optionally restricted to inputs where f(x) = output_value.

    Args:
        f: BooleanFunction to analyze
        output_value: If specified (0 or 1), only consider inputs with this output

    Returns:
        Maximum sensitivity

    References:
        - Tal's BooleanFunc.py: max_sensitivity
        - Huang's Theorem (2019): s(f) ≥ √deg(f)
    """
    n = f.n_vars or 0
    size = 1 << n

    if size == 0:
        return 0

    if output_value is None:
        profile = sensitivity_profile(f)
        return int(np.max(profile))

    # Filter by output value
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)
    max_sens = 0
    for x in range(size):
        if truth_table[x] == output_value:
            max_sens = max(max_sens, sensitivity_at(f, x))

    return max_sens


def min_sensitivity(f: BooleanFunction, output_value: Optional[int] = None) -> int:
    """
    Compute the minimum sensitivity (everywhere sensitivity).

    The minimum sensitivity is es(f) = min_x s(f, x).

    Optionally restricted to inputs where f(x) = output_value.

    Args:
        f: BooleanFunction to analyze
        output_value: If specified (0 or 1), only consider inputs with this output

    Returns:
        Minimum sensitivity

    References:
        - Tal's BooleanFunc.py: min_sensitivity
    """
    n = f.n_vars or 0
    size = 1 << n

    if size == 0:
        return 0

    if output_value is None:
        profile = sensitivity_profile(f)
        return int(np.min(profile))

    # Filter by output value
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)
    min_sens = n + 1  # Sentinel
    for x in range(size):
        if truth_table[x] == output_value:
            min_sens = min(min_sens, sensitivity_at(f, x))

    return min_sens if min_sens <= n else 0


def average_sensitivity(f: BooleanFunction) -> float:
    """
    Compute the average sensitivity as(f).

    The average sensitivity is as(f) = E_x[s(f, x)].

    By the Poincaré inequality, this equals the total influence:
        as(f) = Σ_i Inf_i[f]

    Args:
        f: BooleanFunction to analyze

    Returns:
        Average sensitivity

    References:
        - Tal's BooleanFunc.py: average_sensitivity
        - O'Donnell Theorem 2.14
    """
    profile = sensitivity_profile(f)
    return float(np.mean(profile))


def total_influence_via_sensitivity(f: BooleanFunction) -> float:
    """
    Compute total influence via the average sensitivity definition.

    This is an alias for average_sensitivity, provided for semantic clarity.
    Total influence I[f] = Σ_i Inf_i[f] = E_x[s(f, x)] = as(f).

    Args:
        f: BooleanFunction to analyze

    Returns:
        Total influence
    """
    return average_sensitivity(f)


def average_sensitivity_moment(f: BooleanFunction, t: float) -> float:
    """
    Compute the t-th moment of the sensitivity distribution.

    This computes:
        as_t(f) = E_x[s(f, x)^t]

    The t-th moment captures how spread out the sensitivity distribution is.

    Special cases:
    - t = 0: Always returns 1
    - t = 1: Returns average_sensitivity(f)
    - t = 2: Gives information about variance of sensitivity

    This is Tal's `average_sensitivity_moment` function.

    Args:
        f: BooleanFunction to analyze
        t: Moment parameter (can be non-integer)

    Returns:
        The t-th moment E[s(f,x)^t]

    References:
        - Tal's BooleanFunc.py: average_sensitivity_moment
    """
    profile = sensitivity_profile(f)
    return float(np.mean(profile.astype(float) ** t))


def sensitivity_histogram(f: BooleanFunction) -> np.ndarray:
    """
    Compute histogram of sensitivity values.

    Returns an array H where H[k] = |{x : s(f, x) = k}| / 2^n.

    This gives the distribution of sensitivity values across inputs.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Array of length n+1 where H[k] is fraction of inputs with sensitivity k
    """
    n = f.n_vars or 0
    size = 1 << n

    if n == 0:
        return np.array([1.0])

    profile = sensitivity_profile(f)
    histogram = np.zeros(n + 1)

    for s in profile:
        histogram[s] += 1

    return histogram / size


def arg_max_sensitivity(f: BooleanFunction, output_value: Optional[int] = None) -> Tuple[int, int]:
    """
    Find an input achieving maximum sensitivity.

    Args:
        f: BooleanFunction to analyze
        output_value: If specified, only consider inputs with this output

    Returns:
        Tuple of (input_x, sensitivity) where sensitivity is maximized

    References:
        - Tal's BooleanFunc.py: arg_max_sensitivity
    """
    n = f.n_vars or 0
    size = 1 << n

    if size == 0:
        return (0, 0)

    best_x = 0
    best_sens = 0

    truth_table = None
    if output_value is not None:
        truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)

    for x in range(size):
        if output_value is not None and truth_table is not None and truth_table[x] != output_value:
            continue
        sens = sensitivity_at(f, x)
        if sens > best_sens:
            best_sens = sens
            best_x = x

    return (best_x, best_sens)


def arg_min_sensitivity(f: BooleanFunction, output_value: Optional[int] = None) -> Tuple[int, int]:
    """
    Find an input achieving minimum sensitivity.

    Args:
        f: BooleanFunction to analyze
        output_value: If specified, only consider inputs with this output

    Returns:
        Tuple of (input_x, sensitivity) where sensitivity is minimized
    """
    n = f.n_vars or 0
    size = 1 << n

    if size == 0:
        return (0, 0)

    best_x = 0
    best_sens = n + 1

    truth_table = None
    if output_value is not None:
        truth_table = np.asarray(f.get_representation("truth_table"), dtype=int)

    for x in range(size):
        if output_value is not None and truth_table is not None and truth_table[x] != output_value:
            continue
        sens = sensitivity_at(f, x)
        if sens < best_sens:
            best_sens = sens
            best_x = x

    return (best_x, best_sens if best_sens <= n else 0)
