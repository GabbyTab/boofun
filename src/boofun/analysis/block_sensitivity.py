"""
Block sensitivity analysis for Boolean functions.

Block sensitivity is a fundamental complexity measure that counts the maximum
number of disjoint sensitive blocks at any input. It is closely related to
decision tree complexity through the block sensitivity theorem.

This module provides both exact and efficient algorithms:
- Exact backtracking search for small functions
- Fast minimal block computation using sieve-like filtering
- Optimized max block sensitivity with pruning via sensitivity/certificates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "block_sensitivity_at",
    "max_block_sensitivity",
    "minimal_sensitive_blocks",
    "sensitive_coordinates",
    "block_sensitivity_profile",
]


def _popcnt(x: int) -> int:
    """Population count (number of 1 bits)."""
    return bin(x).count("1")


def sensitive_coordinates(f: "BooleanFunction", x: int) -> List[int]:
    """
    Find coordinates where f is sensitive at input x.

    A coordinate i is sensitive at x if flipping bit i changes f(x).
    This is the set of coordinates that contribute to sensitivity(f, x).

    Args:
        f: BooleanFunction to analyze
        x: Input index

    Returns:
        List of sensitive coordinate indices
    """
    n = f.n_vars or 0
    base = bool(f.evaluate(int(x)))
    sensitive = []

    for i in range(n):
        if bool(f.evaluate(int(x) ^ (1 << i))) != base:
            sensitive.append(i)

    return sensitive


def minimal_sensitive_blocks(f: "BooleanFunction", x: int) -> List[int]:
    """
    Find all minimal sensitive blocks at input x.

    A block B is a minimal sensitive block if flipping all bits in B changes
    the output, but no proper subset of B is sensitive.

    This uses an efficient sieve-like algorithm:
    1. Mark all sensitive blocks
    2. For each block, check if any proper subset is also sensitive
    3. Keep only blocks with no sensitive proper subsets

    Args:
        f: BooleanFunction to analyze
        x: Input index (integer representation)

    Returns:
        List of minimal sensitive blocks (as bitmasks)

    Note:
        Time complexity: O(n * 2^n) where n is the number of variables.
    """
    n = f.n_vars or 0
    if n == 0:
        return []

    # Get truth table for efficient evaluation
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    base_val = truth_table[x]

    size = 1 << n
    # is_sensitive[S] = True if flipping bits in S changes output
    is_sensitive = np.zeros(size, dtype=bool)
    # has_sensitive_subset[S] = True if some proper subset of S is sensitive
    has_sensitive_subset = np.zeros(size, dtype=bool)

    minimal = []

    # Process blocks in order of increasing size
    for S in range(1, size):
        # Check if S is sensitive
        if truth_table[x ^ S] != base_val:
            is_sensitive[S] = True

        # Check if any single-bit subset is sensitive
        for b in range(n):
            if (S >> b) & 1:  # Bit b is in S
                subset = S ^ (1 << b)  # Remove bit b
                if is_sensitive[subset] or has_sensitive_subset[subset]:
                    has_sensitive_subset[S] = True
                    break

        # S is minimal iff it's sensitive and has no sensitive proper subset
        if is_sensitive[S] and not has_sensitive_subset[S]:
            minimal.append(S)

    return minimal


def _max_disjoint(blocks: List[int], start: int, used: int) -> int:
    """
    Find maximum number of pairwise disjoint blocks via backtracking.

    Args:
        blocks: List of blocks (as bitmasks), sorted by size
        start: Starting index in blocks list
        used: Bitmask of coordinates already used

    Returns:
        Maximum number of disjoint blocks from blocks[start:]
    """
    best = 0
    for idx in range(start, len(blocks)):
        b = blocks[idx]
        if used & b:  # Block overlaps with used coordinates
            continue
        best = max(best, 1 + _max_disjoint(blocks, idx + 1, used | b))
    return best


def block_sensitivity_at(f: "BooleanFunction", x: int) -> int:
    """
    Compute the block sensitivity of f at input x.

    The block sensitivity bs(f, x) is the maximum number of pairwise disjoint
    blocks B such that flipping all bits in each B changes f(x).

    This implementation uses minimal blocks for efficiency:
    1. Find all minimal sensitive blocks (fast sieve algorithm)
    2. Find maximum packing of disjoint blocks (backtracking)

    Args:
        f: BooleanFunction to analyze
        x: Input index (integer representation)

    Returns:
        Block sensitivity at x

    Example:
        >>> and_func = bf.create([0, 0, 0, 1])
        >>> block_sensitivity_at(and_func, 3)  # At (1,1)
        2  # Both single-bit blocks are sensitive
    """
    minimal = minimal_sensitive_blocks(f, x)

    if not minimal:
        return 0

    # Sort by block size (smaller first)
    minimal.sort(key=_popcnt)

    return _max_disjoint(minimal, 0, 0)


def max_block_sensitivity(
    f: "BooleanFunction", value: Optional[int] = None, use_pruning: bool = True
) -> int:
    """
    Compute the maximum block sensitivity of f.

    The block sensitivity bs(f) = max_x bs(f, x).

    This implementation uses several optimizations:
    1. Early termination when bs reaches n (the maximum possible)
    2. Optional pruning: skip inputs where sensitivity = certificate
       (since bs(x) >= s(x) and bs(x) <= C(x))

    Args:
        f: BooleanFunction to analyze
        value: If specified (0 or 1), only consider inputs where f(x) = value
        use_pruning: Whether to use sensitivity/certificate pruning

    Returns:
        Maximum block sensitivity across all (relevant) inputs

    Note:
        Block sensitivity satisfies:
        - s(f) <= bs(f) <= C(f) (sensitivity <= block sensitivity <= certificate)
        - D(f) <= bs(f)^2 (Nisan's theorem, superseded)
        - bs(f) = Theta(s(f)^4) (Huang's theorem, 2019)
    """
    n = f.n_vars or 0
    if n == 0:
        return 0

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)

    best = 0
    size = 1 << n

    for x in range(size):
        # Filter by value if specified
        if value is not None and truth_table[x] != bool(value):
            continue

        if use_pruning:
            # Quick sensitivity check - if equal to current best, might skip
            sens = len(sensitive_coordinates(f, x))

            # Block sensitivity is at least sensitivity
            if sens > best:
                best = sens

            # If sensitivity equals n, block sensitivity is also n
            if sens == n:
                break

        # Compute full block sensitivity
        bs = block_sensitivity_at(f, x)

        if bs > best:
            best = bs

        # Early termination
        if best == n:
            break

    return best


def block_sensitivity_profile(f: "BooleanFunction") -> Tuple[int, int, List[int]]:
    """
    Compute block sensitivity profile of f.

    Returns:
        Tuple of (bs0, bs1, per_input_bs) where:
        - bs0: max block sensitivity on inputs where f(x) = 0
        - bs1: max block sensitivity on inputs where f(x) = 1
        - per_input_bs: list of bs(f, x) for each input x
    """
    n = f.n_vars or 0
    if n == 0:
        return (0, 0, [])

    truth_table = np.asarray(f.get_representation("truth_table"), dtype=bool)
    size = 1 << n

    per_input = []
    bs0, bs1 = 0, 0

    for x in range(size):
        bs = block_sensitivity_at(f, x)
        per_input.append(bs)

        if truth_table[x]:
            bs1 = max(bs1, bs)
        else:
            bs0 = max(bs0, bs)

    return (bs0, bs1, per_input)
