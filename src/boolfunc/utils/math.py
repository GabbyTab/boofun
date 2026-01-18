"""Lightweight math helpers shared across BoolFunc modules."""

from __future__ import annotations

from itertools import combinations, product, permutations
from math import comb, factorial
from typing import Iterator, List, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "popcnt",
    "poppar",
    "over",
    "subsets",
    "cartesian",
    "num2bin_list",
    "bits",
    "tensor_product",
    "krawchouk",
    "krawchouk2",
    "hamming_distance",
    "hamming_weight",
    "generate_permutations",
    "int_to_binary_tuple",
    "binary_tuple_to_int",
]


def popcnt(x: int) -> int:
    """Return the population count of an integer."""

    return bin(x).count("1")


def poppar(x: int) -> int:
    """Return the parity of the population count."""

    return popcnt(x) & 1


def over(n: int, k: int) -> int:
    """Safe binomial coefficient with bounds guarding."""

    if k < 0 or k > n:
        return 0
    return comb(n, k)


def subsets(a: Sequence[int] | int, k: int | None = None) -> Iterator[Tuple[int, ...]]:
    """Yield subsets of sequence *a* (optionally fixed size)."""

    base = tuple(range(a)) if isinstance(a, int) else tuple(a)
    if k is None:
        for r in range(len(base) + 1):
            yield from combinations(base, r)
    else:
        yield from combinations(base, k)


def cartesian(seqs: Sequence[Sequence]) -> Iterator[Tuple]:
    """Yield cartesian product rows from sequences."""

    return product(*seqs)


def num2bin_list(num: int, n_digits: int) -> List[int]:
    """Convert *num* to an n-digit binary list (MSB first)."""

    return [(num >> i) & 1 for i in range(n_digits - 1, -1, -1)]


def bits(i: int, n: int) -> List[int]:
    """Return *n* bits of *i* as a list (MSB first)."""

    return list(map(int, bin((1 << n) | (i & ((1 << n) - 1)))[3:]))


def tensor_product(
    A: Union[np.ndarray, Sequence], B: Union[np.ndarray, Sequence]
) -> np.ndarray:
    """Compute the Kronecker product of *A* and *B*."""

    return np.kron(np.asarray(A), np.asarray(B))


def krawchouk(n: int, k: int, x: int) -> int:
    """Classical binary Krawchouk polynomial K_k(x; n)."""

    total = 0
    for j in range(0, k + 1):
        total += ((-1) ** j) * over(x, j) * over(n - x, k - j)
    return total


def krawchouk2(n: int, k: int, x: int) -> int:
    """Legacy variant with (-2)^j weights (kept for completeness)."""

    total = 0
    for j in range(0, k + 1):
        total += ((-2) ** j) * over(x, j) * over(n - j, k - j)
    return total


def hamming_distance(x: int, y: int) -> int:
    """Compute Hamming distance between two integers (number of differing bits)."""
    return popcnt(x ^ y)


def hamming_weight(x: int) -> int:
    """Alias for popcnt - number of 1 bits in x."""
    return popcnt(x)


def generate_permutations(n: int) -> Iterator[Tuple[int, ...]]:
    """Generate all permutations of [0, 1, ..., n-1]."""
    return permutations(range(n))


def int_to_binary_tuple(x: int, n: int) -> Tuple[int, ...]:
    """
    Convert integer x to an n-bit binary tuple (LSB first).
    
    Args:
        x: Integer to convert
        n: Number of bits
        
    Returns:
        Tuple of n bits, e.g., (0, 1, 1) for x=6 with n=3
    """
    return tuple((x >> i) & 1 for i in range(n))


def binary_tuple_to_int(bits: Sequence[int]) -> int:
    """
    Convert binary tuple (LSB first) to integer.
    
    Args:
        bits: Sequence of bits (0 or 1)
        
    Returns:
        Integer value
    """
    return sum(b << i for i, b in enumerate(bits))

