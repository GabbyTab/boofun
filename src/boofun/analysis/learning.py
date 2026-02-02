"""
Learning algorithms for Boolean functions.

This module implements algorithms for learning Boolean functions from
queries or samples, including the famous Goldreich-Levin algorithm.

Key algorithms:
- Goldreich-Levin: Find heavy Fourier coefficients with query access
- Low-degree algorithm: Learn juntas and low-degree functions
- LMN algorithm: Learn decision trees from uniform samples
- Sparse Fourier learning: Learn functions with few Fourier coefficients

References:
- O'Donnell Chapter 3.4: Goldreich-Levin
- O'Donnell Chapter 4: Social choice and learning
- Linial, Mansour, Nisan (1993): Learning decision trees
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "goldreich_levin",
    "estimate_fourier_coefficient",
    "find_heavy_coefficients",
    "learn_sparse_fourier",
    "GoldreichLevinLearner",
]


def _random_subset(n: int, rng: np.random.Generator) -> int:
    """Generate a random subset of [n] as a bitmask.

    Raises:
        ValueError: If n >= 64 (overflow)
    """
    if n >= 64:
        raise ValueError(f"_random_subset requires n < 64, got {n}")
    return rng.integers(0, 1 << n)


def _inner_product_mod2(x: int, s: int) -> int:
    """Compute <x, s> mod 2 (parity of intersection)."""
    return bin(x & s).count("1") % 2


def estimate_fourier_coefficient(
    f: "BooleanFunction", S: int, num_samples: int = 1000, rng: Optional[np.random.Generator] = None
) -> Tuple[float, float]:
    """
    Estimate the Fourier coefficient f̂(S) using random sampling.

    Uses the identity: f̂(S) = E_x[f(x) * χ_S(x)]
    where χ_S(x) = (-1)^{<x,S>}

    Args:
        f: BooleanFunction to analyze (or query function)
        S: Subset (as bitmask) for which to estimate coefficient
        num_samples: Number of random samples
        rng: Random number generator

    Returns:
        Tuple of (estimate, standard_error)

    Raises:
        ValueError: If n >= 64 (use sampling.estimate_fourier_coefficient instead)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars or 0
    if n == 0:
        return (float(f.evaluate(0)), 0.0)

    if n >= 64:
        raise ValueError(
            f"estimate_fourier_coefficient in learning.py requires n < 64 (got n={n}). "
            f"For large n, the sampling range 2^n overflows int64. "
            f"Consider using bit-array based sampling or oracle patterns."
        )

    # Sample and compute estimate
    samples = []
    for _ in range(num_samples):
        # Generate random input (safe for n < 64)
        x = int(rng.integers(0, 1 << n))

        # Evaluate f(x) in ±1 convention
        fx = 1 - 2 * int(f.evaluate(x))  # Convert {0,1} to {+1,-1}

        # Compute χ_S(x) = (-1)^{<x,S>}
        chi_S_x = 1 - 2 * _inner_product_mod2(x, S)

        samples.append(fx * chi_S_x)

    estimate = np.mean(samples)
    stderr = np.std(samples, ddof=1) / np.sqrt(num_samples)

    return (float(estimate), float(stderr))


def goldreich_levin(
    f: "BooleanFunction",
    threshold: float = 0.1,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, float]]:
    """
    Find all heavy Fourier coefficients using the Goldreich-Levin algorithm.

    The algorithm finds all S such that |f̂(S)| >= threshold, with
    high probability.

    This implementation uses the "bucketing" approach:
    1. Hash coefficients into buckets
    2. Find heavy buckets
    3. Iteratively refine to find individual heavy coefficients

    Args:
        f: BooleanFunction to analyze
        threshold: Minimum absolute coefficient value to report
        confidence: Confidence level for the estimates
        rng: Random number generator

    Returns:
        List of (subset_mask, coefficient_estimate) for heavy coefficients

    Note:
        Query complexity: O(n / threshold^2 * log(1/confidence))
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars or 0
    if n == 0:
        return [(0, float(f.evaluate(0)))]

    # Number of samples needed for good estimates
    num_samples = int(4 / (threshold**2) * np.log(2 / (1 - confidence)))
    num_samples = max(num_samples, 100)

    # Phase 1: Find candidate heavy coefficients using direct estimation
    # For small n, we can check all coefficients directly
    heavy = []

    if n <= 10:  # Direct approach for small n
        for S in range(1 << n):
            est, _ = estimate_fourier_coefficient(f, S, num_samples, rng)
            if abs(est) >= threshold * 0.9:  # Slightly lower threshold to not miss
                heavy.append((S, est))
    else:
        # For larger n, use the Goldreich-Levin tree search
        heavy = _gl_tree_search(f, threshold, num_samples, rng)

    return heavy


def _gl_tree_search(
    f: "BooleanFunction", threshold: float, num_samples: int, rng: np.random.Generator
) -> List[Tuple[int, float]]:
    """
    Goldreich-Levin tree search for heavy coefficients.

    The idea: estimate "aggregate" coefficients for prefixes and prune
    branches that can't contain heavy coefficients.
    """
    n = f.n_vars or 0
    heavy = []

    # BFS through the binary tree of subsets
    # Each node represents a partial assignment to variables
    # We estimate the sum of squares of coefficients in the subtree

    queue = [(0, 0)]  # (prefix_mask, prefix_value)

    while queue:
        prefix_mask, prefix_value = queue.pop(0)

        # Count bits set in prefix
        depth = bin(prefix_mask).count("1")

        if depth == n:
            # Leaf node - this is a complete subset S
            S = prefix_value
            est, _ = estimate_fourier_coefficient(f, S, num_samples, rng)
            if abs(est) >= threshold * 0.9:
                heavy.append((S, est))
        else:
            # Find next variable to branch on
            next_var = 0
            while (prefix_mask >> next_var) & 1:
                next_var += 1

            # Estimate potential in each branch
            new_mask = prefix_mask | (1 << next_var)

            # Branch where variable is 0 (not in S)
            est0, _ = _estimate_subtree_weight(
                f, new_mask, prefix_value, threshold, num_samples, rng
            )
            if est0 >= threshold**2 / 4:  # Could contain heavy coefficient
                queue.append((new_mask, prefix_value))

            # Branch where variable is 1 (in S)
            est1, _ = _estimate_subtree_weight(
                f, new_mask, prefix_value | (1 << next_var), threshold, num_samples, rng
            )
            if est1 >= threshold**2 / 4:
                queue.append((new_mask, prefix_value | (1 << next_var)))

    return heavy


def _estimate_subtree_weight(
    f: "BooleanFunction",
    mask: int,
    value: int,
    threshold: float,
    num_samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Estimate the sum of squared coefficients in a subtree.

    Subtree is defined by: S has the same bits as 'value' where 'mask' is set.
    """
    n = f.n_vars or 0

    # Sample pairs (x, y) where x, y agree on masked bits
    samples = []

    for _ in range(num_samples):
        # Generate random x
        x = int(rng.integers(0, 1 << n))

        # Generate y that agrees with x on unmasked bits
        free_bits = int(rng.integers(0, 1 << n))
        y = (x & ~mask) | (free_bits & mask)

        # Force the prefix constraint
        x = (x & ~mask) | value
        y = (y & ~mask) | value

        # f(x) * f(y) contributes to sum of f̂(S)^2 over S in subtree
        fx = 1 - 2 * int(f.evaluate(x))
        fy = 1 - 2 * int(f.evaluate(y))

        samples.append(fx * fy)

    estimate = np.mean(samples)
    stderr = np.std(samples, ddof=1) / np.sqrt(num_samples)

    return (float(estimate), float(stderr))


def find_heavy_coefficients(
    f: "BooleanFunction",
    threshold: float = 0.01,
    num_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    Find all Fourier coefficients with |f̂(S)| >= threshold.

    This is a simpler alternative to Goldreich-Levin that directly
    estimates all coefficients (suitable for small n).

    Args:
        f: BooleanFunction to analyze
        threshold: Minimum absolute coefficient value
        num_samples: Samples per coefficient estimate
        rng: Random number generator

    Returns:
        Dictionary mapping subset masks to coefficient estimates
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars or 0
    heavy = {}

    for S in range(1 << n):
        est, stderr = estimate_fourier_coefficient(f, S, num_samples, rng)
        if abs(est) >= threshold:
            heavy[S] = est

    return heavy


def learn_sparse_fourier(
    f: "BooleanFunction",
    sparsity: int,
    num_samples: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """
    Learn a function assuming it has at most 'sparsity' non-zero Fourier coefficients.

    Uses sampling to find the heavy coefficients, then refines estimates.

    Args:
        f: BooleanFunction to learn
        sparsity: Maximum number of non-zero coefficients
        num_samples: Number of samples for estimation
        rng: Random number generator

    Returns:
        Dictionary of estimated non-zero Fourier coefficients
    """
    if rng is None:
        rng = np.random.default_rng()

    # Threshold based on sparsity: if k coefficients, each at least 1/sqrt(k) contribution
    # to have unit L2 norm
    threshold = 0.5 / np.sqrt(sparsity)

    return find_heavy_coefficients(f, threshold, num_samples, rng)


class GoldreichLevinLearner:
    """
    Interactive Goldreich-Levin learner with query counting.

    This class tracks the number of queries made and allows for
    interactive exploration of heavy Fourier coefficients.
    """

    def __init__(self, f: "BooleanFunction", rng: Optional[np.random.Generator] = None):
        """
        Initialize the learner.

        Args:
            f: BooleanFunction to learn
            rng: Random number generator
        """
        self.function = f
        self.rng = rng or np.random.default_rng()
        self.query_count = 0
        self._cache: Dict[int, int] = {}  # Cache queries

    def query(self, x: int) -> int:
        """Query f(x), with caching."""
        if x not in self._cache:
            self._cache[x] = int(self.function.evaluate(x))
            self.query_count += 1
        return self._cache[x]

    def estimate_coefficient(self, S: int, num_samples: int = 1000) -> float:
        """Estimate f̂(S) using queries."""
        n = self.function.n_vars or 0

        total = 0.0
        for _ in range(num_samples):
            x = int(self.rng.integers(0, 1 << n))
            fx = 1 - 2 * self.query(x)
            chi_S_x = 1 - 2 * _inner_product_mod2(x, S)
            total += fx * chi_S_x

        return total / num_samples

    def find_heavy(self, threshold: float = 0.1) -> List[Tuple[int, float]]:
        """Find heavy coefficients using Goldreich-Levin."""
        return goldreich_levin(self.function, threshold, rng=self.rng)

    def reset_queries(self):
        """Reset the query counter and cache."""
        self.query_count = 0
        self._cache.clear()

    def summary(self) -> str:
        """Get summary of learning progress."""
        return f"GoldreichLevinLearner: {self.query_count} queries made"
