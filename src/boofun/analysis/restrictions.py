"""
Random restrictions for Boolean function analysis.

Random restrictions are a fundamental tool in circuit complexity and DNF analysis,
as described in O'Donnell Chapter 4 and the Switching Lemma.

A p-random restriction ρ ∈ {0, 1, *}^n assigns each variable:
- A fixed value (0 or 1) with probability 1-p each
- The "free" symbol * with probability p

When applied to f, the restriction f|_ρ is a function on the remaining free variables.

Key applications:
- DNF Fourier concentration (Mansour's theorem)
- Decision tree depth reduction
- Switching lemma proofs
- Shrinkage of formulas under random restrictions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "Restriction",
    "random_restriction",
    "apply_restriction",
    "restriction_shrinkage",
    "average_restricted_decision_tree_depth",
    "switching_lemma_probability",
]


@dataclass
class Restriction:
    """
    A restriction that fixes some variables and leaves others free.

    Attributes:
        fixed: Dictionary mapping variable indices to fixed values (0 or 1)
        free: Set of variable indices left free
        n_vars: Original number of variables
    """

    fixed: Dict[int, int]
    free: Set[int]
    n_vars: int

    @property
    def p(self) -> float:
        """Probability parameter (fraction of free variables)."""
        return len(self.free) / self.n_vars if self.n_vars > 0 else 0.0

    def __str__(self) -> str:
        parts = []
        for i in range(self.n_vars):
            if i in self.fixed:
                parts.append(str(self.fixed[i]))
            else:
                parts.append("*")
        return "".join(parts)

    def __repr__(self) -> str:
        return f"Restriction({self})"


def random_restriction(n: int, p: float, rng: Optional[np.random.Generator] = None) -> Restriction:
    """
    Generate a random p-restriction on n variables.

    Each variable is independently:
    - Fixed to 0 with probability (1-p)/2
    - Fixed to 1 with probability (1-p)/2
    - Left free (*) with probability p

    Args:
        n: Number of variables
        p: Probability of leaving a variable free
        rng: Random number generator

    Returns:
        Random Restriction object

    Example:
        >>> rho = random_restriction(10, 0.3)
        >>> len(rho.free)  # Approximately 3 variables
    """
    if rng is None:
        rng = np.random.default_rng()

    if not 0 <= p <= 1:
        raise ValueError(f"Probability p must be in [0,1], got {p}")

    fixed = {}
    free = set()

    for i in range(n):
        r = rng.random()
        if r < p:
            free.add(i)
        elif r < p + (1 - p) / 2:
            fixed[i] = 0
        else:
            fixed[i] = 1

    return Restriction(fixed=fixed, free=free, n_vars=n)


def apply_restriction(f: "BooleanFunction", rho: Restriction) -> "BooleanFunction":
    """
    Apply a restriction to a Boolean function.

    The restricted function f|_ρ is defined only on the free variables.
    Fixed variables are replaced by their assigned values.

    Args:
        f: BooleanFunction to restrict
        rho: Restriction to apply

    Returns:
        Restricted BooleanFunction on the free variables

    Note:
        The resulting function has fewer variables (only the free ones).
    """
    if f.n_vars != rho.n_vars:
        raise ValueError(f"Function has {f.n_vars} vars but restriction has {rho.n_vars}")

    # Use the fix method to apply fixed variables
    result = f

    # Sort fixed variables by index descending to preserve indices
    for var, val in sorted(rho.fixed.items(), reverse=True):
        result = result.fix(var, val)

    return result


def restriction_shrinkage(
    f: "BooleanFunction",
    p: float,
    num_samples: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Estimate how much random restrictions shrink various complexity measures.

    This is key to understanding DNF/decision tree complexity:
    - Random restrictions simplify functions
    - After restriction, DNFs become decision trees of small depth

    Args:
        f: BooleanFunction to analyze
        p: Probability parameter for restrictions
        num_samples: Number of random restrictions to sample
        rng: Random number generator

    Returns:
        Dictionary with statistics about shrinkage
    """
    if rng is None:
        rng = np.random.default_rng()

    from .complexity import decision_tree_depth
    from .gf2 import gf2_degree

    n = f.n_vars or 0

    original_dt_depth = decision_tree_depth(f)
    original_gf2_deg = gf2_degree(f)

    dt_depths = []
    gf2_degrees = []
    num_free_vars = []
    constant_count = 0

    for _ in range(num_samples):
        rho = random_restriction(n, p, rng)

        try:
            f_rho = apply_restriction(f, rho)

            num_free = len(rho.free)
            num_free_vars.append(num_free)

            if num_free == 0:
                constant_count += 1
                dt_depths.append(0)
                gf2_degrees.append(0)
            else:
                dt_depths.append(decision_tree_depth(f_rho))
                gf2_degrees.append(gf2_degree(f_rho))

        except Exception:
            # Skip failed restrictions
            continue

    return {
        "original_dt_depth": original_dt_depth,
        "original_gf2_degree": original_gf2_deg,
        "avg_restricted_dt_depth": np.mean(dt_depths) if dt_depths else 0,
        "avg_restricted_gf2_degree": np.mean(gf2_degrees) if gf2_degrees else 0,
        "avg_free_vars": np.mean(num_free_vars) if num_free_vars else 0,
        "expected_free_vars": n * p,
        "constant_fraction": constant_count / num_samples,
        "depth_shrinkage_factor": (
            np.mean(dt_depths) / original_dt_depth if original_dt_depth > 0 else 0
        ),
    }


def average_restricted_decision_tree_depth(
    f: "BooleanFunction",
    p: float,
    num_samples: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate expected decision tree depth after random p-restriction.

    From O'Donnell Chapter 4: For width-w DNFs, after a p-restriction,
    the expected decision tree depth is O(log(1/p) * p^w).

    Args:
        f: BooleanFunction to analyze
        p: Restriction probability
        num_samples: Number of samples for estimation
        rng: Random number generator

    Returns:
        Estimated expected decision tree depth after restriction
    """
    if rng is None:
        rng = np.random.default_rng()

    from .complexity import decision_tree_depth

    n = f.n_vars or 0
    depths = []

    for _ in range(num_samples):
        rho = random_restriction(n, p, rng)

        try:
            f_rho = apply_restriction(f, rho)
            if f_rho.n_vars > 0:
                depths.append(decision_tree_depth(f_rho))
            else:
                depths.append(0)
        except Exception:
            continue

    return np.mean(depths) if depths else 0.0


def switching_lemma_probability(width: int, p: float, depth_threshold: int) -> float:
    """
    Estimate probability bound from the Switching Lemma.

    The Switching Lemma (Håstad) states that for a width-w DNF f,
    after a random p-restriction:
        Pr[DT-depth(f|_ρ) > s] ≤ (5pw)^s

    Args:
        width: Width of the DNF (max clause size)
        p: Restriction probability
        depth_threshold: Depth threshold s

    Returns:
        Upper bound on probability that restricted DT-depth exceeds threshold

    Note:
        This is the theoretical bound, not empirical measurement.
    """
    return min(1.0, (5 * p * width) ** depth_threshold)


def batch_random_restrictions(
    n: int, p: float, num_restrictions: int, rng: Optional[np.random.Generator] = None
) -> List[Restriction]:
    """
    Generate multiple random restrictions.

    Args:
        n: Number of variables
        p: Probability of leaving variable free
        num_restrictions: Number of restrictions to generate
        rng: Random number generator

    Returns:
        List of Restriction objects
    """
    if rng is None:
        rng = np.random.default_rng()

    return [random_restriction(n, p, rng) for _ in range(num_restrictions)]


def restriction_to_inputs(rho: Restriction, num_inputs: int = None) -> List[int]:
    """
    Generate all inputs consistent with a restriction.

    For a restriction ρ, returns all x such that x agrees with ρ on fixed vars.

    Args:
        rho: Restriction to expand
        num_inputs: Maximum number of inputs to return (None = all)

    Returns:
        List of input indices (as integers)
    """
    free_vars = sorted(rho.free)
    num_free = len(free_vars)

    max_inputs = 1 << num_free
    if num_inputs is not None:
        max_inputs = min(max_inputs, num_inputs)

    result = []

    for i in range(max_inputs):
        # Start with fixed values
        x = 0
        for var, val in rho.fixed.items():
            if val:
                x |= 1 << (rho.n_vars - 1 - var)

        # Add free variable values
        for j, var in enumerate(free_vars):
            if (i >> j) & 1:
                x |= 1 << (rho.n_vars - 1 - var)

        result.append(x)

    return result
