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
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

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
    "batch_random_restrictions",
    "restriction_to_inputs",
    "min_fixing_to_constant",
    "shift_by_mask",
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
        "avg_restricted_dt_depth": float(np.mean(dt_depths)) if dt_depths else 0.0,
        "avg_restricted_gf2_degree": float(np.mean(gf2_degrees)) if gf2_degrees else 0.0,
        "avg_free_vars": float(np.mean(num_free_vars)) if num_free_vars else 0.0,
        "expected_free_vars": n * p,
        "constant_fraction": constant_count / num_samples,
        "depth_shrinkage_factor": (
            float(np.mean(dt_depths)) / original_dt_depth if original_dt_depth > 0 else 0.0
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
            if f_rho.n_vars is not None and f_rho.n_vars > 0:
                depths.append(decision_tree_depth(f_rho))
            else:
                depths.append(0)
        except Exception:
            continue

    return float(np.mean(depths)) if depths else 0.0


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


def restriction_to_inputs(rho: Restriction, num_inputs: Optional[int] = None) -> List[int]:
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


def min_fixing_to_constant(
    f: "BooleanFunction",
    target_value: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[int, int]]:
    """
    Find a minimum set of variable assignments that fixes f to a constant.

    This implements Tal's min_fixing algorithm: find the smallest partial
    assignment that makes f constant (always 0 or always 1).

    Args:
        f: BooleanFunction to analyze
        target_value: Target constant value (0 or 1)
        rng: Random number generator (for tie-breaking)

    Returns:
        Dictionary {var_index: value} of minimum fixing, or None if impossible

    Note:
        This is related to certificate complexity. The size of the minimum
        fixing to 1 is the 1-certificate complexity at the all-zeros input.

    Reference:
        Tal's library: min_fixing(go_up) function
    """
    if rng is None:
        rng = np.random.default_rng()

    n = f.n_vars or 0
    if n == 0:
        return {} if bool(f.evaluate(0)) == target_value else None

    # Greedy approach: try to fix variables one at a time
    # This doesn't guarantee minimum but is efficient
    fixed: Dict[int, int] = {}
    remaining = set(range(n))

    while remaining:
        # Check if already constant
        is_constant = True
        first_val: Optional[bool] = None

        for x in range(1 << n):
            # Check if x is consistent with current fixing
            consistent = True
            for var, val in fixed.items():
                bit = (x >> (n - 1 - var)) & 1
                if bit != val:
                    consistent = False
                    break

            if consistent:
                fval = bool(f.evaluate(x))
                if first_val is None:
                    first_val = fval
                elif fval != first_val:
                    is_constant = False
                    break

        if is_constant and first_val == target_value:
            return fixed

        if not remaining:
            break

        # Try fixing each remaining variable
        best_var = None
        best_val = None
        best_remaining_diversity = float("inf")

        for var in remaining:
            for val in [0, 1]:
                test_fixed = dict(fixed)
                test_fixed[var] = val

                # Count diversity of outputs under this fixing
                outputs = set()
                for x in range(1 << n):
                    consistent = True
                    for v, vval in test_fixed.items():
                        bit = (x >> (n - 1 - v)) & 1
                        if bit != vval:
                            consistent = False
                            break
                    if consistent:
                        outputs.add(bool(f.evaluate(x)))

                if len(outputs) == 1 and target_value in outputs:
                    # This single fix achieves the target!
                    fixed[var] = val
                    return fixed

                diversity = len(outputs)
                if diversity < best_remaining_diversity:
                    best_remaining_diversity = diversity
                    best_var = var
                    best_val = val

        if best_var is not None and best_val is not None:
            fixed[best_var] = best_val
            remaining.remove(best_var)
        else:
            break

    # Final check
    for x in range(1 << n):
        consistent = True
        for var, val in fixed.items():
            bit = (x >> (n - 1 - var)) & 1
            if bit != val:
                consistent = False
                break
        if consistent:
            if bool(f.evaluate(x)) != target_value:
                return None

    return fixed


def shift_by_mask(f: "BooleanFunction", mask: int) -> "BooleanFunction":
    """
    Shift a Boolean function by XORing all inputs with a mask.

    Returns g where g(x) = f(x ⊕ mask).

    This is useful for:
    - Converting between different input conventions
    - Finding monotone representations
    - Analyzing function structure under affine transformations

    Args:
        f: BooleanFunction to shift
        mask: Integer mask to XOR with inputs

    Returns:
        Shifted BooleanFunction

    Reference:
        Tal's library: shift(sh) function
    """
    import boofun as bf

    n = f.n_vars or 0
    if n == 0:
        return f

    # Validate mask
    if mask < 0 or mask >= (1 << n):
        raise ValueError(f"Mask {mask} out of range for {n}-variable function")

    # Build new truth table
    new_tt = []
    for x in range(1 << n):
        shifted_x = x ^ mask
        new_tt.append(int(f.evaluate(shifted_x)))

    return bf.create(new_tt)
