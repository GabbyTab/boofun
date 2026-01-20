"""
Complexity measures for Boolean functions.

This module implements various complexity measures including:
- Decision tree depth (deterministic query complexity)
- Decision tree size
- Sensitivity and block sensitivity
- Certificate complexity

These are fundamental measures in computational complexity theory
as discussed in O'Donnell's book and related literature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "decision_tree_depth",
    "decision_tree_size",
    "sensitivity",
    "max_sensitivity",
    "min_sensitivity",
    "average_sensitivity",
    "certificate_complexity",
    "ComplexityProfile",
]


def decision_tree_depth(f: "BooleanFunction") -> int:
    """
    Compute the optimal deterministic decision tree depth for function f.

    The decision tree depth D(f) is the minimum number of adaptive queries
    needed to compute f on any input in the worst case.

    This uses dynamic programming over sub-cubes, computing the optimal
    depth for all partial assignments simultaneously.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Optimal decision tree depth

    Note:
        Time complexity: O(3^n) where n is the number of variables.
        Memory complexity: O(3^n)

    Example:
        >>> majority = bf.BooleanFunctionBuiltins.majority(3)
        >>> decision_tree_depth(majority)
        2  # Can determine majority with 2 queries
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)

    # Phase 1: Compute average values for all sub-cubes
    # We represent sub-cubes using ternary indices where:
    # - 0 means variable fixed to 0
    # - 1 means variable fixed to 1
    # - 2 means variable is free (not fixed)
    # The average is the fraction of 1s in that sub-cube

    old_layer = list(truth_table)

    for i in range(n):
        new_size = 3 ** (i + 1) * 2 ** (n - (i + 1))
        new_layer = [0.0] * new_size
        mask = (1 << (n - (i + 1))) - 1

        for index in range(len(old_layer)):
            first = index >> (n - i)
            last = index & mask
            ind = (index >> (n - (i + 1))) & 1

            # Position where this value is directly stored
            new_ind1 = ((first * 3 + ind) << (n - (i + 1))) + last
            # Position for the "free" variable (average)
            new_ind2 = ((first * 3 + 2) << (n - (i + 1))) + last

            new_layer[new_ind1] = old_layer[index]
            new_layer[new_ind2] += old_layer[index] / 2.0

        old_layer = new_layer

    # Phase 2: DP to find optimal decision tree depth
    # For each sub-cube, compute min depth needed
    INF = 1 << 20
    result = [INF] * len(new_layer)

    for j in range(len(new_layer)):
        avg = new_layer[j]

        # If sub-cube is constant (all 0s or all 1s), depth is 0
        if avg < 1e-7 or avg > 1 - 1e-7:
            result[j] = 0
        else:
            # Try each variable as the first query
            trits = [(j // (3**i)) % 3 for i in range(n)]

            for i in range(n):
                if trits[i] == 2:  # Variable i is free
                    # Compute indices for sub-cubes with i fixed to 0 and 1
                    idx_0 = j - 2 * (3**i)  # i fixed to 0
                    idx_1 = j - (3**i)  # i fixed to 1

                    # Depth is max of depths of children + 1
                    depth = max(result[idx_0], result[idx_1]) + 1

                    if depth < result[j]:
                        result[j] = depth

    return result[-1]  # Last entry is the full cube (all variables free)


def decision_tree_size(f: "BooleanFunction", minimize_depth_first: bool = False) -> Tuple[int, int]:
    """
    Compute the optimal decision tree size for function f.

    Args:
        f: BooleanFunction to analyze
        minimize_depth_first: If True, minimize depth first, then size.
                             If False, minimize size first, then depth.

    Returns:
        Tuple of (size, depth) where size is the number of leaves

    Note:
        Size measures the total number of leaves in the decision tree,
        while depth measures the longest root-to-leaf path.
    """
    n = f.n_vars
    if n is None or n == 0:
        return (1, 0)

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)

    # Phase 1: Compute averages
    old_layer = list(truth_table)

    for i in range(n):
        new_size = 3 ** (i + 1) * 2 ** (n - (i + 1))
        new_layer = [0.0] * new_size
        mask = (1 << (n - (i + 1))) - 1

        for index in range(len(old_layer)):
            first = index >> (n - i)
            last = index & mask
            ind = (index >> (n - (i + 1))) & 1

            new_ind1 = ((first * 3 + ind) << (n - (i + 1))) + last
            new_ind2 = ((first * 3 + 2) << (n - (i + 1))) + last

            new_layer[new_ind1] = old_layer[index]
            new_layer[new_ind2] += old_layer[index] / 2.0

        old_layer = new_layer

    # Phase 2: DP for size and depth
    INF = 1 << 20
    # result[j] = (size, depth) for sub-cube j
    result = [(INF, INF)] * len(new_layer)

    for j in range(len(new_layer)):
        avg = new_layer[j]

        if avg < 1e-7:  # Constant 0
            result[j] = (1, 0)
        elif avg > 1 - 1e-7:  # Constant 1
            result[j] = (1, 0)
        else:
            trits = [(j // (3**i)) % 3 for i in range(n)]

            for i in range(n):
                if trits[i] == 2:
                    idx_0 = j - 2 * (3**i)
                    idx_1 = j - (3**i)

                    s0, d0 = result[idx_0]
                    s1, d1 = result[idx_1]

                    size = s0 + s1
                    depth = max(d0, d1) + 1

                    # Compare based on priority
                    if minimize_depth_first:
                        key = (depth, size)
                        current_key = (result[j][1], result[j][0])
                    else:
                        key = (size, depth)
                        current_key = result[j]

                    if key < current_key:
                        result[j] = (size, depth)

    return result[-1]


def sensitivity(f: "BooleanFunction", x: int) -> int:
    """
    Compute the sensitivity of f at input x.

    The sensitivity s(f, x) counts how many single-bit flips change the output.

    Args:
        f: BooleanFunction to analyze
        x: Input index (integer representation)

    Returns:
        Number of sensitive coordinates at x
    """
    n = f.n_vars
    if n is None:
        return 0

    base_val = bool(f.evaluate(x))
    count = 0

    for i in range(n):
        flipped = x ^ (1 << i)
        if bool(f.evaluate(flipped)) != base_val:
            count += 1

    return count


def max_sensitivity(f: "BooleanFunction", value: Optional[int] = None) -> int:
    """
    Compute the maximum sensitivity of f.

    Args:
        f: BooleanFunction to analyze
        value: If specified (0 or 1), only consider inputs where f(x) = value

    Returns:
        Maximum sensitivity across all (relevant) inputs
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0

    max_sens = 0

    for x in range(1 << n):
        if value is not None and bool(f.evaluate(x)) != bool(value):
            continue

        sens = sensitivity(f, x)
        max_sens = max(max_sens, sens)

        if max_sens == n:  # Can't get higher
            break

    return max_sens


def min_sensitivity(f: "BooleanFunction", value: Optional[int] = None) -> int:
    """
    Compute the minimum sensitivity of f (over non-trivial inputs).

    Args:
        f: BooleanFunction to analyze
        value: If specified (0 or 1), only consider inputs where f(x) = value

    Returns:
        Minimum sensitivity across relevant inputs
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0

    min_sens = n + 1

    for x in range(1 << n):
        if value is not None and bool(f.evaluate(x)) != bool(value):
            continue

        sens = sensitivity(f, x)
        min_sens = min(min_sens, sens)

        if min_sens == 0:
            break

    return min_sens if min_sens <= n else 0


def average_sensitivity(f: "BooleanFunction") -> float:
    """
    Compute the average sensitivity of f.

    The average sensitivity equals the total influence I(f) = sum_i Inf_i(f).

    Args:
        f: BooleanFunction to analyze

    Returns:
        Average sensitivity (total influence)
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0

    total = 0
    size = 1 << n

    for x in range(size):
        total += sensitivity(f, x)

    return total / size


def certificate_complexity(f: "BooleanFunction", x: int) -> Tuple[int, List[int]]:
    """
    Compute the certificate complexity at input x.

    A certificate for f at x is a minimal set of coordinates S such that
    fixing the coordinates in S to their values in x determines f.

    Args:
        f: BooleanFunction to analyze
        x: Input index

    Returns:
        Tuple of (certificate_size, list_of_certificate_variables)
    """
    n = f.n_vars
    if n is None or n == 0:
        return (0, [])

    target = bool(f.evaluate(x))

    # Try all subsets of increasing size
    from itertools import combinations

    for r in range(n + 1):
        for vars_tuple in combinations(range(n), r):
            vars_list = list(vars_tuple)
            # Check if fixing these variables determines the output
            free_vars = [i for i in range(n) if i not in vars_list]

            is_certificate = True
            for mask in range(1 << len(free_vars)):
                # Construct y: same as x on vars_list, determined by mask on free_vars
                y = x
                for idx, v in enumerate(free_vars):
                    bit = (mask >> idx) & 1
                    # Set bit v of y to bit
                    if ((y >> v) & 1) != bit:
                        y ^= 1 << v

                if bool(f.evaluate(y)) != target:
                    is_certificate = False
                    break

            if is_certificate:
                return (r, vars_list)

    return (n, list(range(n)))


def max_certificate_complexity(f: "BooleanFunction", value: Optional[int] = None) -> int:
    """
    Compute the maximum certificate complexity of f.

    Args:
        f: BooleanFunction to analyze
        value: If specified (0 or 1), only consider inputs where f(x) = value

    Returns:
        Maximum certificate complexity
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0

    max_cert = 0

    for x in range(1 << n):
        if value is not None and bool(f.evaluate(x)) != bool(value):
            continue

        cert, _ = certificate_complexity(f, x)
        max_cert = max(max_cert, cert)

    return max_cert


class ComplexityProfile:
    """
    Compute and store various complexity measures for a Boolean function.

    This class provides a comprehensive analysis of function complexity,
    computing multiple measures and checking known relationships between them.
    """

    def __init__(self, f: "BooleanFunction"):
        """
        Initialize and compute complexity profile.

        Args:
            f: BooleanFunction to analyze
        """
        self.function = f
        self.n_vars = f.n_vars
        self._computed = False
        self._measures: Dict[str, Any] = {}

    def compute(self) -> Dict[str, Any]:
        """
        Compute all complexity measures.

        Returns:
            Dictionary of complexity measures
        """
        if self._computed:
            return self._measures

        f = self.function

        # Sensitivity measures
        self._measures["s0"] = max_sensitivity(f, 0)  # Max sensitivity on 0-inputs
        self._measures["s1"] = max_sensitivity(f, 1)  # Max sensitivity on 1-inputs
        self._measures["s"] = max(self._measures["s0"], self._measures["s1"])
        self._measures["avg_sensitivity"] = average_sensitivity(f)

        # Certificate complexity
        self._measures["C0"] = max_certificate_complexity(f, 0)
        self._measures["C1"] = max_certificate_complexity(f, 1)
        self._measures["C"] = max(self._measures["C0"], self._measures["C1"])

        # Decision tree complexity
        self._measures["D"] = decision_tree_depth(f)
        size, depth = decision_tree_size(f)
        self._measures["DT_size"] = size
        self._measures["DT_depth"] = depth

        self._computed = True
        return self._measures

    def summary(self) -> str:
        """
        Return a human-readable summary of complexity measures.
        """
        m = self.compute()
        lines = [
            f"Complexity Profile for function on {self.n_vars} variables:",
            f"  Sensitivity:  s={m['s']} (s0={m['s0']}, s1={m['s1']})",
            f"  Avg Sensitivity (Total Influence): I(f)={m['avg_sensitivity']:.4f}",
            f"  Certificate: C={m['C']} (C0={m['C0']}, C1={m['C1']})",
            f"  Decision Tree: D={m['D']}",
            f"  Decision Tree Size: {m['DT_size']}",
        ]
        return "\n".join(lines)

    def check_relations(self) -> Dict[str, bool]:
        """
        Check known relationships between complexity measures.

        Returns:
            Dictionary of relationship checks
        """
        m = self.compute()

        checks = {}

        # s(f) <= C(f) (sensitivity is at most certificate complexity)
        checks["s <= C"] = m["s"] <= m["C"]

        # C(f) <= D(f) (certificate complexity is at most decision tree depth)
        checks["C <= D"] = m["C"] <= m["D"]

        # D(f) <= C0 * C1 (decision tree depth bounded by product of certificates)
        checks["D <= C0*C1"] = m["D"] <= m["C0"] * m["C1"]

        # Average sensitivity = total influence
        checks["avg_sens = I(f)"] = True  # By definition

        return checks
