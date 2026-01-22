"""
Decision tree analysis for Boolean functions.

This module implements algorithms for analyzing decision tree complexity,
including dynamic programming approaches for computing optimal trees and
game-theoretic analysis for randomized complexity.

The algorithms in this module are based on Avishay Tal's PhD-era library,
with modernized implementations, type hints, and documentation.

Key concepts:
- D(f): Deterministic decision tree depth (worst-case)
- D_avg(f): Average-case decision tree depth under uniform distribution
- D_μ(f): Average-case under arbitrary distribution μ
- R(f): Randomized decision tree complexity (via minimax theorem)

The core insight is that decision trees correspond to subcubes of {0,1}^n,
and we can use dynamic programming over the "cube lattice" where each
node represents a partial assignment (some variables fixed, others free).

References:
- Tal's PhD library (BooleanFunc.py)
- Buhrman & de Wolf, "Complexity Measures and Decision Tree Complexity" (2002)
- O'Donnell, "Analysis of Boolean Functions" (2014), Chapter 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    # Core algorithms
    "decision_tree_depth_dp",
    "decision_tree_depth_uniform_dp",
    "decision_tree_depth_weighted_dp",
    "decision_tree_size_dp",
    # Tree enumeration
    "enumerate_decision_trees",
    "count_decision_trees",
    # Randomized complexity
    "randomized_complexity_matrix",
    "compute_randomized_complexity",
    # Tree representation
    "DecisionTree",
    "reconstruct_tree",
    # Utilities
    "tree_depth",
    "tree_size",
]


@dataclass
class DecisionTree:
    """
    Representation of a decision tree for a Boolean function.

    A decision tree is either:
    - A leaf with a constant output (0 or 1)
    - An internal node that queries variable `var` and branches to
      `left` (var=0) or `right` (var=1) subtrees

    Attributes:
        var: Variable index to query (-1 for leaf nodes)
        left: Left subtree (when var=0), or None for leaves
        right: Right subtree (when var=1), or None for leaves
        value: Output value for leaf nodes (0 or 1), None for internal nodes
    """

    var: int = -1
    left: Optional["DecisionTree"] = None
    right: Optional["DecisionTree"] = None
    value: Optional[int] = None

    def is_leaf(self) -> bool:
        """Return True if this is a leaf node."""
        return self.var == -1

    def depth(self) -> int:
        """Compute the depth (longest root-to-leaf path) of this tree."""
        if self.is_leaf():
            return 0
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def size(self) -> int:
        """Compute the size (number of leaves) of this tree."""
        if self.is_leaf():
            return 1
        left_size = self.left.size() if self.left else 0
        right_size = self.right.size() if self.right else 0
        return left_size + right_size

    def evaluate(self, x: int, n_vars: int) -> int:
        """
        Evaluate the decision tree on input x.

        Args:
            x: Input as integer (bit i = value of variable i)
            n_vars: Number of variables

        Returns:
            Output value (0 or 1)
        """
        if self.is_leaf():
            return self.value if self.value is not None else 0

        bit = (x >> self.var) & 1
        if bit == 0:
            return self.left.evaluate(x, n_vars) if self.left else 0
        else:
            return self.right.evaluate(x, n_vars) if self.right else 0

    def query_depth(self, x: int, n_vars: int) -> int:
        """
        Return the number of queries made to evaluate x.

        Args:
            x: Input as integer
            n_vars: Number of variables

        Returns:
            Number of queries (depth of path to leaf)
        """
        if self.is_leaf():
            return 0

        bit = (x >> self.var) & 1
        if bit == 0:
            return 1 + (self.left.query_depth(x, n_vars) if self.left else 0)
        else:
            return 1 + (self.right.query_depth(x, n_vars) if self.right else 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        if self.is_leaf():
            return {"type": "leaf", "value": self.value}
        return {
            "type": "internal",
            "var": self.var,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    def __repr__(self) -> str:
        if self.is_leaf():
            return f"Leaf({self.value})"
        return f"Node(x{self.var}, {self.left}, {self.right})"


def tree_depth(tree: Union[DecisionTree, List, Tuple]) -> int:
    """
    Compute the depth of a decision tree.

    Handles both DecisionTree objects and list/tuple representations
    from Tal's original code: [var, left_subtree, right_subtree] or [].

    Args:
        tree: Decision tree in any supported format

    Returns:
        Tree depth (longest root-to-leaf path)
    """
    if isinstance(tree, DecisionTree):
        return tree.depth()

    # Handle list/tuple format from Tal's code
    if not tree or tree == []:
        return 0
    if isinstance(tree, (list, tuple)) and len(tree) == 3:
        var, left, right = tree
        return 1 + max(tree_depth(left), tree_depth(right))
    return 0


def tree_size(tree: Union[DecisionTree, List, Tuple]) -> int:
    """
    Compute the size (number of leaves) of a decision tree.

    Args:
        tree: Decision tree in any supported format

    Returns:
        Number of leaves
    """
    if isinstance(tree, DecisionTree):
        return tree.size()

    if not tree or tree == []:
        return 1
    if isinstance(tree, (list, tuple)) and len(tree) == 3:
        var, left, right = tree
        return tree_size(left) + tree_size(right)
    return 1


def decision_tree_depth_dp(f: "BooleanFunction") -> int:
    """
    Compute optimal decision tree depth using dynamic programming.

    This implements Tal's `calc_decision_tree_DP` algorithm which uses
    a clever representation of subcubes. Each subcube is encoded using
    a ternary representation where each variable can be:
    - 0: fixed to 0
    - 1: fixed to 1
    - 2: free (not yet queried)

    The DP computes the optimal depth for each subcube bottom-up,
    from fully-specified inputs to the full hypercube.

    Time complexity: O(3^n) where n is number of variables.
    Space complexity: O(3^n)

    Args:
        f: BooleanFunction to analyze

    Returns:
        Optimal decision tree depth D(f)
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)

    # Phase 1: Build the cube averages
    # old_layer[x] = fraction of 1s in subcube corresponding to x
    old_layer = truth_table.astype(float).tolist()

    for i in range(n):
        new_layer = [0.0] * (3 ** (i + 1) * 2 ** (n - (i + 1)))
        mask = (1 << (n - (i + 1))) - 1

        for index in range(len(old_layer)):
            first = index >> (n - i)
            last = index & mask
            ind = (index >> (n - (i + 1))) & 1

            # Index where variable i has value ind
            new_ind1 = ((first * 3 + ind) << (n - (i + 1))) + last
            # Index where variable i is free (2)
            new_ind2 = ((first * 3 + 2) << (n - (i + 1))) + last

            new_layer[new_ind1] = old_layer[index]
            new_layer[new_ind2] += old_layer[index] / 2.0

        old_layer = new_layer

    # Phase 2: Compute optimal depths bottom-up
    INF = 1 << 20
    results = [INF] * len(new_layer)

    for j in range(len(new_layer)):
        # If subcube is constant (all 0s or all 1s), depth is 0
        if new_layer[j] < 1e-7 or new_layer[j] > 1 - 1e-7:
            results[j] = 0
        else:
            # Try each free variable as the next query
            trits = [(j // (3**i)) % 3 for i in range(n)]

            for i in range(n):
                if trits[i] == 2:  # Variable i is free
                    num = j
                    # Subcube where variable i = 0
                    num -= 2 * (3**i)
                    dt0 = results[num]
                    # Subcube where variable i = 1
                    num += 3**i
                    dt1 = results[num]

                    # Worst-case depth after querying variable i
                    candidate = max(dt0, dt1) + 1
                    if candidate < results[j]:
                        results[j] = candidate

    return results[-1]


def decision_tree_depth_uniform_dp(
    f: "BooleanFunction", weights: Optional[List[float]] = None
) -> Tuple[float, Optional[DecisionTree]]:
    """
    Compute optimal average-case decision tree depth under uniform distribution.

    This implements Tal's `calc_decision_tree_DP_uniform` algorithm with
    NumPy optimizations for better performance on larger inputs.

    Unlike worst-case depth, this minimizes the expected number of queries
    when inputs are drawn uniformly at random.

    Args:
        f: BooleanFunction to analyze
        weights: Optional per-variable query costs (default: all 1s)

    Returns:
        Tuple of (average depth, optimal tree if reconstructible)
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0, DecisionTree(value=int(f.evaluate(0)))

    if weights is None:
        weights = [1.0] * n

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=np.int8)

    # Use compact representation: 1 = output 0, 2 = output 1, 3 = mixed
    old_layer = np.zeros(3**n, dtype=np.int8)
    for i in range(len(truth_table)):
        old_layer[i] = 1 if truth_table[i] == 0 else 2

    # Build cube indicators
    for i in range(n):
        new_layer = np.zeros(3**n, dtype=np.int8)
        mask = (1 << (n - (i + 1))) - 1

        for first in range(3**i):
            for index in range(2 ** (n - i)):
                last = index & mask
                ind = (index >> (n - (i + 1))) & 1
                old_idx = (first << (n - i)) + index

                new_ind1 = ((first * 3 + ind) << (n - (i + 1))) + last
                new_ind2 = ((first * 3 + 2) << (n - (i + 1))) + last

                new_layer[new_ind1] = old_layer[old_idx]
                new_layer[new_ind2] |= old_layer[old_idx]

        old_layer = new_layer

    # Compute optimal average depths
    INF = 1 << 30
    results = np.zeros(len(new_layer), dtype=np.int32)
    results.fill(INF)
    back_ptr = np.zeros(len(new_layer), dtype=np.int8) - 1

    # Pre-compute trit decompositions for efficiency
    for j in range(len(new_layer)):
        # If subcube is constant, cost is 0
        if new_layer[j] != 3:
            results[j] = 0
        else:
            trits = [(j // (3**i)) % 3 for i in range(n)]
            cnt2 = trits.count(2)  # Number of free variables

            for i in range(n):
                if trits[i] == 2:
                    dt0 = results[j - 2 * (3**i)]
                    dt1 = results[j - (3**i)]
                    # Cost is sum of subtree costs plus query cost scaled by subcube size
                    candidate = (dt0 + dt1) + int(weights[i] * (1 << cnt2))
                    if candidate < results[j]:
                        results[j] = candidate
                        back_ptr[j] = i

    # Average depth = total cost / 2^n
    avg_depth = results[-1] / (1 << n)

    # Reconstruct tree
    tree = reconstruct_tree(back_ptr.tolist(), len(back_ptr) - 1, n, truth_table.tolist())

    return avg_depth, tree


def decision_tree_depth_weighted_dp(
    f: "BooleanFunction", probabilities: List[float], weights: Optional[List[float]] = None
) -> Tuple[float, Optional[DecisionTree]]:
    """
    Compute optimal decision tree depth under arbitrary input distribution.

    This implements Tal's `calc_decision_tree_DP_with_prob` algorithm which
    handles non-uniform input distributions. This is useful for analyzing
    decision tree performance when some inputs are more likely than others.

    Args:
        f: BooleanFunction to analyze
        probabilities: Probability of each input (length 2^n)
        weights: Optional per-variable query costs

    Returns:
        Tuple of (expected depth, optimal tree)
    """
    n = f.n_vars
    if n is None or n == 0:
        return 0.0, DecisionTree(value=int(f.evaluate(0)))

    if weights is None:
        weights = [1.0] * n

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)

    # Phase 1: Build value averages for each subcube
    old_layer = truth_table.tolist()

    for i in range(n):
        new_layer = [0.0] * (3 ** (i + 1) * 2 ** (n - (i + 1)))
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

    value_layer = new_layer[:]

    # Phase 2: Build probability sums for each subcube
    old_layer = list(probabilities)

    for i in range(n):
        new_layer = [0.0] * (3 ** (i + 1) * 2 ** (n - (i + 1)))
        mask = (1 << (n - (i + 1))) - 1

        for index in range(len(old_layer)):
            first = index >> (n - i)
            last = index & mask
            ind = (index >> (n - (i + 1))) & 1

            new_ind1 = ((first * 3 + ind) << (n - (i + 1))) + last
            new_ind2 = ((first * 3 + 2) << (n - (i + 1))) + last

            new_layer[new_ind1] = old_layer[index]
            new_layer[new_ind2] += old_layer[index]

        old_layer = new_layer

    prob_layer = new_layer

    # Phase 3: Compute optimal expected depths
    INF = 1 << 20
    results = [float(INF)] * len(value_layer)
    back_ptr = [-1] * len(value_layer)

    for j in range(len(value_layer)):
        # Constant subcube: expected cost is 0
        if value_layer[j] < 1e-7 or value_layer[j] > 1 - 1e-7:
            results[j] = 0.0
        else:
            trits = [(j // (3**i)) % 3 for i in range(n)]

            for i in range(n):
                if trits[i] == 2:  # Variable i is free
                    p = prob_layer[j]
                    if p == 0.0:
                        results[j] = 0.0
                        continue

                    num = j
                    num -= 2 * (3**i)
                    dt0 = results[num] * prob_layer[num] / p if prob_layer[num] > 0 else 0
                    num += 3**i
                    dt1 = results[num] * prob_layer[num] / p if prob_layer[num] > 0 else 0

                    candidate = dt0 + dt1 + weights[i]
                    if candidate < results[j]:
                        results[j] = candidate
                        back_ptr[j] = i

    tree = reconstruct_tree(back_ptr, len(back_ptr) - 1, n, truth_table.astype(int).tolist())

    return results[-1], tree


def decision_tree_size_dp(
    f: "BooleanFunction", optimize_size_first: bool = True
) -> Tuple[int, int, Optional[DecisionTree]]:
    """
    Compute decision tree optimizing for size (number of leaves).

    This implements Tal's `calc_decision_tree_size_DP` which finds trees
    that minimize the number of leaves, with depth as a tiebreaker.

    Smaller trees are often preferred for interpretability and can
    sometimes be converted to smaller Boolean formulas.

    Args:
        f: BooleanFunction to analyze
        optimize_size_first: If True, minimize size then depth.
                            If False, minimize depth then size.

    Returns:
        Tuple of (size, depth, optimal tree)
    """
    n = f.n_vars
    if n is None or n == 0:
        val = int(f.evaluate(0))
        return 1, 0, DecisionTree(value=val)

    # Get truth table
    truth_table = np.asarray(f.get_representation("truth_table"), dtype=float)

    # Build cube averages
    old_layer = truth_table.tolist()

    for i in range(n):
        new_layer = [0.0] * (3 ** (i + 1) * 2 ** (n - (i + 1)))
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

    # Results: (size, depth, tree)
    INF = 1 << 20
    results: List[Tuple[int, int, Any]] = [(INF, INF, None)] * len(new_layer)

    for j in range(len(new_layer)):
        val = new_layer[j]

        if val < 1e-7:  # All 0s
            results[j] = (1, 0, DecisionTree(value=0))
        elif val > 1 - 1e-7:  # All 1s
            results[j] = (1, 0, DecisionTree(value=1))
        else:
            trits = [(j // (3**i)) % 3 for i in range(n)]

            for i in range(n):
                if trits[i] == 2:
                    num = j - 2 * (3**i)
                    s0, d0, dt0 = results[num]
                    num += 3**i
                    s1, d1, dt1 = results[num]

                    sz = s0 + s1
                    d = max(d0, d1) + 1
                    tree = DecisionTree(var=i, left=dt0, right=dt1)

                    current_size, current_depth, _ = results[j]

                    if optimize_size_first:
                        if (sz, d) < (current_size, current_depth):
                            results[j] = (sz, d, tree)
                    else:
                        if (d, sz) < (current_depth, current_size):
                            results[j] = (sz, d, tree)

    size, depth, tree = results[-1]
    return size, depth, tree


def reconstruct_tree(
    back_ptr: List[int], index: int, n_vars: int, truth_table: List[int]
) -> Optional[DecisionTree]:
    """
    Reconstruct a decision tree from back pointers.

    Args:
        back_ptr: Array of back pointers from DP
        index: Current index in the cube representation
        n_vars: Number of variables
        truth_table: Original truth table

    Returns:
        Reconstructed DecisionTree
    """
    if index < 0 or back_ptr[index] < 0:
        # This is a leaf - determine value from truth table
        # Decode index to find which inputs are in this subcube
        trits = [(index // (3**i)) % 3 for i in range(n_vars)]

        # Find any input in this subcube
        x = 0
        for i in range(n_vars):
            if trits[i] == 1:
                x |= 1 << i

        if x < len(truth_table):
            return DecisionTree(value=truth_table[x])
        return DecisionTree(value=0)

    var = back_ptr[index]

    # Find indices for subtrees
    left_index = index - 2 * (3**var)  # var = 0
    right_index = index - (3**var)  # var = 1

    left_tree = reconstruct_tree(back_ptr, left_index, n_vars, truth_table)
    right_tree = reconstruct_tree(back_ptr, right_index, n_vars, truth_table)

    return DecisionTree(var=var, left=left_tree, right=right_tree)


def enumerate_decision_trees(
    f: "BooleanFunction", prune_dominated: bool = True
) -> List[DecisionTree]:
    """
    Enumerate all valid decision trees for a Boolean function.

    This implements Tal's `all_decision_trees` algorithm with optional
    pruning of dominated trees (trees that are strictly worse than
    another tree on all inputs).

    Warning: The number of trees can be exponential in n!
    Only use for small functions (n ≤ 6 recommended).

    Args:
        f: BooleanFunction to analyze
        prune_dominated: If True, remove dominated trees

    Returns:
        List of decision trees
    """
    n = f.n_vars
    if n is None or n == 0:
        return [DecisionTree(value=int(f.evaluate(0)))]

    if n > 8:
        raise ValueError(f"enumerate_decision_trees is only practical for n ≤ 8, got n={n}")

    # Find influential variables
    truth_table = f.get_representation("truth_table")
    influential = []

    for i in range(n):
        # Check if variable i affects the function
        is_influential = False
        for x in range(1 << n):
            x_flip = x ^ (1 << i)
            if truth_table[x] != truth_table[x_flip]:
                is_influential = True
                break
        if is_influential:
            influential.append(i)

    if not influential:
        # Constant function
        return [DecisionTree(value=int(truth_table[0]))]

    def enumerate_recursive(fixed: Dict[int, int]) -> List[DecisionTree]:
        """Recursively enumerate trees for subcube defined by fixed."""
        # Check if subcube is constant
        first_val = None
        is_constant = True

        for x in range(1 << n):
            # Check if x is in this subcube
            in_subcube = True
            for var, val in fixed.items():
                if ((x >> var) & 1) != val:
                    in_subcube = False
                    break

            if in_subcube:
                if first_val is None:
                    first_val = truth_table[x]
                elif truth_table[x] != first_val:
                    is_constant = False
                    break

        if is_constant:
            return [DecisionTree(value=int(first_val) if first_val is not None else 0)]

        # Try each unfixed influential variable
        trees = []
        for var in influential:
            if var not in fixed:
                # Recurse on both subtrees
                fixed_0 = {**fixed, var: 0}
                fixed_1 = {**fixed, var: 1}

                trees_0 = enumerate_recursive(fixed_0)
                trees_1 = enumerate_recursive(fixed_1)

                for t0 in trees_0:
                    for t1 in trees_1:
                        trees.append(DecisionTree(var=var, left=t0, right=t1))

        return trees

    all_trees = enumerate_recursive({})

    if prune_dominated and len(all_trees) > 1:
        all_trees = _prune_dominated_trees(all_trees, n, influential)

    return all_trees


def _prune_dominated_trees(
    trees: List[DecisionTree], n_vars: int, influential: List[int]
) -> List[DecisionTree]:
    """Remove trees that are dominated by others on all inputs."""
    if len(trees) <= 1:
        return trees

    # Build matrix of query depths
    inputs = []
    for b in range(1 << len(influential)):
        x = 0
        for idx, var in enumerate(influential):
            if (b >> idx) & 1:
                x |= 1 << var
        inputs.append(x)

    depths = [[tree.query_depth(x, n_vars) for x in inputs] for tree in trees]

    # Keep non-dominated trees
    keep = []
    for i in range(len(trees)):
        dominated = False
        for j in range(len(trees)):
            if i == j:
                continue
            # Check if j dominates i (j is at least as good on all inputs, better on some)
            at_least_as_good = all(depths[j][k] <= depths[i][k] for k in range(len(inputs)))
            strictly_better = any(depths[j][k] < depths[i][k] for k in range(len(inputs)))
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(trees[i])

    return keep


def count_decision_trees(f: "BooleanFunction") -> int:
    """
    Count the number of distinct decision trees (without enumeration).

    This is faster than enumerate_decision_trees when you only need the count.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Number of distinct decision trees
    """
    n = f.n_vars
    if n is None or n == 0:
        return 1

    truth_table = f.get_representation("truth_table")

    # Find influential variables
    influential = []
    for i in range(n):
        for x in range(1 << n):
            if truth_table[x] != truth_table[x ^ (1 << i)]:
                influential.append(i)
                break

    k = len(influential)
    if k == 0:
        return 1

    # Use DP with memoization on subcube signatures
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def count_for_subcube(fixed_tuple: Tuple[Tuple[int, int], ...]) -> int:
        fixed = dict(fixed_tuple)

        # Check if constant
        first_val = None
        is_constant = True
        for x in range(1 << n):
            in_subcube = all(((x >> v) & 1) == val for v, val in fixed.items())
            if in_subcube:
                if first_val is None:
                    first_val = truth_table[x]
                elif truth_table[x] != first_val:
                    is_constant = False
                    break

        if is_constant:
            return 1

        # Sum over all choices of root variable
        total = 0
        for var in influential:
            if var not in fixed:
                fixed_0 = tuple(sorted(list(fixed_tuple) + [(var, 0)]))
                fixed_1 = tuple(sorted(list(fixed_tuple) + [(var, 1)]))
                total += count_for_subcube(fixed_0) * count_for_subcube(fixed_1)

        return total

    return count_for_subcube(())


def randomized_complexity_matrix(
    f: "BooleanFunction", output_value: Optional[int] = None
) -> np.ndarray:
    """
    Build the game matrix for randomized decision tree complexity.

    The rows correspond to inputs, columns to decision trees.
    Entry (i, j) is the number of queries tree j makes on input i.

    By the minimax theorem, the randomized complexity R(f) equals
    the value of this matrix game.

    Args:
        f: BooleanFunction to analyze
        output_value: If specified, only include inputs where f(x) = output_value

    Returns:
        NumPy matrix where M[i,j] = depth of tree j on input i
    """
    n = f.n_vars
    if n is None:
        raise ValueError("Function must have defined n_vars")

    trees = enumerate_decision_trees(f)
    truth_table = f.get_representation("truth_table")

    # Select inputs
    if output_value is not None:
        inputs = [x for x in range(1 << n) if truth_table[x] == output_value]
    else:
        inputs = list(range(1 << n))

    # Build matrix
    matrix = np.zeros((len(inputs), len(trees)), dtype=int)
    for i, x in enumerate(inputs):
        for j, tree in enumerate(trees):
            matrix[i, j] = tree.query_depth(x, n)

    return matrix


def compute_randomized_complexity(
    f: "BooleanFunction", output_value: Optional[int] = None
) -> float:
    """
    Compute the randomized decision tree complexity R(f).

    This solves the minimax game between:
    - The algorithm (chooses a distribution over decision trees)
    - The adversary (chooses an input)

    R(f) = min over tree distributions, max over inputs, E[queries]

    By von Neumann's minimax theorem, this equals:
    R(f) = max over input distributions, min over trees, E[queries]

    Requires scipy for linear programming.

    Args:
        f: BooleanFunction to analyze
        output_value: If specified, compute R_b(f) for inputs with f(x)=b

    Returns:
        Randomized query complexity
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError("scipy is required for compute_randomized_complexity")

    matrix = randomized_complexity_matrix(f, output_value)
    m, n_trees = matrix.shape

    if m == 0 or n_trees == 0:
        return 0.0

    # Solve LP: max v s.t. Ax >= v*1, sum(x) = 1, x >= 0
    # This finds the optimal mixed strategy for the row player (adversary)
    # Equivalently: min c'x s.t. A'x <= b, x >= 0

    # Variables: [p_1, ..., p_n, v] where p_i is prob of tree i, v is value
    # Maximize v subject to:
    #   For each input i: sum_j p_j * M[i,j] >= v
    #   sum_j p_j = 1
    #   p_j >= 0

    # Convert to minimization: minimize -v
    c = np.zeros(n_trees + 1)
    c[-1] = -1  # Minimize -v (maximize v)

    # Inequality constraints: v - sum_j p_j * M[i,j] <= 0
    A_ub = np.zeros((m, n_trees + 1))
    A_ub[:, :-1] = -matrix  # -M
    A_ub[:, -1] = 1  # v
    b_ub = np.zeros(m)

    # Equality constraint: sum_j p_j = 1
    A_eq = np.zeros((1, n_trees + 1))
    A_eq[0, :-1] = 1
    b_eq = np.array([1.0])

    # Bounds: p_j >= 0, v unbounded
    bounds = [(0, None)] * n_trees + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if result.success:
        return -result.fun  # Negate because we minimized -v
    else:
        # Fallback: return deterministic complexity
        return float(np.max(np.min(matrix, axis=1)))
