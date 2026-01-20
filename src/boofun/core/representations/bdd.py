"""
Binary Decision Diagram (BDD) representation for Boolean functions.

This module implements Reduced Ordered Binary Decision Diagrams (ROBDDs)
for efficient representation and manipulation of Boolean functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


class BDDNode:
    """
    Node in a Binary Decision Diagram.

    Attributes:
        var: Variable index (None for terminal nodes)
        low: Low branch (False branch)
        high: High branch (True branch)
        is_terminal: Whether this is a terminal node
        value: Terminal value (only for terminal nodes)
    """

    def __init__(
        self,
        var: Optional[int] = None,
        low: Optional["BDDNode"] = None,
        high: Optional["BDDNode"] = None,
        value: Optional[bool] = None,
    ):
        self.var = var
        self.low = low
        self.high = high
        self.is_terminal = var is None
        self.value = value

        # For terminal nodes, ensure value is set
        if self.is_terminal and value is None:
            self.value = False

    def __eq__(self, other):
        if not isinstance(other, BDDNode):
            return False
        if self.is_terminal != other.is_terminal:
            return False
        if self.is_terminal:
            return self.value == other.value
        return self.var == other.var and self.low == other.low and self.high == other.high

    def __hash__(self):
        if self.is_terminal:
            return hash((None, self.value))
        return hash((self.var, id(self.low), id(self.high)))


class BDD:
    """
    Reduced Ordered Binary Decision Diagram.

    Implements a canonical representation of Boolean functions
    with efficient operations for evaluation, manipulation, and analysis.
    """

    def __init__(self, n_vars: int):
        """
        Initialize BDD.

        Args:
            n_vars: Number of variables
        """
        self.n_vars = n_vars
        self.root: Optional[BDDNode] = None
        self.node_cache: Dict[Tuple[int, "BDDNode", "BDDNode"], BDDNode] = {}
        self.terminal_true = BDDNode(value=True)
        self.terminal_false = BDDNode(value=False)

    def create_terminal(self, value: bool) -> BDDNode:
        """Create terminal node."""
        return self.terminal_true if value else self.terminal_false

    def create_node(self, var: int, low: BDDNode, high: BDDNode) -> BDDNode:
        """
        Create or retrieve BDD node (with reduction).

        Args:
            var: Variable index
            low: Low branch
            high: High branch

        Returns:
            BDD node
        """
        # Reduction rule 1: If low == high, return low
        if low == high:
            return low

        # Check cache for existing node
        key = (var, low, high)
        if key in self.node_cache:
            return self.node_cache[key]

        # Create new node
        node = BDDNode(var, low, high)
        self.node_cache[key] = node
        return node

    def evaluate(self, inputs: List[bool]) -> bool:
        """
        Evaluate BDD with given inputs.

        Args:
            inputs: Boolean input values

        Returns:
            Boolean result
        """
        if len(inputs) != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} inputs, got {len(inputs)}")

        if self.root is None:
            return False

        current = self.root
        while not current.is_terminal:
            if inputs[current.var]:
                current = current.high
            else:
                current = current.low

        return current.value

    def get_node_count(self) -> int:
        """Get total number of nodes in BDD."""
        if self.root is None:
            return 0

        visited = set()
        stack = [self.root]

        while stack:
            node = stack.pop()
            if node in visited or node.is_terminal:
                continue

            visited.add(node)
            stack.append(node.low)
            stack.append(node.high)

        return len(visited) + 2  # +2 for terminal nodes


@register_strategy("bdd")
class BDDRepresentation(BooleanFunctionRepresentation[BDD]):
    """BDD representation for Boolean functions."""

    def evaluate(
        self, inputs: np.ndarray, data: BDD, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate BDD representation.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: BDD
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        if inputs.ndim == 0:
            # Single integer index
            binary_input = self._index_to_binary(int(inputs), n_vars)
            return data.evaluate(binary_input)
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                binary_input = inputs.astype(bool)
                return data.evaluate(binary_input)
            else:
                # Array of integer indices
                results = []
                for idx in inputs:
                    binary_input = self._index_to_binary(int(idx), n_vars)
                    results.append(data.evaluate(binary_input))
                return np.array(results, dtype=bool)
        elif inputs.ndim == 2:
            # Batch of binary vectors
            results = []
            for row in inputs:
                binary_input = row.astype(bool)
                results.append(data.evaluate(binary_input))
            return np.array(results, dtype=bool)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _index_to_binary(self, index: int, n_vars: int) -> List[bool]:
        """Convert integer index to binary vector."""
        return [(index >> i) & 1 == 1 for i in range(n_vars - 1, -1, -1)]

    def dump(self, data: BDD, **kwargs) -> Dict[str, Any]:
        """Export BDD representation."""
        return {"type": "bdd", "n_vars": data.n_vars, "node_count": data.get_node_count()}

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> BDD:
        """
        Convert from another representation to BDD.

        Uses truth table to build BDD.
        """
        # Get truth table
        size = 1 << n_vars
        truth_table = []

        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)
            truth_table.append(bool(val))

        # Build BDD from truth table
        return self._build_bdd_from_truth_table(truth_table, n_vars)

    def _build_bdd_from_truth_table(self, truth_table: List[bool], n_vars: int) -> BDD:
        """
        Build BDD from truth table using Shannon expansion.

        Args:
            truth_table: Boolean truth table
            n_vars: Number of variables

        Returns:
            BDD implementing the function
        """
        bdd = BDD(n_vars)

        # Use Shannon expansion recursively
        bdd.root = self._shannon_expansion(truth_table, 0, n_vars, bdd)

        return bdd

    def _shannon_expansion(
        self, truth_table: List[bool], var: int, n_vars: int, bdd: BDD
    ) -> BDDNode:
        """
        Apply Shannon expansion for variable var.

        Args:
            truth_table: Current truth table
            var: Current variable index
            n_vars: Total number of variables
            bdd: BDD instance

        Returns:
            BDD node for this expansion
        """
        if var >= n_vars:
            # Terminal case
            return bdd.create_terminal(truth_table[0])

        # Split truth table for current variable
        half_size = len(truth_table) // 2
        low_table = truth_table[:half_size]
        high_table = truth_table[half_size:]

        # Recursive expansion
        low_node = self._shannon_expansion(low_table, var + 1, n_vars, bdd)
        high_node = self._shannon_expansion(high_table, var + 1, n_vars, bdd)

        # Create node for current variable
        return bdd.create_node(var, low_node, high_node)

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert BDD to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> BDD:
        """Create empty BDD (constant False)."""
        bdd = BDD(n_vars)
        bdd.root = bdd.create_terminal(False)
        return bdd

    def is_complete(self, data: BDD) -> bool:
        """Check if BDD is complete (has root node)."""
        return data.root is not None

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for BDD operations."""
        return {
            "evaluation": 0,  # O(depth) - typically O(n)
            "construction": n_vars,  # O(2^n) - via truth table
            "conversion_from": n_vars,  # O(2^n) - via truth table
            "space_complexity": 0,  # O(nodes) - often much smaller than 2^n
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return storage requirements for BDD representation."""
        return {
            "max_nodes": 2**n_vars,  # Worst case
            "bytes_per_node": 24,  # Rough estimate for BDDNode
            "max_bytes": 2**n_vars * 24,
            "space_complexity": "O(2^n) worst case, often O(n) or O(n²)",
        }


# Export main classes
__all__ = ["BDDNode", "BDD", "BDDRepresentation"]
