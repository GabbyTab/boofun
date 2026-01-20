"""
Linear Threshold Function (LTF) representation for Boolean functions.

A Linear Threshold Function is defined by:
f(x) = sign(w₁x₁ + w₂x₂ + ... + wₙxₙ - θ)

where w = (w₁, w₂, ..., wₙ) are the weights and θ is the threshold.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


@dataclass
class LTFParameters:
    """
    Parameters for a Linear Threshold Function.

    Attributes:
        weights: Array of weights for each variable
        threshold: Threshold value
        n_vars: Number of variables
    """

    weights: np.ndarray
    threshold: float
    n_vars: int

    def __post_init__(self):
        """Validate parameters after initialization."""
        if len(self.weights) != self.n_vars:
            raise ValueError(
                f"Weights length {len(self.weights)} doesn't match n_vars {self.n_vars}"
            )

    def evaluate(self, x: Union[List[int], np.ndarray]) -> bool:
        """
        Evaluate LTF at given input.

        Args:
            x: Binary input vector

        Returns:
            Boolean output
        """
        if len(x) != self.n_vars:
            raise ValueError(f"Input length {len(x)} doesn't match n_vars {self.n_vars}")

        weighted_sum = np.dot(self.weights, x)
        return weighted_sum >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Export LTF parameters to dictionary."""
        return {
            "weights": self.weights.tolist(),
            "threshold": float(self.threshold),
            "n_vars": self.n_vars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LTFParameters":
        """Create LTF parameters from dictionary."""
        return cls(
            weights=np.array(data["weights"]), threshold=data["threshold"], n_vars=data["n_vars"]
        )


@register_strategy("ltf")
class LTFRepresentation(BooleanFunctionRepresentation[LTFParameters]):
    """Linear Threshold Function representation for Boolean functions."""

    def evaluate(
        self, inputs: np.ndarray, data: LTFParameters, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate LTF representation.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: LTF parameters
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
                binary_input = inputs.astype(int)
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
                binary_input = row.astype(int)
                results.append(data.evaluate(binary_input))
            return np.array(results, dtype=bool)
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _index_to_binary(self, index: int, n_vars: int) -> List[int]:
        """Convert integer index to binary vector."""
        return [(index >> i) & 1 for i in range(n_vars - 1, -1, -1)]

    def dump(self, data: LTFParameters, **kwargs) -> Dict[str, Any]:
        """Export LTF representation."""
        result = data.to_dict()
        result["type"] = "ltf"
        return result

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> LTFParameters:
        """
        Convert from another representation to LTF.

        Uses linear programming to find weights and threshold.
        Note: Not all Boolean functions can be represented as LTFs.
        """
        # Get truth table first
        size = 1 << n_vars
        truth_table = []

        for i in range(size):
            val = source_repr.evaluate(i, source_data, space, n_vars)
            truth_table.append(bool(val))

        # Try to find LTF parameters
        return self._find_ltf_parameters(truth_table, n_vars)

    def _find_ltf_parameters(self, truth_table: List[bool], n_vars: int) -> LTFParameters:
        """
        Find LTF parameters using linear programming approach.

        Args:
            truth_table: Boolean truth table
            n_vars: Number of variables

        Returns:
            LTF parameters

        Raises:
            ValueError: If function is not linearly separable
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            # Fallback to simple heuristic method
            return self._find_ltf_heuristic(truth_table, n_vars)

        # Set up linear programming problem
        # Variables: [w₁, w₂, ..., wₙ, θ, slack₁, ..., slack_m]
        # where m = 2^n is the number of constraints

        size = len(truth_table)
        num_vars = n_vars + 1 + size  # weights + threshold + slack variables

        # Objective: minimize sum of slack variables (try to satisfy all constraints)
        c = np.zeros(num_vars)
        c[n_vars + 1 :] = 1  # Only slack variables contribute to objective

        # Constraints: For each input x_i:
        # If f(x_i) = 1: w·x_i - θ ≥ 1 - slack_i  =>  -w·x_i + θ + slack_i ≤ -1
        # If f(x_i) = 0: w·x_i - θ ≤ -1 + slack_i  =>   w·x_i - θ - slack_i ≤ -1

        A_ub = []
        b_ub = []

        for i in range(size):
            # Convert index to binary
            x_i = [(i >> j) & 1 for j in range(n_vars - 1, -1, -1)]

            constraint = np.zeros(num_vars)

            if truth_table[i]:  # f(x_i) = 1
                # -w·x_i + θ + slack_i ≤ -1
                constraint[:n_vars] = -np.array(x_i)  # -w
                constraint[n_vars] = 1  # θ
                constraint[n_vars + 1 + i] = 1  # slack_i
                b_ub.append(-1)
            else:  # f(x_i) = 0
                # w·x_i - θ - slack_i ≤ -1
                constraint[:n_vars] = np.array(x_i)  # w
                constraint[n_vars] = -1  # -θ
                constraint[n_vars + 1 + i] = -1  # -slack_i
                b_ub.append(-1)

            A_ub.append(constraint)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Bounds: slack variables ≥ 0, weights and threshold unbounded
        bounds = [(None, None)] * (n_vars + 1) + [(0, None)] * size

        # Solve linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not result.success or result.fun > 1e-6:
            # Function is not linearly separable
            raise ValueError("Function is not representable as an LTF (not linearly separable)")

        weights = result.x[:n_vars]
        threshold = result.x[n_vars]

        return LTFParameters(weights=weights, threshold=threshold, n_vars=n_vars)

    def _find_ltf_heuristic(self, truth_table: List[bool], n_vars: int) -> LTFParameters:
        """
        Fallback heuristic method when scipy is not available.

        Uses simple perceptron-like learning.
        """
        # Initialize weights and threshold randomly
        weights = np.random.randn(n_vars)
        threshold = 0.0
        learning_rate = 0.1
        max_iterations = 1000

        size = len(truth_table)

        for iteration in range(max_iterations):
            errors = 0

            for i in range(size):
                # Convert index to binary
                x_i = np.array([(i >> j) & 1 for j in range(n_vars - 1, -1, -1)])

                # Compute current output
                output = np.dot(weights, x_i) >= threshold
                target = truth_table[i]

                if output != target:
                    errors += 1
                    # Update weights and threshold
                    if target:  # Should be 1 but got 0
                        weights += learning_rate * x_i
                        threshold -= learning_rate
                    else:  # Should be 0 but got 1
                        weights -= learning_rate * x_i
                        threshold += learning_rate

            if errors == 0:
                break

        if errors > 0:
            raise ValueError("Function is not representable as an LTF (heuristic failed)")

        return LTFParameters(weights=weights, threshold=threshold, n_vars=n_vars)

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert LTF to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> LTFParameters:
        """Create empty LTF (constant False)."""
        # Constant False: all weights = 0, threshold = 1
        weights = np.zeros(n_vars)
        threshold = 1.0
        return LTFParameters(weights=weights, threshold=threshold, n_vars=n_vars)

    def is_complete(self, data: LTFParameters) -> bool:
        """Check if LTF is complete (has valid parameters)."""
        return (
            data.weights is not None
            and data.threshold is not None
            and len(data.weights) == data.n_vars
        )

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for LTF operations."""
        return {
            "evaluation": 0,  # O(n) - linear in number of variables
            "construction": n_vars,  # O(2^n) - via truth table and LP
            "conversion_from": n_vars,  # O(2^n) - via truth table
            "space_complexity": 0,  # O(n) - stores weights and threshold
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return storage requirements for LTF representation."""
        return {
            "weights": n_vars,
            "threshold": 1,
            "total_parameters": n_vars + 1,
            "bytes": (n_vars + 1) * 8,  # Assuming 64-bit floats
            "space_complexity": "O(n)",
        }


# Utility functions for LTF analysis
def is_ltf(truth_table: List[bool], n_vars: int) -> bool:
    """
    Check if a Boolean function can be represented as an LTF.

    Args:
        truth_table: Boolean truth table
        n_vars: Number of variables

    Returns:
        True if function is linearly separable
    """
    try:
        repr_obj = LTFRepresentation()
        repr_obj._find_ltf_parameters(truth_table, n_vars)
        return True
    except ValueError:
        return False


def create_majority_ltf(n_vars: int) -> LTFParameters:
    """
    Create LTF for majority function.

    Args:
        n_vars: Number of variables (must be odd)

    Returns:
        LTF parameters for majority function
    """
    if n_vars % 2 == 0:
        raise ValueError("Majority function requires odd number of variables")

    weights = np.ones(n_vars)
    threshold = (n_vars + 1) / 2

    return LTFParameters(weights=weights, threshold=threshold, n_vars=n_vars)


def create_threshold_ltf(n_vars: int, k: int) -> LTFParameters:
    """
    Create LTF for k-threshold function.

    Args:
        n_vars: Number of variables
        k: Threshold (output 1 if at least k variables are 1)

    Returns:
        LTF parameters for threshold function
    """
    if k <= 0 or k > n_vars:
        raise ValueError(f"Threshold k={k} must be in range [1, {n_vars}]")

    weights = np.ones(n_vars)
    threshold = k

    return LTFParameters(weights=weights, threshold=threshold, n_vars=n_vars)


# Export main classes and functions
__all__ = [
    "LTFParameters",
    "LTFRepresentation",
    "is_ltf",
    "create_majority_ltf",
    "create_threshold_ltf",
]
