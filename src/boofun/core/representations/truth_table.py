import warnings
from typing import Any, Dict

import numpy as np

from ...utils.exceptions import EvaluationError
from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy


@register_strategy("truth_table")
class TruthTableRepresentation(BooleanFunctionRepresentation[np.ndarray]):
    """Truth table representation using NumPy arrays."""

    def evaluate(
        self, inputs: np.ndarray, data: np.ndarray, space: Space, n_vars: int
    ) -> np.ndarray:
        """
        Evaluate the Boolean function using its truth table.

        Args:
            inputs: Input values - can be:
                   - Integer indices (0 to 2^n-1)
                   - Binary vectors (shape: (n,) or (batch, n))
                   - Batch of integer indices (shape: (batch,))
            data: Truth table as boolean array of length 2^n
            space: Evaluation space (affects input interpretation)
            n_vars: Number of Boolean variables

        Returns:
            Boolean result(s) - scalar for single input, array for batch
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        # Handle different input formats
        if inputs.ndim == 0:
            # Single integer index
            index = int(inputs)
            if index < 0 or index >= len(data):
                raise IndexError(f"Index {index} out of range for truth table of size {len(data)}")
            return bool(data[index])

        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector - convert to index
                if space == Space.PLUS_MINUS_CUBE:
                    # Convert {-1,1} to {0,1}
                    inputs = ((inputs + 1) // 2).astype(int)
                index = self._binary_to_index(inputs)
                return bool(data[index])
            else:
                # Array of integer indices
                indices = inputs.astype(int)
                if np.any((indices < 0) | (indices >= len(data))):
                    raise IndexError(
                        f"Some indices out of range for truth table of size {len(data)}"
                    )
                # Convert to Python list to avoid numpy indexing issues, then back to array
                result = np.array([bool(data[idx]) for idx in indices])
                return result

        elif inputs.ndim == 2:
            # Batch of binary vectors (batch_size, n_vars)
            if inputs.shape[1] != n_vars:
                raise ValueError(f"Expected {n_vars} variables, got {inputs.shape[1]}")

            if space == Space.PLUS_MINUS_CUBE:
                # Convert {-1,1} to {0,1}
                inputs = ((inputs + 1) // 2).astype(int)

            # Convert each binary vector to index
            indices = np.array([self._binary_to_index(row) for row in inputs])
            return data[indices].astype(bool)

        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _binary_to_index(self, binary_vector: np.ndarray) -> int:
        """Convert binary vector to integer index using LSB=x₀ convention."""
        # LSB-first: binary_vector[i] corresponds to x_i, so index = Σ x_i * 2^i
        return int(np.dot(binary_vector, 2 ** np.arange(len(binary_vector))))

    def _compute_index(self, bits: np.ndarray) -> int:
        """Optimized bit packing using NumPy"""
        return int(np.packbits(bits.astype(np.uint8), bitorder="big")[0])

    def dump(self, data: np.ndarray, space=None, **kwargs) -> Dict[str, Any]:
        """
        Export the truth table.

        Returns a serializable dictionary containing:
        - 'table': list of booleans
        - 'n_vars': number of variables
        """
        return {
            "type": "truth_table",
            "n": int(np.log2(data.size)),
            "size": data.size,
            "values": data.astype(bool).tolist(),
        }

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Convert from any representation by evaluating all possible inputs.

        This is the universal converter - any representation can be converted
        to truth table by exhaustive evaluation.

        Args:
            source_repr: Source representation strategy
            source_data: Data in source format
            space: Mathematical space
            n_vars: Number of variables
            **kwargs: Additional options:
                - lenient (bool): If True, substitute False for failed evaluations
                  and emit a warning. Default is False (strict mode).

        Returns:
            Truth table as boolean array

        Raises:
            EvaluationError: If evaluation fails at any index (unless lenient=True)
        """
        size = 1 << n_vars  # 2^n
        truth_table = np.zeros(size, dtype=bool)
        lenient = kwargs.get("lenient", False)
        failed_indices = []

        # Generate all possible input indices
        for idx in range(size):
            try:
                value = source_repr.evaluate(idx, source_data, space, n_vars)

                # Handle different return types and spaces
                if isinstance(value, (bool, np.bool_)):
                    truth_table[idx] = bool(value)
                elif isinstance(value, (int, np.integer)):
                    truth_table[idx] = bool(value)
                elif isinstance(value, (float, np.floating)):
                    # For real-valued outputs, convert to boolean
                    # Fourier evaluation returns ±1 values, need to map to {0,1}
                    if abs(value - 1.0) < 1e-10:  # value ≈ 1.0 in ±1 domain
                        truth_table[idx] = False  # 1 in ±1 maps to 0 in {0,1}
                    elif abs(value - (-1.0)) < 1e-10:  # value ≈ -1.0 in ±1 domain
                        truth_table[idx] = True  # -1 in ±1 maps to 1 in {0,1}
                    elif space == Space.PLUS_MINUS_CUBE:
                        truth_table[idx] = value > 0  # ±1 space: positive -> True
                    else:
                        truth_table[idx] = value > 0.5  # [0,1] space: > 0.5 -> True
                else:
                    truth_table[idx] = bool(value)

            except Exception as e:
                if lenient:
                    # Lenient mode: substitute False and track failures
                    truth_table[idx] = False
                    failed_indices.append((idx, str(e)))
                else:
                    # Strict mode (default): raise immediately with context
                    raise EvaluationError(
                        f"Evaluation failed during truth table conversion at index {idx}",
                        input_value=idx,
                        representation=type(source_repr).__name__,
                        context={"n_vars": n_vars, "total_size": size},
                        suggestion="Use lenient=True to substitute False for failed evaluations",
                    ) from e

        # In lenient mode, warn about failures
        if failed_indices:
            warnings.warn(
                f"Truth table conversion: {len(failed_indices)} evaluations failed "
                f"(substituted False). First failure at index {failed_indices[0][0]}: "
                f"{failed_indices[0][1]}",
                UserWarning,
            )

        return truth_table

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert truth table to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> np.ndarray:
        """Create an empty (all-False) truth table for n variables."""
        size = 1 << n_vars
        return np.zeros(size, dtype=bool)

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Storage grows exponentially: 1 byte per entry (packed to bits)."""
        entries = 1 << n_vars
        return {
            "entries": entries,
            "bytes": entries // 8,  # packed bits
            "space_complexity": "O(2^n)",
        }

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for computing/evaluating n variables."""
        return {
            "evaluation": 1,  # O(1) - direct indexing
            "construction": n_vars,  # O(2^n) - exponential in variables
            "conversion_from": n_vars,  # O(2^n) - must evaluate all points
            "space_complexity": n_vars,  # O(2^n) storage
        }

    def is_complete(self, data: np.ndarray) -> bool:
        """Check if the representation contains complete information."""
        # Truth table is complete if it has the right size and contains valid boolean data
        if data is None or len(data) == 0:
            return False
        # Check if size is a power of 2 (valid truth table size)
        size = len(data)
        return size > 0 and (size & (size - 1)) == 0
