"""
Packed truth table representation using bitarray for memory efficiency.

This module provides memory-optimized truth table storage using the bitarray
library, reducing memory usage from 8 bytes per entry (numpy bool) to 1 bit.

For n=20: standard numpy = 8MB, bitarray = 128KB (64x smaller)
For n=24: standard numpy = 128MB, bitarray = 2MB (64x smaller)
"""

from typing import Any, Dict, Union

import numpy as np

from ..spaces import Space
from .base import BooleanFunctionRepresentation
from .registry import register_strategy

# Try to import bitarray
try:
    from bitarray import bitarray

    HAS_BITARRAY = True
except ImportError:
    HAS_BITARRAY = False
    bitarray = None


def is_bitarray_available() -> bool:
    """Check if bitarray is available."""
    return HAS_BITARRAY


@register_strategy("packed_truth_table")
class PackedTruthTableRepresentation(BooleanFunctionRepresentation[Any]):
    """
    Memory-efficient truth table using bitarray (1 bit per entry).

    This is ideal for large Boolean functions (n > 14) where memory
    becomes a concern. Provides 8x memory savings compared to numpy bool arrays.

    Falls back to numpy if bitarray is not installed.
    """

    def evaluate(
        self, inputs: np.ndarray, data: Any, space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate the packed truth table at given inputs.

        Args:
            inputs: Input values (integer indices or binary vectors)
            data: Packed truth table (bitarray or numpy fallback)
            space: Evaluation space
            n_vars: Number of variables

        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        # Get the underlying data
        if HAS_BITARRAY and isinstance(data, dict) and "bitarray" in data:
            ba = data["bitarray"]
            size = len(ba)

            if inputs.ndim == 0:
                index = int(inputs)
                if index < 0 or index >= size:
                    raise IndexError(f"Index {index} out of range")
                return bool(ba[index])

            elif inputs.ndim == 1:
                if len(inputs) == n_vars:
                    index = self._binary_to_index(inputs)
                    return bool(ba[index])
                else:
                    return np.array([bool(ba[int(idx)]) for idx in inputs], dtype=bool)

            elif inputs.ndim == 2:
                results = []
                for row in inputs:
                    index = self._binary_to_index(row)
                    results.append(bool(ba[index]))
                return np.array(results, dtype=bool)
            else:
                raise ValueError(f"Unsupported input shape: {inputs.shape}")
        else:
            # Fallback to numpy array
            arr = data["array"] if isinstance(data, dict) else data

            if inputs.ndim == 0:
                return bool(arr[int(inputs)])
            elif inputs.ndim == 1:
                if len(inputs) == n_vars:
                    index = self._binary_to_index(inputs)
                    return bool(arr[index])
                else:
                    return np.array([bool(arr[int(idx)]) for idx in inputs], dtype=bool)
            elif inputs.ndim == 2:
                indices = [self._binary_to_index(row) for row in inputs]
                return np.array([bool(arr[i]) for i in indices], dtype=bool)
            else:
                raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _binary_to_index(self, binary_vector: np.ndarray) -> int:
        """Convert binary vector to integer index."""
        return int(np.dot(binary_vector, 2 ** np.arange(len(binary_vector) - 1, -1, -1)))

    def dump(self, data: Any, space=None, **kwargs) -> Dict[str, Any]:
        """Export packed truth table in serializable format."""
        if HAS_BITARRAY and isinstance(data, dict) and "bitarray" in data:
            ba = data["bitarray"]
            return {
                "type": "packed_truth_table",
                "n_vars": data.get("n_vars", int(np.log2(len(ba)))),
                "size": len(ba),
                "format": "bitarray",
                "memory_bytes": len(ba) // 8 + 1,
                # Serialize as base64 for compactness
                "values": ba.tobytes().hex(),
            }
        else:
            arr = data["array"] if isinstance(data, dict) else data
            return {
                "type": "packed_truth_table",
                "n_vars": data.get("n_vars", int(np.log2(len(arr)))),
                "size": len(arr),
                "format": "numpy_fallback",
                "memory_bytes": arr.nbytes,
                "values": arr.tolist(),
            }

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert from any representation to packed truth table."""
        size = 1 << n_vars

        if HAS_BITARRAY:
            # Use bitarray for memory efficiency
            ba = bitarray(size)
            ba.setall(False)

            for idx in range(size):
                val = source_repr.evaluate(idx, source_data, space, n_vars)
                ba[idx] = bool(val)

            return {"bitarray": ba, "n_vars": n_vars, "size": size}
        else:
            # Fallback to numpy packed bits
            arr = np.zeros(size, dtype=bool)

            for idx in range(size):
                val = source_repr.evaluate(idx, source_data, space, n_vars)
                arr[idx] = bool(val)

            return {
                "array": np.packbits(arr),  # Pack into bytes
                "n_vars": n_vars,
                "size": size,
                "original_dtype": "packed_uint8",
            }

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert packed truth table to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> Dict[str, Any]:
        """Create empty packed truth table."""
        size = 1 << n_vars

        if HAS_BITARRAY:
            ba = bitarray(size)
            ba.setall(False)
            return {"bitarray": ba, "n_vars": n_vars, "size": size}
        else:
            return {
                "array": np.packbits(np.zeros(size, dtype=bool)),
                "n_vars": n_vars,
                "size": size,
            }

    def is_complete(self, data: Any) -> bool:
        """Check if representation is complete."""
        if isinstance(data, dict):
            return "bitarray" in data or "array" in data
        return data is not None

    def to_numpy(self, data: Any) -> np.ndarray:
        """Convert to numpy boolean array."""
        if HAS_BITARRAY and isinstance(data, dict) and "bitarray" in data:
            return np.array(data["bitarray"].tolist(), dtype=bool)
        elif isinstance(data, dict) and "array" in data:
            size = data["size"]
            unpacked = np.unpackbits(data["array"])[:size]
            return unpacked.astype(bool)
        else:
            return np.asarray(data, dtype=bool)

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for packed operations."""
        return {
            "evaluation": 1,  # O(1) - bit indexing
            "construction": n_vars,  # O(2^n) - must evaluate all
            "conversion_from": n_vars,
            "space_complexity": n_vars - 3,  # O(2^n / 8) bits
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, Any]:
        """Return storage requirements for packed representation."""
        size = 1 << n_vars
        packed_bytes = size // 8 + (1 if size % 8 else 0)
        numpy_bytes = size  # 1 byte per bool in numpy

        return {
            "packed_bytes": packed_bytes,
            "numpy_bool_bytes": numpy_bytes,
            "numpy_int_bytes": size * 8,
            "memory_savings": f"{numpy_bytes / packed_bytes:.1f}x vs numpy bool",
            "space_complexity": "O(2^n / 8) bits",
            "example_n20": "128 KB (vs 1 MB numpy bool)",
            "example_n24": "2 MB (vs 16 MB numpy bool)",
        }


def create_packed_truth_table(truth_table: np.ndarray) -> Dict[str, Any]:
    """
    Create a packed truth table from a numpy array.

    Args:
        truth_table: Boolean numpy array

    Returns:
        Packed truth table data structure
    """
    size = len(truth_table)
    n_vars = int(np.log2(size))

    if HAS_BITARRAY:
        ba = bitarray(truth_table.tolist())
        return {"bitarray": ba, "n_vars": n_vars, "size": size}
    else:
        return {"array": np.packbits(truth_table.astype(bool)), "n_vars": n_vars, "size": size}


def memory_comparison(n_vars: int) -> Dict[str, str]:
    """
    Compare memory usage for different truth table representations.

    Args:
        n_vars: Number of variables

    Returns:
        Dictionary with memory comparisons
    """
    size = 1 << n_vars

    return {
        "n_vars": n_vars,
        "entries": f"{size:,}",
        "packed_bitarray": f"{size // 8:,} bytes ({size // 8 / 1024:.1f} KB)",
        "numpy_bool": f"{size:,} bytes ({size / 1024:.1f} KB)",
        "numpy_int64": f"{size * 8:,} bytes ({size * 8 / 1024:.1f} KB)",
        "savings_vs_bool": f"{8}x",
        "savings_vs_int64": f"{64}x",
    }
