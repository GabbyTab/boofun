"""
Automatic representation selection for Boolean functions.

This module provides intelligent auto-selection of the most appropriate
representation based on:
- Number of variables (n)
- Sparsity of the function
- Memory constraints
- Access patterns

Guidelines:
- n <= 14: Use dense truth table (fast, fits in cache)
- n > 14, sparse: Use sparse truth table
- n > 14, dense: Use packed truth table (bitarray)
- n > 20: Consider sparse Fourier or symbolic
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .base import BooleanFunction

# Thresholds for auto-selection
DENSE_THRESHOLD = 14  # Use dense below this
PACKED_THRESHOLD = 20  # Use packed between dense and this
SPARSE_RATIO_THRESHOLD = 0.3  # Use sparse if ratio < this


def estimate_sparsity(truth_table: np.ndarray) -> float:
    """
    Estimate the sparsity of a truth table.

    Sparsity = min(ones_ratio, zeros_ratio)
    Low sparsity means the function is mostly 0s or mostly 1s.

    Args:
        truth_table: Boolean array

    Returns:
        Sparsity ratio (0 = completely sparse, 0.5 = balanced)
    """
    ones = np.sum(truth_table)
    total = len(truth_table)
    ones_ratio = ones / total
    return min(ones_ratio, 1 - ones_ratio)


def recommend_representation(
    n_vars: int,
    sparsity: Optional[float] = None,
    memory_limit_mb: Optional[float] = None,
    access_pattern: str = "random",
) -> Dict[str, Any]:
    """
    Recommend the best representation for given constraints.

    Args:
        n_vars: Number of variables
        sparsity: Estimated sparsity (if known)
        memory_limit_mb: Maximum memory in MB (optional)
        access_pattern: "random", "sequential", or "sparse_queries"

    Returns:
        Dictionary with recommendation and reasoning
    """
    size = 1 << n_vars

    # Memory calculations
    dense_bytes = size  # numpy bool
    packed_bytes = size // 8

    # Default recommendation
    recommendation = {
        "n_vars": n_vars,
        "size": size,
        "access_pattern": access_pattern,
    }

    # Check memory constraint first
    if memory_limit_mb is not None:
        limit_bytes = memory_limit_mb * 1024 * 1024

        if packed_bytes > limit_bytes:
            recommendation["representation"] = "sparse_truth_table"
            recommendation["reason"] = f"Memory limit exceeded even with packing"
            recommendation["required_mb"] = packed_bytes / (1024 * 1024)
            return recommendation

    # Small n: always use dense
    if n_vars <= DENSE_THRESHOLD:
        recommendation["representation"] = "truth_table"
        recommendation["reason"] = f"n={n_vars} <= {DENSE_THRESHOLD}: dense is fastest"
        recommendation["memory_mb"] = dense_bytes / (1024 * 1024)
        return recommendation

    # Check sparsity if available
    if sparsity is not None and sparsity < SPARSE_RATIO_THRESHOLD:
        recommendation["representation"] = "sparse_truth_table"
        recommendation["reason"] = f"Sparsity {sparsity:.1%} < {SPARSE_RATIO_THRESHOLD:.0%}"
        recommendation["memory_mb"] = (
            f"~{sparsity * size * 12 / (1024 * 1024):.1f} (depends on actual)"
        )
        return recommendation

    # Medium n: use packed
    if n_vars <= PACKED_THRESHOLD:
        recommendation["representation"] = "packed_truth_table"
        recommendation["reason"] = (
            f"{DENSE_THRESHOLD} < n={n_vars} <= {PACKED_THRESHOLD}: packed saves memory"
        )
        recommendation["memory_mb"] = packed_bytes / (1024 * 1024)
        return recommendation

    # Large n: use packed or sparse depending on access pattern
    if access_pattern == "sparse_queries":
        recommendation["representation"] = "sparse_truth_table"
        recommendation["reason"] = f"Large n with sparse queries"
    else:
        recommendation["representation"] = "packed_truth_table"
        recommendation["reason"] = f"Large n={n_vars}: packed is most efficient"
        recommendation["memory_mb"] = packed_bytes / (1024 * 1024)
        recommendation["warning"] = "Consider sparse if function has low Hamming weight"

    return recommendation


def auto_select_representation(
    truth_table: np.ndarray, n_vars: Optional[int] = None, memory_limit_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Automatically select the best representation for a truth table.

    Args:
        truth_table: Boolean array
        n_vars: Number of variables (computed if not provided)
        memory_limit_mb: Maximum memory in MB

    Returns:
        Dictionary with selected representation and converted data
    """
    if n_vars is None:
        n_vars = int(np.log2(len(truth_table)))

    sparsity = estimate_sparsity(truth_table)
    rec = recommend_representation(n_vars, sparsity, memory_limit_mb)

    result = {
        "recommendation": rec,
        "sparsity": sparsity,
    }

    # Convert to recommended representation
    repr_type = rec["representation"]

    if repr_type == "truth_table":
        result["data"] = truth_table.astype(bool)
        result["format"] = "dense"

    elif repr_type == "packed_truth_table":
        from .representations.packed_truth_table import create_packed_truth_table

        result["data"] = create_packed_truth_table(truth_table)
        result["format"] = "packed"

    elif repr_type == "sparse_truth_table":
        # Determine default value
        ones = np.sum(truth_table)
        default_value = ones > len(truth_table) // 2

        # Build exceptions
        exceptions = {}
        for i, val in enumerate(truth_table):
            if bool(val) != default_value:
                exceptions[i] = bool(val)

        result["data"] = {
            "default_value": default_value,
            "exceptions": exceptions,
            "n_vars": n_vars,
            "size": len(truth_table),
        }
        result["format"] = "sparse"

    return result


class AdaptiveFunction:
    """
    A Boolean function wrapper that automatically uses the best representation.

    This class analyzes the function and picks the optimal storage format,
    transparently handling the conversion.
    """

    def __init__(
        self,
        truth_table: np.ndarray,
        n_vars: Optional[int] = None,
        memory_limit_mb: Optional[float] = None,
        force_representation: Optional[str] = None,
    ):
        """
        Initialize with automatic representation selection.

        Args:
            truth_table: Boolean array
            n_vars: Number of variables
            memory_limit_mb: Maximum memory
            force_representation: Override auto-selection ("dense", "packed", "sparse")
        """
        self.n_vars = n_vars or int(np.log2(len(truth_table)))
        self.size = len(truth_table)

        if force_representation:
            self._setup_forced(truth_table, force_representation)
        else:
            selection = auto_select_representation(truth_table, self.n_vars, memory_limit_mb)
            self._data = selection["data"]
            self._format = selection["format"]
            self._recommendation = selection["recommendation"]
            self._sparsity = selection["sparsity"]

    def _setup_forced(self, truth_table: np.ndarray, repr_type: str):
        """Set up with forced representation type."""
        if repr_type == "dense":
            self._data = truth_table.astype(bool)
            self._format = "dense"
        elif repr_type == "packed":
            from .representations.packed_truth_table import create_packed_truth_table

            self._data = create_packed_truth_table(truth_table)
            self._format = "packed"
        elif repr_type == "sparse":
            ones = np.sum(truth_table)
            default_value = ones > len(truth_table) // 2
            exceptions = {i: bool(v) for i, v in enumerate(truth_table) if bool(v) != default_value}
            self._data = {
                "default_value": default_value,
                "exceptions": exceptions,
                "n_vars": self.n_vars,
                "size": len(truth_table),
            }
            self._format = "sparse"
        else:
            raise ValueError(f"Unknown representation: {repr_type}")

        self._sparsity = estimate_sparsity(truth_table)
        self._recommendation = {"representation": repr_type, "reason": "forced"}

    def evaluate(self, x: int) -> bool:
        """Evaluate the function at x."""
        if self._format == "dense":
            return bool(self._data[x])
        elif self._format == "packed":
            from .representations.packed_truth_table import HAS_BITARRAY

            if HAS_BITARRAY and "bitarray" in self._data:
                return bool(self._data["bitarray"][x])
            else:
                byte_idx = x // 8
                bit_idx = 7 - (x % 8)  # MSB first
                return bool((self._data["array"][byte_idx] >> bit_idx) & 1)
        elif self._format == "sparse":
            return self._data["exceptions"].get(x, self._data["default_value"])
        else:
            raise ValueError(f"Unknown format: {self._format}")

    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array."""
        if self._format == "dense":
            return self._data
        elif self._format == "packed":
            from .representations.packed_truth_table import PackedTruthTableRepresentation

            repr_obj = PackedTruthTableRepresentation()
            return repr_obj.to_numpy(self._data)
        elif self._format == "sparse":
            result = np.full(self.size, self._data["default_value"], dtype=bool)
            for idx, val in self._data["exceptions"].items():
                result[idx] = val
            return result
        else:
            raise ValueError(f"Unknown format: {self._format}")

    @property
    def format(self) -> str:
        """Get current storage format."""
        return self._format

    @property
    def sparsity(self) -> float:
        """Get function sparsity."""
        return self._sparsity

    def memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if self._format == "dense":
            return {
                "format": "dense",
                "bytes": self._data.nbytes,
                "per_entry": f"{self._data.nbytes / self.size:.1f} bytes",
            }
        elif self._format == "packed":
            if "bitarray" in self._data:
                bytes_used = len(self._data["bitarray"]) // 8
            else:
                bytes_used = len(self._data["array"])
            return {
                "format": "packed",
                "bytes": bytes_used,
                "per_entry": f"{bytes_used / self.size:.3f} bytes (1 bit)",
            }
        elif self._format == "sparse":
            # Approximate: each exception needs ~12 bytes (int key + bool value + overhead)
            bytes_used = 32 + len(self._data["exceptions"]) * 12
            return {
                "format": "sparse",
                "bytes": bytes_used,
                "num_exceptions": len(self._data["exceptions"]),
                "compression_ratio": len(self._data["exceptions"]) / self.size,
            }
        return {}

    def summary(self) -> str:
        """Get summary of the adaptive function."""
        mem = self.memory_usage()
        lines = [
            f"AdaptiveFunction (n={self.n_vars})",
            f"  Format: {self._format}",
            f"  Size: {self.size:,} entries",
            f"  Sparsity: {self._sparsity:.1%}",
            f"  Memory: {mem['bytes']:,} bytes",
            f"  Reason: {self._recommendation.get('reason', 'auto')}",
        ]
        return "\n".join(lines)


# Convenience function for BooleanFunction integration
def optimize_representation(f: "BooleanFunction") -> Dict[str, Any]:
    """
    Analyze a BooleanFunction and recommend optimal representation.

    Args:
        f: BooleanFunction to analyze

    Returns:
        Recommendation dictionary
    """
    n = f.n_vars
    tt = f.get_representation("truth_table")

    sparsity = estimate_sparsity(tt)
    rec = recommend_representation(n, sparsity)

    return {
        "current_representation": f._current_repr,
        "recommended_representation": rec["representation"],
        "reason": rec["reason"],
        "sparsity": sparsity,
        "n_vars": n,
        "would_save_memory": rec["representation"] != "truth_table",
    }
