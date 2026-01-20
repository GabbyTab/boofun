from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

import numpy as np

from ..spaces import Space

DataType = TypeVar("DataType")


class BooleanFunctionRepresentation(ABC, Generic[DataType]):
    """Abstract base class for all Boolean function representations"""

    @abstractmethod
    def evaluate(self, inputs: np.ndarray, data: Any, space: Space, n_vars: int) -> np.ndarray:
        """
        Evaluate the function on given inputs using the provided data.

        Args:
            inputs: Input values (binary array or boolean values)
            data: Representation-specific data (coefficients, truth table, etc.)

        Returns:
            Boolean result or array of results
        """

    @abstractmethod
    def dump(self, data: DataType, space: Space = None, **kwargs) -> Dict[str, Any]:
        """
        Export the representation data in a serializable format.

        Args:
            data: The representation data to export
            space: Optional space specification (some representations need this)
            **kwargs: Representation-specific options

        Returns:
            Dictionary containing the exported representation
        """

    @abstractmethod
    def convert_from(
        self,
        source_repr: "BooleanFunctionRepresentation",
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Convert from another representation to this representation.

        Args:
            source_repr: Source representation strategy
            source_data: Data in source format
            **kwargs: Conversion options

        Returns:
            Data in this representation's format
        """

    @abstractmethod
    def convert_to(
        self,
        target_repr: "BooleanFunctionRepresentation",
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Convert from this representation to target representation.

        Args:
            target_repr: Target representation strategy
            data: Data in this representation's format
            **kwargs: Conversion options

        Returns:
            Data in target representation's format
        """

    @abstractmethod
    def create_empty(self, n_vars: int, **kwargs) -> DataType:
        """Create empty representation data structure for n variables."""

    @abstractmethod
    def is_complete(self, data: DataType) -> bool:
        """Check if the representation contains complete information."""

    @abstractmethod
    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time_complexity for computing/evaluating n variables."""

    @abstractmethod
    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return memory/storage requirements for n variables."""

    def __str__(self) -> str:
        """String representation for user display."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return f"{self.__class__.__name__}()"


class PartialRepresentation(Generic[DataType]):
    """
    Wrapper for handling partial/incomplete Boolean function data.

    A partial representation is useful when:
    - Only some outputs are known (e.g., from sampling)
    - Data is streaming in incrementally
    - Function is too large to store completely

    Provides:
    - Confidence tracking for known vs unknown inputs
    - Interpolation/estimation for unknown values
    - Completion status and statistics

    Example:
        >>> # Create partial truth table (only some values known)
        >>> known_values = {0: True, 1: False, 3: True}
        >>> partial = PartialRepresentation.from_sparse(
        ...     n_vars=2, known_values=known_values
        ... )
        >>> partial.completeness  # 0.75 (3 of 4 values known)
        >>> partial.evaluate_with_confidence(2)  # (estimated_value, confidence)
    """

    def __init__(
        self,
        strategy: BooleanFunctionRepresentation[DataType],
        data: DataType,
        known_mask: Optional[np.ndarray] = None,
        n_vars: Optional[int] = None,
    ):
        """
        Initialize partial representation.

        Args:
            strategy: The representation strategy to use
            data: The partial data (may have unknown values)
            known_mask: Boolean mask indicating which values are known
            n_vars: Number of variables (inferred from data if not provided)
        """
        self.strategy = strategy
        self.data = data
        self.n_vars = n_vars
        self._confidence_cache: Dict[int, float] = {}

        # Initialize or validate known_mask
        if known_mask is None:
            # Assume all data is known if no mask provided
            if isinstance(data, np.ndarray):
                self.known_mask = np.ones(len(data), dtype=bool)
                if n_vars is None:
                    self.n_vars = int(np.log2(len(data)))
            else:
                self.known_mask = None
        else:
            self.known_mask = np.asarray(known_mask, dtype=bool)
            if n_vars is None and len(known_mask) > 0:
                self.n_vars = int(np.log2(len(known_mask)))

    @classmethod
    def from_sparse(
        cls,
        n_vars: int,
        known_values: Dict[int, bool],
        strategy_name: str = "truth_table",
    ) -> "PartialRepresentation":
        """
        Create partial representation from sparse known values.

        Args:
            n_vars: Number of variables
            known_values: Dict mapping input indices to output values
            strategy_name: Name of representation strategy

        Returns:
            PartialRepresentation with known values filled in
        """
        from .registry import get_strategy

        strategy = get_strategy(strategy_name)
        size = 1 << n_vars

        # Create data array with default values
        data = np.zeros(size, dtype=bool)
        known_mask = np.zeros(size, dtype=bool)

        for idx, value in known_values.items():
            if 0 <= idx < size:
                data[idx] = value
                known_mask[idx] = True

        return cls(strategy, data, known_mask, n_vars)

    @property
    def completeness(self) -> float:
        """Fraction of values that are known (0.0 to 1.0)."""
        if self.known_mask is None:
            return 1.0
        return np.mean(self.known_mask)

    @property
    def is_complete(self) -> bool:
        """Check if all values are known."""
        if self.known_mask is None:
            return True
        return np.all(self.known_mask)

    @property
    def num_known(self) -> int:
        """Number of known values."""
        if self.known_mask is None:
            return len(self.data) if isinstance(self.data, np.ndarray) else 0
        return int(np.sum(self.known_mask))

    @property
    def num_unknown(self) -> int:
        """Number of unknown values."""
        if self.known_mask is None:
            return 0
        return int(np.sum(~self.known_mask))

    def is_known(self, idx: int) -> bool:
        """Check if a specific input's output is known."""
        if self.known_mask is None:
            return True
        if 0 <= idx < len(self.known_mask):
            return bool(self.known_mask[idx])
        return False

    def get_known_indices(self) -> np.ndarray:
        """Get indices of all known values."""
        if self.known_mask is None:
            return np.arange(len(self.data))
        return np.where(self.known_mask)[0]

    def get_unknown_indices(self) -> np.ndarray:
        """Get indices of all unknown values."""
        if self.known_mask is None:
            return np.array([], dtype=int)
        return np.where(~self.known_mask)[0]

    def evaluate(self, inputs: np.ndarray, space: Space) -> Any:
        """
        Evaluate at inputs (returns None for unknown values).

        Args:
            inputs: Input index or array
            space: Mathematical space

        Returns:
            Boolean result or None if unknown
        """
        if self.known_mask is None or self.is_complete:
            return self.strategy.evaluate(inputs, self.data, space, self.n_vars)

        inputs = np.asarray(inputs)
        if inputs.ndim == 0:
            idx = int(inputs)
            if self.is_known(idx):
                return bool(self.data[idx])
            return None

        # Handle array of indices
        results = []
        for idx in inputs.flat:
            if self.is_known(int(idx)):
                results.append(bool(self.data[int(idx)]))
            else:
                results.append(None)
        return results

    def evaluate_with_confidence(
        self,
        inputs: np.ndarray,
        space: Space,
    ) -> tuple[Any, float]:
        """
        Evaluate with confidence measure.

        For known values, confidence is 1.0.
        For unknown values, estimates based on known neighbors.

        Args:
            inputs: Input index or array
            space: Mathematical space

        Returns:
            Tuple of (estimated_value, confidence)
        """
        if self.is_complete:
            result = self.strategy.evaluate(inputs, self.data, space, self.n_vars)
            return result, 1.0

        inputs = np.asarray(inputs)
        idx = int(inputs) if inputs.ndim == 0 else int(inputs.flat[0])

        if self.is_known(idx):
            return bool(self.data[idx]), 1.0

        return self._estimate_with_uncertainty(idx)

    def _estimate_with_uncertainty(self, idx: int) -> tuple[bool, float]:
        """
        Estimate unknown value based on known neighbors.

        Uses Hamming-distance-1 neighbors to estimate.
        Confidence is based on how many neighbors are known.
        """
        if idx in self._confidence_cache:
            # Return cached estimate
            return self._confidence_cache[idx]

        if self.n_vars is None:
            return False, 0.0

        # Find Hamming-distance-1 neighbors
        neighbors = []
        for i in range(self.n_vars):
            neighbor_idx = idx ^ (1 << i)
            if self.is_known(neighbor_idx):
                neighbors.append(bool(self.data[neighbor_idx]))

        if not neighbors:
            # No known neighbors - use global prior (bias of known values)
            known_indices = self.get_known_indices()
            if len(known_indices) == 0:
                return False, 0.0

            bias = np.mean([self.data[i] for i in known_indices])
            estimate = bias > 0.5
            confidence = 0.1  # Very low confidence
        else:
            # Majority vote from neighbors
            ones = sum(neighbors)
            zeros = len(neighbors) - ones
            estimate = ones > zeros

            # Confidence based on agreement and coverage
            agreement = max(ones, zeros) / len(neighbors)
            coverage = len(neighbors) / self.n_vars
            confidence = agreement * coverage * 0.8  # Cap at 0.8 for estimates

        # Cache result
        result = (estimate, confidence)
        self._confidence_cache[idx] = result
        return result

    def add_value(self, idx: int, value: bool) -> None:
        """
        Add a known value to the partial representation.

        Args:
            idx: Input index
            value: Output value
        """
        if isinstance(self.data, np.ndarray) and 0 <= idx < len(self.data):
            self.data[idx] = value
            if self.known_mask is not None:
                self.known_mask[idx] = True
            # Invalidate affected cache entries
            self._confidence_cache.pop(idx, None)
            # Also invalidate neighbors
            if self.n_vars:
                for i in range(self.n_vars):
                    self._confidence_cache.pop(idx ^ (1 << i), None)

    def to_complete(self, default: bool = False) -> np.ndarray:
        """
        Convert to complete representation, filling unknowns with default.

        Args:
            default: Value to use for unknown entries

        Returns:
            Complete truth table as numpy array
        """
        if self.is_complete or self.known_mask is None:
            return np.asarray(self.data, dtype=bool)

        result = np.asarray(self.data, dtype=bool).copy()
        result[~self.known_mask] = default
        return result

    def to_complete_estimated(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert to complete representation using estimation for unknowns.

        Returns:
            Tuple of (values, confidence) arrays
        """
        if self.is_complete or self.known_mask is None:
            return (
                np.asarray(self.data, dtype=bool),
                np.ones(len(self.data), dtype=float),
            )

        values = np.asarray(self.data, dtype=bool).copy()
        confidence = np.ones(len(values), dtype=float)

        for idx in self.get_unknown_indices():
            est_value, est_conf = self._estimate_with_uncertainty(int(idx))
            values[idx] = est_value
            confidence[idx] = est_conf

        return values, confidence

    def __repr__(self) -> str:
        return (
            f"PartialRepresentation("
            f"n_vars={self.n_vars}, "
            f"completeness={self.completeness:.1%}, "
            f"known={self.num_known}, "
            f"unknown={self.num_unknown})"
        )


# Keep old name for backwards compatibility
partial_representation = PartialRepresentation
