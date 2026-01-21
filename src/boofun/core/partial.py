"""
Partial Boolean functions for streaming and incremental specification.

This module provides user-friendly classes for working with Boolean functions
where only some outputs are known. This is useful for:
- Streaming data: adding function values incrementally
- Sampling: knowing only a subset of outputs
- Large functions: avoiding full materialization
- Studying sections of very large objects

Example:
    >>> import boofun as bf
    >>> 
    >>> # Create partial function with some known values
    >>> partial = bf.partial(n=20, known_values={0: True, 1: False, 7: True})
    >>> 
    >>> # Add more values incrementally
    >>> partial.add(5, False)
    >>> partial.add_batch({10: True, 11: True})
    >>> 
    >>> # Query status
    >>> partial.completeness  # Very small fraction
    >>> partial.num_known  # 6
    >>> 
    >>> # Evaluate known and unknown points
    >>> partial.evaluate(0)  # True (known)
    >>> partial.evaluate(100)  # None (unknown)
    >>> 
    >>> # Get estimate with confidence for unknown
    >>> val, conf = partial.evaluate_with_confidence(100)
    >>> 
    >>> # Convert to full function when ready
    >>> f = partial.to_function(fill_unknown=False)

Cross-validation:
    This design is inspired by thomasarmel/boolean_function's partial representation
    capabilities, adapted for Python's streaming/incremental use cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np

from .representations.base import PartialRepresentation
from .spaces import Space

if TYPE_CHECKING:
    from .base import BooleanFunction

__all__ = [
    "PartialBooleanFunction",
    "partial",
]


class PartialBooleanFunction:
    """
    A Boolean function with partial/incomplete specification.
    
    This class wraps PartialRepresentation to provide a user-friendly interface
    for working with Boolean functions where only some outputs are known.
    
    Key features:
    - Incremental value addition (streaming)
    - Completeness tracking
    - Confidence-based estimation for unknown values
    - Conversion to full BooleanFunction when complete
    
    Attributes:
        n_vars: Number of input variables
        size: Total number of possible inputs (2^n_vars)
        completeness: Fraction of known values (0.0 to 1.0)
        num_known: Count of known values
        num_unknown: Count of unknown values
    
    Example:
        >>> partial = PartialBooleanFunction(n_vars=10)
        >>> partial.add(0, True)
        >>> partial.add(1, False)
        >>> partial.completeness
        0.001953125  # 2/1024
    """
    
    def __init__(
        self,
        n_vars: int,
        known_values: Optional[Dict[int, bool]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a partial Boolean function.
        
        Args:
            n_vars: Number of input variables (determines domain size 2^n_vars)
            known_values: Optional dictionary mapping input indices to output values
            name: Optional name for the function
            
        Raises:
            ValueError: If n_vars < 0 or n_vars > 30
        """
        if n_vars < 0:
            raise ValueError(f"n_vars must be non-negative, got {n_vars}")
        if n_vars > 30:
            raise ValueError(f"n_vars too large ({n_vars}), maximum is 30")
        
        self._n_vars = n_vars
        self._size = 1 << n_vars
        self._name = name
        
        # Initialize the underlying PartialRepresentation
        self._partial = PartialRepresentation.from_sparse(
            n_vars=n_vars,
            known_values=known_values or {},
            strategy_name="truth_table",
        )
        
        # Create a default space for evaluation
        self._space = Space.BOOLEAN_CUBE
    
    @property
    def n_vars(self) -> int:
        """Number of input variables."""
        return self._n_vars
    
    @property
    def size(self) -> int:
        """Total number of possible inputs (2^n_vars)."""
        return self._size
    
    @property
    def completeness(self) -> float:
        """Fraction of values that are known (0.0 to 1.0)."""
        return self._partial.completeness
    
    @property
    def num_known(self) -> int:
        """Number of known values."""
        return self._partial.num_known
    
    @property
    def num_unknown(self) -> int:
        """Number of unknown values."""
        return self._partial.num_unknown
    
    @property
    def is_complete(self) -> bool:
        """Check if all values are known."""
        return self._partial.is_complete
    
    @property
    def name(self) -> Optional[str]:
        """Optional name for this function."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the function name."""
        self._name = value
    
    def add(self, idx: int, value: bool) -> None:
        """
        Add a known value at a specific input index.
        
        Args:
            idx: Input index (0 to size-1)
            value: Output value at this input
            
        Raises:
            IndexError: If idx is out of range
            
        Example:
            >>> partial.add(5, True)
            >>> partial.is_known(5)
            True
        """
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range [0, {self._size})")
        self._partial.add_value(idx, bool(value))
    
    def add_batch(self, values: Dict[int, bool]) -> None:
        """
        Add multiple known values at once.
        
        Args:
            values: Dictionary mapping input indices to output values
            
        Raises:
            IndexError: If any index is out of range
            
        Example:
            >>> partial.add_batch({0: True, 1: False, 7: True})
        """
        for idx, value in values.items():
            self.add(idx, value)
    
    def add_from_samples(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
    ) -> None:
        """
        Add values from arrays of inputs and outputs.
        
        Args:
            inputs: Array of input indices or binary vectors
            outputs: Array of corresponding output values
            
        Example:
            >>> inputs = np.array([0, 1, 2, 3])
            >>> outputs = np.array([True, False, False, True])
            >>> partial.add_from_samples(inputs, outputs)
        """
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs, dtype=bool)
        
        if len(inputs) != len(outputs):
            raise ValueError(
                f"inputs and outputs must have same length, "
                f"got {len(inputs)} and {len(outputs)}"
            )
        
        for inp, out in zip(inputs, outputs):
            if isinstance(inp, np.ndarray) and inp.ndim == 1:
                # Binary vector - convert to index
                idx = int(np.dot(inp, 2 ** np.arange(len(inp) - 1, -1, -1)))
            else:
                idx = int(inp)
            self.add(idx, bool(out))
    
    def is_known(self, idx: int) -> bool:
        """
        Check if a specific input's output is known.
        
        Args:
            idx: Input index
            
        Returns:
            True if the output at this index is known
        """
        return self._partial.is_known(idx)
    
    def evaluate(self, idx: int) -> Optional[bool]:
        """
        Evaluate the function at a specific input.
        
        Args:
            idx: Input index
            
        Returns:
            The output value if known, None if unknown
            
        Example:
            >>> partial.add(5, True)
            >>> partial.evaluate(5)
            True
            >>> partial.evaluate(6)
            None
        """
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range [0, {self._size})")
        
        return self._partial.evaluate(np.array(idx), self._space)
    
    def evaluate_with_confidence(
        self,
        idx: int,
    ) -> Tuple[bool, float]:
        """
        Evaluate with confidence estimate for unknown values.
        
        For known values, returns (value, 1.0).
        For unknown values, estimates based on Hamming neighbors.
        
        Args:
            idx: Input index
            
        Returns:
            Tuple of (estimated_value, confidence)
            - confidence is 1.0 for known values
            - confidence < 1.0 for estimated values
            
        Example:
            >>> partial.add(0, True)
            >>> partial.add(2, True)
            >>> val, conf = partial.evaluate_with_confidence(1)
            >>> # val might be True (neighbors are True), conf < 1.0
        """
        if idx < 0 or idx >= self._size:
            raise IndexError(f"Index {idx} out of range [0, {self._size})")
        
        return self._partial.evaluate_with_confidence(np.array(idx), self._space)
    
    def get_known_indices(self) -> np.ndarray:
        """
        Get indices of all known values.
        
        Returns:
            Array of input indices where output is known
        """
        return self._partial.get_known_indices()
    
    def get_unknown_indices(self) -> np.ndarray:
        """
        Get indices of all unknown values.
        
        Returns:
            Array of input indices where output is unknown
        """
        return self._partial.get_unknown_indices()
    
    def get_known_values(self) -> Dict[int, bool]:
        """
        Get dictionary of all known (index, value) pairs.
        
        Returns:
            Dictionary mapping known indices to their values
        """
        indices = self.get_known_indices()
        return {int(i): bool(self._partial.data[i]) for i in indices}
    
    def __getitem__(self, idx: int) -> Optional[bool]:
        """
        Get value at index using bracket notation.
        
        Args:
            idx: Input index
            
        Returns:
            Output value if known, None if unknown
            
        Example:
            >>> partial[5]  # Same as partial.evaluate(5)
        """
        return self.evaluate(idx)
    
    def __setitem__(self, idx: int, value: bool) -> None:
        """
        Set value at index using bracket notation.
        
        Args:
            idx: Input index
            value: Output value
            
        Example:
            >>> partial[5] = True  # Same as partial.add(5, True)
        """
        self.add(idx, value)
    
    def __len__(self) -> int:
        """Return number of known values."""
        return self.num_known
    
    def __iter__(self) -> Iterator[Tuple[int, bool]]:
        """Iterate over known (index, value) pairs."""
        for idx in self.get_known_indices():
            yield (int(idx), bool(self._partial.data[idx]))
    
    def __contains__(self, idx: int) -> bool:
        """Check if index has a known value."""
        return self.is_known(idx)
    
    def to_function(
        self,
        fill_unknown: bool = False,
        estimate_unknown: bool = False,
    ) -> "BooleanFunction":
        """
        Convert to a full BooleanFunction.
        
        Args:
            fill_unknown: Value to use for unknown entries (default False)
            estimate_unknown: If True, estimate unknown values using neighbors
            
        Returns:
            Complete BooleanFunction
            
        Raises:
            ValueError: If not complete and neither fill_unknown nor estimate_unknown
            
        Example:
            >>> partial = bf.partial(n=2, known_values={0: True, 1: False, 2: False, 3: True})
            >>> f = partial.to_function()  # Complete, so this works
        """
        from .base import BooleanFunction
        from .factory import BooleanFunctionFactory
        
        if self.is_complete:
            truth_table = np.asarray(self._partial.data, dtype=bool)
        elif estimate_unknown:
            truth_table, _ = self._partial.to_complete_estimated()
        else:
            truth_table = self._partial.to_complete(default=fill_unknown)
        
        return BooleanFunctionFactory.create(
            BooleanFunction,
            truth_table,
            n=self._n_vars,
            nickname=self._name or f"partial_{self._n_vars}",
        )
    
    def to_dense_array(
        self,
        fill_value: bool = False,
    ) -> np.ndarray:
        """
        Convert to dense numpy array, filling unknown with specified value.
        
        Args:
            fill_value: Value to use for unknown entries
            
        Returns:
            Boolean numpy array of shape (2^n_vars,)
        """
        return self._partial.to_complete(default=fill_value)
    
    def sample_unknown(
        self,
        n_samples: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Randomly sample from unknown indices.
        
        Useful for active learning or guided sampling.
        
        Args:
            n_samples: Number of indices to sample
            seed: Random seed for reproducibility
            
        Returns:
            Array of sampled unknown indices
        """
        unknown = self.get_unknown_indices()
        if len(unknown) == 0:
            return np.array([], dtype=int)
        
        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, len(unknown))
        return rng.choice(unknown, size=n_samples, replace=False)
    
    def summary(self) -> str:
        """
        Return human-readable summary of the partial function.
        
        Returns:
            Multi-line summary string
        """
        lines = [
            f"PartialBooleanFunction (n={self._n_vars})",
            f"  Size: {self._size:,} entries",
            f"  Known: {self.num_known:,} ({self.completeness:.2%})",
            f"  Unknown: {self.num_unknown:,}",
            f"  Complete: {self.is_complete}",
        ]
        if self._name:
            lines.insert(1, f"  Name: {self._name}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        name_str = f", name={self._name!r}" if self._name else ""
        return (
            f"PartialBooleanFunction(n_vars={self._n_vars}, "
            f"known={self.num_known}/{self._size}{name_str})"
        )
    
    def __str__(self) -> str:
        return self.summary()


def partial(
    n: int,
    known_values: Optional[Dict[int, bool]] = None,
    name: Optional[str] = None,
) -> PartialBooleanFunction:
    """
    Create a partial Boolean function.
    
    This is the main factory function for creating partial functions
    where only some outputs are known.
    
    Args:
        n: Number of input variables
        known_values: Optional dictionary mapping input indices to output values
        name: Optional name for the function
        
    Returns:
        PartialBooleanFunction instance
        
    Example:
        >>> import boofun as bf
        >>> 
        >>> # Empty partial function
        >>> p = bf.partial(n=10)
        >>> 
        >>> # With initial values
        >>> p = bf.partial(n=10, known_values={0: True, 1: False})
        >>> 
        >>> # Add more values
        >>> p.add(5, True)
        >>> p.completeness
    """
    return PartialBooleanFunction(n_vars=n, known_values=known_values, name=name)
