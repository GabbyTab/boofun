"""
Sparse truth table representation for Boolean functions.

This module implements memory-efficient sparse representations for Boolean functions
where most outputs are 0 or 1, significantly reducing memory usage for large functions.
"""

import numpy as np
from typing import Any, Dict, Set, Union
from .registry import register_strategy
from .base import BooleanFunctionRepresentation
from ..spaces import Space


@register_strategy("sparse_truth_table")
class SparseTruthTableRepresentation(BooleanFunctionRepresentation[Dict[str, Any]]):
    """
    Sparse truth table representation for memory efficiency.
    
    Stores only non-default values to save memory for large functions.
    Data format: {
        'default_value': bool,  # Value for unspecified indices
        'exceptions': Dict[int, bool],  # Indices with non-default values
        'n_vars': int,  # Number of variables
        'size': int  # Total size (2^n_vars)
    }
    """

    def __init__(self, compression_threshold: float = 0.1):
        """
        Initialize sparse representation.
        
        Args:
            compression_threshold: Use sparse format if non-default ratio < threshold
        """
        self.compression_threshold = compression_threshold

    def evaluate(
        self, inputs: np.ndarray, data: Dict[str, Any], space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """
        Evaluate the sparse truth table at given inputs.
        
        Args:
            inputs: Input values (integer indices or binary vectors)
            data: Sparse truth table data
            space: Evaluation space
            n_vars: Number of variables
            
        Returns:
            Boolean result(s)
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.asarray(inputs)

        default_value = data['default_value']
        exceptions = data['exceptions']
        size = data['size']

        if inputs.ndim == 0:
            # Single input
            index = int(inputs)
            if index < 0 or index >= size:
                raise IndexError(f"Index {index} out of range for size {size}")
            return exceptions.get(index, default_value)
            
        elif inputs.ndim == 1:
            if len(inputs) == n_vars:
                # Single binary vector
                index = self._binary_to_index(inputs)
                return exceptions.get(index, default_value)
            else:
                # Array of indices
                results = []
                for idx in inputs:
                    idx = int(idx)
                    if idx < 0 or idx >= size:
                        raise IndexError(f"Index {idx} out of range for size {size}")
                    results.append(exceptions.get(idx, default_value))
                return np.array(results, dtype=bool)
                
        elif inputs.ndim == 2:
            # Batch of binary vectors
            results = []
            for row in inputs:
                index = self._binary_to_index(row)
                results.append(exceptions.get(index, default_value))
            return np.array(results, dtype=bool)
            
        else:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")

    def _binary_to_index(self, binary_vector: np.ndarray) -> int:
        """Convert binary vector to integer index."""
        return int(np.dot(binary_vector, 2 ** np.arange(len(binary_vector) - 1, -1, -1)))

    def dump(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Export sparse truth table in serializable format."""
        return {
            "type": "sparse_truth_table",
            "default_value": data['default_value'],
            "exceptions": {str(k): v for k, v in data['exceptions'].items()},
            "n_vars": data['n_vars'],
            "size": data['size'],
            "compression_ratio": len(data['exceptions']) / data['size'],
            "memory_saved": 1 - (len(data['exceptions']) / data['size'])
        }

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert from any representation to sparse truth table."""
        size = 1 << n_vars
        
        # Sample a subset to determine the best default value
        sample_size = min(1000, size)
        sample_indices = np.random.choice(size, sample_size, replace=False)
        sample_values = []
        
        for idx in sample_indices:
            val = source_repr.evaluate(idx, source_data, space, n_vars)
            sample_values.append(bool(val))
        
        # Choose default value as the most common value
        true_count = sum(sample_values)
        default_value = true_count > (sample_size // 2)
        
        # Build exceptions dictionary
        exceptions = {}
        for i in range(size):
            val = bool(source_repr.evaluate(i, source_data, space, n_vars))
            if val != default_value:
                exceptions[i] = val
        
        # Check if sparse representation is beneficial
        compression_ratio = len(exceptions) / size
        if compression_ratio > self.compression_threshold:
            # Not worth compressing, fall back to dense representation
            # (In practice, you might want to return a flag or use dense representation)
            pass
        
        return {
            'default_value': default_value,
            'exceptions': exceptions,
            'n_vars': n_vars,
            'size': size
        }

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert sparse truth table to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def create_empty(self, n_vars: int, **kwargs) -> Dict[str, Any]:
        """Create empty sparse truth table (all False)."""
        size = 1 << n_vars
        return {
            'default_value': False,
            'exceptions': {},
            'n_vars': n_vars,
            'size': size
        }

    def is_complete(self, data: Dict[str, Any]) -> bool:
        """Check if representation is complete."""
        return 'default_value' in data and 'size' in data

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity for sparse operations."""
        return {
            "evaluation": 1,  # O(1) - dictionary lookup
            "construction": n_vars,  # O(2^n) - must evaluate all points
            "conversion_from": n_vars,  # O(2^n) - must evaluate all points
            "space_complexity": 0,  # O(k) where k is number of exceptions
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return storage requirements for sparse representation."""
        size = 1 << n_vars
        # Best case: only a few exceptions
        # Worst case: all values are exceptions (worse than dense)
        return {
            "best_case_bytes": 32,  # Just metadata
            "worst_case_bytes": size * 12,  # All exceptions (index + value)
            "dense_equivalent_bytes": size // 8,  # Packed bits
            "space_complexity": "O(k) where k = number of exceptions",
        }

    def get_compression_stats(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Get compression statistics."""
        size = data['size']
        num_exceptions = len(data['exceptions'])
        
        return {
            "compression_ratio": num_exceptions / size,
            "memory_saved": 1 - (num_exceptions / size),
            "sparse_bytes": 32 + num_exceptions * 12,  # Metadata + exceptions
            "dense_bytes": size // 8,  # Packed bits
            "actual_savings": max(0, 1 - (32 + num_exceptions * 12) / (size // 8))
        }


@register_strategy("adaptive_truth_table")
class AdaptiveTruthTableRepresentation(BooleanFunctionRepresentation[Dict[str, Any]]):
    """
    Adaptive truth table that automatically chooses between dense and sparse formats.
    
    Dynamically selects the most memory-efficient representation based on the data.
    """

    def __init__(self, sparse_threshold: float = 0.3):
        """
        Initialize adaptive representation.
        
        Args:
            sparse_threshold: Use sparse format if non-default ratio < threshold
        """
        self.sparse_threshold = sparse_threshold
        self.sparse_repr = SparseTruthTableRepresentation()
        
        # Import dense representation
        from .truth_table import TruthTableRepresentation
        self.dense_repr = TruthTableRepresentation()

    def evaluate(
        self, inputs: np.ndarray, data: Dict[str, Any], space: Space, n_vars: int
    ) -> Union[bool, np.ndarray]:
        """Evaluate using the appropriate internal representation."""
        if data['format'] == 'sparse':
            return self.sparse_repr.evaluate(inputs, data['data'], space, n_vars)
        else:  # dense
            return self.dense_repr.evaluate(inputs, data['data'], space, n_vars)

    def convert_from(
        self,
        source_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convert and choose optimal format."""
        # Try sparse conversion first
        sparse_data = self.sparse_repr.convert_from(source_repr, source_data, space, n_vars)
        compression_ratio = len(sparse_data['exceptions']) / sparse_data['size']
        
        if compression_ratio < self.sparse_threshold:
            # Use sparse format
            return {
                'format': 'sparse',
                'data': sparse_data,
                'compression_ratio': compression_ratio
            }
        else:
            # Use dense format
            dense_data = self.dense_repr.convert_from(source_repr, source_data, space, n_vars)
            return {
                'format': 'dense',
                'data': dense_data,
                'compression_ratio': 1.0
            }

    def convert_to(
        self,
        target_repr: BooleanFunctionRepresentation,
        source_data: Any,
        space: Space,
        n_vars: int,
        **kwargs,
    ) -> np.ndarray:
        """Convert to another representation."""
        return target_repr.convert_from(self, source_data, space, n_vars, **kwargs)

    def dump(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Export adaptive representation."""
        base_info = {
            "type": "adaptive_truth_table",
            "format": data['format'],
            "compression_ratio": data['compression_ratio']
        }
        
        if data['format'] == 'sparse':
            base_info.update(self.sparse_repr.dump(data['data']))
        else:
            base_info.update(self.dense_repr.dump(data['data']))
            
        return base_info

    def create_empty(self, n_vars: int, **kwargs) -> Dict[str, Any]:
        """Create empty adaptive representation."""
        return {
            'format': 'sparse',  # Empty is always sparse-friendly
            'data': self.sparse_repr.create_empty(n_vars),
            'compression_ratio': 0.0
        }

    def is_complete(self, data: Dict[str, Any]) -> bool:
        """Check if representation is complete."""
        return 'format' in data and 'data' in data

    def time_complexity_rank(self, n_vars: int) -> Dict[str, int]:
        """Return time complexity (depends on chosen format)."""
        return {
            "evaluation": 1,  # O(1) for both formats
            "construction": n_vars,  # O(2^n) - must evaluate to choose format
            "conversion_from": n_vars,  # O(2^n)
            "space_complexity": 0,  # O(min(2^n, k)) adaptive
        }

    def get_storage_requirements(self, n_vars: int) -> Dict[str, int]:
        """Return adaptive storage requirements."""
        size = 1 << n_vars
        return {
            "adaptive_bytes": f"min({size // 8}, 32 + k*12)",
            "space_complexity": "O(min(2^n, k)) where k = exceptions",
            "optimal": True
        }
