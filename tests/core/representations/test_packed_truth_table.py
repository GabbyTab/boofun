"""
Tests for core/representations/packed_truth_table module.

Tests for memory-efficient packed truth table representation.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.representations.packed_truth_table import (
    is_bitarray_available,
    PackedTruthTableRepresentation,
    create_packed_truth_table,
    memory_comparison,
)
from boofun.core.spaces import Space


class TestIsBitarrayAvailable:
    """Test is_bitarray_available function."""
    
    def test_returns_bool(self):
        """Should return a boolean."""
        result = is_bitarray_available()
        assert isinstance(result, bool)


class TestPackedTruthTableRepresentation:
    """Test PackedTruthTableRepresentation class."""
    
    def test_class_exists(self):
        """PackedTruthTableRepresentation class should exist."""
        assert PackedTruthTableRepresentation is not None


class TestCreatePackedTruthTable:
    """Test create_packed_truth_table function."""
    
    def test_returns_dict(self):
        """Should return info dictionary."""
        tt = np.array([0, 0, 0, 1])
        result = create_packed_truth_table(tt)
        
        assert isinstance(result, dict)
    
    def test_works_with_numpy_array(self):
        """Should work with numpy array input."""
        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        result = create_packed_truth_table(tt)
        
        assert result is not None


class TestMemoryComparison:
    """Test memory_comparison function."""
    
    def test_returns_dict(self):
        """Should return comparison dictionary."""
        result = memory_comparison(10)
        
        assert isinstance(result, dict)
    
    def test_has_memory_info(self):
        """Should have memory-related information."""
        result = memory_comparison(8)
        
        # Should have some memory-related keys
        assert len(result) > 0
    
    def test_different_sizes(self):
        """Should work with different n_vars."""
        for n in [4, 8, 12]:
            result = memory_comparison(n)
            assert result is not None


class TestPackedTruthTableHelpers:
    """Test helper functions for packed truth tables."""
    
    def test_create_function_callable(self):
        """create_packed_truth_table should be callable."""
        assert callable(create_packed_truth_table)
    
    def test_memory_comparison_callable(self):
        """memory_comparison should be callable."""
        assert callable(memory_comparison)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
