"""
Tests for core/auto_representation module.

Tests for automatic representation selection.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.auto_representation import (
    estimate_sparsity,
    recommend_representation,
    auto_select_representation,
    AdaptiveFunction,
    optimize_representation,
)


class TestEstimateSparsity:
    """Test estimate_sparsity function."""
    
    def test_sparsity_bounded(self):
        """Sparsity should be in [0, 1]."""
        for tt in [
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 0, 1]),
        ]:
            sparsity = estimate_sparsity(tt)
            assert 0.0 <= sparsity <= 1.0
    
    def test_returns_numeric(self):
        """Sparsity should return a numeric value."""
        tt = np.array([0, 0, 0, 1])
        sparsity = estimate_sparsity(tt)
        assert isinstance(sparsity, (int, float, np.number))


class TestRecommendRepresentation:
    """Test recommend_representation function."""
    
    def test_function_callable(self):
        """recommend_representation should be callable."""
        assert callable(recommend_representation)


class TestAutoSelectRepresentation:
    """Test auto_select_representation function."""
    
    def test_function_callable(self):
        """auto_select_representation should be callable."""
        assert callable(auto_select_representation)


class TestAdaptiveFunction:
    """Test AdaptiveFunction class."""
    
    def test_class_exists(self):
        """AdaptiveFunction class should exist."""
        assert AdaptiveFunction is not None


class TestOptimizeRepresentation:
    """Test optimize_representation function."""
    
    def test_function_callable(self):
        """optimize_representation should be callable."""
        assert callable(optimize_representation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
