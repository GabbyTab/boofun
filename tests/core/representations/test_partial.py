"""
Comprehensive tests for partial representations.

Tests cover:
- Creation from sparse data
- Completeness tracking
- Known/unknown value queries
- Confidence-based evaluation
- Estimation for unknown values
- Adding values incrementally
- Conversion to complete representation
"""

import numpy as np
import pytest

from boofun.core.representations.base import PartialRepresentation
from boofun.core.spaces import Space


class TestPartialRepresentationCreation:
    """Tests for creating partial representations."""

    def test_create_from_sparse_dict(self):
        """Create partial representation from sparse dict."""
        known_values = {0: False, 1: True, 3: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        assert partial.n_vars == 2
        assert partial.num_known == 3
        assert partial.num_unknown == 1

    def test_create_empty(self):
        """Create partial representation with no known values."""
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values={}
        )
        
        assert partial.completeness == 0.0
        assert partial.num_unknown == 4

    def test_create_complete(self):
        """Create fully complete representation."""
        known_values = {0: False, 1: True, 2: True, 3: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        assert partial.is_complete
        assert partial.completeness == 1.0

    def test_create_with_numpy_data(self):
        """Create from numpy array with mask."""
        from boofun.core.representations.registry import get_strategy
        
        strategy = get_strategy("truth_table")
        data = np.array([False, True, True, False])
        mask = np.array([True, True, False, True])  # Index 2 unknown
        
        partial = PartialRepresentation(strategy, data, mask, n_vars=2)
        
        assert partial.num_known == 3
        assert partial.num_unknown == 1
        assert partial.is_known(0)
        assert partial.is_known(1)
        assert not partial.is_known(2)
        assert partial.is_known(3)


class TestCompletenessTracking:
    """Tests for completeness metrics."""

    def test_completeness_fraction(self):
        """Test completeness as fraction."""
        known_values = {0: True, 1: False}  # 2 of 4 known
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        assert partial.completeness == 0.5

    def test_completeness_with_all_known(self):
        """Completeness is 1.0 when all known."""
        known_values = {i: bool(i % 2) for i in range(8)}
        partial = PartialRepresentation.from_sparse(
            n_vars=3, known_values=known_values
        )
        
        assert partial.completeness == 1.0
        assert partial.is_complete

    def test_is_known_individual(self):
        """Test is_known for individual indices."""
        known_values = {0: True, 2: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        assert partial.is_known(0)
        assert not partial.is_known(1)
        assert partial.is_known(2)
        assert not partial.is_known(3)

    def test_get_known_indices(self):
        """Test retrieving all known indices."""
        known_values = {0: True, 2: False, 3: True}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        known = partial.get_known_indices()
        assert set(known) == {0, 2, 3}

    def test_get_unknown_indices(self):
        """Test retrieving all unknown indices."""
        known_values = {0: True, 2: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        unknown = partial.get_unknown_indices()
        assert set(unknown) == {1, 3}


class TestEvaluation:
    """Tests for evaluating partial representations."""

    def test_evaluate_known_value(self):
        """Evaluate returns correct value for known inputs."""
        known_values = {0: False, 1: True, 2: True, 3: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        assert partial.evaluate(np.array(0), Space.BOOLEAN_CUBE) == False
        assert partial.evaluate(np.array(1), Space.BOOLEAN_CUBE) == True
        assert partial.evaluate(np.array(2), Space.BOOLEAN_CUBE) == True
        assert partial.evaluate(np.array(3), Space.BOOLEAN_CUBE) == False

    def test_evaluate_unknown_returns_none(self):
        """Evaluate returns None for unknown inputs."""
        known_values = {0: True}  # Only index 0 known
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        assert partial.evaluate(np.array(0), Space.BOOLEAN_CUBE) == True
        assert partial.evaluate(np.array(1), Space.BOOLEAN_CUBE) is None
        assert partial.evaluate(np.array(2), Space.BOOLEAN_CUBE) is None
        assert partial.evaluate(np.array(3), Space.BOOLEAN_CUBE) is None

    def test_evaluate_array_of_indices(self):
        """Evaluate array of indices."""
        known_values = {0: False, 1: True}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        results = partial.evaluate(np.array([0, 1, 2, 3]), Space.BOOLEAN_CUBE)
        
        assert results[0] == False
        assert results[1] == True
        assert results[2] is None
        assert results[3] is None


class TestConfidenceEvaluation:
    """Tests for evaluate_with_confidence."""

    def test_known_value_confidence_one(self):
        """Known values have confidence 1.0."""
        known_values = {0: True, 1: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        value, confidence = partial.evaluate_with_confidence(
            np.array(0), Space.BOOLEAN_CUBE
        )
        
        assert value == True
        assert confidence == 1.0

    def test_unknown_value_estimated(self):
        """Unknown values are estimated with confidence < 1."""
        # XOR function with one value unknown
        known_values = {0: False, 1: True, 3: False}  # Index 2 unknown
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        value, confidence = partial.evaluate_with_confidence(
            np.array(2), Space.BOOLEAN_CUBE
        )
        
        # Value should be estimated from neighbors
        assert isinstance(value, bool)
        assert 0.0 < confidence < 1.0

    def test_estimation_uses_neighbors(self):
        """Estimation considers Hamming-distance-1 neighbors."""
        # Create pattern where neighbors agree
        known_values = {
            0: True,   # Neighbor of 1 (distance 1)
            2: True,   # Neighbor of 3 (distance 1)
            # 1 is unknown, neighbors 0 and 3 are True/unknown
        }
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        # Index 1 has neighbor 0 (True), should estimate True
        value, confidence = partial.evaluate_with_confidence(
            np.array(1), Space.BOOLEAN_CUBE
        )
        
        assert value == True  # Neighbor 0 is True

    def test_estimation_caches_results(self):
        """Estimation results are cached."""
        known_values = {0: True, 2: True}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        # First call
        v1, c1 = partial.evaluate_with_confidence(np.array(1), Space.BOOLEAN_CUBE)
        
        # Second call should use cache
        v2, c2 = partial.evaluate_with_confidence(np.array(1), Space.BOOLEAN_CUBE)
        
        assert v1 == v2
        assert c1 == c2


class TestAddingValues:
    """Tests for incrementally adding values."""

    def test_add_value_increases_known(self):
        """Adding a value increases known count."""
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values={0: True}
        )
        
        assert partial.num_known == 1
        
        partial.add_value(1, False)
        
        assert partial.num_known == 2
        assert partial.is_known(1)

    def test_add_value_updates_data(self):
        """Added value is correctly stored."""
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values={}
        )
        
        partial.add_value(2, True)
        
        assert partial.evaluate(np.array(2), Space.BOOLEAN_CUBE) == True

    def test_add_value_invalidates_cache(self):
        """Adding value invalidates cached estimates for neighbors."""
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values={0: True}
        )
        
        # Get estimate for index 1 (caches result)
        partial.evaluate_with_confidence(np.array(1), Space.BOOLEAN_CUBE)
        
        # Add value at index 3 (neighbor of 1)
        partial.add_value(3, False)
        
        # Cache should be invalidated
        assert 1 not in partial._confidence_cache


class TestConversionToComplete:
    """Tests for converting to complete representation."""

    def test_to_complete_with_default(self):
        """Convert to complete using default for unknowns."""
        known_values = {0: True, 1: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        complete = partial.to_complete(default=False)
        
        assert len(complete) == 4
        assert complete[0] == True
        assert complete[1] == False
        assert complete[2] == False  # Unknown, filled with default
        assert complete[3] == False  # Unknown, filled with default

    def test_to_complete_with_default_true(self):
        """Convert to complete using True as default."""
        known_values = {0: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        complete = partial.to_complete(default=True)
        
        assert complete[0] == False  # Known
        assert complete[1] == True   # Unknown, default
        assert complete[2] == True   # Unknown, default
        assert complete[3] == True   # Unknown, default

    def test_to_complete_estimated(self):
        """Convert to complete using estimation for unknowns."""
        known_values = {0: True, 1: True, 3: True}  # Index 2 unknown
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        values, confidence = partial.to_complete_estimated()
        
        assert len(values) == 4
        assert len(confidence) == 4
        
        # Known values have confidence 1.0
        assert confidence[0] == 1.0
        assert confidence[1] == 1.0
        assert confidence[3] == 1.0
        
        # Unknown value has confidence < 1.0
        assert 0.0 < confidence[2] < 1.0

    def test_complete_representation_unchanged(self):
        """to_complete on complete representation returns same data."""
        known_values = {i: bool(i % 2) for i in range(4)}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        complete = partial.to_complete()
        
        for i in range(4):
            assert complete[i] == known_values[i]


class TestRepr:
    """Tests for string representation."""

    def test_repr_shows_stats(self):
        """repr shows useful statistics."""
        known_values = {0: True, 1: False}
        partial = PartialRepresentation.from_sparse(
            n_vars=2, known_values=known_values
        )
        
        r = repr(partial)
        
        assert "n_vars=2" in r
        assert "50.0%" in r  # completeness
        assert "known=2" in r
        assert "unknown=2" in r


class TestLargerFunctions:
    """Tests with larger functions."""

    def test_partial_3_variables(self):
        """Partial representation with 3 variables."""
        # Know half the values
        known_values = {i: bool(i % 2) for i in range(0, 8, 2)}
        partial = PartialRepresentation.from_sparse(
            n_vars=3, known_values=known_values
        )
        
        assert partial.n_vars == 3
        assert partial.num_known == 4
        assert partial.num_unknown == 4
        assert partial.completeness == 0.5

    def test_estimation_with_3_variables(self):
        """Estimation works correctly with 3 variables."""
        # Create partial with known values
        known_values = {
            0: False,  # 000
            1: True,   # 001
            2: True,   # 010
            4: True,   # 100
            # Indices 3, 5, 6, 7 unknown
        }
        partial = PartialRepresentation.from_sparse(
            n_vars=3, known_values=known_values
        )
        
        # Index 3 (011) has neighbors 1 (001), 2 (010), 7 (111)
        # Neighbors 1 and 2 are known (both True), should estimate True
        value, conf = partial.evaluate_with_confidence(
            np.array(3), Space.BOOLEAN_CUBE
        )
        
        assert value == True

    def test_sparse_representation_efficiency(self):
        """Verify sparse representation handles large n efficiently."""
        # Only store 10 values for n=10 (1024 total)
        known_values = {i: bool(i % 2) for i in range(10)}
        partial = PartialRepresentation.from_sparse(
            n_vars=10, known_values=known_values
        )
        
        assert partial.n_vars == 10
        assert partial.num_known == 10
        assert partial.num_unknown == 1024 - 10
        assert partial.completeness < 0.01  # Less than 1%
