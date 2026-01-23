"""
Tests for families/tracker module.

Tests for growth tracking of Boolean function families.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.families.tracker import (
    GrowthTracker,
    Marker,
    MarkerType,
    PropertyMarker,
    TrackingResult,
)


class TestMarkerType:
    """Test MarkerType enum."""

    def test_marker_types_exist(self):
        """MarkerType should have expected values."""
        assert MarkerType.SCALAR is not None
        assert MarkerType.VECTOR is not None
        assert MarkerType.MATRIX is not None
        assert MarkerType.BOOLEAN is not None

    def test_marker_type_values(self):
        """MarkerType values should be strings."""
        assert MarkerType.SCALAR.value == "scalar"
        assert MarkerType.VECTOR.value == "vector"


class TestMarker:
    """Test Marker dataclass."""

    def test_marker_creation(self):
        """Marker should be creatable."""
        marker = Marker(name="test_marker", compute_fn=lambda f: f.total_influence())

        assert marker.name == "test_marker"
        assert callable(marker.compute_fn)

    def test_marker_with_type(self):
        """Marker should accept marker_type."""
        marker = Marker(
            name="influence",
            compute_fn=lambda f: f.total_influence(),
            marker_type=MarkerType.SCALAR,
        )

        assert marker.marker_type == MarkerType.SCALAR

    def test_marker_compute(self):
        """Marker should compute values."""
        marker = Marker(name="total_influence", compute_fn=lambda f: f.total_influence())

        f = bf.majority(3)
        value = marker.compute(f)

        assert isinstance(value, (int, float))
        assert value > 0

    def test_marker_with_theoretical(self):
        """Marker should accept theoretical function."""
        marker = Marker(
            name="influence",
            compute_fn=lambda f: f.total_influence(),
            theoretical_fn=lambda n: n * 0.5,
        )

        assert marker.theoretical(4) == 2.0


class TestPropertyMarker:
    """Test PropertyMarker class."""

    def test_property_marker_exists(self):
        """PropertyMarker class should exist."""
        assert PropertyMarker is not None


class TestTrackingResult:
    """Test TrackingResult dataclass."""

    def test_tracking_result_exists(self):
        """TrackingResult class should exist."""
        assert TrackingResult is not None


class TestGrowthTracker:
    """Test GrowthTracker class."""

    def test_tracker_creation_with_family(self):
        """GrowthTracker should be creatable with a family."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)

        assert tracker.family is family
        assert hasattr(tracker, "markers")
        assert hasattr(tracker, "results")

    def test_mark_property(self):
        """GrowthTracker should allow marking properties."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)

        tracker.mark("total_influence")

        assert "total_influence" in tracker.markers

    def test_observe_n_values(self):
        """GrowthTracker should observe values across n."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)
        tracker.mark("total_influence")

        results = tracker.observe([3, 5, 7])

        # Results is a dict keyed by marker name
        assert "total_influence" in results
        tracking_result = results["total_influence"]
        # Should have computed values for each n
        assert len(tracking_result.computed_values) == 3
        # Total influence grows with n
        assert tracking_result.computed_values[0] < tracking_result.computed_values[2]


class TestGrowthTrackerWithFamilies:
    """Test GrowthTracker with actual function families."""

    def test_track_majority_influence(self):
        """Track total influence of majority family."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)
        tracker.mark("total_influence")

        results = tracker.observe([3, 5, 7])

        # Get computed values
        values = results["total_influence"].computed_values

        # Majority total influence is approximately sqrt(2n/pi)
        # For n=3: sqrt(6/pi) ≈ 1.38, computed ≈ 1.5
        # For n=5: sqrt(10/pi) ≈ 1.78, computed ≈ 1.875
        # For n=7: sqrt(14/pi) ≈ 2.11, computed ≈ 2.1875
        assert 1.0 < values[0] < 2.0
        assert 1.5 < values[1] < 2.5
        assert 1.8 < values[2] < 2.8

    def test_track_multiple_properties(self):
        """Track multiple properties."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)
        tracker.mark("total_influence")
        tracker.mark("expectation")

        results = tracker.observe([3, 5])

        # Both properties should be present in results
        assert "total_influence" in results
        assert "expectation" in results
        # Majority is balanced, so expectation ≈ 0
        for val in results["expectation"].computed_values:
            assert abs(val) < 0.1


class TestGrowthTrackerAnalysis:
    """Test GrowthTracker analysis features."""

    def test_tracker_stores_results(self):
        """Tracker should store results."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)
        tracker.mark("total_influence")

        tracker.observe([3, 5])

        # Results dict has one entry per marker
        assert "total_influence" in tracker.results
        # Should have 2 computed values (for n=3 and n=5)
        assert len(tracker.results["total_influence"].computed_values) == 2


class TestTrackerEdgeCases:
    """Test edge cases for GrowthTracker."""

    def test_single_n_value(self):
        """Track with single n value."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        tracker = GrowthTracker(family)
        tracker.mark("total_influence")

        results = tracker.observe([3])
        assert "total_influence" in results
        assert len(results["total_influence"].computed_values) == 1
        assert results["total_influence"].computed_values[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
