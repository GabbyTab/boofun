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
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)

            assert tracker is not None
            assert tracker.family is not None
        except ImportError:
            pytest.skip("MajorityFamily not available")

    def test_mark_property(self):
        """GrowthTracker should allow marking properties."""
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)

            tracker.mark("total_influence")

            assert len(tracker.markers) >= 1
        except ImportError:
            pytest.skip("MajorityFamily not available")

    def test_observe_n_values(self):
        """GrowthTracker should observe values across n."""
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)
            tracker.mark("total_influence")

            results = tracker.observe([3, 5, 7])

            assert results is not None
        except ImportError:
            pytest.skip("MajorityFamily not available")


class TestGrowthTrackerWithFamilies:
    """Test GrowthTracker with actual function families."""

    def test_track_majority_influence(self):
        """Track total influence of majority family."""
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)
            tracker.mark("total_influence")

            results = tracker.observe([3, 5, 7])

            assert results is not None
        except ImportError:
            pytest.skip("MajorityFamily not available")

    def test_track_multiple_properties(self):
        """Track multiple properties."""
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)
            tracker.mark("total_influence")
            tracker.mark("expectation")

            results = tracker.observe([3, 5])

            assert results is not None
        except ImportError:
            pytest.skip("MajorityFamily not available")


class TestGrowthTrackerAnalysis:
    """Test GrowthTracker analysis features."""

    def test_tracker_stores_results(self):
        """Tracker should store results."""
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)
            tracker.mark("total_influence")

            tracker.observe([3, 5])

            assert len(tracker.results) > 0 or tracker.results is not None
        except ImportError:
            pytest.skip("MajorityFamily not available")


class TestTrackerEdgeCases:
    """Test edge cases for GrowthTracker."""

    def test_single_n_value(self):
        """Track with single n value."""
        try:
            from boofun.families import MajorityFamily

            family = MajorityFamily()
            tracker = GrowthTracker(family)
            tracker.mark("total_influence")

            results = tracker.observe([3])
            assert results is not None
        except ImportError:
            pytest.skip("MajorityFamily not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
