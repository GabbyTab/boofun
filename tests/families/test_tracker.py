"""
Tests for families/tracker module.

Tests for:
- MarkerType enum
- Marker dataclass
- PropertyMarker factory
- TrackingResult
- GrowthTracker
"""

import pytest
import numpy as np

import boolfunc as bf
from boolfunc.families.tracker import (
    MarkerType,
    Marker,
    PropertyMarker,
    TrackingResult,
    GrowthTracker,
)
from boolfunc.families.builtins import MajorityFamily, ParityFamily, ANDFamily


class TestMarkerType:
    """Tests for MarkerType enum."""
    
    def test_enum_values(self):
        """Enum has expected values."""
        assert MarkerType.SCALAR.value == "scalar"
        assert MarkerType.VECTOR.value == "vector"
        assert MarkerType.BOOLEAN.value == "boolean"
        assert MarkerType.FOURIER.value == "fourier"


class TestMarker:
    """Tests for Marker dataclass."""
    
    def test_creation(self):
        """Create marker with minimal args."""
        marker = Marker(
            name="test",
            compute_fn=lambda f: 1.0
        )
        
        assert marker.name == "test"
        assert marker.marker_type == MarkerType.SCALAR
    
    def test_compute(self):
        """Compute method calls compute_fn."""
        marker = Marker(
            name="test",
            compute_fn=lambda f: f.n_vars
        )
        
        f = bf.AND(3)
        result = marker.compute(f)
        
        assert result == 3
    
    def test_theoretical_none(self):
        """Theoretical returns None if no function."""
        marker = Marker(
            name="test",
            compute_fn=lambda f: 1.0
        )
        
        assert marker.theoretical(5) is None
    
    def test_theoretical_with_function(self):
        """Theoretical calls theoretical_fn."""
        marker = Marker(
            name="test",
            compute_fn=lambda f: 1.0,
            theoretical_fn=lambda n: n * 2
        )
        
        assert marker.theoretical(5) == 10


class TestPropertyMarker:
    """Tests for PropertyMarker factory."""
    
    def test_total_influence(self):
        """Create total_influence marker."""
        marker = PropertyMarker.total_influence()
        
        assert marker.name == "total_influence"
        assert marker.marker_type == MarkerType.SCALAR
        
        # Test computation
        f = bf.parity(3)
        result = marker.compute(f)
        assert isinstance(result, float)
    
    def test_influences_all(self):
        """Create influences marker (all variables)."""
        marker = PropertyMarker.influences()
        
        assert marker.name == "influences"
        assert marker.marker_type == MarkerType.VECTOR
    
    def test_influences_single(self):
        """Create influences marker (single variable)."""
        marker = PropertyMarker.influences(variable=0)
        
        assert marker.name == "influence_0"
        assert marker.marker_type == MarkerType.SCALAR
    
    def test_noise_stability(self):
        """Create noise_stability marker."""
        marker = PropertyMarker.noise_stability(rho=0.5)
        
        assert "noise_stability" in marker.name
        assert marker.params["rho"] == 0.5
    
    def test_fourier_degree(self):
        """Create fourier_degree marker."""
        marker = PropertyMarker.fourier_degree()
        
        assert marker.name == "fourier_degree"
    
    def test_spectral_concentration(self):
        """Create spectral_concentration marker."""
        marker = PropertyMarker.spectral_concentration(k=2)
        
        assert "spectral_concentration" in marker.name
        assert marker.params["k"] == 2
    
    def test_expectation(self):
        """Create expectation marker."""
        marker = PropertyMarker.expectation()
        
        assert marker.name == "expectation"
    
    def test_variance(self):
        """Create variance marker."""
        marker = PropertyMarker.variance()
        
        assert marker.name == "variance"
    
    def test_is_property(self):
        """Create is_property marker."""
        marker = PropertyMarker.is_property("balanced")
        
        assert marker.name == "is_balanced"
        assert marker.marker_type == MarkerType.BOOLEAN
    
    def test_custom(self):
        """Create custom marker."""
        marker = PropertyMarker.custom(
            name="my_marker",
            compute_fn=lambda f: f.n_vars * 2,
            description="Double n_vars"
        )
        
        assert marker.name == "my_marker"
        assert marker.description == "Double n_vars"


class TestTrackingResult:
    """Tests for TrackingResult dataclass."""
    
    def test_creation(self):
        """Create TrackingResult."""
        marker = PropertyMarker.total_influence()
        result = TrackingResult(
            marker=marker,
            n_values=[3, 5, 7],
            computed_values=[1.5, 2.0, 2.5]
        )
        
        assert len(result.n_values) == 3
        assert len(result.computed_values) == 3
    
    def test_to_arrays(self):
        """Convert to numpy arrays."""
        marker = PropertyMarker.total_influence()
        result = TrackingResult(
            marker=marker,
            n_values=[3, 5, 7],
            computed_values=[1.5, 2.0, 2.5],
            theoretical_values=[1.6, 2.1, 2.6]
        )
        
        n_arr, computed_arr, theory_arr = result.to_arrays()
        
        assert isinstance(n_arr, np.ndarray)
        assert isinstance(computed_arr, np.ndarray)
        assert isinstance(theory_arr, np.ndarray)
        assert len(n_arr) == 3


class TestGrowthTracker:
    """Tests for GrowthTracker."""
    
    def test_initialization(self):
        """Initialize GrowthTracker."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        assert tracker.family is family
        assert len(tracker.markers) == 0
        assert len(tracker.results) == 0
    
    def test_mark_total_influence(self):
        """Mark total_influence property."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        result = tracker.mark("total_influence")
        
        assert result is tracker  # Chaining
        assert "total_influence" in tracker.markers
    
    def test_mark_multiple(self):
        """Mark multiple properties."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence").mark("expectation").mark("variance")
        
        assert len(tracker.markers) == 3
    
    def test_mark_noise_stability(self):
        """Mark noise_stability with parameter."""
        family = ANDFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("noise_stability", rho=0.7)
        
        assert "noise_stability_0.7" in tracker.markers
    
    def test_mark_fourier_degree(self):
        """Mark fourier_degree."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("fourier_degree")
        
        assert "fourier_degree" in tracker.markers
    
    def test_mark_is_property(self):
        """Mark is_<property>."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("is_balanced")
        
        assert "is_balanced" in tracker.markers
    
    def test_mark_custom(self):
        """Mark with custom compute function."""
        family = ANDFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("custom", name="my_prop", compute_fn=lambda f: f.n_vars)
        
        assert "my_prop" in tracker.markers
    
    def test_mark_unknown_raises(self):
        """Unknown property without compute_fn raises."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        with pytest.raises(ValueError, match="Unknown property"):
            tracker.mark("unknown_property_xyz")
    
    def test_observe(self):
        """Observe family growth."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence")
        results = tracker.observe(n_values=[3, 5])
        
        assert "total_influence" in results
        assert len(results["total_influence"].n_values) == 2
    
    def test_observe_with_range(self):
        """Observe with range parameters."""
        family = ANDFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("expectation")
        results = tracker.observe(n_min=2, n_max=4, step=1)
        
        assert "expectation" in results
    
    def test_get_result(self):
        """Get result for marker."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("variance")
        tracker.observe(n_values=[3, 5])
        
        result = tracker.get_result("variance")
        
        assert result is not None
        assert isinstance(result, TrackingResult)
    
    def test_get_result_missing(self):
        """Get result for non-existent marker."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        result = tracker.get_result("nonexistent")
        
        assert result is None
    
    def test_summary(self):
        """Generate summary."""
        family = ANDFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence")
        tracker.observe(n_values=[2, 3])
        
        summary = tracker.summary()
        
        assert isinstance(summary, str)
        assert "total_influence" in summary
    
    def test_clear(self):
        """Clear tracking data."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("expectation")
        tracker.observe(n_values=[3, 5])
        
        assert len(tracker.results) > 0
        
        tracker.clear()
        
        assert len(tracker.results) == 0
        assert len(tracker._functions_cache) == 0


class TestGrowthTrackerWithFamilies:
    """Integration tests with different families."""
    
    def test_majority_family(self):
        """Track majority family."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence")
        results = tracker.observe(n_values=[3, 5, 7])
        
        # Majority has total influence that grows with n
        values = results["total_influence"].computed_values
        assert values[0] < values[1] < values[2]
    
    def test_parity_family(self):
        """Track parity family."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("fourier_degree")
        results = tracker.observe(n_values=[3, 4, 5])
        
        # Parity has degree = n
        values = results["fourier_degree"].computed_values
        assert values == [3, 4, 5]
    
    def test_and_family(self):
        """Track AND family."""
        family = ANDFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("expectation")
        results = tracker.observe(n_values=[2, 3, 4])
        
        # AND expectation decreases with n
        values = results["expectation"].computed_values
        assert all(isinstance(v, (int, float)) for v in values)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_observe(self):
        """Observe with no markers."""
        family = MajorityFamily()
        tracker = GrowthTracker(family)
        
        results = tracker.observe(n_values=[3, 5])
        
        assert results == {}
    
    def test_single_n(self):
        """Observe single n value."""
        family = ParityFamily()
        tracker = GrowthTracker(family)
        
        tracker.mark("total_influence")
        results = tracker.observe(n_values=[5])
        
        assert len(results["total_influence"].n_values) == 1
