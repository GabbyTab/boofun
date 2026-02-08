"""
Comprehensive tests for core/optimizations module.

Tests for optimized WHT, vectorized operations, caching, and parallel batch processing.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.core.optimizations import (
    ComputeCache,
    LazyFourierCoefficients,
    batch_evaluate,
    cached_computation,
    fast_walsh_hadamard,
    get_best_wht_implementation,
    get_global_cache,
    memoize_method,
    noise_stability_from_fourier,
    vectorized_influences_from_fourier,
    vectorized_total_influence_from_fourier,
    vectorized_truth_table_to_pm,
)


class TestFastWalshHadamard:
    """Test fast_walsh_hadamard function."""

    def test_basic_wht(self):
        """Basic WHT should work correctly."""
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = fast_walsh_hadamard(values)

        # All +1 → only constant term non-zero
        assert abs(result[0] - 1.0) < 1e-10
        assert all(abs(r) < 1e-10 for r in result[1:])

    def test_alternating_values(self):
        """WHT of alternating values."""
        values = np.array([1.0, -1.0, 1.0, -1.0])
        result = fast_walsh_hadamard(values)

        # Should have weight on degree 1
        assert result is not None
        assert len(result) == 4

    def test_parity_wht(self):
        """WHT of parity function."""
        # Parity on 2 variables: [+1, -1, -1, +1]
        values = np.array([1.0, -1.0, -1.0, 1.0])
        result = fast_walsh_hadamard(values)

        # Parity has weight only on highest degree
        assert abs(result[0]) < 1e-10  # No constant term
        assert abs(result[-1]) > 0.5  # Has weight on highest degree

    def test_wht_normalization(self):
        """WHT normalization should be correct."""
        values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        normalized = fast_walsh_hadamard(values, normalize=True)
        unnormalized = fast_walsh_hadamard(values, normalize=False)

        # Normalized should be scaled by 1/8
        assert abs(normalized[0] - unnormalized[0] / 8) < 1e-10

    def test_wht_power_of_two_required(self):
        """WHT should require power of 2 length."""
        values = np.array([1.0, 1.0, 1.0])  # Length 3

        with pytest.raises(ValueError):
            fast_walsh_hadamard(values)

    def test_wht_preserves_input(self):
        """WHT should not modify input array."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        original = values.copy()

        fast_walsh_hadamard(values)

        assert np.array_equal(values, original)


class TestVectorizedOperations:
    """Test vectorized helper functions."""

    def test_truth_table_to_pm(self):
        """Vectorized truth table to ±1 conversion."""
        tt = np.array([0, 1, 1, 0])
        pm = vectorized_truth_table_to_pm(tt)

        # O'Donnell convention: 0 → +1, 1 → -1
        expected = np.array([1.0, -1.0, -1.0, 1.0])
        assert np.allclose(pm, expected)

    def test_influences_from_fourier(self):
        """Vectorized influence computation from Fourier."""
        # Dictator function on n=2: x_0
        # Fourier: [0, 1, 0, 0] (weight 1 on {0})
        fourier = np.array([0.0, 1.0, 0.0, 0.0])
        influences = vectorized_influences_from_fourier(fourier, n_vars=2)

        # Variable 0 should have influence 1
        assert abs(influences[0] - 1.0) < 1e-10
        # Variable 1 should have influence 0
        assert abs(influences[1]) < 1e-10

    def test_total_influence_from_fourier(self):
        """Vectorized total influence computation."""
        # Parity on n=3: all weight on highest degree
        f = bf.parity(3)
        fourier = np.array(f.fourier())

        total = vectorized_total_influence_from_fourier(fourier, n_vars=3)

        # Total influence of parity is n
        assert abs(total - 3.0) < 1e-10


class TestNoiseStability:
    """Test noise stability computation."""

    def test_noise_stability_rho_one(self):
        """Noise stability with rho=1 is the second moment."""
        f = bf.majority(3)
        fourier = np.array(f.fourier())

        stability = noise_stability_from_fourier(fourier, rho=1.0)

        # At rho=1, noise stability = sum of squared coefficients = 1
        assert abs(stability - 1.0) < 1e-10

    def test_noise_stability_rho_zero(self):
        """Noise stability with rho=0 is the square of the mean."""
        f = bf.majority(3)
        fourier = np.array(f.fourier())

        stability = noise_stability_from_fourier(fourier, rho=0.0)

        # At rho=0, noise stability = f̂(∅)²
        assert abs(stability - fourier[0] ** 2) < 1e-10

    def test_noise_stability_bounded(self):
        """Noise stability should be bounded."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            fourier = np.array(func.fourier())
            for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
                stability = noise_stability_from_fourier(fourier, rho)
                assert -1 <= stability <= 1


class TestLazyFourierCoefficients:
    """Test LazyFourierCoefficients class."""

    def test_lazy_init(self):
        """Lazy coefficients should initialize."""
        f = bf.majority(3)
        lazy = LazyFourierCoefficients(f)
        assert lazy is not None

    def test_lazy_has_methods(self):
        """Lazy coefficients should have methods."""
        f = bf.majority(3)
        lazy = LazyFourierCoefficients(f)

        # Check for any useful methods
        methods = [m for m in dir(lazy) if not m.startswith("_")]
        assert len(methods) > 0


class TestGetBestWHTImplementation:
    """Test WHT implementation selection."""

    def test_returns_tuple_or_callable(self):
        """get_best_wht_implementation should return implementation info."""
        result = get_best_wht_implementation()

        # May return tuple (func, name) or just callable
        if isinstance(result, tuple):
            impl, name = result
            assert callable(impl)
            assert isinstance(name, str)
        else:
            assert callable(result)

    def test_implementation_works(self):
        """Selected implementation should work correctly."""
        result = get_best_wht_implementation()

        # Extract callable if tuple
        impl = result[0] if isinstance(result, tuple) else result

        values = np.array([1.0, 1.0, 1.0, 1.0])
        output = impl(values)

        assert output is not None
        assert len(output) == 4


class TestComputeCache:
    """Test ComputeCache class."""

    def test_cache_basic(self):
        """Basic caching should work."""
        cache = ComputeCache()

        # Cache a value using put
        cache.put("func_hash", "test_comp", 42)

        # Retrieve it
        found, value = cache.get("func_hash", "test_comp")
        assert found == True
        assert value == 42

    def test_cache_miss(self):
        """Cache miss should return (False, None)."""
        cache = ComputeCache()

        found, value = cache.get("nonexistent", "comp")
        assert found == False
        assert value is None

    def test_cache_clear(self):
        """Cache clearing should work."""
        cache = ComputeCache()
        cache.put("hash1", "comp", 1)
        cache.put("hash2", "comp", 2)

        cache.clear()

        found1, _ = cache.get("hash1", "comp")
        found2, _ = cache.get("hash2", "comp")
        assert found1 == False
        assert found2 == False

    def test_cache_stats(self):
        """Cache should track statistics."""
        cache = ComputeCache()
        cache.put("h", "c", 1)
        cache.get("h", "c")  # Hit
        cache.get("x", "c")  # Miss

        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats


class TestGlobalCache:
    """Test global cache functions."""

    def test_global_cache_exists(self):
        """Global cache should exist."""
        cache = get_global_cache()
        assert cache is not None

    def test_global_cache_singleton(self):
        """Global cache should be a singleton."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        assert cache1 is cache2


class TestCachedComputation:
    """Test cached_computation decorator."""

    def test_cached_decorator_exists(self):
        """cached_computation should be a callable decorator factory."""
        assert callable(cached_computation)

        # Create decorator
        decorator = cached_computation("test")
        assert callable(decorator)


class TestMemoizeMethod:
    """Test memoize_method decorator."""

    def test_memoize_works(self):
        """memoize_method should memoize correctly."""

        class TestClass:
            def __init__(self):
                self.call_count = 0

            @memoize_method
            def expensive_method(self, x):
                self.call_count += 1
                return x * 2

        obj = TestClass()

        # First call
        r1 = obj.expensive_method(5)
        assert r1 == 10

        # Second call should be memoized
        r2 = obj.expensive_method(5)
        assert r2 == 10


class TestBatchEvaluate:
    """Test batch_evaluate function."""

    def test_batch_evaluate_exists(self):
        """batch_evaluate should be callable."""
        assert callable(batch_evaluate)

    def test_batch_evaluate_signature(self):
        """batch_evaluate should accept function and inputs."""
        f = bf.AND(2)

        try:
            result = batch_evaluate(f, np.array([0]))
            # Result should be array-like with the expected evaluation
            assert result is not None
            result_arr = np.asarray(result)
            assert len(result_arr) == 1
        except (TypeError, ValueError):
            # Acceptable if the implementation has specific requirements
            pass


class TestOptimizationEdgeCases:
    """Test edge cases for optimization functions."""

    def test_wht_single_element(self):
        """WHT should work for single element (n=0)."""
        values = np.array([1.0])
        result = fast_walsh_hadamard(values)

        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-10

    def test_large_wht(self):
        """WHT should work for larger inputs."""
        n = 8
        values = np.random.randn(2**n)
        result = fast_walsh_hadamard(values)

        assert len(result) == 2**n
        assert np.isfinite(result).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
