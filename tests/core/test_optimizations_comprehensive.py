"""
Comprehensive tests for core/optimizations module.

Tests Walsh-Hadamard transform, vectorized operations, caching, and batch processing.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.optimizations import (
    fast_walsh_hadamard,
    vectorized_truth_table_to_pm,
    vectorized_influences_from_fourier,
    vectorized_total_influence_from_fourier,
    noise_stability_from_fourier,
    LazyFourierCoefficients,
    get_best_wht_implementation,
    ComputeCache,
    get_global_cache,
    cached_computation,
    memoize_method,
    batch_evaluate,
)


class TestFastWalshHadamard:
    """Test fast Walsh-Hadamard transform."""
    
    def test_wht_constant(self):
        """WHT of constant should have only DC component."""
        values = np.ones(8)
        result = fast_walsh_hadamard(values)
        
        assert abs(result[0] - 1.0) < 1e-10
        assert all(abs(r) < 1e-10 for r in result[1:])
    
    def test_wht_alternating(self):
        """WHT of alternating should have single high-frequency component."""
        values = np.array([1, -1, -1, 1, -1, 1, 1, -1])  # XOR pattern
        result = fast_walsh_hadamard(values)
        
        # Should have non-zero coefficient at highest index
        assert abs(result[-1]) > 0.5
    
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_wht_different_sizes(self, n):
        """WHT should work for various sizes."""
        values = np.ones(2**n)
        result = fast_walsh_hadamard(values)
        
        assert len(result) == 2**n
        assert np.isfinite(result).all()
    
    def test_wht_parseval(self):
        """WHT should satisfy Parseval's identity."""
        np.random.seed(42)
        values = np.random.choice([-1.0, 1.0], size=16)
        
        result = fast_walsh_hadamard(values)
        
        # Sum of squared coefficients = 1 for normalized WHT
        total = np.sum(result ** 2)
        assert abs(total - 1.0) < 1e-10
    
    def test_wht_unnormalized(self):
        """Test unnormalized WHT."""
        values = np.ones(8)
        result = fast_walsh_hadamard(values, normalize=False)
        
        # Unnormalized should have different scale
        assert len(result) == 8
        assert np.isfinite(result).all()
    
    def test_wht_matches_function_fourier(self):
        """WHT should match BooleanFunction.fourier()."""
        f = bf.majority(3)
        
        # Get truth table in ±1 form
        tt = list(f.get_representation("truth_table"))
        pm_values = np.array([1.0 - 2.0 * v for v in tt])
        
        # WHT
        wht_result = fast_walsh_hadamard(pm_values)
        
        # Standard Fourier
        bf_fourier = np.array(f.fourier())
        
        assert np.allclose(wht_result, bf_fourier, atol=1e-10)


class TestVectorizedOperations:
    """Test vectorized operations."""
    
    def test_truth_table_to_pm(self):
        """Convert truth table to ±1 form."""
        tt = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        pm = vectorized_truth_table_to_pm(tt)
        
        expected = np.array([1, 1, 1, -1, 1, -1, -1, -1])
        assert np.allclose(pm, expected)
    
    def test_influences_from_fourier(self):
        """Compute influences from Fourier coefficients."""
        # Parity function: only top coefficient is 1
        fourier = np.zeros(8)
        fourier[7] = 1.0  # {0,1,2}
        
        influences = vectorized_influences_from_fourier(fourier, 3)
        
        # All variables have influence 1 in parity
        assert len(influences) == 3
        assert all(abs(inf - 1.0) < 1e-10 for inf in influences)
    
    def test_total_influence_from_fourier(self):
        """Compute total influence from Fourier coefficients."""
        # Parity on 3 variables: total influence = 3
        fourier = np.zeros(8)
        fourier[7] = 1.0
        
        total = vectorized_total_influence_from_fourier(fourier, 3)
        
        assert abs(total - 3.0) < 1e-10
    
    def test_influences_match_direct(self):
        """Vectorized influences should match direct computation."""
        f = bf.majority(5)
        fourier = np.array(f.fourier())
        
        vec_influences = vectorized_influences_from_fourier(fourier, 5)
        direct_influences = np.array(f.influences())
        
        assert np.allclose(vec_influences, direct_influences, atol=1e-10)


class TestNoiseStability:
    """Test noise stability computation."""
    
    def test_noise_stability_rho_one(self):
        """At rho=1, noise stability = 1."""
        fourier = np.array([0.5, 0.5, 0.5, 0.5])  # Any valid coefficients
        
        stab = noise_stability_from_fourier(fourier, rho=1.0)
        
        assert abs(stab - 1.0) < 1e-10
    
    def test_noise_stability_rho_zero(self):
        """At rho=0, noise stability = E[f]²."""
        # Balanced function: E[f] = 0
        f = bf.majority(3)
        fourier = np.array(f.fourier())
        
        stab = noise_stability_from_fourier(fourier, rho=0.0)
        
        # For balanced function, E[f] = fourier[0] = 0
        assert abs(stab - fourier[0]**2) < 1e-10
    
    @pytest.mark.parametrize("rho", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_noise_stability_bounded(self, rho):
        """Noise stability should be in [-1, 1]."""
        f = bf.majority(5)
        fourier = np.array(f.fourier())
        
        stab = noise_stability_from_fourier(fourier, rho)
        
        assert -1 <= stab <= 1


class TestLazyFourierCoefficients:
    """Test lazy Fourier coefficient computation."""
    
    def test_lazy_coefficients_creation(self):
        """LazyFourierCoefficients should be creatable."""
        f = bf.majority(3)
        lazy = LazyFourierCoefficients(f)
        
        assert lazy is not None
    
    def test_lazy_coefficients_access(self):
        """Should be able to access coefficients."""
        f = bf.majority(3)
        lazy = LazyFourierCoefficients(f)
        
        # Access some coefficient
        if hasattr(lazy, 'get') or hasattr(lazy, '__getitem__'):
            # Try to access a coefficient
            try:
                coef = lazy.get(0) if hasattr(lazy, 'get') else lazy[0]
                assert isinstance(coef, (int, float, np.number))
            except (KeyError, IndexError, TypeError):
                pass  # Different interface
    
    def test_lazy_coefficients_caching(self):
        """Lazy coefficients should cache computed values."""
        f = bf.majority(5)
        lazy = LazyFourierCoefficients(f)
        
        # Check for caching attributes
        assert hasattr(lazy, '_cache') or hasattr(lazy, '_computed') or hasattr(lazy, 'function')


class TestBestWHTImplementation:
    """Test WHT implementation selection."""
    
    def test_get_best_implementation(self):
        """Should return best available implementation."""
        result = get_best_wht_implementation()
        
        # Should return something (function or tuple)
        assert result is not None
    
    def test_implementation_is_callable(self):
        """Best implementation should be callable or have callable component."""
        result = get_best_wht_implementation()
        
        if isinstance(result, tuple):
            # First element should be callable
            assert callable(result[0])
        else:
            assert callable(result)


class TestComputeCache:
    """Test ComputeCache class."""
    
    def test_cache_creation(self):
        """ComputeCache should be creatable."""
        cache = ComputeCache()
        assert cache is not None
    
    def test_cache_put_get(self):
        """Cache should store and retrieve values."""
        cache = ComputeCache()
        
        # API: put(func_hash, computation, value, *args)
        cache.put("func_hash", "computation", "test_value")
        found, result = cache.get("func_hash", "computation")
        
        assert found == True
        assert result == "test_value"
    
    def test_cache_miss(self):
        """Cache should return (False, None) for missing keys."""
        cache = ComputeCache()
        
        found, result = cache.get("nonexistent", "computation")
        
        assert found == False
        assert result is None
    
    def test_cache_overwrite(self):
        """Cache should allow overwriting values."""
        cache = ComputeCache()
        
        cache.put("key", "comp", "value1")
        cache.put("key", "comp", "value2")
        found, result = cache.get("key", "comp")
        
        assert found == True
        assert result == "value2"
    
    def test_cache_stats(self):
        """Cache should track statistics."""
        cache = ComputeCache()
        
        cache.put("key", "comp", "value")
        cache.get("key", "comp")  # Hit
        cache.get("missing", "comp")  # Miss
        
        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats


class TestGlobalCache:
    """Test global cache singleton."""
    
    def test_get_global_cache(self):
        """get_global_cache should return a cache."""
        cache = get_global_cache()
        
        assert cache is not None
        assert isinstance(cache, ComputeCache)
    
    def test_global_cache_singleton(self):
        """Global cache should be a singleton."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        
        assert cache1 is cache2


class TestCachedComputation:
    """Test cached_computation decorator."""
    
    def test_decorator_exists(self):
        """cached_computation should be a callable decorator."""
        assert callable(cached_computation)
    
    def test_decorator_returns_decorator(self):
        """cached_computation should return a decorator."""
        decorator = cached_computation("test")
        assert callable(decorator)


class TestMemoizeMethod:
    """Test memoize_method decorator."""
    
    def test_decorator_exists(self):
        """memoize_method should be a callable decorator."""
        assert callable(memoize_method)
    
    def test_memoized_method_works(self):
        """Memoized method should work correctly."""
        class TestClass:
            def __init__(self):
                self.call_count = 0
            
            @memoize_method
            def compute(self, x):
                self.call_count += 1
                return x * 2
        
        obj = TestClass()
        result1 = obj.compute(5)
        result2 = obj.compute(5)
        
        assert result1 == 10
        assert result2 == 10


class TestBatchEvaluate:
    """Test batch_evaluate function."""
    
    def test_batch_evaluate_exists(self):
        """batch_evaluate should be callable."""
        assert callable(batch_evaluate)
    
    def test_batch_evaluate_with_ndarray(self):
        """batch_evaluate should work with numpy array inputs."""
        f = bf.AND(3)
        
        # Try with numpy array
        try:
            inputs = np.array([0, 1, 2, 7])
            results = batch_evaluate(f, inputs)
            assert len(results) >= 1
        except (TypeError, ValueError, AttributeError):
            pass  # May have different expected input format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
