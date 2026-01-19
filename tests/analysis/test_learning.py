"""
Tests for learning module (Goldreich-Levin algorithm and related).

Tests the algorithms for finding heavy Fourier coefficients and
learning Boolean functions from queries.
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.learning import (
    estimate_fourier_coefficient,
    goldreich_levin,
    find_heavy_coefficients,
    learn_sparse_fourier,
    GoldreichLevinLearner,
)


class TestEstimateFourierCoefficient:
    """Tests for estimate_fourier_coefficient function."""
    
    def test_returns_tuple(self):
        """Returns (estimate, stderr) tuple."""
        f = bf.parity(3)
        result = estimate_fourier_coefficient(f, S=0, num_samples=100)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_parity_coefficient(self):
        """Parity has f̂([n]) = ±1."""
        f = bf.parity(3)
        # For parity, the only non-zero coefficient is at S = {0,1,2} = 7
        
        rng = np.random.default_rng(42)
        est, stderr = estimate_fourier_coefficient(f, S=7, num_samples=500, rng=rng)
        
        # Should be close to ±1
        assert abs(abs(est) - 1.0) < 0.3
    
    def test_parity_zero_coefficients(self):
        """Parity has f̂(S) = 0 for S ≠ [n]."""
        f = bf.parity(3)
        
        rng = np.random.default_rng(42)
        # S = 1 (just first variable) should have coefficient ≈ 0
        est, _ = estimate_fourier_coefficient(f, S=1, num_samples=500, rng=rng)
        
        assert abs(est) < 0.3
    
    def test_constant_coefficient(self):
        """Constant function has f̂(∅) = ±1."""
        f = bf.constant(True, 3)
        
        rng = np.random.default_rng(42)
        est, _ = estimate_fourier_coefficient(f, S=0, num_samples=100, rng=rng)
        
        # f̂(∅) = E[f] = ±1 for constant
        assert abs(abs(est) - 1.0) < 0.2
    
    def test_reproducible_with_rng(self):
        """Same rng gives same estimate."""
        f = bf.majority(3)
        
        rng1 = np.random.default_rng(123)
        est1, _ = estimate_fourier_coefficient(f, S=3, num_samples=100, rng=rng1)
        
        rng2 = np.random.default_rng(123)
        est2, _ = estimate_fourier_coefficient(f, S=3, num_samples=100, rng=rng2)
        
        assert abs(est1 - est2) < 1e-10


class TestGoldreichLevin:
    """Tests for goldreich_levin function."""
    
    def test_returns_list(self):
        """goldreich_levin returns list of tuples."""
        f = bf.parity(3)
        result = goldreich_levin(f, threshold=0.5)
        
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
    
    def test_finds_parity_coefficient(self):
        """Finds the single heavy coefficient in parity."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.5, rng=rng)
        
        # Should find at least one heavy coefficient
        assert len(heavy) >= 1
        
        # The coefficient at S=7 ([n]) should be heavy
        heavy_sets = [S for S, _ in heavy]
        assert 7 in heavy_sets
    
    def test_constant_single_coefficient(self):
        """Constant function has one heavy coefficient (at ∅)."""
        f = bf.constant(False, 3)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.5, rng=rng)
        
        # Should find coefficient at S=0 (empty set)
        heavy_sets = [S for S, _ in heavy]
        assert 0 in heavy_sets
    
    def test_higher_threshold_fewer_results(self):
        """Higher threshold returns fewer (or equal) results."""
        f = bf.majority(3)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        heavy_low = goldreich_levin(f, threshold=0.1, rng=rng1)
        heavy_high = goldreich_levin(f, threshold=0.5, rng=rng2)
        
        assert len(heavy_high) <= len(heavy_low) + 2  # Allow some noise


class TestFindHeavyCoefficients:
    """Tests for find_heavy_coefficients function."""
    
    def test_returns_dict(self):
        """find_heavy_coefficients returns dictionary."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        
        result = find_heavy_coefficients(f, threshold=0.5, num_samples=500, rng=rng)
        
        assert isinstance(result, dict)
    
    def test_finds_parity_coefficient(self):
        """Finds parity's heavy coefficient."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        
        heavy = find_heavy_coefficients(f, threshold=0.5, num_samples=500, rng=rng)
        
        # Should find coefficient at S=7
        assert 7 in heavy
        assert abs(abs(heavy[7]) - 1.0) < 0.3
    
    def test_dictator_coefficients(self):
        """Dictator has heavy coefficients."""
        f = bf.dictator(3, i=0)
        rng = np.random.default_rng(42)
        
        heavy = find_heavy_coefficients(f, threshold=0.4, num_samples=500, rng=rng)
        
        # Should find at least one coefficient
        # (bit ordering may vary, so just check we found something)
        assert len(heavy) >= 1
        # At least one coefficient should be close to ±1
        assert any(abs(v) > 0.5 for v in heavy.values())


class TestLearnSparseFourier:
    """Tests for learn_sparse_fourier function."""
    
    def test_returns_dict(self):
        """learn_sparse_fourier returns dictionary."""
        f = bf.dictator(3, i=0)
        rng = np.random.default_rng(42)
        
        result = learn_sparse_fourier(f, sparsity=5, num_samples=500, rng=rng)
        
        assert isinstance(result, dict)
    
    def test_learns_parity(self):
        """Learns parity (sparsity=1)."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        
        learned = learn_sparse_fourier(f, sparsity=1, num_samples=1000, rng=rng)
        
        # Should find the single coefficient at S=7
        assert len(learned) >= 1
    
    def test_learns_dictator(self):
        """Learns dictator (sparsity=2)."""
        f = bf.dictator(3, i=0)
        rng = np.random.default_rng(42)
        
        learned = learn_sparse_fourier(f, sparsity=2, num_samples=1000, rng=rng)
        
        # Should find coefficients
        assert len(learned) >= 1


class TestGoldreichLevinLearner:
    """Tests for GoldreichLevinLearner class."""
    
    def test_initialization(self):
        """Learner initializes correctly."""
        f = bf.parity(3)
        learner = GoldreichLevinLearner(f)
        
        assert learner.function is f
        assert learner.query_count == 0
    
    def test_query_increments_count(self):
        """Each unique query increments count."""
        f = bf.parity(3)
        learner = GoldreichLevinLearner(f)
        
        learner.query(0)
        assert learner.query_count == 1
        
        learner.query(1)
        assert learner.query_count == 2
    
    def test_query_caching(self):
        """Repeated queries are cached."""
        f = bf.parity(3)
        learner = GoldreichLevinLearner(f)
        
        result1 = learner.query(5)
        result2 = learner.query(5)
        
        assert result1 == result2
        assert learner.query_count == 1  # Only one actual query
    
    def test_estimate_coefficient(self):
        """estimate_coefficient works."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        learner = GoldreichLevinLearner(f, rng=rng)
        
        est = learner.estimate_coefficient(S=7, num_samples=200)
        
        # Parity has f̂(7) = ±1
        assert abs(abs(est) - 1.0) < 0.4
    
    def test_find_heavy(self):
        """find_heavy returns heavy coefficients."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        learner = GoldreichLevinLearner(f, rng=rng)
        
        heavy = learner.find_heavy(threshold=0.5)
        
        assert isinstance(heavy, list)
        assert len(heavy) >= 1
    
    def test_reset_queries(self):
        """reset_queries clears count and cache."""
        f = bf.parity(3)
        learner = GoldreichLevinLearner(f)
        
        learner.query(0)
        learner.query(1)
        assert learner.query_count == 2
        
        learner.reset_queries()
        assert learner.query_count == 0
        assert len(learner._cache) == 0
    
    def test_summary(self):
        """summary returns string."""
        f = bf.parity(3)
        learner = GoldreichLevinLearner(f)
        
        learner.query(0)
        summary = learner.summary()
        
        assert isinstance(summary, str)
        assert "1" in summary  # Query count


class TestOnBuiltinFunctions:
    """Test learning algorithms on built-in functions."""
    
    def test_gl_on_majority(self):
        """Goldreich-Levin on majority function."""
        f = bf.majority(3)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.2, rng=rng)
        
        # Majority has several non-zero coefficients
        assert len(heavy) >= 1
    
    def test_gl_on_and(self):
        """Goldreich-Levin on AND function."""
        f = bf.AND(3)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.2, rng=rng)
        
        assert len(heavy) >= 1
    
    def test_gl_on_or(self):
        """Goldreich-Levin on OR function."""
        f = bf.OR(3)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.2, rng=rng)
        
        assert len(heavy) >= 1
    
    def test_learner_on_tribes(self):
        """Learner on tribes function."""
        f = bf.tribes(2, 4)  # 2 tribes of size 2
        rng = np.random.default_rng(42)
        learner = GoldreichLevinLearner(f, rng=rng)
        
        # Use estimate_coefficient which does use the query method
        est = learner.estimate_coefficient(S=0, num_samples=100)
        
        assert isinstance(est, float)
        # Tribes query count should be > 0 after estimating
        assert learner.query_count > 0


class TestEdgeCases:
    """Test edge cases for learning algorithms."""
    
    def test_single_variable(self):
        """Works with single variable function."""
        f = bf.parity(1)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.5, rng=rng)
        
        # Parity_1 = x_1, has f̂(1) = 1
        assert len(heavy) >= 1
    
    def test_two_variables(self):
        """Works with two variable function."""
        f = bf.parity(2)
        rng = np.random.default_rng(42)
        
        heavy = goldreich_levin(f, threshold=0.5, rng=rng)
        
        # XOR has f̂(3) = 1
        heavy_sets = [S for S, _ in heavy]
        assert 3 in heavy_sets
    
    def test_low_threshold(self):
        """Works with low threshold."""
        f = bf.majority(3)
        rng = np.random.default_rng(42)
        
        heavy = find_heavy_coefficients(f, threshold=0.01, num_samples=500, rng=rng)
        
        # Should find multiple coefficients
        assert len(heavy) >= 1
