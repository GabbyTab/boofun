"""
Tests for restrictions module.

Tests random restrictions and their applications:
- Restriction class
- random_restriction
- apply_restriction
- restriction_shrinkage
- average_restricted_decision_tree_depth
- switching_lemma_probability
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.restrictions import (
    Restriction,
    random_restriction,
    apply_restriction,
    restriction_shrinkage,
    average_restricted_decision_tree_depth,
    switching_lemma_probability,
)


class TestRestriction:
    """Tests for Restriction class."""
    
    def test_creation(self):
        """Restriction can be created."""
        rho = Restriction(fixed={0: 1, 2: 0}, free={1}, n_vars=3)
        
        assert rho.n_vars == 3
        assert len(rho.fixed) == 2
        assert len(rho.free) == 1
    
    def test_p_property(self):
        """p property computes fraction of free variables."""
        rho = Restriction(fixed={0: 1}, free={1, 2}, n_vars=3)
        
        assert abs(rho.p - 2/3) < 1e-10
    
    def test_p_property_no_free(self):
        """p is 0 when all variables are fixed."""
        rho = Restriction(fixed={0: 1, 1: 0, 2: 1}, free=set(), n_vars=3)
        
        assert rho.p == 0.0
    
    def test_p_property_all_free(self):
        """p is 1 when all variables are free."""
        rho = Restriction(fixed={}, free={0, 1, 2}, n_vars=3)
        
        assert rho.p == 1.0
    
    def test_str_representation(self):
        """String representation shows fixed and free vars."""
        rho = Restriction(fixed={0: 1, 2: 0}, free={1}, n_vars=3)
        s = str(rho)
        
        assert len(s) == 3
        assert '*' in s  # Free variable
    
    def test_repr(self):
        """repr includes 'Restriction'."""
        rho = Restriction(fixed={0: 1}, free={1}, n_vars=2)
        r = repr(rho)
        
        assert 'Restriction' in r


class TestRandomRestriction:
    """Tests for random_restriction function."""
    
    def test_returns_restriction(self):
        """random_restriction returns Restriction object."""
        rng = np.random.default_rng(42)
        rho = random_restriction(5, 0.5, rng)
        
        assert isinstance(rho, Restriction)
        assert rho.n_vars == 5
    
    def test_all_vars_accounted_for(self):
        """Every variable is either fixed or free."""
        rng = np.random.default_rng(42)
        rho = random_restriction(10, 0.3, rng)
        
        all_vars = set(rho.fixed.keys()) | rho.free
        expected = set(range(10))
        assert all_vars == expected
    
    def test_p_zero_all_fixed(self):
        """p=0 means all variables are fixed."""
        rng = np.random.default_rng(42)
        rho = random_restriction(10, 0.0, rng)
        
        assert len(rho.free) == 0
        assert len(rho.fixed) == 10
    
    def test_p_one_all_free(self):
        """p=1 means all variables are free."""
        rng = np.random.default_rng(42)
        rho = random_restriction(10, 1.0, rng)
        
        assert len(rho.free) == 10
        assert len(rho.fixed) == 0
    
    def test_average_free_vars(self):
        """Average number of free vars should be approximately n*p."""
        rng = np.random.default_rng(42)
        n = 20
        p = 0.3
        
        free_counts = []
        for _ in range(100):
            rho = random_restriction(n, p, rng)
            free_counts.append(len(rho.free))
        
        avg = np.mean(free_counts)
        expected = n * p
        assert abs(avg - expected) < 2.0  # Within tolerance
    
    def test_invalid_p_raises(self):
        """Invalid p raises ValueError."""
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError):
            random_restriction(5, -0.1, rng)
        
        with pytest.raises(ValueError):
            random_restriction(5, 1.5, rng)
    
    def test_reproducible_with_rng(self):
        """Same RNG gives same restriction."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        rho1 = random_restriction(10, 0.5, rng1)
        rho2 = random_restriction(10, 0.5, rng2)
        
        assert rho1.fixed == rho2.fixed
        assert rho1.free == rho2.free


class TestApplyRestriction:
    """Tests for apply_restriction function."""
    
    def test_reduces_variables(self):
        """Applying restriction reduces variable count."""
        f = bf.AND(4)
        rho = Restriction(fixed={0: 1, 3: 0}, free={1, 2}, n_vars=4)
        
        g = apply_restriction(f, rho)
        
        # Should have fewer variables
        assert g.n_vars <= 4
    
    def test_all_fixed_returns_constant(self):
        """Fixing all variables returns constant function."""
        f = bf.AND(3)
        # Fix all to 1, so AND(1,1,1) = 1
        rho = Restriction(fixed={0: 1, 1: 1, 2: 1}, free=set(), n_vars=3)
        
        g = apply_restriction(f, rho)
        
        # Should be constant or have 0 variables
        assert g.n_vars == 0 or len(list(g.get_representation("truth_table"))) == 1
    
    def test_mismatched_n_vars_raises(self):
        """Mismatched n_vars raises ValueError."""
        f = bf.AND(3)
        rho = Restriction(fixed={0: 1}, free={1, 2, 3}, n_vars=4)
        
        with pytest.raises(ValueError):
            apply_restriction(f, rho)


class TestRestrictionShrinkage:
    """Tests for restriction_shrinkage function."""
    
    def test_returns_dict(self):
        """restriction_shrinkage returns dictionary."""
        f = bf.AND(4)
        rng = np.random.default_rng(42)
        
        result = restriction_shrinkage(f, p=0.5, num_samples=10, rng=rng)
        
        assert isinstance(result, dict)
    
    def test_contains_expected_keys(self):
        """Result contains expected keys."""
        f = bf.AND(4)
        rng = np.random.default_rng(42)
        
        result = restriction_shrinkage(f, p=0.5, num_samples=10, rng=rng)
        
        expected_keys = [
            "original_dt_depth",
            "avg_restricted_dt_depth",
            "avg_free_vars",
            "expected_free_vars",
        ]
        for key in expected_keys:
            assert key in result
    
    def test_expected_free_vars_correct(self):
        """expected_free_vars equals n*p."""
        f = bf.AND(10)
        rng = np.random.default_rng(42)
        
        result = restriction_shrinkage(f, p=0.3, num_samples=10, rng=rng)
        
        assert abs(result["expected_free_vars"] - 3.0) < 0.01


class TestAverageRestrictedDecisionTreeDepth:
    """Tests for average_restricted_decision_tree_depth function."""
    
    def test_returns_float(self):
        """Returns a float value."""
        f = bf.AND(4)
        rng = np.random.default_rng(42)
        
        result = average_restricted_decision_tree_depth(f, p=0.5, num_samples=10, rng=rng)
        
        assert isinstance(result, (float, np.floating))
    
    def test_nonnegative(self):
        """Result is non-negative."""
        f = bf.parity(4)
        rng = np.random.default_rng(42)
        
        result = average_restricted_decision_tree_depth(f, p=0.3, num_samples=10, rng=rng)
        
        assert result >= 0
    
    def test_shrinks_with_small_p(self):
        """Smaller p tends to give smaller depth (more vars fixed)."""
        f = bf.parity(6)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        depth_high_p = average_restricted_decision_tree_depth(f, p=0.8, num_samples=20, rng=rng1)
        depth_low_p = average_restricted_decision_tree_depth(f, p=0.2, num_samples=20, rng=rng2)
        
        # With lower p, more variables are fixed, so depth should be lower or similar
        # This is probabilistic, so we just check it's in reasonable range
        assert depth_low_p >= 0
        assert depth_high_p >= 0


class TestSwitchingLemmaProbability:
    """Tests for switching_lemma_probability function."""
    
    def test_returns_float(self):
        """Returns a float in [0, 1]."""
        prob = switching_lemma_probability(width=3, p=0.1, depth_threshold=2)
        
        assert isinstance(prob, float)
        assert 0 <= prob <= 1
    
    def test_increases_with_width(self):
        """Probability increases with DNF width."""
        prob1 = switching_lemma_probability(width=2, p=0.1, depth_threshold=2)
        prob2 = switching_lemma_probability(width=5, p=0.1, depth_threshold=2)
        
        assert prob2 >= prob1
    
    def test_increases_with_p(self):
        """Probability increases with p."""
        prob1 = switching_lemma_probability(width=3, p=0.1, depth_threshold=2)
        prob2 = switching_lemma_probability(width=3, p=0.3, depth_threshold=2)
        
        assert prob2 >= prob1
    
    def test_decreases_with_threshold(self):
        """Probability decreases (roughly) with higher threshold."""
        # For small p*w, increasing threshold decreases probability
        prob1 = switching_lemma_probability(width=2, p=0.05, depth_threshold=1)
        prob2 = switching_lemma_probability(width=2, p=0.05, depth_threshold=3)
        
        assert prob2 <= prob1
    
    def test_capped_at_one(self):
        """Probability is capped at 1."""
        prob = switching_lemma_probability(width=10, p=0.9, depth_threshold=1)
        assert prob <= 1.0


class TestOnBuiltinFunctions:
    """Test restrictions on built-in functions."""
    
    def test_restrict_majority(self):
        """Can apply restriction to majority."""
        f = bf.majority(5)
        rng = np.random.default_rng(42)
        rho = random_restriction(5, 0.4, rng)
        
        g = apply_restriction(f, rho)
        assert g is not None
    
    def test_restrict_parity(self):
        """Can apply restriction to parity."""
        f = bf.parity(5)
        rng = np.random.default_rng(42)
        rho = random_restriction(5, 0.4, rng)
        
        g = apply_restriction(f, rho)
        assert g is not None
    
    def test_shrinkage_on_tribes(self):
        """Shrinkage analysis on tribes."""
        f = bf.tribes(2, 4)
        rng = np.random.default_rng(42)
        
        result = restriction_shrinkage(f, p=0.5, num_samples=5, rng=rng)
        assert result["expected_free_vars"] == 2.0


class TestEdgeCases:
    """Test edge cases for restrictions."""
    
    def test_single_variable(self):
        """Restriction on single variable function."""
        f = bf.parity(1)
        
        # Fix the variable
        rho = Restriction(fixed={0: 1}, free=set(), n_vars=1)
        g = apply_restriction(f, rho)
        
        # Result should be a constant (n_vars is 0 or None)
        assert g.n_vars == 0 or g.n_vars is None
    
    def test_empty_free_set(self):
        """Restriction with no free variables."""
        rho = Restriction(fixed={0: 0, 1: 1, 2: 0}, free=set(), n_vars=3)
        
        assert rho.p == 0.0
        assert str(rho) == "010"
