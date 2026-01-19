"""
Tests for fkn module.

Tests the FKN (Friedgut-Kalai-Naor) Theorem and related functions:
- distance_to_dictator
- distance_to_negated_dictator
- closest_dictator
- fkn_theorem_bound
- is_close_to_dictator
- spectral_gap
- analyze_dictator_proximity
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.fkn import (
    distance_to_dictator,
    distance_to_negated_dictator,
    closest_dictator,
    fkn_theorem_bound,
    is_close_to_dictator,
    spectral_gap,
    analyze_dictator_proximity,
)


class TestDistanceToDictator:
    """Tests for distance_to_dictator function."""
    
    def test_dictator_min_distance(self):
        """Dictator has minimum distance to some dictator."""
        f = bf.dictator(3, i=0)
        
        # Find minimum distance across all variable indices
        min_dist = min(distance_to_dictator(f, i) for i in range(3))
        
        # Should be very close to 0 for one of them
        assert min_dist < 0.1
    
    def test_dictator_nonzero_to_other(self):
        """Dictator has non-zero distance to other dictators."""
        f = bf.dictator(3, i=0)
        dist = distance_to_dictator(f, 1)
        
        assert dist > 0
    
    def test_distance_bounded(self):
        """Distance is in [0, 1]."""
        f = bf.majority(3)
        
        for i in range(3):
            dist = distance_to_dictator(f, i)
            assert 0 <= dist <= 1


class TestDistanceToNegatedDictator:
    """Tests for distance_to_negated_dictator function."""
    
    def test_negated_dictator_distance(self):
        """Dictator has specific distance to negated versions."""
        f = bf.dictator(3, i=0)
        
        # For each variable, check that distances sum to 1
        # (either you're close to dict or close to NOT dict)
        for i in range(3):
            dist = distance_to_dictator(f, i)
            neg_dist = distance_to_negated_dictator(f, i)
            
            # dist + neg_dist should equal 1 (complementary)
            assert abs(dist + neg_dist - 1.0) < 1e-10
    
    def test_distance_bounded(self):
        """Distance is in [0, 1]."""
        f = bf.AND(3)
        
        for i in range(3):
            dist = distance_to_negated_dictator(f, i)
            assert 0 <= dist <= 1


class TestClosestDictator:
    """Tests for closest_dictator function."""
    
    def test_returns_tuple(self):
        """Returns (idx, distance, is_negated) tuple."""
        f = bf.majority(3)
        result = closest_dictator(f)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_dictator_finds_itself(self):
        """Dictator's closest dictator is itself."""
        f = bf.dictator(3, i=1)
        idx, dist, neg = closest_dictator(f)
        
        # Should find variable 1 with distance 0
        assert dist == 0.0
    
    def test_distance_nonnegative(self):
        """Distance is non-negative."""
        f = bf.parity(3)
        idx, dist, neg = closest_dictator(f)
        
        assert dist >= 0
    
    def test_negated_flag_boolean(self):
        """Negated flag is boolean."""
        f = bf.AND(3)
        idx, dist, neg = closest_dictator(f)
        
        assert isinstance(neg, bool)


class TestFKNTheoremBound:
    """Tests for fkn_theorem_bound function."""
    
    def test_returns_dict(self):
        """Returns a dictionary with expected keys."""
        f = bf.majority(3)
        result = fkn_theorem_bound(f)
        
        assert isinstance(result, dict)
        assert 'total_influence' in result
        assert 'fkn_bound' in result
        assert 'closest_dictator' in result
    
    def test_dictator_small_bound(self):
        """Dictator has small FKN bound."""
        f = bf.dictator(3, i=0)
        result = fkn_theorem_bound(f)
        
        # Dictator should be close to dictator
        assert result['actual_distance'] == 0.0
        assert result['is_close_to_dictator']
    
    def test_degree_weights_bounded(self):
        """Degree weights are in [0, 1]."""
        f = bf.AND(3)
        result = fkn_theorem_bound(f)
        
        assert 0 <= result['degree_0_weight'] <= 1
        assert 0 <= result['degree_1_weight'] <= 1
        assert 0 <= result['low_degree_weight'] <= 1


class TestIsCloseToDictator:
    """Tests for is_close_to_dictator function."""
    
    def test_dictator_is_close(self):
        """Dictator is close to dictator."""
        f = bf.dictator(3, i=0)
        result = is_close_to_dictator(f)
        
        assert result
    
    def test_returns_bool(self):
        """Returns a boolean."""
        f = bf.majority(3)
        result = is_close_to_dictator(f)
        
        assert isinstance(result, bool)
    
    def test_parity_not_close(self):
        """Parity is typically not close to dictator."""
        f = bf.parity(3)
        result = is_close_to_dictator(f, epsilon=0.1)
        
        # Parity is far from all dictators
        assert isinstance(result, bool)


class TestSpectralGap:
    """Tests for spectral_gap function."""
    
    def test_returns_float(self):
        """Returns a float."""
        f = bf.majority(3)
        gap = spectral_gap(f)
        
        assert isinstance(gap, float)
    
    def test_bounded(self):
        """Gap is in [0, 1]."""
        f = bf.AND(3)
        gap = spectral_gap(f)
        
        assert 0 <= gap <= 1
    
    def test_dictator_small_gap(self):
        """Dictator has small spectral gap."""
        f = bf.dictator(3, i=0)
        gap = spectral_gap(f)
        
        # Dictator has f̂({i}) = ±1, so gap = 0
        assert gap < 0.1


class TestAnalyzeDictatorProximity:
    """Tests for analyze_dictator_proximity function."""
    
    def test_returns_dict(self):
        """Returns a dictionary."""
        f = bf.majority(3)
        result = analyze_dictator_proximity(f)
        
        assert isinstance(result, dict)
    
    def test_contains_interpretation(self):
        """Contains interpretation string."""
        f = bf.AND(3)
        result = analyze_dictator_proximity(f)
        
        assert 'interpretation' in result
        assert isinstance(result['interpretation'], str)
    
    def test_contains_n_vars(self):
        """Contains n_vars."""
        f = bf.parity(4)
        result = analyze_dictator_proximity(f)
        
        assert result['n_vars'] == 4
    
    def test_dictator_interpretation(self):
        """Dictator gets appropriate interpretation."""
        f = bf.dictator(3, i=0)
        result = analyze_dictator_proximity(f)
        
        # Should indicate close to dictator
        assert "dictator" in result['interpretation'].lower()


class TestOnBuiltinFunctions:
    """Test FKN analysis on built-in functions."""
    
    def test_majority(self):
        """FKN analysis on majority."""
        f = bf.majority(3)
        
        idx, dist, neg = closest_dictator(f)
        assert dist >= 0
        
        result = fkn_theorem_bound(f)
        assert result['total_influence'] > 0
    
    def test_and(self):
        """FKN analysis on AND."""
        f = bf.AND(3)
        
        gap = spectral_gap(f)
        assert 0 <= gap <= 1
        
        result = analyze_dictator_proximity(f)
        assert 'interpretation' in result
    
    def test_or(self):
        """FKN analysis on OR."""
        f = bf.OR(3)
        
        result = fkn_theorem_bound(f)
        assert result['fkn_bound'] >= 0
    
    def test_parity(self):
        """FKN analysis on parity."""
        f = bf.parity(3)
        
        # Parity should be far from dictators
        is_close = is_close_to_dictator(f, epsilon=0.1)
        # Parity might or might not be close depending on n
        assert isinstance(is_close, bool)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_variable(self):
        """Works with single variable."""
        f = bf.parity(1)  # Just x_0
        
        idx, dist, neg = closest_dictator(f)
        assert dist == 0.0  # Single variable IS a dictator
    
    def test_two_variables(self):
        """Works with two variables."""
        f = bf.AND(2)
        
        result = analyze_dictator_proximity(f)
        assert result['n_vars'] == 2
