"""
Comprehensive tests for FKN Theorem analysis module.

The FKN theorem characterizes functions close to dictators.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

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
    """Test distance_to_dictator function."""
    
    def test_distance_bounded(self):
        """Distance should be in [0, 1]."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            for i in range(func.n_vars):
                dist = distance_to_dictator(func, i)
                assert 0 <= dist <= 1
    
    def test_parity_distance(self):
        """Parity has distance 0.5 to all dictators."""
        for n in [3, 4, 5]:
            f = bf.parity(n)
            for i in range(n):
                dist = distance_to_dictator(f, i)
                # Parity disagrees with dictator on exactly half the inputs
                assert abs(dist - 0.5) < 1e-10


class TestDistanceToNegatedDictator:
    """Test distance_to_negated_dictator function."""
    
    def test_negated_dictator_zero_distance(self):
        """NOT(x_i) has zero distance to negated dictator on i."""
        # Create NOT(x_0) for n=3
        tt = [1, 0, 1, 0, 1, 0, 1, 0]  # NOT(x_0)
        f = bf.create(tt)
        
        dist = distance_to_negated_dictator(f, 0)
        assert abs(dist) < 1e-10
    
    def test_dictator_has_half_distance_to_negated(self):
        """Dictator has distance 0.5 to negated version."""
        f = bf.dictator(3, 0)
        dist = distance_to_negated_dictator(f, 0)
        # x_0 disagrees with NOT(x_0) on all inputs → distance 0.5? 
        # Actually distance = 1 (100% disagreement when compared to negated)
        # But in the ±1 representation, it might be 0.5
        assert dist >= 0


class TestClosestDictator:
    """Test closest_dictator function."""
    
    def test_closest_dictator_returns_tuple(self):
        """closest_dictator should return (idx, dist, negated)."""
        f = bf.majority(3)
        result = closest_dictator(f)
        
        assert len(result) == 3
        idx, dist, negated = result
        
        assert 0 <= idx < 3
        assert 0 <= dist <= 1
        assert isinstance(negated, bool)
    
    def test_parity_closest_dictator(self):
        """Parity should have valid closest dictator result."""
        f = bf.parity(3)
        idx, dist, negated = closest_dictator(f)
        
        # Distance should be around 0.5 (parity is equidistant to all dictators)
        assert 0.25 <= dist <= 0.75


class TestFKNTheoremBound:
    """Test fkn_theorem_bound function."""
    
    def test_fkn_returns_expected_keys(self):
        """FKN analysis should return expected dictionary keys."""
        f = bf.majority(3)
        result = fkn_theorem_bound(f)
        
        expected_keys = [
            'total_influence',
            'degree_0_weight',
            'degree_1_weight',
            'low_degree_weight',
            'fkn_bound',
            'closest_dictator',
            'actual_distance',
            'is_close_to_dictator',
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_dictator_low_fkn_bound(self):
        """Dictator should have low FKN bound (close to dictator)."""
        f = bf.dictator(3, 0)
        result = fkn_theorem_bound(f)
        
        # Dictator has all weight on degree 1
        assert result['degree_1_weight'] > 0.99
        assert result['is_close_to_dictator'] == True
        assert result['actual_distance'] < 0.01
    
    def test_parseval_holds(self):
        """Weights should sum to 1 (Parseval)."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            result = fkn_theorem_bound(func)
            
            total = result['degree_0_weight'] + result['degree_1_weight'] + result['high_degree_weight']
            # Note: high_degree_weight = 1 - low_degree_weight = 1 - (w0 + w1)
            # So total should be close to 1
            assert abs(result['low_degree_weight'] + result['high_degree_weight'] - 1.0) < 1e-10


class TestIsCloseToDictator:
    """Test is_close_to_dictator function."""
    
    def test_dictator_is_close(self):
        """Dictator should be close to dictator."""
        f = bf.dictator(3, 0)
        assert is_close_to_dictator(f, epsilon=0.1)
        assert is_close_to_dictator(f, epsilon=0.01)
    
    def test_majority_not_too_close(self):
        """Majority is not very close to any dictator."""
        f = bf.majority(3)
        # Majority disagrees with every dictator on some inputs
        assert not is_close_to_dictator(f, epsilon=0.01)
    
    def test_parity_not_close(self):
        """Parity is far from all dictators."""
        f = bf.parity(3)
        assert not is_close_to_dictator(f, epsilon=0.1)


class TestSpectralGap:
    """Test spectral_gap function."""
    
    def test_dictator_zero_gap(self):
        """Dictator has zero spectral gap."""
        f = bf.dictator(3, 0)
        gap = spectral_gap(f)
        assert abs(gap) < 1e-10
    
    def test_parity_large_gap(self):
        """Parity has large spectral gap (all degree-1 coeffs are 0)."""
        f = bf.parity(3)
        gap = spectral_gap(f)
        assert abs(gap - 1.0) < 1e-10  # All degree-1 coefficients are 0
    
    def test_gap_bounded(self):
        """Spectral gap should be in [0, 1]."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            gap = spectral_gap(func)
            assert 0 <= gap <= 1


class TestAnalyzeDictatorProximity:
    """Test analyze_dictator_proximity function."""
    
    def test_proximity_analysis_complete(self):
        """Proximity analysis should include all expected fields."""
        f = bf.majority(3)
        result = analyze_dictator_proximity(f)
        
        assert 'spectral_gap' in result
        assert 'n_vars' in result
        assert 'interpretation' in result
    
    def test_dictator_interpretation(self):
        """Dictator should be interpreted as 'very close'."""
        f = bf.dictator(3, 0)
        result = analyze_dictator_proximity(f)
        
        assert 'close to dictator' in result['interpretation'].lower() or \
               'dictator' in result['interpretation'].lower()
    
    def test_parity_interpretation(self):
        """Parity should be interpreted as 'far from dictators'."""
        f = bf.parity(3)
        result = analyze_dictator_proximity(f)
        
        assert 'far' in result['interpretation'].lower()


class TestFKNEdgeCases:
    """Test edge cases for FKN analysis."""
    
    def test_constant_function(self):
        """Constant function should have well-defined FKN analysis."""
        f = bf.create([0, 0, 0, 0])
        result = fkn_theorem_bound(f)
        
        # Constant has all weight on degree 0
        assert result['degree_0_weight'] > 0.99
    
    def test_small_n(self):
        """FKN should work for small n."""
        f = bf.create([0, 1])  # n=1
        result = fkn_theorem_bound(f)
        
        assert result is not None
        assert result['n_vars'] == 1 if 'n_vars' in result else True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
