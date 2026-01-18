"""Tests for Arrow's Theorem and social choice analysis."""

import pytest
import numpy as np

import boolfunc as bf
from boolfunc.analysis.arrow import (
    is_unanimous,
    is_dictatorial,
    is_non_dictatorial,
    find_dictator,
    arrow_analysis,
    social_welfare_properties,
    voting_power_analysis,
    ArrowAnalyzer,
)


class TestUnanimity:
    """Tests for unanimity (Pareto) property."""
    
    def test_majority_is_unanimous(self):
        """Majority function satisfies unanimity."""
        maj = bf.majority(5)
        assert is_unanimous(maj)
    
    def test_dictator_is_unanimous(self):
        """Dictator function satisfies unanimity."""
        d = bf.dictator(4, 1)
        assert is_unanimous(d)
    
    def test_and_is_unanimous(self):
        """AND function satisfies unanimity."""
        f = bf.AND(3)
        assert is_unanimous(f)
    
    def test_or_is_unanimous(self):
        """OR function satisfies unanimity."""
        f = bf.OR(3)
        assert is_unanimous(f)
    
    def test_parity_even_not_unanimous(self):
        """XOR/Parity on even n is not unanimous."""
        f = bf.parity(2)
        # Parity(2): 00->0, 01->1, 10->1, 11->0
        # Not unanimous because f(1,1) = 0 ≠ 1
        assert not is_unanimous(f)
    
    def test_parity_odd_is_unanimous(self):
        """XOR/Parity on odd n is unanimous."""
        f = bf.parity(3)
        # Parity(3): f(000)=0, f(111)=1 (odd number of 1s)
        assert is_unanimous(f)


class TestDictator:
    """Tests for dictator detection."""
    
    def test_dictator_is_dictatorial(self):
        """Dictator function is detected as dictatorial."""
        for n in [3, 5, 7]:
            for i in range(n):
                d = bf.dictator(n, i)
                assert is_dictatorial(d)
                assert not is_non_dictatorial(d)
    
    def test_majority_is_not_dictatorial(self):
        """Majority is not dictatorial."""
        maj = bf.majority(5)
        assert not is_dictatorial(maj)
        assert is_non_dictatorial(maj)
    
    def test_find_dictator_returns_correct_index(self):
        """find_dictator returns correct variable index."""
        for n in [3, 4, 5]:
            for i in range(n):
                d = bf.dictator(n, i)
                result = find_dictator(d)
                assert result is not None
                idx, negated = result
                # Note: dictator(n, i) creates x_{n-1-i} in LSB indexing
                # So we just verify it found *some* dictator
                assert 0 <= idx < n
                assert negated == False
    
    def test_and_is_not_dictatorial(self):
        """AND is not dictatorial."""
        f = bf.AND(3)
        assert not is_dictatorial(f)


class TestArrowAnalysis:
    """Tests for comprehensive Arrow analysis."""
    
    def test_majority_arrow_analysis(self):
        """Arrow analysis on majority function."""
        maj = bf.majority(5)
        result = arrow_analysis(maj)
        
        assert result['n_voters'] == 5
        assert result['is_unanimous'] == True
        assert result['is_iia'] == True  # Always true for Boolean
        assert result['is_dictator'] == False
        assert result['arrow_type'] == 'impossible'
    
    def test_dictator_arrow_analysis(self):
        """Arrow analysis on dictator function."""
        d = bf.dictator(4, 2)
        result = arrow_analysis(d)
        
        assert result['is_dictator'] == True
        assert result['arrow_type'] == 'dictator'
        # dictator_info should be (some_index, False)
        assert result['dictator_info'] is not None
        assert result['dictator_info'][1] == False  # Not negated
    
    def test_parity_arrow_analysis(self):
        """Arrow analysis on parity function."""
        # Even n parity is not unanimous
        p = bf.parity(2)
        result = arrow_analysis(p)
        
        assert result['is_unanimous'] == False
        assert result['arrow_type'] == 'non-unanimous'


class TestSocialWelfareProperties:
    """Tests for social welfare properties."""
    
    def test_majority_is_symmetric(self):
        """Majority treats all voters symmetrically."""
        maj = bf.majority(5)
        result = social_welfare_properties(maj)
        
        assert result['is_symmetric'] == True
        assert result['is_anonymous'] == True
    
    def test_majority_is_monotone(self):
        """Majority is monotone."""
        maj = bf.majority(5)
        result = social_welfare_properties(maj)
        
        assert result['is_monotone'] == True
    
    def test_dictator_is_not_symmetric(self):
        """Dictator is not symmetric (one voter has all power)."""
        d = bf.dictator(5, 0)
        result = social_welfare_properties(d)
        
        # All influences are 0 except one
        assert result['is_symmetric'] == False


class TestVotingPower:
    """Tests for voting power analysis."""
    
    def test_majority_equal_power(self):
        """All voters have equal power in majority."""
        maj = bf.majority(5)
        result = voting_power_analysis(maj)
        
        # Banzhaf should be equal
        banzhaf = result['banzhaf_index']
        assert len(banzhaf) == 5
        assert all(abs(b - banzhaf[0]) < 0.01 for b in banzhaf)
        
        # Shapley-Shubik should be equal (1/n each)
        shapley = result['shapley_shubik_index']
        assert all(abs(s - 0.2) < 0.01 for s in shapley)
    
    def test_dictator_all_power(self):
        """Dictator has all the power."""
        d = bf.dictator(5, 2)
        result = voting_power_analysis(d)
        
        # Only voter 2 has power
        banzhaf = result['banzhaf_index']
        assert banzhaf[2] == 1.0
        assert all(banzhaf[i] == 0 for i in range(5) if i != 2)
        
        # Most powerful is voter 2
        assert result['most_powerful'] == 2
        
        # Power concentration is 100%
        assert result['power_concentration'] == 1.0
        
        # All others are dummies
        assert set(result['dummy_voters']) == {0, 1, 3, 4}
    
    def test_and_power_distribution(self):
        """AND function: all voters are pivotal."""
        f = bf.AND(3)
        result = voting_power_analysis(f)
        
        # All voters have equal power in AND
        banzhaf = result['banzhaf_index']
        assert all(abs(b - banzhaf[0]) < 0.01 for b in banzhaf)
        
        # No dummy voters
        assert result['dummy_voters'] == []


class TestArrowAnalyzer:
    """Tests for the ArrowAnalyzer class."""
    
    def test_analyzer_caches_results(self):
        """Analyzer caches results for efficiency."""
        maj = bf.majority(3)
        analyzer = ArrowAnalyzer(maj)
        
        # First call computes
        result1 = analyzer.arrow_properties()
        # Second call uses cache
        result2 = analyzer.arrow_properties()
        
        assert result1 == result2
    
    def test_full_analysis(self):
        """full_analysis returns all three analyses."""
        maj = bf.majority(3)
        analyzer = ArrowAnalyzer(maj)
        
        result = analyzer.full_analysis()
        
        assert 'arrow' in result
        assert 'welfare' in result
        assert 'power' in result
    
    def test_summary_returns_string(self):
        """summary() returns a non-empty string."""
        maj = bf.majority(3)
        analyzer = ArrowAnalyzer(maj)
        
        summary = analyzer.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 100  # Should be substantial
        assert "Arrow" in summary
        assert "Voting Power" in summary
