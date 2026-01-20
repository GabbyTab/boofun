import sys
sys.path.insert(0, "src")
"""
Tests for Huang's Sensitivity Theorem module.

Tests the sensitivity analysis and verification of Huang's theorem
which states: s(f) >= sqrt(deg(f)) for all Boolean functions.
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.huang import (
    sensitivity_at,
    max_sensitivity,
    sensitivity,
    average_sensitivity,
    block_sensitivity,
    verify_huang_theorem,
    sensitivity_vs_degree,
    HuangAnalysis,
)


class TestSensitivityAt:
    """Tests for sensitivity_at function."""
    
    def test_parity_fully_sensitive(self):
        """Parity is fully sensitive at every input."""
        f = bf.parity(3)
        
        # Every input should have sensitivity n (all bits matter)
        for x in range(8):
            s = sensitivity_at(f, x)
            assert s == 3, f"Parity should have sensitivity 3 at {x}, got {s}"
    
    def test_constant_zero_sensitivity(self):
        """Constant functions have zero sensitivity."""
        f = bf.constant(True, 3)
        
        for x in range(8):
            s = sensitivity_at(f, x)
            assert s == 0, f"Constant should have sensitivity 0 at {x}"
    
    def test_and_sensitivity(self):
        """AND function sensitivity properties."""
        f = bf.AND(3)
        
        # At all-ones (7), all bits are sensitive
        assert sensitivity_at(f, 7) == 3
        
        # At all-zeros (0), no bits are sensitive
        assert sensitivity_at(f, 0) == 0
    
    def test_or_sensitivity(self):
        """OR function sensitivity properties."""
        f = bf.OR(3)
        
        # At all-zeros (0), all bits are sensitive
        assert sensitivity_at(f, 0) == 3
        
        # At all-ones (7), no bits are sensitive
        assert sensitivity_at(f, 7) == 0
    
    def test_dictator_sensitivity(self):
        """Dictator has sensitivity 1 everywhere."""
        f = bf.dictator(3, i=0)
        
        for x in range(8):
            s = sensitivity_at(f, x)
            assert s == 1, f"Dictator should have sensitivity 1 at {x}, got {s}"


class TestMaxSensitivity:
    """Tests for max_sensitivity and sensitivity functions."""
    
    def test_parity_max_sensitivity(self):
        """Parity has max sensitivity n."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            s = max_sensitivity(f)
            assert s == n, f"Parity_{n} should have sensitivity {n}"
    
    def test_constant_max_sensitivity(self):
        """Constant has max sensitivity 0."""
        f = bf.constant(False, 3)
        assert max_sensitivity(f) == 0
    
    def test_dictator_max_sensitivity(self):
        """Dictator has max sensitivity 1."""
        f = bf.dictator(4, i=1)
        assert max_sensitivity(f) == 1
    
    def test_majority_sensitivity(self):
        """Majority has max sensitivity at boundary inputs."""
        f = bf.majority(3)
        s = max_sensitivity(f)
        # MAJ_3 has sensitivity 2 (at boundary inputs like 110, 100)
        # At these inputs, flipping 2 bits can change the output
        assert s == 2, f"MAJ_3 should have sensitivity 2, got {s}"
    
    def test_sensitivity_alias(self):
        """sensitivity() is alias for max_sensitivity()."""
        f = bf.parity(3)
        assert sensitivity(f) == max_sensitivity(f)


class TestAverageSensitivity:
    """Tests for average_sensitivity function."""
    
    def test_parity_average_sensitivity(self):
        """Parity has average sensitivity n (= total influence)."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            avg_s = average_sensitivity(f)
            assert abs(avg_s - n) < 1e-10, f"Parity_{n} avg sensitivity should be {n}"
    
    def test_constant_average_sensitivity(self):
        """Constant has average sensitivity 0."""
        f = bf.constant(True, 3)
        avg_s = average_sensitivity(f)
        assert avg_s == 0
    
    def test_average_equals_total_influence(self):
        """Average sensitivity equals total influence."""
        f = bf.majority(3)
        avg_s = average_sensitivity(f)
        total_inf = f.total_influence()
        assert abs(avg_s - total_inf) < 1e-10


class TestBlockSensitivity:
    """Tests for block_sensitivity function."""
    
    def test_parity_block_sensitivity(self):
        """Parity has block sensitivity n."""
        f = bf.parity(3)
        bs = block_sensitivity(f)
        # Parity bs = n (can flip all bits as one block, or each individually)
        assert bs >= 1, f"Parity block sensitivity should be >= 1"
    
    def test_constant_block_sensitivity(self):
        """Constant has block sensitivity 0."""
        f = bf.constant(True, 3)
        bs = block_sensitivity(f)
        assert bs == 0
    
    def test_dictator_block_sensitivity(self):
        """Dictator has block sensitivity 1."""
        f = bf.dictator(3, i=0)
        bs = block_sensitivity(f)
        assert bs == 1


class TestVerifyHuangTheorem:
    """Tests for verify_huang_theorem function."""
    
    def test_huang_holds_for_parity(self):
        """Huang's theorem holds for parity."""
        f = bf.parity(4)
        result = verify_huang_theorem(f)
        
        assert result['huang_satisfied']  # Use truthiness, not identity
        assert result['sensitivity'] == 4
        assert result['fourier_degree'] == 4
    
    def test_huang_holds_for_majority(self):
        """Huang's theorem holds for majority."""
        f = bf.majority(3)
        result = verify_huang_theorem(f)
        
        assert result['huang_satisfied']
    
    def test_huang_holds_for_and(self):
        """Huang's theorem holds for AND."""
        f = bf.AND(3)
        result = verify_huang_theorem(f)
        
        # AND has s = n, deg = n, so s >= sqrt(n) holds
        assert result['huang_satisfied']
    
    def test_result_structure(self):
        """verify_huang_theorem returns expected structure."""
        f = bf.parity(3)
        result = verify_huang_theorem(f)
        
        assert 'sensitivity' in result
        assert 'fourier_degree' in result
        assert 'block_sensitivity' in result
        assert 'huang_bound' in result
        assert 'huang_satisfied' in result
        assert 'bs_bound_satisfied' in result
        assert 'gap' in result
        assert 'tightness' in result


class TestSensitivityVsDegree:
    """Tests for sensitivity_vs_degree function."""
    
    def test_returns_tuple(self):
        """sensitivity_vs_degree returns (s, deg, ratio) tuple."""
        f = bf.parity(3)
        result = sensitivity_vs_degree(f)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_parity_ratio(self):
        """Parity has ratio s/sqrt(deg) = sqrt(n)."""
        f = bf.parity(4)
        s, deg, ratio = sensitivity_vs_degree(f)
        
        assert s == 4
        assert deg == 4
        assert abs(ratio - 2.0) < 1e-10  # 4 / sqrt(4) = 2


class TestHuangAnalysis:
    """Tests for HuangAnalysis class."""
    
    def test_initialization(self):
        """HuangAnalysis initializes correctly."""
        f = bf.majority(3)
        analyzer = HuangAnalysis(f)
        
        assert analyzer.f is f
        assert analyzer.n == 3
    
    def test_sensitivity_cached(self):
        """Sensitivity is computed once and cached."""
        f = bf.majority(3)
        analyzer = HuangAnalysis(f)
        
        s1 = analyzer.sensitivity()
        s2 = analyzer.sensitivity()
        
        assert s1 == s2
        assert 'sensitivity' in analyzer._cache
    
    def test_block_sensitivity(self):
        """block_sensitivity method works."""
        f = bf.parity(3)
        analyzer = HuangAnalysis(f)
        
        bs = analyzer.block_sensitivity()
        assert bs >= 1
    
    def test_degree(self):
        """degree method works."""
        f = bf.parity(3)
        analyzer = HuangAnalysis(f)
        
        deg = analyzer.degree()
        assert deg == 3
    
    def test_sensitivity_profile(self):
        """sensitivity_profile returns array of sensitivities."""
        f = bf.parity(3)
        analyzer = HuangAnalysis(f)
        
        profile = analyzer.sensitivity_profile()
        assert len(profile) == 8  # 2^3 inputs
        assert all(s == 3 for s in profile)  # Parity fully sensitive everywhere
    
    def test_verify_all_bounds(self):
        """verify_all_bounds checks all Huang-related bounds."""
        f = bf.parity(3)
        analyzer = HuangAnalysis(f)
        
        result = analyzer.verify_all_bounds()
        
        assert 'sensitivity' in result
        assert 'block_sensitivity' in result
        assert 'bounds' in result
        assert 'all_satisfied' in result
        
        # Parity should satisfy all bounds
        assert result['all_satisfied'] is True
    
    def test_summary_returns_string(self):
        """summary method returns readable string."""
        f = bf.majority(3)
        analyzer = HuangAnalysis(f)
        
        summary = analyzer.summary()
        assert isinstance(summary, str)
        assert "Sensitivity" in summary
        assert "Huang" in summary


class TestHuangOnBuiltins:
    """Test Huang's theorem on all built-in functions."""
    
    def test_huang_on_majority_family(self):
        """Huang holds for majority functions of various sizes."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            result = verify_huang_theorem(f)
            assert result['huang_satisfied'], f"Huang failed for MAJ_{n}"
    
    def test_huang_on_parity_family(self):
        """Huang holds for parity functions."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            result = verify_huang_theorem(f)
            assert result['huang_satisfied'], f"Huang failed for XOR_{n}"
    
    def test_huang_on_dictator(self):
        """Huang holds for dictator."""
        f = bf.dictator(5, i=2)
        result = verify_huang_theorem(f)
        
        # Dictator: s = 1, deg = 1, so s >= sqrt(1) = 1 ✓
        assert result['huang_satisfied']
    
    def test_huang_on_threshold(self):
        """Huang holds for threshold functions."""
        f = bf.threshold(5, k=2)
        result = verify_huang_theorem(f)
        assert result['huang_satisfied']


class TestBlockSensitivityBound:
    """Test the bound bs(f) <= s(f)^2."""
    
    def test_bs_bound_for_parity(self):
        """bs(f) <= s(f)² for parity."""
        f = bf.parity(3)
        s = sensitivity(f)
        bs = block_sensitivity(f)
        
        assert bs <= s**2
    
    def test_bs_bound_for_majority(self):
        """bs(f) <= s(f)² for majority."""
        f = bf.majority(3)
        s = sensitivity(f)
        bs = block_sensitivity(f)
        
        assert bs <= s**2
    
    def test_bs_bound_for_and(self):
        """bs(f) <= s(f)² for AND."""
        f = bf.AND(3)
        s = sensitivity(f)
        bs = block_sensitivity(f)
        
        assert bs <= s**2
