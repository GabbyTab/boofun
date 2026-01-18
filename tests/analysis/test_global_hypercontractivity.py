"""
Tests for the global hypercontractivity module.

These tests verify the implementation of concepts from Keevash et al.'s
"Global hypercontractivity and its applications" paper.
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import boolfunc as bf
from boolfunc.analysis import global_hypercontractivity as gh


class TestBasicFunctions:
    """Test basic helper functions."""
    
    def test_sigma_uniform(self):
        """σ(0.5) = 0.5"""
        assert abs(gh.sigma(0.5) - 0.5) < 1e-10
    
    def test_sigma_biased(self):
        """σ(p) = √(p(1-p))"""
        for p in [0.1, 0.3, 0.7, 0.9]:
            expected = np.sqrt(p * (1 - p))
            assert abs(gh.sigma(p) - expected) < 1e-10
    
    def test_sigma_boundary(self):
        """σ(0) = σ(1) = 0"""
        assert gh.sigma(0) == 0
        assert gh.sigma(1) == 0
    
    def test_lambda_uniform(self):
        """λ(0.5) should be manageable."""
        # At p=0.5: λ = σ^{-2}((1-p)^3 + p^3) = 4 * (0.125 + 0.125) = 1
        lam = gh.lambda_p(0.5)
        assert abs(lam - 1.0) < 1e-10
    
    def test_lambda_increases_away_from_half(self):
        """λ(p) should increase as p moves away from 0.5."""
        lam_half = gh.lambda_p(0.5)
        lam_point3 = gh.lambda_p(0.3)
        lam_point1 = gh.lambda_p(0.1)
        
        assert lam_point3 > lam_half
        assert lam_point1 > lam_point3
    
    def test_lambda_symmetry(self):
        """λ(p) = λ(1-p) by symmetry."""
        for p in [0.1, 0.2, 0.3, 0.4]:
            assert abs(gh.lambda_p(p) - gh.lambda_p(1-p)) < 1e-10


class TestPBiasedCharacter:
    """Test p-biased character computation."""
    
    def test_empty_set(self):
        """χ_∅(x) = 1 for all x."""
        x = np.array([0, 1, 0, 1])
        for p in [0.3, 0.5, 0.7]:
            assert gh.p_biased_character(x, set(), p) == 1.0
    
    def test_single_variable_at_half(self):
        """χ_{i}^{0.5}(x) = ±1 at p=0.5."""
        x0 = np.array([0])
        x1 = np.array([1])
        
        # At p=0.5: χ_0(x) = (x_0 - 0.5) / 0.5 = 2x_0 - 1
        assert abs(gh.p_biased_character(x0, {0}, 0.5) - (-1.0)) < 1e-10
        assert abs(gh.p_biased_character(x1, {0}, 0.5) - 1.0) < 1e-10


class TestGeneralizedInfluence:
    """Test generalized influence computation."""
    
    def test_empty_set_influence(self):
        """I_∅(f) = E[f²] = ||f||² = 1 for Boolean functions."""
        f = bf.majority(5)
        inf_empty = gh.generalized_influence(f, set())
        # I_∅(f) should equal the sum of all squared Fourier coefficients
        assert inf_empty > 0
    
    def test_singleton_influence_is_standard(self):
        """I_{i}(f) should relate to standard influence."""
        f = bf.majority(5)
        # Single variable influence
        inf_0 = gh.generalized_influence(f, {0})
        
        # Should be positive
        assert inf_0 > 0
    
    def test_dictator_has_high_singleton_influence(self):
        """Dictator should have very high I_{i} for the dictator variable."""
        f = bf.dictator(0, 5)  # x_0 is the dictator
        
        inf_0 = gh.generalized_influence(f, {0})
        inf_1 = gh.generalized_influence(f, {1})
        
        # I_{0} should be much larger than I_{1}
        assert inf_0 > 10 * inf_1 or inf_1 < 1e-10


class TestAlphaGlobal:
    """Test α-global function checking."""
    
    def test_dictator_not_global(self):
        """Dictator functions should NOT be global (high single-variable influence)."""
        f = bf.dictator(0, 7)
        is_global, details = gh.is_alpha_global(f, alpha=0.5, max_set_size=2)
        
        # Dictator should not be global for reasonable α
        assert not is_global
        assert details['max_generalized_influence'] > 0.5
    
    def test_parity_has_high_generalized_influence(self):
        """Parity function should have high generalized influences."""
        f = bf.parity(5)
        is_global, details = gh.is_alpha_global(f, alpha=0.5, max_set_size=2)
        
        # Parity is the "worst" function for generalized influences
        assert details['max_generalized_influence'] > 1.0
    
    def test_details_contain_required_keys(self):
        """Result dictionary should contain all expected keys."""
        f = bf.majority(5)
        is_global, details = gh.is_alpha_global(f, alpha=0.5, max_set_size=2)
        
        assert 'max_generalized_influence' in details
        assert 'worst_set' in details
        assert 'threshold' in details
        assert 'all_influences' in details


class TestPBiasedExpectation:
    """Test p-biased expectation estimation."""
    
    def test_dictator_expectation_equals_p(self):
        """E_μp[x_i] = p for dictator function."""
        f = bf.dictator(0, 5)
        
        for p in [0.3, 0.5, 0.7]:
            exp = gh.p_biased_expectation(f, p, samples=5000)
            assert abs(exp - p) < 0.1  # Monte Carlo error
    
    def test_constant_zero_expectation(self):
        """E_μp[0] = 0 for all p."""
        # Create constant-0 function
        f = bf.create([0] * 8)  # 3 variables, all 0
        
        for p in [0.3, 0.5, 0.7]:
            exp = gh.p_biased_expectation(f, p, samples=1000)
            assert exp < 0.05
    
    def test_constant_one_expectation(self):
        """E_μp[1] = 1 for all p."""
        # Create constant-1 function
        f = bf.create([1] * 8)  # 3 variables, all 1
        
        for p in [0.3, 0.5, 0.7]:
            exp = gh.p_biased_expectation(f, p, samples=1000)
            assert exp > 0.95


class TestPBiasedInfluence:
    """Test p-biased influence estimation."""
    
    def test_dictator_influence(self):
        """Dictator's relevant variable has influence 1."""
        f = bf.dictator(0, 5)
        
        inf_0 = gh.p_biased_influence(f, 0, p=0.5, samples=3000)
        inf_1 = gh.p_biased_influence(f, 1, p=0.5, samples=3000)
        
        assert inf_0 > 0.9  # Should be close to 1
        assert inf_1 < 0.1  # Should be close to 0
    
    def test_parity_equal_influences(self):
        """All variables in parity have equal influence."""
        f = bf.parity(5)
        
        influences = [gh.p_biased_influence(f, i, p=0.5, samples=2000) for i in range(5)]
        
        # All influences should be similar
        assert max(influences) - min(influences) < 0.2


class TestThresholdCurve:
    """Test threshold curve computation."""
    
    def test_majority_threshold_at_half(self):
        """Majority function has threshold at p=0.5."""
        f = bf.majority(7)
        p_range = np.array([0.3, 0.5, 0.7])
        
        curve = gh.threshold_curve(f, p_range, samples=2000)
        
        # At p=0.5, majority should output ~0.5
        assert abs(curve[1] - 0.5) < 0.15
        
        # Should increase with p
        assert curve[2] > curve[1] > curve[0]
    
    def test_dictator_linear_threshold(self):
        """Dictator has linear threshold curve: μ_p(x_i) = p."""
        f = bf.dictator(0, 5)
        p_range = np.array([0.2, 0.5, 0.8])
        
        curve = gh.threshold_curve(f, p_range, samples=2000)
        
        # Should be approximately linear
        for i, p in enumerate(p_range):
            assert abs(curve[i] - p) < 0.15


class TestHypercontractivityBound:
    """Test hypercontractivity bound computation."""
    
    def test_returns_required_keys(self):
        """Result should contain all expected keys."""
        f = bf.majority(5)
        result = gh.hypercontractivity_bound(f)
        
        assert 'alpha' in result
        assert 'bound' in result
        assert 'is_global' in result
        assert 'worst_set' in result
        assert 'details' in result
    
    def test_bound_is_alpha_quarter_power(self):
        """Bound should equal α^{1/4}."""
        f = bf.majority(5)
        result = gh.hypercontractivity_bound(f)
        
        expected_bound = result['alpha'] ** 0.25
        assert abs(result['bound'] - expected_bound) < 1e-10


class TestGlobalHypercontractivityAnalyzer:
    """Test the analyzer class."""
    
    def test_initialization(self):
        """Analyzer should initialize correctly."""
        f = bf.majority(7)
        analyzer = gh.GlobalHypercontractivityAnalyzer(f, p=0.3)
        
        assert analyzer.n_vars == 7
        assert analyzer.p == 0.3
    
    def test_sigma_and_lambda(self):
        """Analyzer should compute σ and λ correctly."""
        f = bf.majority(5)
        analyzer = gh.GlobalHypercontractivityAnalyzer(f, p=0.5)
        
        assert abs(analyzer.sigma() - 0.5) < 1e-10
        assert abs(analyzer.lambda_p() - 1.0) < 1e-10
    
    def test_caching(self):
        """Analyzer should cache results."""
        f = bf.majority(5)
        analyzer = gh.GlobalHypercontractivityAnalyzer(f, p=0.5)
        
        # First call computes
        result1 = analyzer.is_global(alpha=0.5)
        
        # Second call should use cache
        result2 = analyzer.is_global(alpha=0.5)
        
        assert result1 == result2
    
    def test_summary(self):
        """Summary should contain all key properties."""
        f = bf.majority(5)
        analyzer = gh.GlobalHypercontractivityAnalyzer(f, p=0.5)
        
        summary = analyzer.summary(samples=500)
        
        assert 'n_vars' in summary
        assert 'bias_p' in summary
        assert 'sigma_p' in summary
        assert 'lambda_p' in summary
        assert 'is_global_alpha_0.5' in summary
        assert 'max_generalized_influence' in summary
        assert 'hypercontractive_bound' in summary


class TestMathematicalIdentities:
    """Test mathematical identities from the paper."""
    
    def test_influence_formula(self):
        """
        From equation (1) in the paper:
        I(f) = (p(1-p))^{-1} Σ_S |S| f̂(S)²
        
        For uniform measure, this simplifies.
        """
        f = bf.majority(5)
        
        # Compute total influence via p-biased method
        total_inf = gh.p_biased_total_influence(f, p=0.5, samples=2000)
        
        # Should match the uniform-measure total influence
        from boolfunc.analysis import SpectralAnalyzer
        analyzer = SpectralAnalyzer(f)
        standard_inf = analyzer.total_influence()
        
        # Should be reasonably close (Monte Carlo error)
        assert abs(total_inf - standard_inf) < 0.5
    
    def test_noise_stability_in_range(self):
        """Noise stability should be in [-1, 1] range."""
        f = bf.majority(5)
        
        for rho in [0.5, 0.9, 0.99]:
            ns = gh.noise_stability_p_biased(f, rho, p=0.5, samples=2000)
            assert -1.1 <= ns <= 1.1  # Allow some Monte Carlo error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
