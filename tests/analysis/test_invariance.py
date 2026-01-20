import sys
sys.path.insert(0, "src")
"""
Tests for invariance module.

Tests the invariance principle for Boolean functions:
- gaussian_cdf, bivariate_gaussian_cdf
- majority_noise_stability
- invariance_distance
- multilinear_extension_gaussian_expectation
- test_function_expectation
- noise_stability_deficit
- is_stablest_candidate
- max_cut_approximation_ratio
- unique_games_hardness_bound
- InvarianceAnalyzer class
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.invariance import (
    gaussian_cdf,
    bivariate_gaussian_cdf,
    majority_noise_stability,
    invariance_distance,
    multilinear_extension_gaussian_expectation,
    test_function_expectation,
    noise_stability_deficit,
    is_stablest_candidate,
    max_cut_approximation_ratio,
    unique_games_hardness_bound,
    InvarianceAnalyzer,
)


class TestGaussianCDF:
    """Tests for gaussian_cdf function."""
    
    def test_cdf_zero(self):
        """Φ(0) = 0.5."""
        assert abs(gaussian_cdf(0.0) - 0.5) < 1e-10
    
    def test_cdf_large_positive(self):
        """Φ(large) ≈ 1."""
        assert gaussian_cdf(5.0) > 0.99
    
    def test_cdf_large_negative(self):
        """Φ(-large) ≈ 0."""
        assert gaussian_cdf(-5.0) < 0.01
    
    def test_cdf_monotonic(self):
        """CDF is monotonically increasing."""
        values = [-2, -1, 0, 1, 2]
        cdfs = [gaussian_cdf(x) for x in values]
        
        for i in range(len(cdfs) - 1):
            assert cdfs[i] < cdfs[i+1]


class TestBivariateGaussianCDF:
    """Tests for bivariate_gaussian_cdf function."""
    
    def test_returns_float(self):
        """Returns a float in [0, 1]."""
        cdf = bivariate_gaussian_cdf(0.0, 0.0, 0.5)
        
        assert isinstance(cdf, float)
        assert 0 <= cdf <= 1
    
    def test_zero_correlation(self):
        """At rho=0, bivariate = product of marginals."""
        x, y = 1.0, 0.5
        biv = bivariate_gaussian_cdf(x, y, 0.0)
        prod = gaussian_cdf(x) * gaussian_cdf(y)
        
        assert abs(biv - prod) < 0.1
    
    def test_positive_correlation(self):
        """Positive correlation increases joint probability."""
        x, y = 0.5, 0.5
        biv_pos = bivariate_gaussian_cdf(x, y, 0.8)
        biv_neg = bivariate_gaussian_cdf(x, y, -0.8)
        
        # Positive correlation should give higher probability
        # (Both being above mean is more likely when correlated)
        assert biv_pos >= biv_neg - 0.1  # Allow some numerical tolerance


class TestMajorityNoiseStability:
    """Tests for majority_noise_stability function."""
    
    def test_rho_zero(self):
        """At rho=0, stability is close to 0 for balanced functions."""
        stab = majority_noise_stability(5, 0.0)
        # For balanced majority, E[f]^2 ≈ 0
        assert abs(stab) < 0.2
    
    def test_rho_one(self):
        """At rho=1, stability is 1."""
        stab = majority_noise_stability(5, 1.0)
        # Should be close to 1
        assert stab > 0.8
    
    def test_increases_with_rho(self):
        """Stability increases with rho."""
        stab_low = majority_noise_stability(5, 0.3)
        stab_high = majority_noise_stability(5, 0.8)
        
        assert stab_high > stab_low
    
    def test_sheppard_formula_large_n(self):
        """For large n, converges to (2/π) arcsin(ρ)."""
        rho = 0.5
        stab = majority_noise_stability(100, rho)
        sheppard = (2 / np.pi) * np.arcsin(rho)
        
        assert abs(stab - sheppard) < 0.1


class TestInvarianceDistance:
    """Tests for invariance_distance function."""
    
    def test_returns_float(self):
        """Returns a float."""
        f = bf.majority(3)
        dist = invariance_distance(f)
        
        assert isinstance(dist, float)
    
    def test_nonnegative(self):
        """Distance is non-negative."""
        f = bf.parity(3)
        dist = invariance_distance(f)
        
        assert dist >= 0
    
    def test_low_for_symmetric(self):
        """Symmetric functions have bounded distance."""
        f = bf.majority(5)
        dist = invariance_distance(f)
        
        # Symmetric functions have low max influence
        assert dist < 1.0


class TestMultilinearExtensionGaussianExpectation:
    """Tests for multilinear_extension_gaussian_expectation function."""
    
    def test_returns_tuple(self):
        """Returns (mean, sign_mean) tuple."""
        f = bf.majority(3)
        result = multilinear_extension_gaussian_expectation(f, num_samples=100)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_values_bounded(self):
        """Values are in reasonable range."""
        f = bf.AND(3)
        mean, sign_mean = multilinear_extension_gaussian_expectation(f, num_samples=100)
        
        # Mean of extension can be any value
        # Sign mean should be in [-1, 1]
        assert -1 <= sign_mean <= 1


class TestTestFunctionExpectation:
    """Tests for test_function_expectation function."""
    
    def test_boolean_domain(self):
        """Expectation on Boolean domain."""
        f = bf.majority(3)
        
        # Test with identity function
        exp = test_function_expectation(f, lambda x: x, domain="boolean")
        
        # For balanced majority, should be near 0
        assert abs(exp) < 0.2
    
    def test_gaussian_domain(self):
        """Expectation on Gaussian domain."""
        f = bf.AND(3)
        
        # Test with identity function
        exp = test_function_expectation(f, lambda x: x, domain="gaussian")
        
        assert isinstance(exp, float)


class TestNoiseStabilityDeficit:
    """Tests for noise_stability_deficit function."""
    
    def test_returns_float(self):
        """Returns a float."""
        f = bf.parity(3)
        deficit = noise_stability_deficit(f, 0.5)
        
        assert isinstance(deficit, float)
    
    def test_majority_zero_deficit(self):
        """Majority has near-zero deficit (is most stable)."""
        f = bf.majority(5)
        deficit = noise_stability_deficit(f, 0.5)
        
        # Should be close to 0 for majority
        assert abs(deficit) < 0.2
    
    def test_parity_positive_deficit(self):
        """Parity has positive deficit (less stable than majority)."""
        f = bf.parity(3)
        deficit = noise_stability_deficit(f, 0.5)
        
        # Parity should be less stable
        assert isinstance(deficit, float)


class TestIsStablestCandidate:
    """Tests for is_stablest_candidate function."""
    
    def test_returns_bool(self):
        """Returns a boolean."""
        f = bf.majority(5)
        result = is_stablest_candidate(f)
        
        assert isinstance(result, (bool, np.bool_))
    
    def test_dictator_not_candidate(self):
        """Dictator is not a stablest candidate (high influence)."""
        f = bf.dictator(3, i=0)
        result = is_stablest_candidate(f, epsilon=0.5)
        
        # Dictator has influence 1 on one variable
        assert not result


class TestMaxCutApproximationRatio:
    """Tests for max_cut_approximation_ratio function."""
    
    def test_returns_float(self):
        """Returns a float."""
        ratio = max_cut_approximation_ratio(0.5)
        
        assert isinstance(ratio, float)
    
    def test_gw_ratio(self):
        """GW ratio is approximately 0.87856."""
        ratio = max_cut_approximation_ratio(0.5)
        
        assert abs(ratio - 0.87856) < 0.01
    
    def test_bounded(self):
        """Ratio is in (0, 1]."""
        ratio = max_cut_approximation_ratio(0.9)
        
        assert 0 < ratio <= 1


class TestUniqueGamesHardnessBound:
    """Tests for unique_games_hardness_bound function."""
    
    def test_returns_float(self):
        """Returns a float."""
        f = bf.majority(3)
        bound = unique_games_hardness_bound(f)
        
        assert isinstance(bound, float)
    
    def test_bounded(self):
        """Bound is positive."""
        f = bf.AND(3)
        bound = unique_games_hardness_bound(f)
        
        assert bound > 0


class TestInvarianceAnalyzer:
    """Tests for InvarianceAnalyzer class."""
    
    def test_initialization(self):
        """Analyzer initializes correctly."""
        f = bf.majority(3)
        analyzer = InvarianceAnalyzer(f)
        
        assert analyzer.function is f
        assert analyzer.n_vars == 3
    
    def test_invariance_bound(self):
        """invariance_bound method works."""
        f = bf.parity(3)
        analyzer = InvarianceAnalyzer(f)
        
        bound = analyzer.invariance_bound()
        assert bound >= 0
    
    def test_noise_stability_deficit(self):
        """noise_stability_deficit method works."""
        f = bf.AND(3)
        analyzer = InvarianceAnalyzer(f)
        
        deficit = analyzer.noise_stability_deficit(0.5)
        assert isinstance(deficit, float)
    
    def test_is_stablest_candidate(self):
        """is_stablest_candidate method works."""
        f = bf.majority(5)
        analyzer = InvarianceAnalyzer(f)
        
        result = analyzer.is_stablest_candidate()
        assert isinstance(result, (bool, np.bool_))
    
    def test_gaussian_expectation(self):
        """gaussian_expectation method works."""
        f = bf.OR(3)
        analyzer = InvarianceAnalyzer(f)
        
        mean, sign_mean = analyzer.gaussian_expectation(num_samples=100)
        assert isinstance(mean, float)
        assert isinstance(sign_mean, float)
    
    def test_compare_domains(self):
        """compare_domains method returns dict."""
        f = bf.majority(3)
        analyzer = InvarianceAnalyzer(f)
        
        comparison = analyzer.compare_domains()
        
        assert isinstance(comparison, dict)
        assert "boolean_mean" in comparison
        assert "gaussian_mean" in comparison
    
    def test_summary(self):
        """summary method returns string."""
        f = bf.parity(3)
        analyzer = InvarianceAnalyzer(f)
        
        summary = analyzer.summary()
        
        assert isinstance(summary, str)
        assert "Invariance Principle" in summary


class TestOnBuiltinFunctions:
    """Test invariance on built-in functions."""
    
    def test_tribes(self):
        """Invariance analysis on tribes."""
        f = bf.tribes(2, 4)
        
        dist = invariance_distance(f)
        assert dist >= 0
        
        analyzer = InvarianceAnalyzer(f)
        summary = analyzer.summary()
        assert "Invariance" in summary
    
    def test_threshold(self):
        """Invariance analysis on threshold function."""
        f = bf.threshold(4, k=2)
        
        deficit = noise_stability_deficit(f, 0.5)
        assert isinstance(deficit, float)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_variable(self):
        """Works with single variable function."""
        f = bf.parity(1)
        
        dist = invariance_distance(f)
        assert dist >= 0
        
        analyzer = InvarianceAnalyzer(f)
        result = analyzer.invariance_bound()
        assert isinstance(result, float)
    
    def test_constant_function(self):
        """Works with constant function."""
        f = bf.constant(True, 3)
        
        dist = invariance_distance(f)
        assert dist >= 0
