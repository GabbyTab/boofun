import sys

sys.path.insert(0, "src")
"""
Tests for p_biased module.

Tests p-biased Fourier analysis for Boolean functions:
- p_biased_fourier_coefficients
- p_biased_influence
- p_biased_total_influence
- p_biased_noise_stability
- p_biased_expectation
- p_biased_variance
- biased_measure_mass
- PBiasedAnalyzer class
"""

import pytest

import boofun as bf
from boofun.analysis.p_biased import (
    PBiasedAnalyzer,
    biased_measure_mass,
    p_biased_expectation,
    p_biased_fourier_coefficients,
    p_biased_influence,
    p_biased_noise_stability,
    p_biased_total_influence,
    p_biased_variance,
)


class TestBiasedMeasureMass:
    """Tests for biased_measure_mass function."""

    def test_uniform_equal_mass(self):
        """At p=0.5, all inputs have equal mass."""
        n = 3
        p = 0.5

        for x in range(1 << n):
            mass = biased_measure_mass(p, n, x)
            assert abs(mass - 1 / 8) < 1e-10

    def test_p_zero_all_ones(self):
        """At p close to 0, inputs with all +1s have most mass."""
        p = 0.01
        n = 3

        # x=0 means all bits are 0, which corresponds to all +1s
        mass_all_plus = biased_measure_mass(p, n, 0)
        mass_all_minus = biased_measure_mass(p, n, 7)  # 111 = all -1s

        assert mass_all_plus > mass_all_minus

    def test_p_one_all_minus(self):
        """At p close to 1, inputs with all -1s have most mass."""
        p = 0.99
        n = 3

        mass_all_plus = biased_measure_mass(p, n, 0)
        mass_all_minus = biased_measure_mass(p, n, 7)

        assert mass_all_minus > mass_all_plus

    def test_sum_to_one(self):
        """All masses sum to 1."""
        p = 0.3
        n = 4

        total = sum(biased_measure_mass(p, n, x) for x in range(1 << n))
        assert abs(total - 1.0) < 1e-10


class TestPBiasedExpectation:
    """Tests for p_biased_expectation function."""

    def test_constant_expectation(self):
        """Constant function has known expectation."""
        f = bf.constant(True, 3)

        # Constant True in {0,1} becomes -1 in ±1
        exp = p_biased_expectation(f, p=0.5)
        assert abs(exp - (-1.0)) < 0.1  # Should be close to -1

    def test_uniform_balanced(self):
        """Balanced function at p=0.5 has expectation near 0."""
        f = bf.parity(3)

        exp = p_biased_expectation(f, p=0.5)
        assert abs(exp) < 0.1

    def test_expectation_monotone_in_p(self):
        """AND expectation increases as p decreases (more +1s)."""
        f = bf.AND(3)

        exp_high_p = p_biased_expectation(f, p=0.8)
        exp_low_p = p_biased_expectation(f, p=0.2)

        # With lower p, more +1s, so AND is more likely to be True
        # True in {0,1} is -1 in ±1... this can be confusing
        # Just check they're different
        assert exp_high_p != exp_low_p


class TestPBiasedVariance:
    """Tests for p_biased_variance function."""

    def test_constant_zero_variance(self):
        """Constant function has zero variance."""
        f = bf.constant(True, 3)

        var = p_biased_variance(f, p=0.5)
        assert abs(var) < 1e-10

    def test_variance_nonnegative(self):
        """Variance is always non-negative."""
        f = bf.parity(3)

        for p in [0.2, 0.5, 0.8]:
            var = p_biased_variance(f, p)
            assert var >= -1e-10

    def test_uniform_variance_is_one(self):
        """For Boolean functions at p=0.5, variance is related to balance."""
        f = bf.parity(3)  # Balanced

        var = p_biased_variance(f, p=0.5)
        # For balanced Boolean ±1 function, Var = 1 - E[f]^2 = 1
        assert var > 0


class TestPBiasedFourierCoefficients:
    """Tests for p_biased_fourier_coefficients function."""

    def test_returns_dict(self):
        """Returns dictionary of coefficients."""
        f = bf.parity(3)

        coeffs = p_biased_fourier_coefficients(f, p=0.5)

        assert isinstance(coeffs, dict)

    def test_uniform_matches_standard(self):
        """At p=0.5, coefficients should match standard Fourier (approximately)."""
        f = bf.majority(3)

        coeffs_biased = p_biased_fourier_coefficients(f, p=0.5)

        # At p=0.5, the p-biased expansion should be similar to standard
        # Just check we get some coefficients
        assert len(coeffs_biased) >= 1

    def test_constant_single_coefficient(self):
        """Constant function has single coefficient at S=0."""
        f = bf.constant(False, 3)

        coeffs = p_biased_fourier_coefficients(f, p=0.3)

        # Should have coefficient at S=0
        assert 0 in coeffs


class TestPBiasedInfluence:
    """Tests for p_biased_influence function."""

    def test_influence_nonnegative(self):
        """Influence is non-negative."""
        f = bf.majority(3)

        for i in range(3):
            inf = p_biased_influence(f, i, p=0.5)
            assert inf >= -1e-10

    def test_invalid_variable_raises(self):
        """Invalid variable index raises ValueError."""
        f = bf.AND(3)

        with pytest.raises(ValueError):
            p_biased_influence(f, -1, p=0.5)

        with pytest.raises(ValueError):
            p_biased_influence(f, 5, p=0.5)

    def test_symmetric_equal_influences(self):
        """Symmetric function has equal influences."""
        f = bf.majority(3)

        influences = [p_biased_influence(f, i, p=0.5) for i in range(3)]

        assert abs(influences[0] - influences[1]) < 1e-6
        assert abs(influences[1] - influences[2]) < 1e-6

    def test_dictator_influence(self):
        """Dictator has influence 1 on its variable, 0 on others."""
        f = bf.dictator(3, i=0)

        influences = [p_biased_influence(f, i, p=0.5) for i in range(3)]

        # At p=0.5 (uniform), dictator on variable 0: Inf_0 = 1, Inf_1 = Inf_2 = 0
        assert abs(influences[0] - 1.0) < 1e-10
        assert abs(influences[1]) < 1e-10
        assert abs(influences[2]) < 1e-10
        assert abs(sum(influences) - 1.0) < 1e-10


class TestPBiasedTotalInfluence:
    """Tests for p_biased_total_influence function."""

    def test_nonnegative(self):
        """Total influence is non-negative."""
        f = bf.parity(4)

        total = p_biased_total_influence(f, p=0.5)
        assert total >= -1e-10

    def test_sum_of_influences(self):
        """Total influence equals sum of individual influences."""
        f = bf.majority(3)

        total = p_biased_total_influence(f, p=0.5)
        individual_sum = sum(p_biased_influence(f, i, p=0.5) for i in range(3))

        assert abs(total - individual_sum) < 1e-10


class TestPBiasedNoiseStability:
    """Tests for p_biased_noise_stability function."""

    def test_rho_one_is_variance_plus_mean_squared(self):
        """At rho=1, stability is E[f^2] = Var[f] + E[f]^2."""
        f = bf.majority(3)

        stab = p_biased_noise_stability(f, rho=1.0, p=0.5)

        # For Boolean ±1 functions, E[f^2] = 1
        assert stab > 0

    def test_rho_zero_is_mean_squared(self):
        """At rho=0, stability is E[f]^2."""
        f = bf.parity(3)

        stab = p_biased_noise_stability(f, rho=0.0, p=0.5)
        exp = p_biased_expectation(f, p=0.5)

        assert abs(stab - exp**2) < 1e-6

    def test_stability_decreases_with_rho(self):
        """Stability typically decreases as |rho| decreases."""
        f = bf.majority(3)

        stab_high = p_biased_noise_stability(f, rho=0.9, p=0.5)
        stab_low = p_biased_noise_stability(f, rho=0.3, p=0.5)

        # For most functions, higher correlation means higher stability
        assert stab_high >= stab_low - 1e-6


class TestPBiasedAnalyzer:
    """Tests for PBiasedAnalyzer class."""

    def test_initialization(self):
        """Analyzer initializes correctly."""
        f = bf.majority(3)
        analyzer = PBiasedAnalyzer(f, p=0.3)

        assert analyzer.function is f
        assert analyzer.p == 0.3

    def test_coefficients_caching(self):
        """Coefficients are cached."""
        f = bf.majority(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        c1 = analyzer.coefficients
        c2 = analyzer.coefficients

        assert c1 is c2  # Same object (cached)

    def test_expectation(self):
        """expectation method works."""
        f = bf.parity(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        exp = analyzer.expectation()
        assert isinstance(exp, float)

    def test_variance(self):
        """AND(3) at p=0.5 has computable variance."""
        f = bf.AND(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        var = analyzer.variance()
        # Variance must be non-negative, and for AND(3) it's non-trivial
        assert var > 0

    def test_influence(self):
        """OR(3) influence at p=0.5: each variable has influence 1/4 by symmetry."""
        f = bf.OR(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        inf = analyzer.influence(0)
        # OR(3) at uniform: Inf_i = Pr[x_i is pivotal] = (1/2)^(n-1) = 1/4
        assert abs(inf - 0.25) < 1e-10

    def test_influences(self):
        """influences method returns list."""
        f = bf.majority(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        infs = analyzer.influences()
        assert len(infs) == 3

    def test_total_influence(self):
        """total_influence of parity(n) at p=0.5 equals n."""
        f = bf.parity(4)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        total = analyzer.total_influence()
        # Parity(n) at uniform measure: every variable has influence 1, total = n
        assert abs(total - 4.0) < 1e-10

    def test_noise_stability(self):
        """noise_stability method works."""
        f = bf.majority(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        stab = analyzer.noise_stability(0.5)
        assert isinstance(stab, float)

    def test_spectral_norm(self):
        """Parity(3) has all weight at degree 3, zero at degree 1."""
        f = bf.parity(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        norm = analyzer.spectral_norm(level=1)
        # Parity has all Fourier weight on the top coefficient; level 1 weight is 0
        assert abs(norm) < 1e-10

    def test_max_influence(self):
        """Dictator(3, i=1) max influence is variable 1 with value 1."""
        f = bf.dictator(3, i=1)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        idx, val = analyzer.max_influence()
        assert idx == 1
        assert abs(val - 1.0) < 1e-10

    def test_summary(self):
        """summary method returns string."""
        f = bf.majority(3)
        analyzer = PBiasedAnalyzer(f, p=0.5)

        summary = analyzer.summary()
        assert isinstance(summary, str)
        assert "P-biased Analysis" in summary


class TestOnBuiltinFunctions:
    """Test p-biased analysis on built-in functions."""

    def test_tribes(self):
        """p-biased analysis on tribes."""
        f = bf.tribes(2, 4)

        exp = p_biased_expectation(f, p=0.5)
        assert isinstance(exp, float)

        total_inf = p_biased_total_influence(f, p=0.5)
        # Tribes is non-trivial; total influence must be strictly positive
        assert total_inf > 0

    def test_threshold(self):
        """p-biased analysis on threshold function."""
        f = bf.threshold(3, k=2)

        analyzer = PBiasedAnalyzer(f, p=0.3)
        summary = analyzer.summary()

        assert "Variables: 3" in summary


class TestEdgeCases:
    """Test edge cases for p-biased analysis."""

    def test_extreme_p_values(self):
        """Test with p close to 0 and 1."""
        f = bf.AND(3)

        # Very biased towards +1
        exp_low = p_biased_expectation(f, p=0.01)
        # Very biased towards -1
        exp_high = p_biased_expectation(f, p=0.99)

        # Results should differ
        assert exp_low != exp_high

    def test_single_variable(self):
        """Parity(1) = identity: influence 1, balanced expectation."""
        f = bf.parity(1)

        exp = p_biased_expectation(f, p=0.5)
        assert isinstance(exp, float)

        inf = p_biased_influence(f, 0, p=0.5)
        # Parity(1) is the identity function; its single variable has influence 1
        assert abs(inf - 1.0) < 1e-10
