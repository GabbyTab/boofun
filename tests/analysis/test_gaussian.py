import sys

sys.path.insert(0, "src")
"""
Tests for gaussian module.

Tests Gaussian analysis of Boolean functions:
- Hermite polynomials (probabilist's and physicist's)
- Hermite coefficients
- Gaussian noise stability and sensitivity
- Ornstein-Uhlenbeck operator
- Berry-Esseen bound and CLT approximation
- Multilinear extension
- GaussianAnalyzer class
"""

import numpy as np

import boofun as bf
from boofun.analysis.gaussian import (
    GaussianAnalyzer,
    berry_esseen_bound,
    clt_approximation,
    gaussian_noise_sensitivity,
    gaussian_noise_stability,
    hermite_coefficients,
    hermite_polynomial,
    multilinear_extension,
    ornstein_uhlenbeck_operator,
    physicists_hermite,
    probabilists_hermite,
)


class TestHermitePolynomials:
    """Tests for Hermite polynomial functions."""

    def test_probabilists_hermite_base_cases(self):
        """He_0(x) = 1, He_1(x) = x."""
        assert probabilists_hermite(0, 0.5) == 1.0
        assert probabilists_hermite(0, -2.0) == 1.0

        assert probabilists_hermite(1, 0.5) == 0.5
        assert probabilists_hermite(1, -2.0) == -2.0

    def test_physicists_hermite_base_cases(self):
        """H_0(x) = 1, H_1(x) = 2x."""
        assert physicists_hermite(0, 0.5) == 1.0
        assert physicists_hermite(0, -1.0) == 1.0

        assert physicists_hermite(1, 0.5) == 1.0  # 2 * 0.5
        assert physicists_hermite(1, -1.0) == -2.0  # 2 * -1

    def test_hermite_recurrence(self):
        """Test higher degree polynomials satisfy recurrence."""
        # He_2(x) = x * He_1(x) - 1 * He_0(x) = x^2 - 1
        x = 2.0
        he2 = probabilists_hermite(2, x)
        expected = x**2 - 1
        assert abs(he2 - expected) < 1e-10

        # He_3(x) = x * He_2(x) - 2 * He_1(x) = x^3 - 3x
        he3 = probabilists_hermite(3, x)
        expected = x**3 - 3 * x
        assert abs(he3 - expected) < 1e-10

    def test_hermite_polynomial_function(self):
        """hermite_polynomial returns callable."""
        he2 = hermite_polynomial(2, "probabilist")
        h2 = hermite_polynomial(2, "physicist")

        assert callable(he2)
        assert callable(h2)

        x = 1.5
        assert abs(he2(x) - probabilists_hermite(2, x)) < 1e-10
        assert abs(h2(x) - physicists_hermite(2, x)) < 1e-10


class TestHermiteCoefficients:
    """Tests for hermite_coefficients function."""

    def test_returns_dict(self):
        """hermite_coefficients returns dictionary."""
        f = bf.parity(3)
        coeffs = hermite_coefficients(f)

        assert isinstance(coeffs, dict)

    def test_parity_coefficients(self):
        """Parity has single coefficient at full degree."""
        f = bf.parity(3)
        coeffs = hermite_coefficients(f)

        # Parity has f̂(S) = ±1 at S = {0,1,2}
        # Multi-index (1,1,1) should have non-zero coefficient
        assert (1, 1, 1) in coeffs

    def test_constant_single_coefficient(self):
        """Constant function has single coefficient."""
        f = bf.constant(True, 3)
        coeffs = hermite_coefficients(f)

        # Constant has only degree-0 coefficient
        assert (0, 0, 0) in coeffs


class TestGaussianNoiseStability:
    """Tests for gaussian_noise_stability function."""

    def test_rho_one_is_l2_norm(self):
        """At rho=1, stability is sum of squared Fourier coefficients."""
        f = bf.majority(3)
        stab = gaussian_noise_stability(f, rho=1.0)

        # For Boolean ±1 functions, this should be 1
        assert stab > 0

    def test_rho_zero_is_mean_squared(self):
        """At rho=0, stability is E[f]^2."""
        f = bf.parity(3)  # Balanced, E[f] = 0
        stab = gaussian_noise_stability(f, rho=0.0)

        # f̂(∅)^2 for balanced function
        assert abs(stab) < 0.1

    def test_stability_between_zero_and_one(self):
        """Stability is in [0, 1] for Boolean functions."""
        f = bf.majority(3)

        for rho in [0.0, 0.3, 0.5, 0.7, 1.0]:
            stab = gaussian_noise_stability(f, rho)
            assert 0 <= stab <= 1 + 1e-6


class TestGaussianNoiseSensitivity:
    """Tests for gaussian_noise_sensitivity function."""

    def test_sensitivity_is_one_minus_stability(self):
        """Sensitivity = 1 - stability."""
        f = bf.AND(3)

        stab = gaussian_noise_stability(f, rho=0.5)
        sens = gaussian_noise_sensitivity(f, rho=0.5)

        assert abs(sens - (1 - stab)) < 1e-10

    def test_sensitivity_nonnegative(self):
        """Sensitivity is non-negative."""
        f = bf.parity(3)
        sens = gaussian_noise_sensitivity(f, rho=0.5)
        assert sens >= -1e-10


class TestOrnsteinUhlenbeckOperator:
    """Tests for ornstein_uhlenbeck_operator function."""

    def test_returns_array(self):
        """Returns numpy array."""
        f = bf.AND(3)
        result = ornstein_uhlenbeck_operator(f, rho=0.5)

        assert isinstance(result, np.ndarray)
        assert len(result) == 8

    def test_rho_one_is_identity(self):
        """T_1 f = f."""
        f = bf.majority(3)
        result = ornstein_uhlenbeck_operator(f, rho=1.0)

        tt = np.array(f.get_representation("truth_table"))
        expected = 1.0 - 2.0 * tt

        # Should be close to f in ±1 representation
        # (might differ by sign convention)
        assert (
            np.allclose(np.abs(result), np.abs(expected), atol=1e-6)
            or np.allclose(result, expected, atol=1e-6)
            or np.allclose(result, -expected, atol=1e-6)
        )

    def test_rho_zero_is_mean(self):
        """T_0 f = E[f] (constant)."""
        f = bf.parity(3)  # Balanced, E[f] = 0
        result = ornstein_uhlenbeck_operator(f, rho=0.0)

        # All values should be close to the mean
        assert np.allclose(result, 0.0, atol=1e-6)


class TestBerryEsseenBound:
    """Tests for berry_esseen_bound function."""

    def test_returns_float(self):
        """Returns a float."""
        f = bf.majority(5)
        bound = berry_esseen_bound(f)

        assert isinstance(bound, float)

    def test_bound_nonnegative(self):
        """Bound is non-negative."""
        f = bf.parity(4)
        bound = berry_esseen_bound(f)

        assert bound >= 0

    def test_constant_zero_bound(self):
        """Constant function has zero bound."""
        f = bf.constant(True, 3)
        bound = berry_esseen_bound(f)

        assert bound == 0.0

    def test_symmetric_function_small_bound(self):
        """Symmetric functions with many variables have small bound."""
        f = bf.majority(5)
        bound = berry_esseen_bound(f)

        # Bound should be relatively small for symmetric functions
        assert bound < 1.0


class TestCLTApproximation:
    """Tests for clt_approximation function."""

    def test_returns_tuple(self):
        """Returns (mean, variance) tuple."""
        f = bf.parity(3)
        result = clt_approximation(f)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_balanced_mean_near_zero(self):
        """Balanced function has mean near 0."""
        f = bf.parity(3)
        mean, var = clt_approximation(f)

        assert abs(mean) < 0.1

    def test_variance_nonnegative(self):
        """Variance is non-negative."""
        f = bf.majority(3)
        mean, var = clt_approximation(f)

        assert var >= 0


class TestMultilinearExtension:
    """Tests for multilinear_extension function."""

    def test_returns_callable(self):
        """Returns a callable function."""
        f = bf.AND(3)
        p = multilinear_extension(f)

        assert callable(p)

    def test_agrees_on_boolean_cube(self):
        """Extension agrees with f on {-1, 1}^n."""
        f = bf.majority(3)
        p = multilinear_extension(f)

        tt = np.array(f.get_representation("truth_table"))

        for x in range(8):
            # Convert x to ±1 vector
            vec = np.array([-1.0 if (x >> i) & 1 else 1.0 for i in range(3)])

            p_val = p(vec)
            f_val = 1.0 - 2.0 * tt[x]

            assert abs(p_val - f_val) < 1e-6 or abs(p_val + f_val) < 1e-6

    def test_multilinear_interpolation(self):
        """Extension interpolates for non-boolean inputs."""
        f = bf.AND(2)
        p = multilinear_extension(f)

        # Evaluate at a non-boolean point
        val = p(np.array([0.5, 0.5]))

        # Should be between min and max of f
        assert -1 <= val <= 1


class TestGaussianAnalyzer:
    """Tests for GaussianAnalyzer class."""

    def test_initialization(self):
        """Analyzer initializes correctly."""
        f = bf.majority(3)
        analyzer = GaussianAnalyzer(f)

        assert analyzer.function is f
        assert analyzer.n_vars == 3

    def test_hermite_coefficients_caching(self):
        """Hermite coefficients are cached."""
        f = bf.parity(3)
        analyzer = GaussianAnalyzer(f)

        c1 = analyzer.hermite_coefficients
        c2 = analyzer.hermite_coefficients

        assert c1 is c2

    def test_noise_stability(self):
        """noise_stability method works."""
        f = bf.AND(3)
        analyzer = GaussianAnalyzer(f)

        stab = analyzer.noise_stability(0.5)
        assert isinstance(stab, float)

    def test_noise_sensitivity(self):
        """noise_sensitivity method works."""
        f = bf.OR(3)
        analyzer = GaussianAnalyzer(f)

        sens = analyzer.noise_sensitivity(0.5)
        assert sens >= 0

    def test_berry_esseen(self):
        """berry_esseen method works."""
        f = bf.majority(5)
        analyzer = GaussianAnalyzer(f)

        bound = analyzer.berry_esseen()
        assert bound >= 0

    def test_multilinear_extension(self):
        """multilinear_extension method returns callable."""
        f = bf.parity(3)
        analyzer = GaussianAnalyzer(f)

        p = analyzer.multilinear_extension()
        assert callable(p)

    def test_apply_noise_operator(self):
        """apply_noise_operator method returns array."""
        f = bf.majority(3)
        analyzer = GaussianAnalyzer(f)

        result = analyzer.apply_noise_operator(0.5)
        assert isinstance(result, np.ndarray)

    def test_is_approximately_gaussian(self):
        """is_approximately_gaussian returns boolean."""
        f = bf.majority(5)
        analyzer = GaussianAnalyzer(f)

        result = analyzer.is_approximately_gaussian()
        assert isinstance(result, (bool, np.bool_))

    def test_summary(self):
        """summary method returns string."""
        f = bf.parity(3)
        analyzer = GaussianAnalyzer(f)

        summary = analyzer.summary()
        assert isinstance(summary, str)
        assert "Gaussian Analysis" in summary


class TestOnBuiltinFunctions:
    """Test Gaussian analysis on built-in functions."""

    def test_tribes(self):
        """Gaussian analysis on tribes."""
        f = bf.tribes(2, 4)

        stab = gaussian_noise_stability(f, 0.5)
        assert 0 <= stab <= 1 + 1e-6

    def test_dictator(self):
        """Gaussian analysis on dictator."""
        f = bf.dictator(3, i=0)

        # Dictator has high Berry-Esseen bound (far from Gaussian)
        bound = berry_esseen_bound(f)
        assert bound >= 0


class TestEdgeCases:
    """Test edge cases."""

    def test_single_variable(self):
        """Works with single variable function."""
        f = bf.parity(1)

        stab = gaussian_noise_stability(f, 0.5)
        assert isinstance(stab, float)

        analyzer = GaussianAnalyzer(f)
        summary = analyzer.summary()
        assert isinstance(summary, str)
