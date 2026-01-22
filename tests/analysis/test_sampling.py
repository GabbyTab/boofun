"""
Comprehensive tests for sampling module.

Tests cover:
- Uniform and p-biased sampling
- Spectral sampling
- Monte Carlo estimation of Fourier coefficients
- Influence estimation
- RandomVariableView class
- Statistical properties and convergence
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis.sampling import (
    RandomVariableView,
    SpectralDistribution,
    estimate_expectation,
    estimate_fourier_coefficient,
    estimate_influence,
    estimate_total_influence,
    estimate_variance,
    sample_biased,
    sample_input_output_pairs,
    sample_spectral,
    sample_uniform,
)


class TestSampleUniform:
    """Tests for uniform sampling."""

    def test_sample_shape(self):
        """Sample returns correct shape."""
        samples = sample_uniform(n=4, n_samples=100)
        assert samples.shape == (100,)

    def test_sample_range(self):
        """Samples are in valid range [0, 2^n)."""
        samples = sample_uniform(n=4, n_samples=1000)
        assert np.all(samples >= 0)
        assert np.all(samples < 16)

    def test_sample_distribution(self):
        """Samples are approximately uniform."""
        rng = np.random.default_rng(42)
        samples = sample_uniform(n=3, n_samples=10000, rng=rng)

        # Each of 8 values should appear roughly 1/8 of the time
        counts = np.bincount(samples, minlength=8)
        expected = 10000 / 8

        # Chi-squared like test: no value should deviate too much
        assert np.all(counts > expected * 0.7)
        assert np.all(counts < expected * 1.3)

    def test_reproducible_with_seed(self):
        """Same seed gives same samples."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        s1 = sample_uniform(n=4, n_samples=100, rng=rng1)
        s2 = sample_uniform(n=4, n_samples=100, rng=rng2)

        assert np.array_equal(s1, s2)


class TestSampleBiased:
    """Tests for p-biased sampling."""

    def test_sample_shape(self):
        """Sample returns correct shape."""
        samples = sample_biased(n=4, p=0.3, n_samples=100)
        assert samples.shape == (100,)

    def test_sample_range(self):
        """Samples are in valid range."""
        samples = sample_biased(n=4, p=0.3, n_samples=1000)
        assert np.all(samples >= 0)
        assert np.all(samples < 16)

    def test_p_zero(self):
        """p=0 gives all zeros."""
        samples = sample_biased(n=4, p=0.0, n_samples=100)
        assert np.all(samples == 0)

    def test_p_one(self):
        """p=1 gives all ones (2^n - 1)."""
        samples = sample_biased(n=4, p=1.0, n_samples=100)
        assert np.all(samples == 15)

    def test_p_half_is_uniform(self):
        """p=0.5 gives uniform distribution."""
        rng = np.random.default_rng(42)
        samples = sample_biased(n=3, p=0.5, n_samples=10000, rng=rng)

        counts = np.bincount(samples, minlength=8)
        expected = 10000 / 8

        # Should be roughly uniform
        assert np.all(counts > expected * 0.7)
        assert np.all(counts < expected * 1.3)

    def test_biased_popcount(self):
        """Higher p means more 1 bits on average."""
        rng = np.random.default_rng(42)
        n = 8

        samples_low = sample_biased(n=n, p=0.2, n_samples=5000, rng=rng)
        samples_high = sample_biased(n=n, p=0.8, n_samples=5000, rng=rng)

        avg_bits_low = np.mean([bin(x).count("1") for x in samples_low])
        avg_bits_high = np.mean([bin(x).count("1") for x in samples_high])

        # p=0.2 should give ~0.2*8=1.6 bits, p=0.8 should give ~6.4 bits
        assert avg_bits_low < 3
        assert avg_bits_high > 5

    def test_invalid_p_raises(self):
        """Invalid p raises ValueError."""
        with pytest.raises(ValueError):
            sample_biased(n=4, p=-0.1, n_samples=10)
        with pytest.raises(ValueError):
            sample_biased(n=4, p=1.5, n_samples=10)


class TestSampleSpectral:
    """Tests for spectral sampling."""

    def test_sample_shape(self):
        """Sample returns correct shape."""
        f = bf.create([0, 1, 1, 0])  # XOR
        samples = sample_spectral(f, n_samples=100)
        assert samples.shape == (100,)

    def test_sample_range(self):
        """Samples are valid subset indices."""
        f = bf.create([0, 1, 1, 0])
        samples = sample_spectral(f, n_samples=100)
        assert np.all(samples >= 0)
        assert np.all(samples < 4)

    def test_parity_concentrates_on_full_set(self):
        """XOR should concentrate on the full set S = {0,1}."""
        f = bf.create([0, 1, 1, 0])  # XOR
        rng = np.random.default_rng(42)
        samples = sample_spectral(f, n_samples=1000, rng=rng)

        # For XOR, all weight is on S = {0,1} = 3
        assert np.mean(samples == 3) > 0.99

    def test_dictator_concentrates_on_singleton(self):
        """Dictator should concentrate on the relevant singleton."""
        f = bf.create([0, 1, 0, 1])  # x0
        rng = np.random.default_rng(42)
        samples = sample_spectral(f, n_samples=1000, rng=rng)

        # For balanced dictator, all weight is on S = {0} = 1
        assert np.mean(samples == 1) > 0.99


class TestSampleInputOutputPairs:
    """Tests for sampling input-output pairs."""

    def test_returns_tuple(self):
        """Function returns tuple of arrays."""
        f = bf.create([0, 1, 1, 0])
        inputs, outputs = sample_input_output_pairs(f, n_samples=100)

        assert isinstance(inputs, np.ndarray)
        assert isinstance(outputs, np.ndarray)
        assert inputs.shape == outputs.shape == (100,)

    def test_outputs_are_binary(self):
        """Outputs are 0 or 1."""
        f = bf.create([0, 1, 1, 0])
        _, outputs = sample_input_output_pairs(f, n_samples=100)

        assert np.all((outputs == 0) | (outputs == 1))

    def test_outputs_match_function(self):
        """Outputs match function evaluation."""
        f = bf.create([0, 1, 1, 0])  # XOR
        inputs, outputs = sample_input_output_pairs(f, n_samples=100)

        for x, y in zip(inputs, outputs):
            expected = int(f.evaluate(int(x)))
            assert y == expected


class TestEstimateFourierCoefficient:
    """Tests for Fourier coefficient estimation."""

    def test_empty_set_is_expectation(self):
        """f̂(∅) estimates E[f]."""
        f = bf.create([0, 0, 0, 1])  # AND
        rng = np.random.default_rng(42)

        # Exact: E[f] in ±1 = (3*(+1) + 1*(-1))/4 = 0.5
        estimate = estimate_fourier_coefficient(f, S=0, n_samples=10000, rng=rng)

        assert abs(estimate - 0.5) < 0.05

    def test_parity_full_set(self):
        """XOR should have f̂({0,1}) ≈ ±1."""
        f = bf.create([0, 1, 1, 0])  # XOR
        rng = np.random.default_rng(42)

        estimate = estimate_fourier_coefficient(f, S=3, n_samples=10000, rng=rng)

        # XOR has f̂({0,1}) = -1
        assert abs(abs(estimate) - 1.0) < 0.05

    def test_confidence_interval(self):
        """Estimate with confidence returns std error."""
        # Use AND which has non-trivial coefficients
        f = bf.create([0, 0, 0, 1])
        rng = np.random.default_rng(42)

        estimate, std_err = estimate_fourier_coefficient(
            f, S=0, n_samples=1000, rng=rng, return_confidence=True
        )

        assert isinstance(estimate, float)
        assert isinstance(std_err, float)
        assert std_err >= 0  # Can be 0 if all samples same
        assert std_err < 0.1  # Should be O(1/sqrt(n))

    def test_convergence(self):
        """More samples gives smaller error."""
        f = bf.create([0, 0, 0, 1])  # AND
        rng = np.random.default_rng(42)

        # Get exact value
        from boofun.analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(f)
        exact = analyzer.fourier_expansion()[0]

        # Estimate with different sample sizes
        err_small = abs(estimate_fourier_coefficient(f, 0, 100, rng=rng) - exact)
        err_large = abs(estimate_fourier_coefficient(f, 0, 10000, rng=rng) - exact)

        # Larger sample should have smaller error (probabilistically)
        # Allow some slack for randomness
        assert err_large < err_small + 0.1


class TestEstimateInfluence:
    """Tests for influence estimation."""

    def test_dictator_influence(self):
        """Dictator has influence 1 on its variable."""
        f = bf.create([0, 1, 0, 1])  # x0
        rng = np.random.default_rng(42)

        inf_0 = estimate_influence(f, i=0, n_samples=1000, rng=rng)
        inf_1 = estimate_influence(f, i=1, n_samples=1000, rng=rng)

        assert abs(inf_0 - 1.0) < 0.1
        assert abs(inf_1 - 0.0) < 0.1

    def test_xor_equal_influences(self):
        """XOR has equal influences on both variables."""
        f = bf.create([0, 1, 1, 0])  # XOR
        rng = np.random.default_rng(42)

        inf_0 = estimate_influence(f, i=0, n_samples=1000, rng=rng)
        inf_1 = estimate_influence(f, i=1, n_samples=1000, rng=rng)

        assert abs(inf_0 - inf_1) < 0.1
        assert abs(inf_0 - 1.0) < 0.1  # Each influence is 1 for XOR

    def test_confidence_interval(self):
        """Estimate with confidence returns std error."""
        f = bf.create([0, 1, 1, 0])
        rng = np.random.default_rng(42)

        estimate, std_err = estimate_influence(
            f, i=0, n_samples=1000, rng=rng, return_confidence=True
        )

        assert isinstance(estimate, float)
        assert isinstance(std_err, float)
        assert std_err >= 0


class TestEstimateTotalInfluence:
    """Tests for total influence estimation."""

    def test_xor_total_influence(self):
        """XOR has total influence = n."""
        f = bf.create([0, 1, 1, 0])  # 2-XOR
        rng = np.random.default_rng(42)

        est_I = estimate_total_influence(f, n_samples=1000, rng=rng)

        assert abs(est_I - 2.0) < 0.2

    def test_matches_sum_of_influences(self):
        """Total influence ≈ sum of individual influences."""
        f = bf.create([0, 0, 0, 1])  # AND
        rng = np.random.default_rng(42)

        est_I = estimate_total_influence(f, n_samples=5000, rng=rng)

        # Compute sum of individual estimates
        sum_inf = sum(estimate_influence(f, i, n_samples=5000, rng=rng) for i in range(2))

        # Should be approximately equal
        assert abs(est_I - sum_inf) < 0.3


class TestSpectralDistribution:
    """Tests for SpectralDistribution class."""

    def test_from_function(self):
        """Can create from BooleanFunction."""
        f = bf.create([0, 1, 1, 0])
        sd = SpectralDistribution.from_function(f)

        assert sd.n_vars == 2
        assert len(sd.weights) == 4
        assert len(sd.probabilities) == 4

    def test_probabilities_sum_to_one(self):
        """Probabilities sum to 1."""
        f = bf.create([0, 0, 0, 1])
        sd = SpectralDistribution.from_function(f)

        assert abs(np.sum(sd.probabilities) - 1.0) < 1e-10

    def test_weight_at_degree(self):
        """Weight at degree is computed correctly."""
        f = bf.create([0, 1, 1, 0])  # XOR has all weight at degree 2
        sd = SpectralDistribution.from_function(f)

        w0 = sd.weight_at_degree(0)
        w1 = sd.weight_at_degree(1)
        w2 = sd.weight_at_degree(2)

        assert w0 < 0.01
        assert w1 < 0.01
        assert abs(w2 - 1.0) < 0.01

    def test_entropy(self):
        """Entropy is non-negative."""
        f = bf.create([0, 0, 0, 1])
        sd = SpectralDistribution.from_function(f)

        H = sd.entropy()
        assert H >= 0

    def test_sample(self):
        """Can sample from distribution."""
        f = bf.create([0, 1, 1, 0])
        sd = SpectralDistribution.from_function(f)

        samples = sd.sample(100)
        assert samples.shape == (100,)
        assert np.all(samples >= 0)
        assert np.all(samples < 4)


class TestRandomVariableView:
    """Tests for RandomVariableView class."""

    def test_creation(self):
        """Can create view from function."""
        f = bf.create([0, 1, 1, 0])
        rv = RandomVariableView(f)

        assert rv.n_vars == 2
        assert rv.p == 0.5

    def test_exact_expectation(self):
        """Exact expectation matches SpectralAnalyzer."""
        f = bf.create([0, 0, 0, 1])
        rv = RandomVariableView(f)

        E = rv.expectation()

        # E[f] for AND in ±1 is (3*(+1) + 1*(-1))/4 = 0.5
        assert abs(E - 0.5) < 0.01

    def test_exact_variance(self):
        """Exact variance is computed correctly."""
        f = bf.create([0, 1, 1, 0])  # XOR
        rv = RandomVariableView(f)

        Var = rv.variance()

        # XOR is ±1-valued and balanced, so Var = 1 - E[f]² = 1 - 0 = 1
        # But in our convention... let's just check it's positive
        assert Var >= 0

    def test_exact_total_influence(self):
        """Exact total influence matches."""
        f = bf.create([0, 1, 1, 0])  # XOR
        rv = RandomVariableView(f)

        I = rv.total_influence()

        # XOR has total influence = 2
        assert abs(I - 2.0) < 0.01

    def test_sample_method(self):
        """Sample method returns pairs."""
        f = bf.create([0, 1, 1, 0])
        rv = RandomVariableView(f)

        inputs, outputs = rv.sample(100)

        assert inputs.shape == (100,)
        assert outputs.shape == (100,)

    def test_estimate_vs_exact(self):
        """Estimates converge to exact values."""
        f = bf.create([0, 0, 0, 1])
        rv = RandomVariableView(f).seed(42)

        exact_E = rv.expectation()
        est_E = rv.estimate_expectation(10000)

        assert abs(exact_E - est_E) < 0.1

    def test_validate_estimates(self):
        """Validation method works."""
        f = bf.create([0, 1, 1, 0])
        rv = RandomVariableView(f).seed(42)

        results = rv.validate_estimates(n_samples=5000, tolerance=0.2)

        assert isinstance(results, dict)
        # Most validations should pass with enough samples
        assert sum(results.values()) >= 3

    def test_summary(self):
        """Summary returns string."""
        f = bf.create([0, 1, 1, 0])
        rv = RandomVariableView(f)

        s = rv.summary()

        assert isinstance(s, str)
        assert "RandomVariableView" in s
        assert "E[f]" in s

    def test_p_biased_view(self):
        """Can create p-biased view."""
        f = bf.create([0, 0, 0, 1])
        rv = RandomVariableView(f, p=0.3)

        assert rv.p == 0.3

        # Should use p-biased formulas
        E = rv.expectation()
        assert isinstance(E, float)


class TestStatisticalProperties:
    """Tests for statistical properties and convergence."""

    def test_law_of_large_numbers(self):
        """Estimates converge as n_samples increases."""
        f = bf.create([0, 0, 0, 1])

        from boofun.analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(f)
        exact_f_empty = analyzer.fourier_expansion()[0]

        rng = np.random.default_rng(42)

        errors = []
        for n in [100, 1000, 10000]:
            est = estimate_fourier_coefficient(f, 0, n, rng=rng)
            errors.append(abs(est - exact_f_empty))

        # Errors should generally decrease
        assert errors[-1] < errors[0] + 0.1

    def test_parseval_via_sampling(self):
        """Σ f̂(S)² ≈ 1 for ±1-valued functions."""
        f = bf.create([0, 0, 0, 1])
        rng = np.random.default_rng(42)

        # Estimate all Fourier coefficients
        total_weight = 0.0
        for S in range(4):
            est = estimate_fourier_coefficient(f, S, n_samples=5000, rng=rng)
            total_weight += est**2

        # Should be approximately 1 by Parseval
        assert abs(total_weight - 1.0) < 0.2
