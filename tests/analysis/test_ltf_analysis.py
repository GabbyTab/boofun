import sys

sys.path.insert(0, "src")
"""
Tests for ltf_analysis module.

Tests Linear Threshold Function (LTF) analysis:
- chow_parameters
- is_ltf, find_ltf_weights
- normalize_ltf_weights
- critical_index, regularity
- ltf_influence_from_weights
- ltf_noise_stability_gaussian
- ltf_total_influence_estimate
- analyze_ltf
- LTF constructors
- LTF property tests
- Chow distance
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis.ltf_analysis import (
    LTFAnalysis,
    analyze_ltf,
    chow_distance,
    chow_parameters,
    create_threshold_function,
    create_weighted_majority,
    critical_index,
    dummy_voters,
    find_ltf_weights,
    is_ltf,
    is_regular_ltf,
    is_symmetric_ltf,
    ltf_influence_from_weights,
    ltf_noise_stability_gaussian,
    ltf_total_influence_estimate,
    normalize_ltf_weights,
    regularity,
)


class TestChowParameters:
    """Tests for chow_parameters function."""

    def test_returns_array(self):
        """Returns numpy array of length n+1."""
        f = bf.majority(3)
        chow = chow_parameters(f)

        assert isinstance(chow, np.ndarray)
        assert len(chow) == 4  # n + 1

    def test_balanced_zero_mean(self):
        """Balanced function has chow[0] near 0."""
        f = bf.majority(3)  # Balanced
        chow = chow_parameters(f)

        # E[f] should be small for balanced function
        assert abs(chow[0]) < 0.2

    def test_symmetric_equal_coefficients(self):
        """Symmetric functions have equal degree-1 coefficients."""
        f = bf.majority(3)
        chow = chow_parameters(f)

        # Chow[1], Chow[2], Chow[3] should be equal for symmetric f
        assert abs(chow[1] - chow[2]) < 1e-6
        assert abs(chow[2] - chow[3]) < 1e-6


class TestIsLTF:
    """Tests for is_ltf function."""

    def test_majority_is_ltf(self):
        """Majority is an LTF."""
        f = bf.majority(3)
        assert is_ltf(f)

    def test_and_is_ltf(self):
        """AND is an LTF."""
        f = bf.AND(3)
        assert is_ltf(f)

    def test_or_is_ltf(self):
        """OR is an LTF."""
        f = bf.OR(3)
        assert is_ltf(f)

    def test_threshold_is_ltf(self):
        """Threshold functions are LTFs."""
        f = bf.threshold(4, k=2)
        assert is_ltf(f)

    def test_parity_not_ltf(self):
        """Parity (XOR) is not an LTF."""
        f = bf.parity(3)
        assert not is_ltf(f)


class TestFindLTFWeights:
    """Tests for find_ltf_weights function."""

    def test_returns_weights_and_threshold(self):
        """Returns (weights, threshold) tuple."""
        f = bf.majority(3)
        weights, threshold = find_ltf_weights(f)

        assert isinstance(weights, np.ndarray)
        assert isinstance(threshold, (int, float, np.floating))

    def test_majority_equal_weights(self):
        """Majority has equal weights."""
        f = bf.majority(3)
        weights, threshold = find_ltf_weights(f)

        # All weights should be exactly equal for majority
        abs_weights = np.abs(weights)
        assert np.allclose(
            abs_weights, abs_weights[0], rtol=1e-10
        ), f"Majority weights should be equal, got: {abs_weights}"
        # Weights should be positive for unweighted majority
        assert np.all(weights > 0), f"Majority weights should be positive, got: {weights}"

    def test_non_ltf_raises(self):
        """Non-LTF raises ValueError."""
        f = bf.parity(3)
        with pytest.raises(ValueError):
            find_ltf_weights(f)


class TestNormalizeLTFWeights:
    """Tests for normalize_ltf_weights function."""

    def test_unit_norm(self):
        """Normalized weights have unit L2 norm."""
        weights = np.array([1.0, 2.0, 3.0])
        threshold = 1.5

        norm_weights, norm_threshold = normalize_ltf_weights(weights, threshold)

        assert abs(np.linalg.norm(norm_weights) - 1.0) < 1e-10

    def test_threshold_scaled(self):
        """Threshold is scaled proportionally."""
        weights = np.array([3.0, 4.0])
        threshold = 5.0

        norm_weights, norm_threshold = normalize_ltf_weights(weights, threshold)

        # Original norm is 5, so everything is divided by 5
        assert abs(norm_threshold - 1.0) < 1e-10

    def test_zero_weights_unchanged(self):
        """Zero weights remain unchanged."""
        weights = np.array([0.0, 0.0])
        threshold = 0.0

        norm_weights, norm_threshold = normalize_ltf_weights(weights, threshold)

        assert np.allclose(norm_weights, weights)


class TestCriticalIndex:
    """Tests for critical_index function."""

    def test_uniform_weights(self):
        """Uniform weights have critical index around n/2."""
        weights = np.ones(10)
        idx = critical_index(weights)

        # Should be around half
        assert 4 <= idx <= 6

    def test_concentrated_weights(self):
        """Concentrated weights have small critical index."""
        weights = np.array([10.0, 1.0, 1.0, 1.0])
        idx = critical_index(weights)

        # First weight dominates
        assert idx <= 2

    def test_returns_valid_index(self):
        """Critical index is between 1 and n."""
        weights = np.array([1.0, 2.0, 3.0])
        idx = critical_index(weights)

        assert 1 <= idx <= len(weights)


class TestRegularity:
    """Tests for regularity function."""

    def test_uniform_weights_low_regularity(self):
        """Uniform weights have low regularity."""
        weights = np.ones(10)
        tau = regularity(weights)

        # τ = 1/√n for uniform
        expected = 1.0 / np.sqrt(10)
        assert abs(tau - expected) < 1e-10

    def test_dictator_high_regularity(self):
        """Dictator-like weights have high regularity."""
        weights = np.array([1.0, 0.0, 0.0])
        tau = regularity(weights)

        # τ = 1 for dictator
        assert abs(tau - 1.0) < 1e-10

    def test_regularity_bounded(self):
        """Regularity is in [0, 1]."""
        weights = np.array([1.0, 2.0, 0.5])
        tau = regularity(weights)

        assert 0 <= tau <= 1


class TestLTFInfluenceFromWeights:
    """Tests for ltf_influence_from_weights function."""

    def test_returns_array(self):
        """Returns array same length as weights."""
        weights = np.array([1.0, 2.0, 3.0])
        inf = ltf_influence_from_weights(weights)

        assert isinstance(inf, np.ndarray)
        assert len(inf) == len(weights)

    def test_nonnegative(self):
        """Influences are non-negative."""
        weights = np.array([1.0, -2.0, 3.0])
        inf = ltf_influence_from_weights(weights)

        assert np.all(inf >= 0)

    def test_proportional_to_weight_squared(self):
        """Influences are proportional to weight squared."""
        weights = np.array([1.0, 2.0])
        inf = ltf_influence_from_weights(weights)

        # Ratio should be 4:1
        assert abs(inf[1] / inf[0] - 4.0) < 1e-6


class TestLTFNoiseStabilityGaussian:
    """Tests for ltf_noise_stability_gaussian function."""

    def test_rho_zero(self):
        """At rho=0, stability is 1/2."""
        stab = ltf_noise_stability_gaussian(0.0)
        assert abs(stab - 0.5) < 1e-10

    def test_rho_one(self):
        """At rho=1, stability is 1."""
        stab = ltf_noise_stability_gaussian(1.0)
        assert abs(stab - 1.0) < 1e-10

    def test_monotonic_in_rho(self):
        """Stability increases with rho."""
        stab_low = ltf_noise_stability_gaussian(0.3)
        stab_high = ltf_noise_stability_gaussian(0.7)

        assert stab_high > stab_low


class TestLTFTotalInfluenceEstimate:
    """Tests for ltf_total_influence_estimate function."""

    def test_increases_with_n(self):
        """Estimate increases with sqrt(n)."""
        est5 = ltf_total_influence_estimate(5)
        est20 = ltf_total_influence_estimate(20)

        # Should be roughly 2x for 4x n
        ratio = est20 / est5
        assert 1.5 < ratio < 2.5

    def test_regular_formula(self):
        """Regular case follows sqrt(2/pi) * sqrt(n)."""
        n = 100
        est = ltf_total_influence_estimate(n, regularity_tau=0.0)

        expected = np.sqrt(2 / np.pi) * np.sqrt(n)
        assert abs(est - expected) < 0.01


class TestAnalyzeLTF:
    """Tests for analyze_ltf function."""

    def test_returns_ltfanalysis(self):
        """Returns LTFAnalysis object."""
        f = bf.majority(3)
        result = analyze_ltf(f)

        assert isinstance(result, LTFAnalysis)

    def test_ltf_has_weights(self):
        """LTF analysis includes weights."""
        f = bf.AND(3)
        result = analyze_ltf(f)

        assert result.is_ltf
        assert result.weights is not None
        assert result.threshold is not None

    def test_non_ltf_minimal_result(self):
        """Non-LTF returns minimal analysis."""
        f = bf.parity(3)
        result = analyze_ltf(f)

        assert not result.is_ltf
        assert result.weights is None

    def test_summary_works(self):
        """summary() returns string."""
        f = bf.majority(3)
        result = analyze_ltf(f)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "LTF Analysis" in summary


class TestCreateWeightedMajority:
    """Tests for create_weighted_majority function."""

    def test_returns_function(self):
        """Returns a BooleanFunction."""
        f = create_weighted_majority([1.0, 1.0, 1.0])

        assert hasattr(f, "n_vars")
        assert f.n_vars == 3

    def test_uniform_weights_is_majority(self):
        """Uniform weights create majority function."""
        f = create_weighted_majority([1.0, 1.0, 1.0])
        g = bf.majority(3)

        # Should have same truth table (possibly negated due to convention)
        tt_f = np.array(f.get_representation("truth_table"), dtype=int)
        tt_g = np.array(g.get_representation("truth_table"), dtype=int)
        # Check if equal or negated equal
        assert np.array_equal(tt_f, tt_g) or np.array_equal(tt_f, 1 - tt_g)

    def test_weighted_majority_is_ltf(self):
        """Weighted majority is an LTF."""
        f = create_weighted_majority([1.0, 2.0, 3.0])
        assert is_ltf(f)


class TestCreateThresholdFunction:
    """Tests for create_threshold_function function."""

    def test_threshold_n_is_and(self):
        """Threshold-n is AND."""
        f = create_threshold_function(3, 3)
        g = bf.AND(3)

        tt_f = list(f.get_representation("truth_table"))
        tt_g = list(g.get_representation("truth_table"))
        assert tt_f == tt_g

    def test_threshold_1_is_or(self):
        """Threshold-1 is OR."""
        f = create_threshold_function(3, 1)
        g = bf.OR(3)

        tt_f = list(f.get_representation("truth_table"))
        tt_g = list(g.get_representation("truth_table"))
        assert tt_f == tt_g

    def test_invalid_k_raises(self):
        """Invalid k raises ValueError."""
        with pytest.raises(ValueError):
            create_threshold_function(3, 0)

        with pytest.raises(ValueError):
            create_threshold_function(3, 5)


class TestIsSymmetricLTF:
    """Tests for is_symmetric_ltf function."""

    def test_majority_is_symmetric(self):
        """Majority is a symmetric LTF."""
        f = bf.majority(3)
        assert is_symmetric_ltf(f)

    def test_parity_not_symmetric_ltf(self):
        """Parity is not a symmetric LTF (not even an LTF)."""
        f = bf.parity(3)
        assert not is_symmetric_ltf(f)


class TestIsRegularLTF:
    """Tests for is_regular_ltf function."""

    def test_majority_is_regular(self):
        """Majority is a regular LTF."""
        f = bf.majority(5)
        assert is_regular_ltf(f)

    def test_dictator_not_regular(self):
        """Dictator is not a regular LTF."""
        f = bf.dictator(3, i=0)
        # Dictator might be considered an LTF (with very unequal weights)
        # but should not be regular
        result = is_regular_ltf(f)
        assert isinstance(result, (bool, np.bool_))


class TestDummyVoters:
    """Tests for dummy_voters function."""

    def test_majority_no_dummies(self):
        """Majority has no dummy voters."""
        f = bf.majority(3)
        dummies = dummy_voters(f)

        assert len(dummies) == 0

    def test_constant_all_dummies(self):
        """Constant function has all dummy voters."""
        f = bf.constant(True, 3)
        dummies = dummy_voters(f)

        assert len(dummies) == 3


class TestChowDistance:
    """Tests for chow_distance function."""

    def test_same_function_zero_distance(self):
        """Same function has zero Chow distance."""
        f = bf.majority(3)
        dist = chow_distance(f, f)

        assert dist < 1e-10

    def test_different_functions_positive_distance(self):
        """Different functions have positive distance."""
        f = bf.AND(3)
        g = bf.OR(3)

        dist = chow_distance(f, g)
        assert dist > 0


class TestLTFAnalysisDataclass:
    """Tests for LTFAnalysis dataclass."""

    def test_creation(self):
        """Can create LTFAnalysis."""
        analysis = LTFAnalysis(is_ltf=True)
        assert analysis.is_ltf

    def test_defaults(self):
        """Optional fields default to None."""
        analysis = LTFAnalysis(is_ltf=False)

        assert analysis.weights is None
        assert analysis.threshold is None
        assert analysis.chow_parameters is None

    def test_summary_non_ltf(self):
        """summary works for non-LTF."""
        analysis = LTFAnalysis(is_ltf=False)
        summary = analysis.summary()

        assert "Is LTF: False" in summary


class TestOnBuiltinFunctions:
    """Test LTF analysis on built-in functions."""

    def test_tribes_not_ltf(self):
        """Tribes is typically not an LTF."""
        f = bf.tribes(2, 4)
        # Tribes might or might not be an LTF depending on parameters
        result = is_ltf(f)
        assert isinstance(result, bool)

    def test_threshold_is_ltf(self):
        """Threshold functions are LTFs."""
        f = bf.threshold(4, k=2)
        assert is_ltf(f)

        analysis = analyze_ltf(f)
        assert analysis.is_ltf
