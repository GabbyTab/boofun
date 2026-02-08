import sys

sys.path.insert(0, "src")
"""
Tests for analysis/sensitivity module.

Tests sensitivity analysis functions:
- sensitivity_at
- sensitive_coordinates
- sensitivity_profile
- max_sensitivity, min_sensitivity
- average_sensitivity
- total_influence_via_sensitivity
- average_sensitivity_moment
- sensitivity_histogram
- arg_max_sensitivity, arg_min_sensitivity
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.sensitivity import (
    arg_max_sensitivity,
    arg_min_sensitivity,
    average_sensitivity,
    average_sensitivity_moment,
    max_sensitivity,
    min_sensitivity,
    sensitive_coordinates,
    sensitivity_at,
    sensitivity_histogram,
    sensitivity_profile,
    total_influence_via_sensitivity,
)


# ---------------------------------------------------------------------------
# Tests: sensitivity_at
# ---------------------------------------------------------------------------


class TestSensitivityAt:
    """Tests for sensitivity_at function."""

    def test_parity_full_sensitivity(self):
        """Parity(n): s(f, x) = n for all x."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            for x in range(1 << n):
                assert sensitivity_at(f, x) == n

    def test_and_at_all_ones(self):
        """AND(3) at (1,1,1)=7: every bit is sensitive."""
        f = bf.AND(3)
        assert sensitivity_at(f, 7) == 3

    def test_and_at_zero(self):
        """AND(3) at (0,0,0)=0: no single flip activates AND."""
        f = bf.AND(3)
        assert sensitivity_at(f, 0) == 0

    def test_constant_zero_everywhere(self):
        """Constant function: s(f, x) = 0 for all x."""
        f = bf.constant(True, 3)
        for x in range(8):
            assert sensitivity_at(f, x) == 0

    def test_dictator_one_everywhere(self):
        """Dictator: s(f, x) = 1 for all x."""
        f = bf.dictator(3, i=0)
        for x in range(8):
            assert sensitivity_at(f, x) == 1

    def test_majority3_known_values(self):
        """MAJ(3): sensitivity at all-zeros = 0, at all-ones = 3."""
        f = bf.majority(3)
        # At (0,0,0)=0: f=0. Flipping any single bit -> Hamming weight 1, still minority
        assert sensitivity_at(f, 0) == 0
        # At (1,1,1)=7: f=1. Flipping any bit -> Hamming weight 2, still majority
        assert sensitivity_at(f, 7) == 0
        # At (1,1,0)=3: f=1. Flip bit0 -> (0,1,0)=2, f=0; flip bit1 -> (1,0,0)=1, f=0; flip bit2 -> (1,1,1)=7, f=1
        assert sensitivity_at(f, 3) == 2


# ---------------------------------------------------------------------------
# Tests: sensitive_coordinates
# ---------------------------------------------------------------------------


class TestSensitiveCoordinates:
    """Tests for sensitive_coordinates function."""

    def test_matches_sensitivity_at(self):
        """len(sensitive_coordinates) == sensitivity_at for all inputs."""
        f = bf.majority(3)
        for x in range(8):
            assert len(sensitive_coordinates(f, x)) == sensitivity_at(f, x)

    def test_parity_all_coordinates(self):
        """Parity: all coordinates are sensitive at every input."""
        f = bf.parity(3)
        for x in range(8):
            assert sorted(sensitive_coordinates(f, x)) == [0, 1, 2]

    def test_dictator_single_coordinate(self):
        """Dictator on variable i: only i is sensitive."""
        for i in range(3):
            f = bf.dictator(3, i=i)
            for x in range(8):
                assert sensitive_coordinates(f, x) == [i]


# ---------------------------------------------------------------------------
# Tests: sensitivity_profile
# ---------------------------------------------------------------------------


class TestSensitivityProfile:
    """Tests for sensitivity_profile function."""

    def test_length(self):
        """Profile has 2^n entries."""
        f = bf.majority(3)
        prof = sensitivity_profile(f)
        assert len(prof) == 8

    def test_parity_uniform(self):
        """Parity profile is all n."""
        f = bf.parity(3)
        prof = sensitivity_profile(f)
        assert np.all(prof == 3)

    def test_matches_pointwise(self):
        """Profile matches individual sensitivity_at calls."""
        f = bf.AND(3)
        prof = sensitivity_profile(f)
        for x in range(8):
            assert prof[x] == sensitivity_at(f, x)

    def test_constant_all_zeros(self):
        """Constant function profile is all zeros."""
        f = bf.constant(False, 3)
        prof = sensitivity_profile(f)
        assert np.all(prof == 0)


# ---------------------------------------------------------------------------
# Tests: max_sensitivity and min_sensitivity
# ---------------------------------------------------------------------------


class TestMaxSensitivity:
    """Tests for max_sensitivity function."""

    def test_parity_max_is_n(self):
        """Parity(n): max sensitivity = n."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            assert max_sensitivity(f) == n

    def test_and_max_is_n(self):
        """AND(n): max sensitivity = n (at all-ones)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            assert max_sensitivity(f) == n

    def test_constant_max_is_zero(self):
        """Constant: max sensitivity = 0."""
        f = bf.constant(True, 3)
        assert max_sensitivity(f) == 0

    def test_dictator_max_is_one(self):
        """Dictator: max sensitivity = 1."""
        f = bf.dictator(3, i=0)
        assert max_sensitivity(f) == 1

    def test_output_filter(self):
        """Filtering by output value restricts the max."""
        f = bf.AND(3)
        # s on inputs where f=1 (only x=7): s(7) = 3
        assert max_sensitivity(f, output_value=1) == 3
        # s on inputs where f=0: max among 0..6
        s0 = max_sensitivity(f, output_value=0)
        assert s0 >= 0


class TestMinSensitivity:
    """Tests for min_sensitivity function."""

    def test_parity_min_is_n(self):
        """Parity(n): min sensitivity = n (same as max)."""
        f = bf.parity(3)
        assert min_sensitivity(f) == 3

    def test_constant_min_is_zero(self):
        """Constant: min sensitivity = 0."""
        f = bf.constant(True, 3)
        assert min_sensitivity(f) == 0

    def test_dictator_min_is_one(self):
        """Dictator: min sensitivity = 1 (everywhere sensitivity)."""
        f = bf.dictator(3, i=0)
        assert min_sensitivity(f) == 1

    def test_min_leq_max(self):
        """min sensitivity <= max sensitivity."""
        for factory in [bf.majority, bf.parity, bf.AND]:
            f = factory(3)
            assert min_sensitivity(f) <= max_sensitivity(f)


# ---------------------------------------------------------------------------
# Tests: average_sensitivity and total_influence_via_sensitivity
# ---------------------------------------------------------------------------


class TestAverageSensitivity:
    """Tests for average_sensitivity function."""

    def test_parity_as_equals_n(self):
        """Parity(n): average sensitivity = n."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            assert abs(average_sensitivity(f) - n) < 1e-10

    def test_constant_as_zero(self):
        """Constant: average sensitivity = 0."""
        f = bf.constant(True, 3)
        assert abs(average_sensitivity(f)) < 1e-10

    def test_dictator_as_one(self):
        """Dictator: average sensitivity = 1."""
        f = bf.dictator(3, i=0)
        assert abs(average_sensitivity(f) - 1.0) < 1e-10

    def test_as_bounded(self):
        """Average sensitivity is between 0 and n."""
        f = bf.majority(3)
        as_val = average_sensitivity(f)
        assert 0 <= as_val <= 3

    def test_total_influence_alias(self):
        """total_influence_via_sensitivity == average_sensitivity."""
        f = bf.majority(3)
        assert abs(total_influence_via_sensitivity(f) - average_sensitivity(f)) < 1e-10


# ---------------------------------------------------------------------------
# Tests: average_sensitivity_moment
# ---------------------------------------------------------------------------


class TestAverageSensitivityMoment:
    """Tests for average_sensitivity_moment function."""

    def test_zeroth_moment_is_one(self):
        """E[s(f,x)^0] = 1 for any function."""
        f = bf.majority(3)
        assert abs(average_sensitivity_moment(f, 0) - 1.0) < 1e-10

    def test_first_moment_is_as(self):
        """E[s(f,x)^1] = average_sensitivity."""
        f = bf.majority(3)
        assert abs(average_sensitivity_moment(f, 1) - average_sensitivity(f)) < 1e-10

    def test_second_moment_geq_first_squared(self):
        """E[s^2] >= E[s]^2 (Jensen's inequality)."""
        f = bf.majority(3)
        m1 = average_sensitivity_moment(f, 1)
        m2 = average_sensitivity_moment(f, 2)
        assert m2 >= m1**2 - 1e-10

    def test_parity_all_moments(self):
        """Parity(n): s(x) = n everywhere, so E[s^t] = n^t."""
        n = 3
        f = bf.parity(n)
        for t in [0, 1, 2, 3]:
            assert abs(average_sensitivity_moment(f, t) - n**t) < 1e-10

    def test_constant_all_moments_zero(self):
        """Constant: s(x) = 0 everywhere, so E[s^t] = 0 for t > 0."""
        f = bf.constant(True, 3)
        for t in [1, 2, 3]:
            assert abs(average_sensitivity_moment(f, t)) < 1e-10


# ---------------------------------------------------------------------------
# Tests: sensitivity_histogram
# ---------------------------------------------------------------------------


class TestSensitivityHistogram:
    """Tests for sensitivity_histogram function."""

    def test_sums_to_one(self):
        """Histogram entries sum to 1."""
        f = bf.majority(3)
        hist = sensitivity_histogram(f)
        assert abs(np.sum(hist) - 1.0) < 1e-10

    def test_length(self):
        """Histogram has n+1 entries (sensitivity can be 0..n)."""
        f = bf.majority(3)
        hist = sensitivity_histogram(f)
        assert len(hist) == 4  # n+1 = 4

    def test_parity_concentrated(self):
        """Parity(3): all mass at sensitivity = 3."""
        f = bf.parity(3)
        hist = sensitivity_histogram(f)
        assert abs(hist[3] - 1.0) < 1e-10
        assert abs(hist[0]) < 1e-10

    def test_constant_concentrated_at_zero(self):
        """Constant: all mass at sensitivity = 0."""
        f = bf.constant(True, 3)
        hist = sensitivity_histogram(f)
        assert abs(hist[0] - 1.0) < 1e-10

    def test_nonnegative(self):
        """All histogram entries are non-negative."""
        f = bf.majority(3)
        hist = sensitivity_histogram(f)
        assert np.all(hist >= -1e-10)


# ---------------------------------------------------------------------------
# Tests: arg_max_sensitivity and arg_min_sensitivity
# ---------------------------------------------------------------------------


class TestArgMaxSensitivity:
    """Tests for arg_max_sensitivity function."""

    def test_returns_correct_max(self):
        """Returned sensitivity equals max_sensitivity."""
        f = bf.majority(3)
        x, s = arg_max_sensitivity(f)
        assert s == max_sensitivity(f)

    def test_returned_input_achieves_max(self):
        """The returned input actually achieves the max sensitivity."""
        f = bf.AND(3)
        x, s = arg_max_sensitivity(f)
        assert sensitivity_at(f, x) == s

    def test_with_output_filter(self):
        """Filtering by output value works correctly."""
        f = bf.AND(3)
        x, s = arg_max_sensitivity(f, output_value=1)
        # Only x=7 has f(x)=1 for AND(3)
        assert x == 7
        assert s == 3


class TestArgMinSensitivity:
    """Tests for arg_min_sensitivity function."""

    def test_returns_correct_min(self):
        """Returned sensitivity equals min_sensitivity."""
        f = bf.majority(3)
        x, s = arg_min_sensitivity(f)
        assert s == min_sensitivity(f)

    def test_returned_input_achieves_min(self):
        """The returned input actually achieves the min sensitivity."""
        f = bf.AND(3)
        x, s = arg_min_sensitivity(f)
        assert sensitivity_at(f, x) == s

    def test_with_output_filter(self):
        """Filtering by output value works correctly."""
        f = bf.AND(3)
        x, s = arg_min_sensitivity(f, output_value=1)
        # Only x=7 has f(x)=1, so min on f=1 inputs is s(7) = 3
        assert x == 7
        assert s == 3


# ---------------------------------------------------------------------------
# Tests: on built-in functions
# ---------------------------------------------------------------------------


class TestOnBuiltinFunctions:
    """Test sensitivity on various built-in functions."""

    @pytest.mark.parametrize("n", [3, 5])
    def test_majority_average_sensitivity(self, n):
        """MAJ(n) average sensitivity = n * C(n-1, (n-1)/2) / 2^(n-1)."""
        from math import comb

        f = bf.majority(n)
        expected = n * comb(n - 1, (n - 1) // 2) / (2 ** (n - 1))
        assert abs(average_sensitivity(f) - expected) < 1e-10

    def test_tribes_positive_sensitivity(self):
        """Tribes has positive max sensitivity."""
        f = bf.tribes(2, 4)
        assert max_sensitivity(f) > 0

    def test_threshold_monotonicity(self):
        """Threshold(3, k=2) is the same as majority(3)."""
        f = bf.threshold(3, k=2)
        g = bf.majority(3)
        assert max_sensitivity(f) == max_sensitivity(g)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
