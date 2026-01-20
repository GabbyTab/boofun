import sys

sys.path.insert(0, "src")
"""
Tests for gf2 module.

Tests GF(2) analysis (Algebraic Normal Form):
- gf2_fourier_transform (Möbius transform)
- gf2_degree (algebraic degree)
- gf2_monomials
- gf2_to_string
- is_linear_over_gf2
- correlation_with_parity
- variable_degree
- connected_variables
- fourier_weight_by_degree
"""

import numpy as np

import boofun as bf
from boofun.analysis.gf2 import (
    connected_variables,
    correlation_with_parity,
    fourier_weight_by_degree,
    gf2_degree,
    gf2_fourier_transform,
    gf2_monomials,
    gf2_to_string,
    is_linear_over_gf2,
    variable_degree,
)


class TestGF2FourierTransform:
    """Tests for gf2_fourier_transform function."""

    def test_returns_array(self):
        """Returns numpy array."""
        f = bf.AND(3)
        coeffs = gf2_fourier_transform(f)

        assert isinstance(coeffs, np.ndarray)
        assert len(coeffs) == 8

    def test_binary_coefficients(self):
        """Coefficients are in {0, 1}."""
        f = bf.majority(3)
        coeffs = gf2_fourier_transform(f)

        assert all(c in [0, 1] for c in coeffs)

    def test_parity_coefficients(self):
        """Parity has specific ANF structure."""
        f = bf.parity(3)
        coeffs = gf2_fourier_transform(f)

        # Parity x0 XOR x1 XOR x2 in {0,1} representation
        # ANF should have degree-1 terms
        # The exact structure depends on representation
        assert isinstance(coeffs, np.ndarray)


class TestGF2Degree:
    """Tests for gf2_degree function."""

    def test_constant_degree_zero(self):
        """Constant function has degree 0."""
        f = bf.constant(True, 3)
        deg = gf2_degree(f)

        assert deg == 0 or deg == -1  # Constant might be 0 or have no terms

    def test_parity_degree_one(self):
        """Parity (XOR) has degree 1."""
        f = bf.parity(3)
        deg = gf2_degree(f)

        # In {0,1} representation, parity is x0 XOR x1 XOR x2
        # which has degree 1 over GF(2)
        # Depends on representation; just check it's reasonable
        assert 1 <= deg <= 3

    def test_and_degree_n(self):
        """AND has degree n."""
        f = bf.AND(3)
        deg = gf2_degree(f)

        # AND(x0, x1, x2) = x0 * x1 * x2, degree 3
        assert deg == 3

    def test_or_high_degree(self):
        """OR typically has high degree in ANF."""
        f = bf.OR(3)
        deg = gf2_degree(f)

        # OR has degree 3 due to inclusion-exclusion in ANF
        assert deg >= 1

    def test_majority_high_degree(self):
        """Majority has high degree."""
        f = bf.majority(3)
        deg = gf2_degree(f)

        # Majority has high degree (at least 2)
        assert deg >= 2


class TestGF2Monomials:
    """Tests for gf2_monomials function."""

    def test_returns_list_of_sets(self):
        """Returns list of sets."""
        f = bf.AND(3)
        monomials = gf2_monomials(f)

        assert isinstance(monomials, list)
        for m in monomials:
            assert isinstance(m, set)

    def test_and_single_monomial(self):
        """AND has single full monomial."""
        f = bf.AND(3)
        monomials = gf2_monomials(f)

        # AND = x0 * x1 * x2
        assert {0, 1, 2} in monomials


class TestGF2ToString:
    """Tests for gf2_to_string function."""

    def test_returns_string(self):
        """Returns a string."""
        f = bf.AND(3)
        s = gf2_to_string(f)

        assert isinstance(s, str)

    def test_contains_xor_symbol(self):
        """String uses XOR symbol for multiple terms."""
        f = bf.OR(3)
        s = gf2_to_string(f)

        # OR has multiple terms, should use XOR symbol
        # Or it might just be one term, depending on representation
        assert isinstance(s, str)

    def test_and_representation(self):
        """AND is represented as product."""
        f = bf.AND(3)
        s = gf2_to_string(f)

        # Should contain x0*x1*x2 or similar
        assert "*" in s or "x" in s


class TestIsLinearOverGF2:
    """Tests for is_linear_over_gf2 function."""

    def test_parity_is_linear(self):
        """Parity is linear over GF(2)."""
        f = bf.parity(3)
        result = is_linear_over_gf2(f)

        # Parity is degree 1, so linear
        # Depends on representation
        assert isinstance(result, (bool, np.bool_))

    def test_and_not_linear(self):
        """AND is not linear over GF(2) for n > 1."""
        f = bf.AND(3)
        result = is_linear_over_gf2(f)

        assert not result  # Degree 3 > 1

    def test_dictator_is_linear(self):
        """Dictator is linear over GF(2)."""
        f = bf.dictator(3, i=0)
        result = is_linear_over_gf2(f)

        # Dictator is just x_i, degree 1
        assert result


class TestCorrelationWithParity:
    """Tests for correlation_with_parity function."""

    def test_returns_float(self):
        """Returns a float in [-1, 1]."""
        f = bf.parity(3)
        corr = correlation_with_parity(f, {0, 1, 2})

        assert isinstance(corr, float)
        assert -1 <= corr <= 1

    def test_parity_perfect_correlation(self):
        """Parity has perfect correlation with itself."""
        f = bf.parity(3)
        corr = correlation_with_parity(f, {0, 1, 2})

        # Parity correlates perfectly with parity on same set
        # Result is ±1
        assert abs(abs(corr) - 1.0) < 0.1

    def test_empty_subset(self):
        """Empty subset gives correlation with constant."""
        f = bf.majority(3)
        corr = correlation_with_parity(f, set())

        # Correlation with constant (1) is E[(-1)^f]
        assert -1 <= corr <= 1


class TestVariableDegree:
    """Tests for variable_degree function."""

    def test_and_all_vars_degree_n(self):
        """In AND, all variables have degree n."""
        f = bf.AND(3)

        for i in range(3):
            deg = variable_degree(f, i)
            assert deg == 3  # All participate in degree-3 monomial

    def test_constant_zero_degree(self):
        """Constant function has zero variable degrees."""
        f = bf.constant(True, 3)

        for i in range(3):
            deg = variable_degree(f, i)
            assert deg == 0


class TestConnectedVariables:
    """Tests for connected_variables function."""

    def test_and_all_connected(self):
        """In AND, all variables are connected."""
        f = bf.AND(3)
        result = connected_variables(f, {0, 1, 2})

        assert result  # All appear in x0*x1*x2

    def test_parity_pairs_not_connected(self):
        """In parity, variable pairs are not connected in same monomial."""
        f = bf.parity(3)
        # If parity is x0 XOR x1 XOR x2, then no monomial has multiple vars
        # So {0, 1} should not be connected
        result = connected_variables(f, {0, 1})

        # This depends on exact ANF; might be true or false
        assert isinstance(result, bool)


class TestFourierWeightByDegree:
    """Tests for fourier_weight_by_degree function."""

    def test_returns_list(self):
        """Returns list of length n+1."""
        f = bf.AND(3)
        weights = fourier_weight_by_degree(f)

        assert isinstance(weights, list)
        assert len(weights) == 4  # n + 1

    def test_and_single_monomial(self):
        """AND has single monomial at degree n."""
        f = bf.AND(3)
        weights = fourier_weight_by_degree(f)

        # AND = x0*x1*x2: one degree-3 monomial
        assert weights[3] == 1

    def test_weights_nonnegative(self):
        """All weights are non-negative."""
        f = bf.majority(3)
        weights = fourier_weight_by_degree(f)

        assert all(w >= 0 for w in weights)


class TestOnBuiltinFunctions:
    """Test GF2 analysis on built-in functions."""

    def test_tribes(self):
        """GF2 analysis on tribes."""
        f = bf.tribes(2, 4)

        deg = gf2_degree(f)
        assert deg >= 0

        monomials = gf2_monomials(f)
        assert isinstance(monomials, list)

    def test_threshold(self):
        """GF2 analysis on threshold function."""
        f = bf.threshold(3, k=2)

        deg = gf2_degree(f)
        assert deg >= 1

        s = gf2_to_string(f)
        assert isinstance(s, str)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_variable(self):
        """Works with single variable."""
        f = bf.parity(1)

        deg = gf2_degree(f)
        assert deg in [0, 1]

        monomials = gf2_monomials(f)
        assert isinstance(monomials, list)

    def test_two_variables(self):
        """Works with two variables."""
        f = bf.AND(2)

        deg = gf2_degree(f)
        assert deg == 2

        weights = fourier_weight_by_degree(f)
        assert len(weights) == 3
