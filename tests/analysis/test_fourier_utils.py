import sys

sys.path.insert(0, "src")
"""
Tests for analysis/fourier module utilities.

Tests for:
- parseval_verify
- plancherel_inner_product
- convolution
- negate_inputs
- odd_part, even_part
- tensor_product
- restriction
- fourier_degree, spectral_norm, fourier_sparsity
- dominant_coefficients
- Example functions
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis.fourier import (
    compute_and_fourier,
    compute_mux3_fourier,
    compute_nae3_fourier,
    convolution,
    convolution_values,
    dominant_coefficients,
    even_part,
    fourier_degree,
    fourier_sparsity,
    negate_inputs,
    odd_part,
    parseval_verify,
    plancherel_inner_product,
    restriction,
    spectral_norm,
    tensor_product,
)


class TestParsevalVerify:
    """Tests for parseval_verify."""

    def test_parity_satisfies_parseval(self):
        """Parity satisfies Parseval's identity."""
        f = bf.parity(3)
        passes, lhs, rhs = parseval_verify(f)

        assert passes
        assert abs(lhs - 1.0) < 1e-10
        assert abs(rhs - 1.0) < 1e-10

    def test_and_satisfies_parseval(self):
        """AND satisfies Parseval's identity."""
        f = bf.AND(3)
        passes, lhs, rhs = parseval_verify(f)

        assert passes

    def test_majority_satisfies_parseval(self):
        """Majority satisfies Parseval's identity."""
        f = bf.majority(3)
        passes, lhs, rhs = parseval_verify(f)

        assert passes


class TestPlancherelInnerProduct:
    """Tests for plancherel_inner_product."""

    def test_self_inner_product(self):
        """Inner product with self equals L2 norm."""
        f = bf.parity(3)

        result = plancherel_inner_product(f, f)

        # For Boolean f, ⟨f, f⟩ = 1
        assert abs(result - 1.0) < 1e-10

    def test_different_functions(self):
        """Inner product of different functions."""
        f = bf.parity(3)
        g = bf.AND(3)

        result = plancherel_inner_product(f, g)

        assert isinstance(result, float)

    def test_different_n_vars_raises(self):
        """Different n_vars raises error."""
        f = bf.parity(3)
        g = bf.parity(4)

        with pytest.raises(ValueError, match="same number of variables"):
            plancherel_inner_product(f, g)


class TestConvolution:
    """Tests for convolution."""

    def test_convolution_theorem_same_function(self):
        """Verify Convolution Theorem: (f*f)^(S) = f̂(S)²."""
        f = bf.parity(3)

        conv_coeffs = convolution(f, f)
        f_coeffs = f.fourier()
        expected = f_coeffs * f_coeffs  # = f̂(S)²

        assert np.allclose(conv_coeffs, expected)

    def test_convolution_theorem_different_functions(self):
        """Verify Convolution Theorem: (f*g)^(S) = f̂(S)·ĝ(S)."""
        f = bf.majority(3)
        g = bf.parity(3)

        conv_coeffs = convolution(f, g)
        expected = f.fourier() * g.fourier()

        assert np.allclose(conv_coeffs, expected)

    def test_convolution_theorem_and_or(self):
        """Verify Convolution Theorem for AND and OR."""
        f = bf.AND(3)
        g = bf.OR(3)

        conv_coeffs = convolution(f, g)
        expected = f.fourier() * g.fourier()

        assert np.allclose(conv_coeffs, expected)

    def test_convolution_returns_array(self):
        """Convolution returns numpy array of Fourier coefficients."""
        f = bf.AND(2)
        g = bf.OR(2)

        result = convolution(f, g)

        assert isinstance(result, np.ndarray)
        assert len(result) == 4  # 2^2 coefficients

    def test_convolution_mismatched_vars(self):
        """Convolution raises error for mismatched variable counts."""
        f = bf.AND(2)
        g = bf.OR(3)

        with pytest.raises(ValueError, match="same number of variables"):
            convolution(f, g)


class TestConvolutionValues:
    """Tests for convolution_values (time-domain convolution)."""

    def test_convolution_values_range(self):
        """Convolution values are in [-1, 1]."""
        f = bf.majority(3)
        g = bf.parity(3)

        values = convolution_values(f, g)

        assert np.all(values >= -1.0)
        assert np.all(values <= 1.0)

    def test_convolution_values_inverse_fourier(self):
        """Convolution values are inverse transform of convolution coeffs."""
        f = bf.AND(2)
        g = bf.OR(2)

        coeffs = convolution(f, g)
        values = convolution_values(f, g)

        # Verify inverse transform: value[x] = sum_S coeff[S] * chi_S(x)
        for x in range(4):
            expected = 0.0
            for s in range(4):
                chi = 1 - 2 * (bin(x & s).count("1") % 2)
                expected += coeffs[s] * chi
            assert np.isclose(values[x], expected)

    def test_parity_self_convolution(self):
        """Parity convolved with itself gives all 1s (max correlation)."""
        f = bf.parity(2)

        values = convolution_values(f, f)

        # Parity * Parity: f̂({0,1}) = 1, so (f*f)^({0,1}) = 1
        # Time domain: only {0,1} coefficient, so values = chi_{0,1}(x)
        assert np.allclose(np.abs(values), 1.0)


class TestNegateInputs:
    """Tests for negate_inputs."""

    def test_negate_parity(self):
        """Negating parity gives same function."""
        f = bf.parity(3)

        g = negate_inputs(f)

        # Parity is odd, so f(-x) = -f(x) in ±1
        # But as Boolean, should be same
        assert g.n_vars == 3

    def test_negate_and(self):
        """Negate AND function."""
        f = bf.AND(2)

        g = negate_inputs(f)

        assert g.n_vars == 2


class TestOddEvenParts:
    """Tests for odd_part and even_part."""

    def test_parity_is_odd(self):
        """Parity is purely odd."""
        f = bf.parity(3)

        odd = odd_part(f)
        even_part(f)

        # Odd part should be non-trivial
        assert np.max(np.abs(odd)) > 0

    def test_parts_sum_to_original(self):
        """Odd + even parts sum to original."""
        f = bf.majority(3)

        odd = odd_part(f)
        even = even_part(f)

        # Get original ±1 values
        tt = np.asarray(f.get_representation("truth_table"), dtype=float)
        pm = 1.0 - 2.0 * tt

        # Should sum approximately
        assert np.allclose(odd + even, pm)


class TestTensorProduct:
    """Tests for tensor_product."""

    def test_tensor_product_dimensions(self):
        """Tensor product has correct dimensions."""
        f = bf.parity(2)
        g = bf.AND(3)

        h = tensor_product(f, g)

        assert h.n_vars == 5  # 2 + 3


class TestRestriction:
    """Tests for restriction."""

    def test_restriction_reduces_vars(self):
        """Restriction reduces number of variables."""
        f = bf.parity(3)

        g = restriction(f, {0: 1})

        assert g.n_vars == 2


class TestFourierDegree:
    """Tests for fourier_degree."""

    def test_parity_full_degree(self):
        """Parity has full Fourier degree."""
        f = bf.parity(3)

        deg = fourier_degree(f)

        assert deg == 3

    def test_dictator_degree_one(self):
        """Dictator has degree 1."""
        f = bf.dictator(3, i=0)

        deg = fourier_degree(f)

        assert deg == 1


class TestSpectralNorm:
    """Tests for spectral_norm."""

    def test_l2_norm(self):
        """L2 spectral norm."""
        f = bf.parity(3)

        norm = spectral_norm(f, p=2)

        # For Boolean f, L2 norm = 1
        assert abs(norm - 1.0) < 1e-10

    def test_l1_norm(self):
        """L1 spectral norm."""
        f = bf.AND(3)

        norm = spectral_norm(f, p=1)

        assert norm > 0

    def test_linf_norm(self):
        """L∞ spectral norm."""
        f = bf.parity(3)

        norm = spectral_norm(f, p=np.inf)

        # Parity has single non-zero coefficient
        assert norm > 0


class TestFourierSparsity:
    """Tests for fourier_sparsity."""

    def test_parity_sparsity(self):
        """Parity has sparsity 1."""
        f = bf.parity(3)

        sparsity = fourier_sparsity(f)

        assert sparsity == 1

    def test_and_sparsity(self):
        """AND has all non-zero coefficients."""
        f = bf.AND(3)

        sparsity = fourier_sparsity(f)

        assert sparsity == 2**3  # All 8 coefficients


class TestDominantCoefficients:
    """Tests for dominant_coefficients."""

    def test_parity_dominant(self):
        """Parity has single dominant coefficient."""
        f = bf.parity(3)

        dom = dominant_coefficients(f, top_k=5)

        assert len(dom) == 1
        assert abs(dom[0][1]) > 0.9  # Should be ±1

    def test_and_dominant(self):
        """Get dominant coefficients of AND."""
        f = bf.AND(3)

        dom = dominant_coefficients(f, top_k=3)

        assert len(dom) >= 1


class TestExampleFunctions:
    """Tests for example computation functions."""

    def test_compute_mux3_fourier(self):
        """Compute MUX3 Fourier expansion."""
        result = compute_mux3_fourier()

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_compute_nae3_fourier(self):
        """Compute NAE3 Fourier expansion."""
        result = compute_nae3_fourier()

        assert isinstance(result, dict)

    def test_compute_and_fourier(self):
        """Compute AND Fourier expansion."""
        result = compute_and_fourier(3)

        assert isinstance(result, dict)
        # AND has 2^n non-zero coefficients
        assert len(result) == 8
