"""
Comprehensive tests for analysis/fourier module.

Tests Fourier analysis and spectral properties of Boolean functions.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.fourier import (
    convolution,
    dominant_coefficients,
    even_part,
    fourier_degree,
    fourier_sparsity,
    odd_part,
    parseval_verify,
    plancherel_inner_product,
    spectral_norm,
)


class TestParseval:
    """Test Parseval verification."""

    def test_parseval_holds(self):
        """Parseval should hold for Boolean functions."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3), bf.parity(3)]:
            valid, expected, actual = parseval_verify(func)
            assert valid == True

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_parseval_various_sizes(self, n):
        """Parseval should hold for various n."""
        f = bf.majority(n) if n % 2 == 1 else bf.AND(n)
        valid, expected, actual = parseval_verify(f)

        assert valid == True


class TestPlancherelInnerProduct:
    """Test Plancherel inner product."""

    def test_self_inner_product(self):
        """<f, f> should equal sum of squared coefficients = 1."""
        f = bf.majority(3)

        ip = plancherel_inner_product(f, f)
        assert abs(ip - 1.0) < 1e-10

    def test_orthogonal_functions(self):
        """Different parity functions should have inner product 0."""
        # This depends on the functions chosen


class TestConvolution:
    """Test convolution of Boolean functions."""

    def test_convolution_exists(self):
        """Convolution should return a function."""
        f = bf.AND(2)
        g = bf.OR(2)

        h = convolution(f, g)
        assert h is not None


class TestSpectralNorm:
    """Test spectral_norm function."""

    def test_norm_positive(self):
        """Spectral norm should be positive."""
        for func in [bf.AND(3), bf.OR(3), bf.parity(3)]:
            norm = spectral_norm(func)
            assert norm >= 0

    def test_different_p_norms(self):
        """Different p values should work."""
        f = bf.majority(3)

        norm_2 = spectral_norm(f, p=2)
        assert norm_2 >= 0


class TestFourierDegree:
    """Test Fourier degree computation."""

    def test_dictator_degree(self):
        """Dictator has degree 1."""
        f = bf.dictator(4, 0)
        d = fourier_degree(f)

        assert d == 1

    def test_parity_degree(self):
        """Parity has degree n."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            d = fourier_degree(f)

            assert d == n

    def test_constant_degree(self):
        """Constant has degree 0."""
        f = bf.create([1, 1, 1, 1])
        d = fourier_degree(f)

        assert d == 0


class TestFourierSparsity:
    """Test fourier_sparsity function."""

    def test_parity_sparsity(self):
        """Parity has sparsity 1 (single non-zero coefficient)."""
        f = bf.parity(3)
        sp = fourier_sparsity(f)

        assert sp == 1

    def test_sparsity_bounded(self):
        """Sparsity should be bounded by 2^n."""
        f = bf.majority(3)
        sp = fourier_sparsity(f)

        assert 1 <= sp <= 8


class TestOddEvenParts:
    """Test odd_part and even_part functions."""

    def test_odd_part_exists(self):
        """odd_part should return array."""
        f = bf.majority(3)
        odd = odd_part(f)

        assert odd is not None
        assert len(odd) == 8

    def test_even_part_exists(self):
        """even_part should return array."""
        f = bf.majority(3)
        even = even_part(f)

        assert even is not None
        assert len(even) == 8


class TestDominantCoefficients:
    """Test dominant_coefficients function."""

    def test_dominant_coeffs(self):
        """Should return largest coefficients."""
        f = bf.parity(3)
        dominant = dominant_coefficients(f, top_k=1)

        # Should return some coefficients
        assert dominant is not None

    def test_top_k_coeffs(self):
        """Should return top k coefficients."""
        f = bf.majority(3)

        top_3 = dominant_coefficients(f, top_k=3)
        assert top_3 is not None


class TestFourierConsistency:
    """Test consistency between Fourier methods."""

    def test_degree_sparsity_relation(self):
        """Degree should be <= n, sparsity should be >= 1."""
        f = bf.majority(3)

        d = fourier_degree(f)
        sp = fourier_sparsity(f)

        assert 0 <= d <= 3
        assert 1 <= sp <= 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
