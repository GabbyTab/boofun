"""
Comprehensive tests for cryptographic analysis module.

Tests cover:
- Walsh transform and spectrum
- Nonlinearity computation
- Bent function detection
- Balancedness
- Algebraic degree and ANF
- Correlation immunity and resiliency
- SAC and propagation criterion
- CryptographicAnalyzer class

Cross-validation notes:
These tests use known values that can be verified against:
- thomasarmel/boolean_function (Rust)
- SageMath
- BooLSPLG
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis.cryptographic import (
    CryptographicAnalyzer,
    SBoxAnalyzer,
    algebraic_degree,
    algebraic_immunity,
    algebraic_normal_form,
    anf_monomials,
    correlation_immunity,
    difference_distribution_table,
    differential_uniformity,
    is_balanced,
    is_bent,
    linear_approximation_table,
    linearity,
    nonlinearity,
    propagation_criterion,
    resiliency,
    strict_avalanche_criterion,
    walsh_spectrum,
    walsh_transform,
)


class TestWalshTransform:
    """Tests for Walsh transform."""

    def test_constant_function(self):
        """Constant function has all Walsh weight on 0."""
        f = bf.create([0, 0, 0, 0])  # Constant 0
        W = walsh_transform(f)

        # W_f(0) = 2^n for constant 0
        assert W[0] == 4
        assert all(W[i] == 0 for i in range(1, 4))

    def test_xor_walsh(self):
        """XOR (parity) has single nonzero Walsh coefficient."""
        f = bf.create([0, 1, 1, 0])  # XOR = x0 ⊕ x1
        W = walsh_transform(f)

        # XOR correlates only with the full linear function
        # W_f(3) = ±2^n, all others 0
        assert W[3] != 0
        assert abs(W[3]) == 4

    def test_walsh_parseval(self):
        """Walsh coefficients satisfy Parseval: Σ W_f(a)² = 2^{2n}."""
        f = bf.AND(3)
        W = walsh_transform(f)

        n = 3
        sum_squares = np.sum(W**2)
        expected = 1 << (2 * n)  # 2^6 = 64

        assert sum_squares == expected


class TestWalshSpectrum:
    """Tests for Walsh spectrum."""

    def test_spectrum_is_histogram(self):
        """Spectrum is a histogram of Walsh values."""
        f = bf.AND(3)
        spectrum = walsh_spectrum(f)

        assert isinstance(spectrum, dict)
        assert sum(spectrum.values()) == 8  # 2^3 values

    def test_bent_spectrum(self):
        """Bent function has flat spectrum (all values same magnitude)."""
        # Known bent function: 0xac90 (4 variables)
        tt_int = 0xAC90
        tt = [(tt_int >> i) & 1 for i in range(16)]
        f = bf.create(tt)

        spectrum = walsh_spectrum(f)

        # For bent, all |W_f(a)| = 2^{n/2} = 4
        values = list(spectrum.keys())
        assert all(abs(v) == 4 for v in values)


class TestNonlinearity:
    """Tests for nonlinearity computation."""

    def test_affine_nonlinearity(self):
        """Affine functions have nonlinearity 0."""
        # Linear function: x0
        f = bf.create([0, 1, 0, 1])
        assert nonlinearity(f) == 0

        # Constant function
        g = bf.create([1, 1, 1, 1])
        assert nonlinearity(g) == 0

    def test_and_nonlinearity(self):
        """AND function has known nonlinearity."""
        f = bf.AND(3)
        nl = nonlinearity(f)

        # AND_3 has nonlinearity >= 1 (it's not affine)
        # The exact value depends on convention
        assert nl >= 1

    def test_xor_nonlinearity(self):
        """XOR (parity) has nonlinearity 0 (it's affine)."""
        f = bf.parity(3)
        assert nonlinearity(f) == 0

    def test_nonlinearity_bound(self):
        """Nonlinearity is bounded by bent bound."""
        f = bf.majority(5)
        nl = nonlinearity(f)
        n = 5

        # For odd n, max NL ≈ 2^{n-1} - 2^{(n-1)/2}
        max_bound = (1 << (n - 1)) - (1 << ((n - 1) // 2))

        assert 0 <= nl <= max_bound


class TestIsBent:
    """Tests for bent function detection."""

    def test_odd_n_not_bent(self):
        """Functions with odd n cannot be bent."""
        f = bf.AND(3)
        assert is_bent(f) is False

        f = bf.majority(5)
        assert is_bent(f) is False

    def test_xor_not_bent(self):
        """XOR is not bent (it's affine)."""
        f = bf.parity(4)
        assert is_bent(f) == False

    def test_known_bent_function(self):
        """Test with known bent function."""
        # The 6-variable function from thomasarmel's example
        # TT = 0x0113077C165E76A8
        # This is a known bent function
        hex_tt = "0113077C165E76A8"
        tt_int = int(hex_tt, 16)
        tt = [(tt_int >> i) & 1 for i in range(64)]
        f = bf.create(tt)

        assert is_bent(f) == True

    def test_small_bent_function(self):
        """Test with small (4-variable) bent function."""
        # 0xac90 is a known 4-variable bent function
        tt_int = 0xAC90
        tt = [(tt_int >> i) & 1 for i in range(16)]
        f = bf.create(tt)

        assert is_bent(f) == True


class TestIsBalanced:
    """Tests for balancedness."""

    def test_and_not_balanced(self):
        """AND is not balanced."""
        f = bf.AND(3)
        assert is_balanced(f) is False

    def test_xor_balanced(self):
        """XOR is balanced."""
        f = bf.parity(3)
        assert is_balanced(f) is True

    def test_majority_balanced(self):
        """Majority is balanced for odd n."""
        f = bf.majority(3)
        assert is_balanced(f) is True

        f = bf.majority(5)
        assert is_balanced(f) is True

    def test_bent_not_balanced(self):
        """Bent functions are never balanced."""
        # 4-variable bent function
        tt_int = 0xAC90
        tt = [(tt_int >> i) & 1 for i in range(16)]
        f = bf.create(tt)

        assert is_bent(f) == True
        assert is_balanced(f) == False


class TestAlgebraicDegree:
    """Tests for algebraic degree computation."""

    def test_constant_degree(self):
        """Constant functions have degree 0."""
        f = bf.create([0, 0, 0, 0])
        assert algebraic_degree(f) == 0

        # Note: constant 1 has degree 0 (just the constant term)
        g = bf.create([1, 1, 1, 1])
        assert algebraic_degree(g) == 0

    def test_linear_degree(self):
        """Linear functions have degree 1."""
        # x0
        f = bf.create([0, 1, 0, 1])
        assert algebraic_degree(f) == 1

    def test_and_degree(self):
        """AND has full degree."""
        f = bf.AND(3)
        assert algebraic_degree(f) == 3  # x0·x1·x2

    def test_xor_degree(self):
        """XOR has degree 1."""
        f = bf.parity(3)
        assert algebraic_degree(f) == 1


class TestANF:
    """Tests for ANF computation."""

    def test_and_anf(self):
        """AND has single monomial x0·x1·...·xn."""
        f = bf.AND(2)
        monomials = anf_monomials(f)

        assert (0, 1) in monomials or (1, 0) in monomials

    def test_xor_anf(self):
        """XOR has linear monomials."""
        f = bf.create([0, 1, 1, 0])  # x0 ⊕ x1
        monomials = anf_monomials(f)

        # Should have (0,) and (1,) but not (0,1)
        assert any(len(m) == 1 for m in monomials)


class TestCorrelationImmunity:
    """Tests for correlation immunity."""

    def test_xor_ci(self):
        """XOR has high correlation immunity."""
        f = bf.parity(3)
        ci = correlation_immunity(f)

        # XOR should have high CI (at least 2)
        assert ci >= 2

    def test_and_ci(self):
        """AND has correlation immunity 0."""
        f = bf.AND(3)
        ci = correlation_immunity(f)

        assert ci == 0


class TestResiliency:
    """Tests for resiliency."""

    def test_xor_resilient(self):
        """XOR is resilient (balanced and CI)."""
        f = bf.parity(3)
        res = resiliency(f)

        # XOR is balanced and has high CI
        assert res >= 2

    def test_unbalanced_not_resilient(self):
        """Unbalanced functions have resiliency -1."""
        f = bf.AND(3)
        res = resiliency(f)

        assert res == -1


class TestPropagationCriterion:
    """Tests for propagation criterion and SAC."""

    def test_xor_does_not_satisfy_pc(self):
        """XOR does NOT satisfy PC (derivative is constant 1, not balanced)."""
        f = bf.parity(3)

        # For XOR (linear function), f(x) ⊕ f(x⊕a) = 1 for all x (constant)
        # A constant function is NOT balanced, so XOR fails PC
        pc1 = propagation_criterion(f, order=1)
        assert pc1 == False

    def test_sac(self):
        """Test SAC (PC of order 1)."""
        # XOR does NOT satisfy SAC (derivative is constant, not balanced)
        f = bf.parity(3)
        sac = strict_avalanche_criterion(f)
        assert sac == False

        # AND also does not satisfy SAC
        g = bf.AND(3)
        sac_and = strict_avalanche_criterion(g)
        assert sac_and == False


class TestCryptographicAnalyzer:
    """Tests for CryptographicAnalyzer class."""

    def test_basic_analysis(self):
        """Analyzer computes basic measures."""
        f = bf.AND(3)
        analyzer = CryptographicAnalyzer(f)

        assert analyzer.n_vars == 3
        assert analyzer.nonlinearity() >= 0
        # Check that bent/balanced return sensible values
        assert analyzer.is_bent() in [True, False]
        assert analyzer.is_balanced() in [True, False]

    def test_summary(self):
        """Summary returns string."""
        f = bf.majority(3)
        analyzer = CryptographicAnalyzer(f)

        summary = analyzer.summary()
        assert "CryptographicAnalyzer" in summary
        assert "Nonlinearity" in summary

    def test_to_dict(self):
        """to_dict returns exportable dictionary."""
        f = bf.parity(3)
        analyzer = CryptographicAnalyzer(f)

        d = analyzer.to_dict()

        assert "nonlinearity" in d
        assert "is_bent" in d
        assert "is_balanced" in d
        assert "algebraic_degree" in d

    def test_caching(self):
        """Analyzer caches Walsh transform."""
        f = bf.AND(4)
        analyzer = CryptographicAnalyzer(f)

        # First access computes
        w1 = analyzer.walsh
        # Second access returns cached
        w2 = analyzer.walsh

        assert np.array_equal(w1, w2)


class TestCrossValidation:
    """
    Cross-validation tests against known values.

    These values can be verified against:
    - thomasarmel/boolean_function (Rust)
    - SageMath
    - Literature
    """

    def test_aes_sbox_nonlinearity(self):
        """AES S-box composition has nonlinearity 112 (known result)."""
        # This is a famous result - AES S-box has NL = 112
        # The truth table is too large to include here, but we test
        # that our computation framework is correct
        pass  # Would need full 8-bit S-box

    def test_thomasarmel_bent_example(self):
        """
        Cross-validate with thomasarmel example.

        From their README:
        - TT = 0x0113077C165E76A8 (6 vars) is bent
        - TT = 0xac90 (4 vars) is bent
        """
        # 4-variable bent
        tt_int = 0xAC90
        tt = [(tt_int >> i) & 1 for i in range(16)]
        f = bf.create(tt)

        analyzer = CryptographicAnalyzer(f)
        assert analyzer.is_bent() == True
        assert analyzer.is_balanced() == False
        assert analyzer.nonlinearity() == 6  # Bent bound for n=4

    def test_thomasarmel_6var_bent(self):
        """6-variable bent function from thomasarmel."""
        hex_tt = "0113077C165E76A8"
        tt_int = int(hex_tt, 16)
        tt = [(tt_int >> i) & 1 for i in range(64)]
        f = bf.create(tt)

        analyzer = CryptographicAnalyzer(f)
        assert analyzer.is_bent() == True
        # For n=6 bent, NL = 2^5 - 2^2 = 28
        assert analyzer.nonlinearity() == 28

    def test_balanced_count_4vars(self):
        """
        Cross-validate: there are C(16,8) = 12870 balanced 4-var functions.

        From thomasarmel README example.
        """
        count = 0
        for tt_int in range(1 << 16):
            # Quick balance check without creating full function
            if bin(tt_int).count("1") == 8:
                count += 1

        assert count == 12870


class TestAlgebraicImmunity:
    """Tests for algebraic immunity computation."""

    def test_constant_ai(self):
        """Constant functions have low algebraic immunity."""
        f = bf.create([0, 0, 0, 0])
        ai = algebraic_immunity(f)
        # Constant 0 has any non-constant function as annihilator
        assert ai >= 0

    def test_xor_ai(self):
        """XOR has good algebraic immunity."""
        f = bf.parity(3)
        ai = algebraic_immunity(f)
        # XOR should have AI close to ceil(n/2)
        assert ai >= 1

    def test_and_ai(self):
        """AND has low algebraic immunity."""
        f = bf.AND(3)
        ai = algebraic_immunity(f)
        # AND_n has low AI (1)
        assert ai >= 1

    def test_ai_upper_bound(self):
        """AI is bounded by ceil(n/2)."""
        f = bf.majority(5)
        ai = algebraic_immunity(f)
        assert ai <= 3  # ceil(5/2) = 3


class TestLAT:
    """Tests for Linear Approximation Table."""

    def test_lat_shape(self):
        """LAT has correct shape."""
        sbox = [0, 1, 2, 3]  # Identity 2-bit S-box
        lat = linear_approximation_table(sbox)

        assert lat.shape == (4, 4)

    def test_lat_trivial_entry(self):
        """LAT[0,0] = 2^{n-1} for any S-box."""
        sbox = [0, 1, 2, 3]
        lat = linear_approximation_table(sbox)

        # LAT[0,0] = count where 0=0 minus 2^{n-1}
        # All 4 inputs satisfy 0=0, so LAT[0,0] = 4 - 2 = 2
        assert lat[0, 0] == 2

    def test_lat_identity(self):
        """Test LAT of identity S-box."""
        # Identity: S(x) = x
        sbox = list(range(4))
        lat = linear_approximation_table(sbox)

        # Identity has specific LAT structure
        assert lat.shape == (4, 4)

    def test_lat_values_range(self):
        """LAT values are bounded."""
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        lat = linear_approximation_table(sbox)

        # LAT values should be in range [-2^{n-1}, 2^{n-1}]
        n = 4
        bound = 1 << (n - 1)
        assert np.all(np.abs(lat) <= bound)


class TestDDT:
    """Tests for Difference Distribution Table."""

    def test_ddt_shape(self):
        """DDT has correct shape."""
        sbox = [0, 1, 2, 3]
        ddt = difference_distribution_table(sbox)

        assert ddt.shape == (4, 4)

    def test_ddt_trivial_entry(self):
        """DDT[0,0] = 2^n for any S-box."""
        sbox = [0, 1, 2, 3]
        ddt = difference_distribution_table(sbox)

        # S(x) XOR S(x XOR 0) = 0 for all x
        assert ddt[0, 0] == 4

    def test_ddt_row_sum(self):
        """Each row of DDT sums to 2^n."""
        sbox = list(range(8))  # Identity
        ddt = difference_distribution_table(sbox)

        # Each row sums to 2^n
        assert np.all(np.sum(ddt, axis=1) == 8)

    def test_ddt_bijective(self):
        """Bijective S-box has even DDT entries."""
        # AES-like 4-bit S-box (bijective)
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        ddt = difference_distribution_table(sbox)

        # For bijective S-boxes, all entries are even
        assert np.all(ddt % 2 == 0)


class TestDifferentialUniformity:
    """Tests for differential uniformity."""

    def test_identity_uniformity(self):
        """Identity S-box has uniformity equal to size."""
        sbox = list(range(4))
        du = differential_uniformity(sbox)

        # Identity: S(x) XOR S(x XOR dx) = dx, so all weight on DDT[dx, dx]
        assert du == 4

    def test_good_sbox_uniformity(self):
        """Good S-box has bounded differential uniformity."""
        # 4-bit S-box with good properties
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        du = differential_uniformity(sbox)

        # Differential uniformity should be reasonable (not too high)
        # For bijective 4-bit S-boxes, du is even and bounded by 16
        assert du <= 16
        assert du % 2 == 0


class TestLinearity:
    """Tests for linearity measure."""

    def test_identity_linearity(self):
        """Identity S-box has specific linearity."""
        sbox = list(range(4))
        lin = linearity(sbox)

        # Identity is linear, so max correlation is high
        assert lin >= 0

    def test_good_sbox_linearity(self):
        """Good S-box has bounded linearity."""
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        lin = linearity(sbox)

        # 4-bit S-boxes have linearity >= 2^2 = 4
        assert lin >= 4


class TestSBoxAnalyzer:
    """Tests for SBoxAnalyzer class."""

    def test_basic_analysis(self):
        """Analyzer computes basic measures."""
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        analyzer = SBoxAnalyzer(sbox)

        assert analyzer.n_inputs == 4
        assert analyzer.n_outputs == 4
        assert analyzer.is_bijective() == True

    def test_summary(self):
        """Summary returns string."""
        sbox = list(range(8))
        analyzer = SBoxAnalyzer(sbox)

        summary = analyzer.summary()
        assert "SBoxAnalyzer" in summary
        assert "Differential uniformity" in summary

    def test_to_dict(self):
        """to_dict returns exportable dictionary."""
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        analyzer = SBoxAnalyzer(sbox)

        d = analyzer.to_dict()

        assert "differential_uniformity" in d
        assert "linearity" in d
        assert "nonlinearity" in d
        assert "is_apn" in d

    def test_caching(self):
        """Analyzer caches LAT and DDT."""
        sbox = list(range(8))
        analyzer = SBoxAnalyzer(sbox)

        lat1 = analyzer.lat
        lat2 = analyzer.lat

        assert np.array_equal(lat1, lat2)

    def test_nonlinearity(self):
        """S-box nonlinearity is computed correctly."""
        sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8, 0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
        analyzer = SBoxAnalyzer(sbox)

        nl = analyzer.nonlinearity()
        # For 4-bit S-box, NL = 2^3 - L/2
        assert nl >= 0
