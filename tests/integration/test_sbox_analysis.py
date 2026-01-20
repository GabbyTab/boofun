"""
Integration tests for S-box (cryptographic) analysis.

S-boxes are substitution boxes used in block ciphers (AES, DES).
Their cryptographic properties can be analyzed using Boolean function theory.

This validates:
1. Nonlinearity computation matches known values
2. Property testing works on S-box component functions
3. Fourier analysis reveals security properties
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis import PropertyTester

# AES S-box (8-bit input/output)
# We analyze component functions (projections onto single output bits)
AES_SBOX = [
    0x63,
    0x7C,
    0x77,
    0x7B,
    0xF2,
    0x6B,
    0x6F,
    0xC5,
    0x30,
    0x01,
    0x67,
    0x2B,
    0xFE,
    0xD7,
    0xAB,
    0x76,
    0xCA,
    0x82,
    0xC9,
    0x7D,
    0xFA,
    0x59,
    0x47,
    0xF0,
    0xAD,
    0xD4,
    0xA2,
    0xAF,
    0x9C,
    0xA4,
    0x72,
    0xC0,
    0xB7,
    0xFD,
    0x93,
    0x26,
    0x36,
    0x3F,
    0xF7,
    0xCC,
    0x34,
    0xA5,
    0xE5,
    0xF1,
    0x71,
    0xD8,
    0x31,
    0x15,
    0x04,
    0xC7,
    0x23,
    0xC3,
    0x18,
    0x96,
    0x05,
    0x9A,
    0x07,
    0x12,
    0x80,
    0xE2,
    0xEB,
    0x27,
    0xB2,
    0x75,
    0x09,
    0x83,
    0x2C,
    0x1A,
    0x1B,
    0x6E,
    0x5A,
    0xA0,
    0x52,
    0x3B,
    0xD6,
    0xB3,
    0x29,
    0xE3,
    0x2F,
    0x84,
    0x53,
    0xD1,
    0x00,
    0xED,
    0x20,
    0xFC,
    0xB1,
    0x5B,
    0x6A,
    0xCB,
    0xBE,
    0x39,
    0x4A,
    0x4C,
    0x58,
    0xCF,
    0xD0,
    0xEF,
    0xAA,
    0xFB,
    0x43,
    0x4D,
    0x33,
    0x85,
    0x45,
    0xF9,
    0x02,
    0x7F,
    0x50,
    0x3C,
    0x9F,
    0xA8,
    0x51,
    0xA3,
    0x40,
    0x8F,
    0x92,
    0x9D,
    0x38,
    0xF5,
    0xBC,
    0xB6,
    0xDA,
    0x21,
    0x10,
    0xFF,
    0xF3,
    0xD2,
    0xCD,
    0x0C,
    0x13,
    0xEC,
    0x5F,
    0x97,
    0x44,
    0x17,
    0xC4,
    0xA7,
    0x7E,
    0x3D,
    0x64,
    0x5D,
    0x19,
    0x73,
    0x60,
    0x81,
    0x4F,
    0xDC,
    0x22,
    0x2A,
    0x90,
    0x88,
    0x46,
    0xEE,
    0xB8,
    0x14,
    0xDE,
    0x5E,
    0x0B,
    0xDB,
    0xE0,
    0x32,
    0x3A,
    0x0A,
    0x49,
    0x06,
    0x24,
    0x5C,
    0xC2,
    0xD3,
    0xAC,
    0x62,
    0x91,
    0x95,
    0xE4,
    0x79,
    0xE7,
    0xC8,
    0x37,
    0x6D,
    0x8D,
    0xD5,
    0x4E,
    0xA9,
    0x6C,
    0x56,
    0xF4,
    0xEA,
    0x65,
    0x7A,
    0xAE,
    0x08,
    0xBA,
    0x78,
    0x25,
    0x2E,
    0x1C,
    0xA6,
    0xB4,
    0xC6,
    0xE8,
    0xDD,
    0x74,
    0x1F,
    0x4B,
    0xBD,
    0x8B,
    0x8A,
    0x70,
    0x3E,
    0xB5,
    0x66,
    0x48,
    0x03,
    0xF6,
    0x0E,
    0x61,
    0x35,
    0x57,
    0xB9,
    0x86,
    0xC1,
    0x1D,
    0x9E,
    0xE1,
    0xF8,
    0x98,
    0x11,
    0x69,
    0xD9,
    0x8E,
    0x94,
    0x9B,
    0x1E,
    0x87,
    0xE9,
    0xCE,
    0x55,
    0x28,
    0xDF,
    0x8C,
    0xA1,
    0x89,
    0x0D,
    0xBF,
    0xE6,
    0x42,
    0x68,
    0x41,
    0x99,
    0x2D,
    0x0F,
    0xB0,
    0x54,
    0xBB,
    0x16,
]


def get_sbox_component(sbox, output_bit):
    """
    Extract a component function from an S-box.

    The component function is f(x) = bit_i(S(x))
    where S is the S-box and bit_i extracts the i-th bit.

    Args:
        sbox: List of S-box values
        output_bit: Which output bit to extract (0-7 for 8-bit)

    Returns:
        BooleanFunction representing the component
    """
    n = int(np.log2(len(sbox)))  # 8 for AES
    truth_table = [(sbox[x] >> output_bit) & 1 for x in range(2**n)]
    return bf.create(truth_table)


def compute_nonlinearity(f):
    """
    Compute nonlinearity = 2^(n-1) - max|Walsh(a)|/2.

    Higher nonlinearity = more resistant to linear cryptanalysis.
    """
    fourier = f.fourier()
    n = f.n_vars

    # Maximum absolute Walsh coefficient (excluding constant)
    max_walsh = max(abs(fourier[s]) for s in range(1, 2**n))

    # Nonlinearity formula
    nonlinearity = 2 ** (n - 1) - int(max_walsh * 2 ** (n - 1))
    return nonlinearity


class TestSboxBasics:
    """Test basic S-box component function analysis."""

    def test_aes_sbox_component_sizes(self):
        """AES S-box components should be 8-variable functions."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            assert f.n_vars == 8

    def test_aes_sbox_components_balanced(self):
        """AES S-box components should be balanced."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            assert f.is_balanced(), f"Component {bit} not balanced"

    def test_aes_sbox_components_nonlinear(self):
        """AES S-box components should be highly nonlinear."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)

            # AES S-box has nonlinearity 112 (maximum for 8-bit is 120)
            nl = compute_nonlinearity(f)
            assert nl >= 100, f"Component {bit} nonlinearity {nl} too low"


class TestSboxPropertyTesting:
    """Test property testing on S-box components."""

    def test_sbox_not_linear(self):
        """S-box components should fail linearity test."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)

        # AES S-box is highly nonlinear
        assert not tester.blr_linearity_test(num_queries=50)

    def test_sbox_not_affine(self):
        """S-box components should fail affine test."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)

        assert not tester.affine_test(num_queries=50)

    def test_sbox_not_symmetric(self):
        """S-box components should not be symmetric."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)

        assert not tester.symmetry_test(num_queries=50)

    def test_sbox_not_monotone(self):
        """S-box components should not be monotone."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)

        assert not tester.monotonicity_test(num_queries=50)


class TestSboxFourierAnalysis:
    """Test Fourier analysis on S-box components."""

    def test_sbox_high_degree(self):
        """AES S-box components have high Fourier degree."""
        f = get_sbox_component(AES_SBOX, 0)

        # AES S-box components should have high degree (near n=8)
        # Note: Fourier degree can differ from algebraic degree
        degree = f.degree()
        assert degree >= 6, f"Expected high degree, got {degree}"

    def test_sbox_parseval_identity(self):
        """Parseval's identity should hold."""
        f = get_sbox_component(AES_SBOX, 0)
        fourier = f.fourier()

        sum_sq = sum(c**2 for c in fourier)
        assert abs(sum_sq - 1.0) < 1e-10

    def test_sbox_spectral_weight_distribution(self):
        """S-box should have weight spread across degrees."""
        f = get_sbox_component(AES_SBOX, 0)
        weights = f.spectral_weight_by_degree()

        # Should have weight at multiple degrees (not concentrated)
        nonzero_degrees = sum(1 for w in weights.values() if w > 0.01)
        assert nonzero_degrees >= 3, "S-box spectral weight too concentrated"

    def test_sbox_variance(self):
        """S-box variance should be 1 (balanced function)."""
        f = get_sbox_component(AES_SBOX, 0)
        var = f.variance()

        # For balanced ±1 function, Var[f] = 1
        assert abs(var - 1.0) < 0.01


class TestSboxInfluences:
    """Test influence analysis on S-box components."""

    def test_sbox_total_influence(self):
        """S-box should have reasonably high total influence."""
        f = get_sbox_component(AES_SBOX, 0)

        # High degree functions typically have high total influence
        total_inf = f.total_influence()
        assert total_inf >= 2.0, f"Total influence {total_inf} too low"

    def test_sbox_influences_spread(self):
        """S-box influences should be spread across variables."""
        f = get_sbox_component(AES_SBOX, 0)
        influences = f.influences()

        # Check that no single variable dominates
        max_inf = max(influences)
        total_inf = sum(influences)

        # No variable should have more than 50% of total influence
        assert max_inf < 0.5 * total_inf


class TestSboxQueryComplexity:
    """Test query complexity measures on S-box components."""

    def test_sbox_decision_tree_depth(self):
        """S-box components should need many queries."""
        from boofun.analysis.query_complexity import deterministic_query_complexity

        f = get_sbox_component(AES_SBOX, 0)
        D = deterministic_query_complexity(f)

        # High degree function, should need close to n queries
        assert D >= 5, f"D(f)={D} unexpectedly low"

    def test_sbox_sensitivity(self):
        """S-box components should have high sensitivity."""
        f = get_sbox_component(AES_SBOX, 0)
        sens = f.sensitivity()

        # Good S-boxes have high sensitivity
        assert sens >= 4, f"Sensitivity {sens} too low"


class TestSimpleSbox:
    """Tests using smaller S-boxes for faster execution."""

    def test_4bit_sbox_analysis(self):
        """Test analysis on a 4-bit S-box (faster)."""
        # Simple 4-bit S-box
        sbox_4bit = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        f = bf.create(sbox_4bit)

        assert f.n_vars == 4
        assert f.is_balanced()

        # Fourier analysis
        fourier = f.fourier()
        assert abs(sum(c**2 for c in fourier) - 1.0) < 1e-10

        # Property testing
        tester = PropertyTester(f, random_seed=42)
        results = tester.run_all_tests()

        assert "linear" in results
        assert "monotone" in results

    def test_balanced_4bit(self):
        """Test a balanced 4-bit function."""
        # A balanced function
        balanced = [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
        f = bf.create(balanced)

        # Should be balanced (8 zeros, 8 ones)
        assert f.is_balanced()

        # Test that nonlinearity is non-negative
        nl = compute_nonlinearity(f)
        assert nl >= 0


class TestCrossValidation:
    """Cross-validate S-box properties with known results."""

    def test_aes_known_properties(self):
        """AES S-box has well-documented properties."""
        # AES S-box component 0 properties from literature
        f = get_sbox_component(AES_SBOX, 0)

        # Known: high degree (Fourier degree can be 8, algebraic degree 7)
        assert f.degree() >= 6

        # Known: balanced
        assert f.is_balanced()

        # Known: nonlinearity = 112
        nl = compute_nonlinearity(f)
        assert nl == 112, f"Expected nonlinearity 112, got {nl}"

    def test_all_aes_components_nl_112(self):
        """All AES S-box components should have nonlinearity 112."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            nl = compute_nonlinearity(f)
            assert nl == 112, f"Component {bit}: expected NL=112, got {nl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
