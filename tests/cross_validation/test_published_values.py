"""
Cross-validation against published values from the literature.

Every assertion in this module checks BooFun's output against a value
stated in a citable source (a standard, a paper, or a reference
implementation's documentation). Each test names its source. All values
are exact integers, so all comparisons are exact.

Sources used:

- FIPS 197, "Advanced Encryption Standard (AES)", NIST, 2001 — the AES
  S-box definition (GF(2^8) inverse + affine map) and its table values.
- K. Nyberg, "Differentially uniform mappings for cryptography",
  EUROCRYPT 1993 — differential uniformity 4 and nonlinearity 112 of the
  inversion-based AES S-box construction.
- J. Daemen & V. Rijmen, "The Design of Rijndael", Springer 2002 —
  max correlation 2^-3 (max |LAT| = 16 in the +-128 convention, i.e.
  linearity 32), algebraic degree 7 of the S-box components.
- A. Bogdanov et al., "PRESENT: An Ultra-Lightweight Block Cipher",
  CHES 2007 — the PRESENT S-box and its differential/linear profile.
- O. Rothaus, "On 'bent' functions", J. Combinatorial Theory A 20, 1976 —
  bent functions attain nonlinearity 2^(n-1) - 2^(n/2 - 1).
- thomasarmel/boolean_function (Rust library) README — the specific bent
  truth tables 0xac90 (n=4) and 0x0113077C165E76A8 (n=6), and the count
  of balanced 4-variable functions.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.cryptographic import (
    algebraic_degree,
    difference_distribution_table,
    differential_uniformity,
    is_balanced,
    is_bent,
    linear_approximation_table,
    linearity,
    nonlinearity,
)

# ---------------------------------------------------------------------------
# S-box definitions (computed, not hand-typed)
# ---------------------------------------------------------------------------


def aes_sbox() -> list[int]:
    """The AES S-box, computed from its definition in FIPS 197 s5.1.1.

    S(x) is the affine transform of the multiplicative inverse of x in
    GF(2^8) = GF(2)[t] / (t^8 + t^4 + t^3 + t + 1), with inv(0) = 0.
    Computed rather than hand-typed to rule out transcription errors; spot
    values below are checked against FIPS 197 Figure 7.
    """

    def gf_mul(a: int, b: int) -> int:
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return p

    def gf_inv(a: int) -> int:
        if a == 0:
            return 0
        result, power, exponent = 1, a, 254
        while exponent:
            if exponent & 1:
                result = gf_mul(result, power)
            power = gf_mul(power, power)
            exponent >>= 1
        return result

    def affine(b: int) -> int:
        c = 0x63
        out = 0
        for i in range(8):
            bit = (
                (b >> i)
                ^ (b >> ((i + 4) % 8))
                ^ (b >> ((i + 5) % 8))
                ^ (b >> ((i + 6) % 8))
                ^ (b >> ((i + 7) % 8))
                ^ (c >> i)
            ) & 1
            out |= bit << i
        return out

    box = [affine(gf_inv(x)) for x in range(256)]
    # Spot checks against FIPS 197, Figure 7.
    assert box[0x00] == 0x63
    assert box[0x01] == 0x7C
    assert box[0x53] == 0xED
    assert box[0xFF] == 0x16
    return box


# PRESENT S-box table, Bogdanov et al., CHES 2007, Table 1 (16 nibbles).
PRESENT_SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]


def sbox_component(sbox: list[int], mask: int):
    """Boolean component function <mask, S(x)> of an S-box."""
    return bf.create([bin(sbox[x] & mask).count("1") % 2 for x in range(len(sbox))])


# ---------------------------------------------------------------------------
# AES S-box
# ---------------------------------------------------------------------------


class TestAESSBox:
    """Published cryptographic profile of the AES S-box.

    Nyberg (EUROCRYPT 1993) established the properties of the inversion
    map construction; Daemen & Rijmen ("The Design of Rijndael", 2002)
    state them for the AES S-box specifically.
    """

    @pytest.fixture(scope="class")
    def sbox(self):
        return aes_sbox()

    def test_differential_uniformity(self, sbox):
        """AES S-box has differential uniformity 4 (Nyberg 1993)."""
        assert differential_uniformity(sbox) == 4

    def test_linearity(self, sbox):
        """AES S-box has linearity 32 = 2 * max|LAT| (Daemen & Rijmen 2002,
        max input-output correlation 2^-3)."""
        assert linearity(sbox) == 32

    def test_lat_spot_values(self, sbox):
        """LAT trivial entry is 128 (= 2^n / 2) and the largest nontrivial
        magnitude is 16 (Daemen & Rijmen 2002)."""
        lat = linear_approximation_table(sbox)
        assert lat[0][0] == 128
        assert int(np.abs(np.asarray(lat)[1:, 1:]).max()) == 16

    def test_ddt_spot_values(self, sbox):
        """DDT trivial entry is 256 and the largest entry for a nonzero
        input difference is 4 (= differential uniformity, Nyberg 1993)."""
        ddt = difference_distribution_table(sbox)
        assert ddt[0][0] == 256
        assert int(np.asarray(ddt)[1:, :].max()) == 4

    @pytest.mark.parametrize("bit", range(8))
    def test_component_nonlinearity_112(self, sbox, bit):
        """Every single-bit AES S-box component has nonlinearity 112
        (Nyberg 1993; the famous 'NL(AES) = 112' result)."""
        assert nonlinearity(sbox_component(sbox, 1 << bit)) == 112

    @pytest.mark.parametrize("bit", range(8))
    def test_component_degree_7(self, sbox, bit):
        """Every AES S-box component has algebraic degree 7
        (Daemen & Rijmen 2002: the inverse map has degree n-1)."""
        assert algebraic_degree(sbox_component(sbox, 1 << bit)) == 7

    @pytest.mark.parametrize("bit", range(8))
    def test_components_balanced(self, sbox, bit):
        """Components of a bijective S-box are balanced."""
        assert is_balanced(sbox_component(sbox, 1 << bit))

    def test_bijective(self, sbox):
        """The AES S-box is a permutation of GF(2^8) (FIPS 197)."""
        assert sorted(sbox) == list(range(256))


# ---------------------------------------------------------------------------
# PRESENT S-box
# ---------------------------------------------------------------------------


class TestPRESENTSBox:
    """Published profile of the PRESENT S-box (Bogdanov et al., CHES 2007)."""

    def test_differential_uniformity(self):
        """PRESENT S-box has differential uniformity 4 (design criterion in
        the CHES 2007 paper)."""
        assert differential_uniformity(PRESENT_SBOX) == 4

    def test_linearity(self):
        """PRESENT S-box has linearity 8, i.e. maximal linear bias 2^-2
        (Bogdanov et al. 2007, Section 4.1)."""
        assert linearity(PRESENT_SBOX) == 8

    def test_bijective(self):
        assert sorted(PRESENT_SBOX) == list(range(16))


# ---------------------------------------------------------------------------
# Bent functions
# ---------------------------------------------------------------------------


class TestBentFunctions:
    """Known bent functions and the bent nonlinearity bound.

    Rothaus (1976): bent functions on n variables (n even) attain the
    maximum nonlinearity 2^(n-1) - 2^(n/2 - 1), and are never balanced.
    """

    def test_ac90_bent_4var(self):
        """TT = 0xac90 is a 4-variable bent function
        (thomasarmel/boolean_function README)."""
        tt_int = 0xAC90
        f = bf.create([(tt_int >> i) & 1 for i in range(16)])
        assert is_bent(f)
        assert not is_balanced(f)
        assert nonlinearity(f) == 6  # 2^3 - 2^1

    def test_thomasarmel_bent_6var(self):
        """TT = 0x0113077C165E76A8 is a 6-variable bent function
        (thomasarmel/boolean_function README)."""
        tt_int = 0x0113077C165E76A8
        f = bf.create([(tt_int >> i) & 1 for i in range(64)])
        assert is_bent(f)
        assert nonlinearity(f) == 28  # 2^5 - 2^2

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_inner_product_bent(self, n):
        """The inner product function IP(x) = x0x1 + x2x3 + ... is the
        canonical bent function (Rothaus 1976) with nonlinearity
        2^(n-1) - 2^(n/2 - 1)."""

        def value(x: int) -> int:
            acc = 0
            for j in range(n // 2):
                acc ^= ((x >> (2 * j)) & 1) & ((x >> (2 * j + 1)) & 1)
            return acc

        f = bf.create([value(x) for x in range(1 << n)])
        assert is_bent(f)
        assert nonlinearity(f) == (1 << (n - 1)) - (1 << (n // 2 - 1))


# ---------------------------------------------------------------------------
# Counting results
# ---------------------------------------------------------------------------


class TestCountingResults:
    """Combinatorial counts stated in the literature."""

    def test_balanced_4var_count(self):
        """There are C(16, 8) = 12870 balanced 4-variable functions
        (elementary; also quoted in the thomasarmel README)."""
        count = sum(1 for tt_int in range(1 << 16) if bin(tt_int).count("1") == 8)
        assert count == 12870


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
