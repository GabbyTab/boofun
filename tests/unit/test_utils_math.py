import sys

sys.path.insert(0, "src")
"""
Tests for utils/math module.

Tests lightweight math helpers:
- popcnt, poppar
- over (binomial coefficient)
- subsets, cartesian
- num2bin_list, bits
- tensor_product
- krawchouk, krawchouk2
- hamming_distance, hamming_weight
- generate_permutations
- int_to_binary_tuple, binary_tuple_to_int
"""

import pytest
import numpy as np

from boofun.utils import math as math_utils


# ---------------------------------------------------------------------------
# Tests: popcnt and poppar
# ---------------------------------------------------------------------------


class TestPopcnt:
    """Tests for popcnt and poppar."""

    def test_popcnt_basic(self):
        """Population count of known values."""
        assert math_utils.popcnt(0b1011) == 3
        assert math_utils.popcnt(0) == 0
        assert math_utils.popcnt(0b1111) == 4
        assert math_utils.popcnt(1) == 1

    def test_poppar_basic(self):
        """Parity of population count."""
        assert math_utils.poppar(0b1011) == 1  # 3 bits -> odd
        assert math_utils.poppar(0b1001) == 0  # 2 bits -> even
        assert math_utils.poppar(0) == 0

    def test_popcnt_large(self):
        """Population count of larger numbers."""
        assert math_utils.popcnt(255) == 8
        assert math_utils.popcnt(256) == 1


# ---------------------------------------------------------------------------
# Tests: over (binomial coefficient)
# ---------------------------------------------------------------------------


class TestOver:
    """Tests for over (safe binomial coefficient)."""

    def test_standard_values(self):
        """Known binomial coefficients."""
        assert math_utils.over(5, 2) == 10
        assert math_utils.over(4, 2) == 6
        assert math_utils.over(10, 0) == 1
        assert math_utils.over(10, 10) == 1

    def test_out_of_bounds(self):
        """Out-of-bounds returns 0."""
        assert math_utils.over(5, -1) == 0
        assert math_utils.over(5, 6) == 0

    def test_symmetry(self):
        """C(n,k) = C(n, n-k)."""
        assert math_utils.over(7, 2) == math_utils.over(7, 5)


# ---------------------------------------------------------------------------
# Tests: subsets and cartesian
# ---------------------------------------------------------------------------


class TestSubsets:
    """Tests for subsets generator."""

    def test_all_subsets(self):
        """All subsets of [0,1,2] has 2^3 = 8 elements."""
        all_subs = list(math_utils.subsets([0, 1, 2]))
        assert len(all_subs) == 8

    def test_fixed_size(self):
        """Size-2 subsets of [0,1,2]."""
        two_subs = list(math_utils.subsets([0, 1, 2], 2))
        assert sorted(two_subs) == [(0, 1), (0, 2), (1, 2)]

    def test_integer_input(self):
        """Subsets accept integer as shorthand for range(n)."""
        all_subs = list(math_utils.subsets(3))
        assert len(all_subs) == 8

    def test_empty_subsets(self):
        """Size-0 subsets is just the empty tuple."""
        subs = list(math_utils.subsets([0, 1], 0))
        assert subs == [()]


class TestCartesian:
    """Tests for cartesian product."""

    def test_basic_product(self):
        """Cartesian product of [0,1] x [0,1]."""
        result = list(math_utils.cartesian([[0, 1], [0, 1]]))
        assert len(result) == 4
        assert (0, 0) in result
        assert (1, 1) in result

    def test_different_sizes(self):
        """Product of sequences with different sizes."""
        result = list(math_utils.cartesian([[0, 1], [0, 1, 2]]))
        assert len(result) == 6


# ---------------------------------------------------------------------------
# Tests: num2bin_list and bits
# ---------------------------------------------------------------------------


class TestNum2BinList:
    """Tests for num2bin_list (MSB first)."""

    def test_basic_conversion(self):
        """5 in 3-digit binary is [1, 0, 1]."""
        assert math_utils.num2bin_list(5, 3) == [1, 0, 1]

    def test_zero(self):
        """0 in any width is all zeros."""
        assert math_utils.num2bin_list(0, 4) == [0, 0, 0, 0]

    def test_all_ones(self):
        """7 in 3-digit binary is [1, 1, 1]."""
        assert math_utils.num2bin_list(7, 3) == [1, 1, 1]


class TestBits:
    """Tests for bits function (MSB first)."""

    def test_basic(self):
        """bits(5, 3) = [1, 0, 1]."""
        assert math_utils.bits(5, 3) == [1, 0, 1]

    def test_zero(self):
        """bits(0, 3) = [0, 0, 0]."""
        assert math_utils.bits(0, 3) == [0, 0, 0]

    def test_length(self):
        """Output always has n elements."""
        for n in range(1, 6):
            assert len(math_utils.bits(0, n)) == n


# ---------------------------------------------------------------------------
# Tests: tensor_product
# ---------------------------------------------------------------------------


class TestTensorProduct:
    """Tests for tensor_product (Kronecker product)."""

    def test_matches_numpy_kron(self):
        """Should match np.kron for matrices."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[0, 5], [6, 7]])
        assert np.allclose(math_utils.tensor_product(a, b), np.kron(a, b))

    def test_identity_kron(self):
        """I_2 kron I_2 = I_4."""
        I2 = np.eye(2)
        result = math_utils.tensor_product(I2, I2)
        # I_2 kron I_2 is a block diagonal with I_2 blocks
        assert result.shape == (4, 4)

    def test_accepts_lists(self):
        """Accepts plain Python lists as input."""
        result = math_utils.tensor_product([[1, 0], [0, 1]], [[1]])
        # kron of 2x2 identity with 1x1 scalar [[1]] gives the same 2x2 matrix
        assert result.shape == (2, 2)
        assert np.allclose(result, np.eye(2))


# ---------------------------------------------------------------------------
# Tests: krawchouk polynomials
# ---------------------------------------------------------------------------


class TestKrawchouk:
    """Tests for Krawchouk polynomial K_k(x; n)."""

    def test_known_values(self):
        """K_1(x; 3) = 3 - 2x."""
        values = [math_utils.krawchouk(3, 1, x) for x in range(4)]
        assert values == [3, 1, -1, -3]

    def test_k_zero(self):
        """K_0(x; n) = 1 for all x."""
        for x in range(5):
            assert math_utils.krawchouk(4, 0, x) == 1

    def test_krawchouk2_basic(self):
        """Legacy variant with (-2)^j weights."""
        assert math_utils.krawchouk2(3, 0, 2) == 1


# ---------------------------------------------------------------------------
# Tests: hamming_distance and hamming_weight
# ---------------------------------------------------------------------------


class TestHammingDistance:
    """Tests for hamming_distance."""

    def test_same_inputs(self):
        """Distance between x and x is 0."""
        assert math_utils.hamming_distance(5, 5) == 0

    def test_known_distance(self):
        """Distance between 0b1010 and 0b0110 is 2."""
        assert math_utils.hamming_distance(0b1010, 0b0110) == 2

    def test_symmetry(self):
        """hamming_distance(x, y) == hamming_distance(y, x)."""
        assert math_utils.hamming_distance(3, 7) == math_utils.hamming_distance(7, 3)

    def test_zero_and_all_ones(self):
        """Distance between 0 and 2^n - 1 is n."""
        assert math_utils.hamming_distance(0, 0b1111) == 4

    def test_triangle_inequality(self):
        """d(x,z) <= d(x,y) + d(y,z)."""
        x, y, z = 0b101, 0b110, 0b011
        dxz = math_utils.hamming_distance(x, z)
        dxy = math_utils.hamming_distance(x, y)
        dyz = math_utils.hamming_distance(y, z)
        assert dxz <= dxy + dyz


class TestHammingWeight:
    """Tests for hamming_weight (alias for popcnt)."""

    def test_matches_popcnt(self):
        """hamming_weight is the same as popcnt."""
        for x in [0, 1, 5, 255, 1023]:
            assert math_utils.hamming_weight(x) == math_utils.popcnt(x)

    def test_known_weights(self):
        """Known weight values."""
        assert math_utils.hamming_weight(0) == 0
        assert math_utils.hamming_weight(7) == 3
        assert math_utils.hamming_weight(255) == 8


# ---------------------------------------------------------------------------
# Tests: generate_permutations
# ---------------------------------------------------------------------------


class TestGeneratePermutations:
    """Tests for generate_permutations."""

    def test_count(self):
        """n! permutations of [0..n-1]."""
        perms = list(math_utils.generate_permutations(3))
        assert len(perms) == 6  # 3!

    def test_elements(self):
        """Each permutation contains exactly [0..n-1]."""
        for perm in math_utils.generate_permutations(3):
            assert sorted(perm) == [0, 1, 2]

    def test_single(self):
        """1 permutation of [0]."""
        perms = list(math_utils.generate_permutations(1))
        assert perms == [(0,)]

    def test_all_unique(self):
        """All permutations are distinct."""
        perms = list(math_utils.generate_permutations(4))
        assert len(perms) == len(set(perms))


# ---------------------------------------------------------------------------
# Tests: int_to_binary_tuple and binary_tuple_to_int
# ---------------------------------------------------------------------------


class TestIntToBinaryTuple:
    """Tests for int_to_binary_tuple (LSB first)."""

    def test_basic(self):
        """6 in 3 bits LSB-first: (0, 1, 1)."""
        assert math_utils.int_to_binary_tuple(6, 3) == (0, 1, 1)

    def test_zero(self):
        """0 in any width is all zeros."""
        assert math_utils.int_to_binary_tuple(0, 4) == (0, 0, 0, 0)

    def test_all_ones(self):
        """7 in 3 bits is (1, 1, 1)."""
        assert math_utils.int_to_binary_tuple(7, 3) == (1, 1, 1)

    def test_single_bit(self):
        """1 in 1 bit is (1,)."""
        assert math_utils.int_to_binary_tuple(1, 1) == (1,)


class TestBinaryTupleToInt:
    """Tests for binary_tuple_to_int (LSB first)."""

    def test_basic(self):
        """(0, 1, 1) LSB-first = 6."""
        assert math_utils.binary_tuple_to_int((0, 1, 1)) == 6

    def test_zero(self):
        """All zeros = 0."""
        assert math_utils.binary_tuple_to_int((0, 0, 0)) == 0

    def test_all_ones(self):
        """(1, 1, 1) = 7."""
        assert math_utils.binary_tuple_to_int((1, 1, 1)) == 7


class TestBinaryRoundtrip:
    """Test that int_to_binary_tuple and binary_tuple_to_int are inverses."""

    @pytest.mark.parametrize("x", range(16))
    def test_roundtrip(self, x):
        """int -> tuple -> int roundtrip preserves value."""
        n = 4
        t = math_utils.int_to_binary_tuple(x, n)
        assert math_utils.binary_tuple_to_int(t) == x

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_all_values_roundtrip(self, n):
        """All 2^n values roundtrip correctly."""
        for x in range(1 << n):
            t = math_utils.int_to_binary_tuple(x, n)
            assert math_utils.binary_tuple_to_int(t) == x
