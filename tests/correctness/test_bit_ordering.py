"""
Comprehensive bit-ordering convention tests.

BooFun uses LSB=x₀ convention:
- Index i corresponds to x₀=(i>>0)&1, x₁=(i>>1)&1, x₂=(i>>2)&1, ...
- Truth table position 5 (binary 101) means x₀=1, x₁=0, x₂=1

These tests verify this convention is consistent across:
- Truth table construction
- Function evaluation
- Fourier transforms
- Variable influences
- Representation conversions
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import boofun as bf

# Load golden test data
GOLDEN_PATH = Path(__file__).parent.parent / "golden" / "bit_ordering_golden.json"


@pytest.fixture
def golden_data():
    """Load golden test data."""
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestBitOrderingConvention:
    """Test that bit-ordering convention is LSB=x₀."""

    def test_index_to_bits_convention(self):
        """Verify the index-to-bits mapping follows LSB=x₀."""
        # Index 5 in 3 variables = binary 101 = x₀=1, x₁=0, x₂=1
        n = 3
        index = 5

        # Expected: x₀=1 (bit 0), x₁=0 (bit 1), x₂=1 (bit 2)
        expected_bits = [1, 0, 1]

        # Verify using bit extraction
        actual_bits = [(index >> i) & 1 for i in range(n)]
        assert actual_bits == expected_bits

    def test_dictator_truth_table(self):
        """Verify dictator functions have correct truth tables."""
        # x₀ dictator: output = x₀, so TT[i] = (i >> 0) & 1
        x0 = bf.dictator(3, 0)
        tt_x0 = list(x0.get_representation("truth_table"))
        expected_x0 = [0, 1, 0, 1, 0, 1, 0, 1]  # x₀ values for i=0..7
        assert tt_x0 == expected_x0, f"x₀ dictator: got {tt_x0}, expected {expected_x0}"

        # x₁ dictator: output = x₁, so TT[i] = (i >> 1) & 1
        x1 = bf.dictator(3, 1)
        tt_x1 = list(x1.get_representation("truth_table"))
        expected_x1 = [0, 0, 1, 1, 0, 0, 1, 1]  # x₁ values for i=0..7
        assert tt_x1 == expected_x1, f"x₁ dictator: got {tt_x1}, expected {expected_x1}"

        # x₂ dictator: output = x₂, so TT[i] = (i >> 2) & 1
        x2 = bf.dictator(3, 2)
        tt_x2 = list(x2.get_representation("truth_table"))
        expected_x2 = [0, 0, 0, 0, 1, 1, 1, 1]  # x₂ values for i=0..7
        assert tt_x2 == expected_x2, f"x₂ dictator: got {tt_x2}, expected {expected_x2}"

    def test_and_gate_truth_table(self):
        """Verify AND gate follows x₀ ∧ x₁ convention."""
        and_func = bf.AND(2)
        tt = list(and_func.get_representation("truth_table"))

        # AND(x₀, x₁) = 1 iff x₀=1 AND x₁=1, which is index 3
        expected = [0, 0, 0, 1]
        assert tt == expected, f"AND(2): got {tt}, expected {expected}"

    def test_or_gate_truth_table(self):
        """Verify OR gate follows x₀ ∨ x₁ convention."""
        or_func = bf.OR(2)
        tt = list(or_func.get_representation("truth_table"))

        # OR(x₀, x₁) = 1 iff x₀=1 OR x₁=1
        expected = [0, 1, 1, 1]
        assert tt == expected, f"OR(2): got {tt}, expected {expected}"

    def test_evaluate_by_index(self):
        """Verify evaluation by index is consistent."""
        f = bf.AND(3)

        # AND(3) is 1 only when all inputs are 1, i.e., index 7
        for i in range(8):
            expected = 1 if i == 7 else 0
            actual = int(f.evaluate(i))
            assert actual == expected, f"AND(3)[{i}]: got {actual}, expected {expected}"

    def test_evaluate_by_bits(self):
        """Verify evaluation by bit array is consistent with indexing."""
        f = bf.AND(3)

        for i in range(8):
            bits = np.array([(i >> j) & 1 for j in range(3)])
            by_index = int(f.evaluate(i))
            by_bits = int(f.evaluate(bits, bit_strings=True))
            assert (
                by_index == by_bits
            ), f"AND(3): index {i} gave {by_index}, bits {bits} gave {by_bits}"


class TestDirectBinaryVectorEvaluation:
    """Test direct binary vector evaluation (without bit_strings=True).

    This tests the _binary_to_index method in representation strategies,
    verifying the LSB=x₀ convention is correctly implemented.
    """

    def test_dictator_binary_vector_lsb_convention(self):
        """Verify dictator correctly interprets binary vectors.

        For dictator x₀, passing [1, 0, 0] (x₀=1, x₁=0, x₂=0) should return 1.
        This tests that _binary_to_index uses LSB-first convention.
        """
        f = bf.dictator(3, 0)  # x₀ dictator

        # Binary vector [1, 0, 0] means x₀=1, x₁=0, x₂=0
        # Correct index = 1*2^0 + 0*2^1 + 0*2^2 = 1
        # If bug exists (MSB-first), would compute 1*2^2 + 0*2^1 + 0*2^0 = 4
        bits = np.array([1, 0, 0])
        result = int(f.evaluate(bits, bit_strings=True))

        # x₀ dictator should return 1 when x₀=1
        assert result == 1, f"dictator(3,0) with bits [1,0,0] gave {result}, expected 1"

    def test_dictator_x1_binary_vector(self):
        """Verify x₁ dictator with binary vector [0, 1, 0]."""
        f = bf.dictator(3, 1)  # x₁ dictator

        # Binary vector [0, 1, 0] means x₀=0, x₁=1, x₂=0
        bits = np.array([0, 1, 0])
        result = int(f.evaluate(bits, bit_strings=True))

        # x₁ dictator should return 1 when x₁=1
        assert result == 1, f"dictator(3,1) with bits [0,1,0] gave {result}, expected 1"

    def test_dictator_x2_binary_vector(self):
        """Verify x₂ dictator with binary vector [0, 0, 1]."""
        f = bf.dictator(3, 2)  # x₂ dictator

        # Binary vector [0, 0, 1] means x₀=0, x₁=0, x₂=1
        bits = np.array([0, 0, 1])
        result = int(f.evaluate(bits, bit_strings=True))

        # x₂ dictator should return 1 when x₂=1
        assert result == 1, f"dictator(3,2) with bits [0,0,1] gave {result}, expected 1"

    def test_non_symmetric_binary_vectors(self):
        """Test with asymmetric inputs that would fail with wrong bit ordering."""
        f = bf.dictator(3, 0)  # x₀ dictator

        # [1, 1, 0] means x₀=1, x₁=1, x₂=0, index=3
        # x₀=1, so dictator should return 1
        assert int(f.evaluate(np.array([1, 1, 0]), bit_strings=True)) == 1

        # [0, 1, 1] means x₀=0, x₁=1, x₂=1, index=6
        # x₀=0, so dictator should return 0
        assert int(f.evaluate(np.array([0, 1, 1]), bit_strings=True)) == 0

    def test_all_dictators_all_inputs(self):
        """Comprehensive test: all dictators, all inputs."""
        for n in [2, 3, 4]:
            for dictator_var in range(n):
                f = bf.dictator(n, dictator_var)

                for index in range(2**n):
                    # Create binary vector for this index
                    bits = np.array([(index >> i) & 1 for i in range(n)])

                    by_index = int(f.evaluate(index))
                    by_bits = int(f.evaluate(bits, bit_strings=True))

                    # Both should agree
                    assert by_index == by_bits, (
                        f"dictator({n}, {dictator_var}): "
                        f"index {index} gave {by_index}, bits {bits} gave {by_bits}"
                    )

                    # And both should equal the value of the dictator variable
                    expected = (index >> dictator_var) & 1
                    assert by_index == expected, (
                        f"dictator({n}, {dictator_var})[{index}]: "
                        f"got {by_index}, expected {expected}"
                    )


class TestGoldenBitOrdering:
    """Test bit ordering against golden data."""

    def test_all_golden_truth_tables(self, golden_data):
        """Verify all golden truth tables match."""
        for case in golden_data["cases"]:
            n = case["n"]
            expected_tt = case["truth_table"]

            # Create function from truth table and verify round-trip
            f = bf.create(expected_tt)
            actual_tt = [int(x) for x in f.get_representation("truth_table")]
            assert actual_tt == expected_tt, f"{case['name']}: truth table mismatch"

    def test_all_golden_evaluations_by_index(self, golden_data):
        """Verify all golden evaluations by index match."""
        for case in golden_data["cases"]:
            f = bf.create(case["truth_table"])

            for ev in case["evaluations"]:
                index = ev["index"]
                expected_output = ev["output"]

                # Test by index
                actual = int(f.evaluate(index))
                assert (
                    actual == expected_output
                ), f"{case['name']}: evaluate({index}) = {actual}, expected {expected_output}"

    def test_golden_evaluations_by_bitstring(self, golden_data):
        """Verify evaluation by bitstring works correctly."""
        for case in golden_data["cases"]:
            f = bf.create(case["truth_table"])

            for ev in case["evaluations"]:
                index = ev["index"]
                bits = np.array(ev["bits"])
                expected_output = ev["output"]

                # Test by bits using bit_strings=True parameter
                actual_bits = int(f.evaluate(bits, bit_strings=True))
                assert (
                    actual_bits == expected_output
                ), f"{case['name']}: evaluate({bits}, bit_strings=True) = {actual_bits}, expected {expected_output}"


class TestFourierBitOrdering:
    """Test that Fourier transform respects bit ordering."""

    def test_dictator_fourier_coefficient(self):
        """Dictator x_i should have only coefficient f̂({i}) non-zero."""
        for n in [2, 3, 4]:
            for i in range(n):
                f = bf.dictator(n, i)
                coeffs = f.fourier()

                # Only coefficient 2^i should be non-zero (= 1.0 for dictator)
                expected_idx = 1 << i
                for s in range(len(coeffs)):
                    if s == expected_idx:
                        assert (
                            abs(coeffs[s] - 1.0) < 1e-10
                        ), f"dictator({n}, {i}): f̂({s}) = {coeffs[s]}, expected 1.0"
                    else:
                        assert (
                            abs(coeffs[s]) < 1e-10
                        ), f"dictator({n}, {i}): f̂({s}) = {coeffs[s]}, expected 0.0"

    def test_parity_fourier_coefficient(self):
        """Parity should have only full-set coefficient non-zero."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            coeffs = f.fourier()

            # Only coefficient 2^n - 1 (all variables) should be non-zero
            full_set = (1 << n) - 1
            for s in range(len(coeffs)):
                if s == full_set:
                    assert (
                        abs(coeffs[s] - 1.0) < 1e-10 or abs(coeffs[s] + 1.0) < 1e-10
                    ), f"parity({n}): f̂({s}) = {coeffs[s]}, expected ±1.0"
                else:
                    assert (
                        abs(coeffs[s]) < 1e-10
                    ), f"parity({n}): f̂({s}) = {coeffs[s]}, expected 0.0"

    def test_golden_fourier_coefficients(self, golden_data):
        """Verify Fourier coefficients against golden data."""
        for name, case in golden_data["fourier_golden"].items():
            f = bf.create(case["truth_table"])
            coeffs = f.fourier()
            expected = np.array(case["fourier_coeffs"])

            np.testing.assert_allclose(
                coeffs,
                expected,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"{name}: Fourier coefficients mismatch",
            )


class TestInfluenceBitOrdering:
    """Test that influence computation respects bit ordering."""

    def test_dictator_influence(self):
        """Dictator x_i should have influence 1 for variable i, 0 otherwise."""
        for n in [2, 3, 4]:
            for i in range(n):
                f = bf.dictator(n, i)
                influences = f.influences()

                for j in range(n):
                    expected = 1.0 if i == j else 0.0
                    assert (
                        abs(influences[j] - expected) < 1e-10
                    ), f"dictator({n}, {i}): Inf[{j}] = {influences[j]}, expected {expected}"

    def test_and_influences(self):
        """AND function has specific influence values."""
        f = bf.AND(3)
        influences = f.influences()

        # For AND, Inf_i = 2^(1-n) = 0.25 for n=3
        expected = 0.25
        for i in range(3):
            assert (
                abs(influences[i] - expected) < 1e-10
            ), f"AND(3): Inf[{i}] = {influences[i]}, expected {expected}"


class TestConversionBitOrdering:
    """Test that representation conversions preserve bit ordering."""

    def test_truth_table_round_trip(self):
        """Creating from truth table and getting it back should be identity."""
        for n in [2, 3, 4]:
            rng = np.random.default_rng(42 + n)
            tt = rng.integers(0, 2, size=2**n).tolist()

            f = bf.create(tt)
            tt_back = [int(x) for x in f.get_representation("truth_table")]

            assert tt_back == tt, f"n={n}: truth table round-trip failed"

    def test_fourier_preserves_bit_ordering(self):
        """Fourier transform should preserve bit ordering."""
        # Create specific function
        tt = [0, 1, 1, 0]  # XOR
        f = bf.create(tt)

        # Get Fourier coefficients
        coeffs = f.fourier()

        # Reconstruct truth table from Fourier
        n = 2
        reconstructed = []
        for x in range(4):
            val = 0.0
            for s in range(4):
                # Chi_S(x) = product of (-1)^{x_i} for i in S
                chi = 1.0
                for i in range(n):
                    if (s >> i) & 1:
                        chi *= 1 - 2 * ((x >> i) & 1)
                val += coeffs[s] * chi
            # Convert from {-1, +1} to {0, 1}
            reconstructed.append(int(round((1 - val) / 2)))

        assert (
            reconstructed == tt
        ), f"Fourier reconstruction changed truth table: {reconstructed} vs {tt}"


class TestBuiltinBitOrdering:
    """Test that built-in functions follow bit ordering convention."""

    def test_majority_symmetry(self):
        """Majority function should give same result for inputs with same Hamming weight."""
        f = bf.majority(3)

        # All inputs with 2 ones should give 1
        for idx in range(8):
            hw = bin(idx).count("1")
            expected = 1 if hw > 1 else 0
            assert int(f.evaluate(idx)) == expected, f"majority(3)[{idx}] wrong"

    def test_tribes_output_depends_on_groups(self):
        """Tribes function computes OR within groups, AND across groups."""
        # tribes(k, n) with k=2, n=4: (x_0 OR x_1) AND (x_2 OR x_3)
        f = bf.tribes(2, 4)

        for idx in range(16):
            bits = [(idx >> i) & 1 for i in range(4)]
            group0 = bits[0] | bits[1]  # x_0 OR x_1
            group1 = bits[2] | bits[3]  # x_2 OR x_3
            expected = group0 & group1

            actual = int(f.evaluate(idx))
            assert actual == expected, f"tribes(2,4)[{idx}] = {actual}, expected {expected}"

    def test_threshold_counts_correctly(self):
        """Threshold function should count ones correctly."""
        f = bf.threshold(3, 2)  # Output 1 iff at least 2 inputs are 1

        for idx in range(8):
            hw = bin(idx).count("1")
            expected = 1 if hw >= 2 else 0
            actual = int(f.evaluate(idx))
            assert actual == expected, f"threshold(3,2)[{idx}] = {actual}, expected {expected}"

    def test_and_all_ones_required(self):
        """AND function should be 1 only when all inputs are 1."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            for idx in range(2**n):
                expected = 1 if idx == (2**n - 1) else 0
                actual = int(f.evaluate(idx))
                assert actual == expected, f"AND({n})[{idx}] = {actual}, expected {expected}"

    def test_or_any_one_sufficient(self):
        """OR function should be 1 when any input is 1."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            for idx in range(2**n):
                expected = 0 if idx == 0 else 1
                actual = int(f.evaluate(idx))
                assert actual == expected, f"OR({n})[{idx}] = {actual}, expected {expected}"
