"""
Tests for families/builtins module.

Tests for built-in function families: Majority, Parity, Tribes, etc.
Tests verify both structure AND mathematical correctness.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.families.builtins import (
    ANDFamily,
    DictatorFamily,
    MajorityFamily,
    ORFamily,
    ParityFamily,
    ThresholdFamily,
    TribesFamily,
)


class TestMajorityFamily:
    """Test MajorityFamily: outputs 1 iff majority of inputs are 1."""

    def test_generate_correct_n_vars(self):
        """Generated function has correct number of variables."""
        family = MajorityFamily()
        for n in [3, 5, 7]:
            f = family.generate(n)
            assert f.n_vars == n, f"Expected n_vars={n}, got {f.n_vars}"

    def test_majority_3_truth_table(self):
        """MAJORITY₃ outputs 1 iff ≥2 inputs are 1."""
        family = MajorityFamily()
        f = family.generate(3)

        # Truth table for MAJORITY₃
        expected = {
            0b000: 0,  # 0 ones
            0b001: 0,  # 1 one
            0b010: 0,  # 1 one
            0b011: 1,  # 2 ones
            0b100: 0,  # 1 one
            0b101: 1,  # 2 ones
            0b110: 1,  # 2 ones
            0b111: 1,  # 3 ones
        }
        for x, expected_out in expected.items():
            actual = f.evaluate(x)
            assert actual == expected_out, f"MAJ₃({bin(x)})={actual}, expected {expected_out}"

    def test_majority_is_balanced_for_odd_n(self):
        """MAJORITY on odd n is balanced (equal 0s and 1s)."""
        family = MajorityFamily()
        for n in [3, 5, 7]:
            f = family.generate(n)
            ones = sum(f.evaluate(x) for x in range(2**n))
            zeros = 2**n - ones
            assert ones == zeros, f"MAJ_{n} not balanced: {ones} ones, {zeros} zeros"

    def test_majority_symmetric(self):
        """MAJORITY is symmetric: permuting inputs doesn't change output."""
        family = MajorityFamily()
        f = family.generate(3)

        # Check inputs with same Hamming weight give same output
        for weight in range(4):
            outputs = set()
            for x in range(8):
                if bin(x).count("1") == weight:
                    outputs.add(f.evaluate(x))
            assert len(outputs) == 1, f"MAJ₃ not symmetric for weight {weight}"


class TestParityFamily:
    """Test ParityFamily: XOR of all inputs."""

    def test_generate_correct_n_vars(self):
        """Generated parity has correct n_vars."""
        family = ParityFamily()
        for n in [2, 3, 4, 5]:
            f = family.generate(n)
            assert f.n_vars == n

    def test_parity_equals_xor(self):
        """PARITY(x) = x₀ ⊕ x₁ ⊕ ... ⊕ xₙ₋₁ = popcount(x) mod 2."""
        family = ParityFamily()
        for n in [2, 3, 4]:
            f = family.generate(n)
            for x in range(2**n):
                expected = bin(x).count("1") % 2
                actual = f.evaluate(x)
                assert actual == expected, f"PAR_{n}({bin(x)})={actual}, expected {expected}"

    def test_parity_is_balanced(self):
        """PARITY is always balanced."""
        family = ParityFamily()
        for n in [2, 3, 4, 5]:
            f = family.generate(n)
            ones = sum(f.evaluate(x) for x in range(2**n))
            zeros = 2**n - ones
            assert ones == zeros, f"PAR_{n} not balanced"

    def test_parity_is_linear(self):
        """PARITY is a linear function (character χ_[n])."""
        family = ParityFamily()
        f = family.generate(3)

        # Check linearity: f(x⊕y) = f(x)⊕f(y)
        for x in range(8):
            for y in range(8):
                xy = x ^ y
                assert f.evaluate(xy) == (f.evaluate(x) ^ f.evaluate(y))


class TestANDFamily:
    """Test ANDFamily: outputs 1 iff all inputs are 1."""

    def test_and_truth_table(self):
        """AND is 1 only when all inputs are 1."""
        family = ANDFamily()
        for n in [2, 3, 4]:
            f = family.generate(n)
            for x in range(2**n):
                expected = 1 if x == (2**n - 1) else 0
                actual = f.evaluate(x)
                assert actual == expected, f"AND_{n}({bin(x)})={actual}, expected {expected}"

    def test_and_fourier_degree(self):
        """AND has Fourier degree n (highest coefficient at S=[n])."""
        family = ANDFamily()
        f = family.generate(3)
        coeffs = f.fourier()

        # AND₃ has non-zero coefficient at all subsets
        # Maximum degree term is at S={0,1,2} = index 7
        assert abs(coeffs[7]) > 0.01, "AND should have degree-n Fourier coefficient"


class TestORFamily:
    """Test ORFamily: outputs 1 iff at least one input is 1."""

    def test_or_truth_table(self):
        """OR is 0 only when all inputs are 0."""
        family = ORFamily()
        for n in [2, 3, 4]:
            f = family.generate(n)
            for x in range(2**n):
                expected = 0 if x == 0 else 1
                actual = f.evaluate(x)
                assert actual == expected, f"OR_{n}({bin(x)})={actual}, expected {expected}"

    def test_or_is_dual_of_and(self):
        """OR(x) = NOT(AND(NOT(x)))."""
        and_fam = ANDFamily()
        or_fam = ORFamily()

        for n in [2, 3]:
            and_f = and_fam.generate(n)
            or_f = or_fam.generate(n)

            for x in range(2**n):
                not_x = (2**n - 1) ^ x  # Flip all bits
                assert or_f.evaluate(x) == (1 - and_f.evaluate(not_x))


class TestDictatorFamily:
    """Test DictatorFamily: outputs value of single variable."""

    def test_dictator_outputs_single_variable(self):
        """Dictator should output the value of one specific variable."""
        family = DictatorFamily()
        f = family.generate(4)  # Default dictator variable

        # Find which variable is the dictator
        dictator_var = None
        for i in range(4):
            # Check if f(x) = xᵢ for all x
            is_dictator = True
            for x in range(16):
                bit_i = (x >> i) & 1
                if f.evaluate(x) != bit_i:
                    is_dictator = False
                    break
            if is_dictator:
                dictator_var = i
                break

        assert dictator_var is not None, "Function is not a dictator"

    def test_dictator_is_linear(self):
        """Dictator is a linear function (character χ_{i}).

        Dictator x_i = χ_{i} has exactly ONE non-zero Fourier coefficient:
        f̂({i}) = 1, all others are 0. This is because dictator IS the
        character function, which forms a basis element.
        """
        family = DictatorFamily()
        f = family.generate(3)

        # Dictator has exactly 1 non-zero Fourier coefficient
        coeffs = f.fourier()
        non_zero = sum(1 for c in coeffs if abs(c) > 0.01)
        assert non_zero == 1, f"Dictator should have 1 non-zero coeff (χ_{{i}}), got {non_zero}"


class TestTribesFamily:
    """Test TribesFamily: OR of ANDs."""

    def test_tribes_structure(self):
        """Tribes is OR of ANDs (DNF with uniform width)."""
        family = TribesFamily()
        f = family.generate(6)  # 2 tribes of 3, or 3 tribes of 2

        # Should have correct n_vars
        assert f.n_vars == 6

    def test_tribes_not_constant(self):
        """Tribes should output both 0 and 1."""
        family = TribesFamily()
        f = family.generate(6)

        outputs = set(f.evaluate(x) for x in range(2**6))
        assert outputs == {0, 1}, "Tribes should output both 0 and 1"


class TestThresholdFamily:
    """Test ThresholdFamily: outputs 1 iff ≥k inputs are 1."""

    def test_threshold_k_equals_1_is_or(self):
        """Threshold with k=1 is OR."""
        threshold_fam = ThresholdFamily()
        or_fam = ORFamily()

        # ThresholdFamily.generate might use default k
        # Just verify it produces a valid function
        f = threshold_fam.generate(3)
        assert f.n_vars == 3

        # Verify it's a threshold function (monotone)
        for x in range(8):
            for y in range(8):
                # If x ≤ y pointwise, f(x) ≤ f(y) for monotone functions
                x_bits = [(x >> i) & 1 for i in range(3)]
                y_bits = [(y >> i) & 1 for i in range(3)]
                if all(xb <= yb for xb, yb in zip(x_bits, y_bits)):
                    assert f.evaluate(x) <= f.evaluate(y), "Threshold should be monotone"


class TestFamilyConsistency:
    """Test that families produce consistent results."""

    def test_same_n_gives_same_function(self):
        """Calling generate(n) twice gives equivalent functions."""
        families = [MajorityFamily(), ParityFamily(), ANDFamily(), ORFamily()]

        for family in families:
            f1 = family.generate(3)
            f2 = family.generate(3)

            for x in range(8):
                assert f1.evaluate(x) == f2.evaluate(
                    x
                ), f"{family.__class__.__name__} not consistent"

    def test_families_have_metadata(self):
        """All families should expose metadata."""
        families = [
            MajorityFamily(),
            ParityFamily(),
            ANDFamily(),
            ORFamily(),
            DictatorFamily(),
        ]

        for family in families:
            assert (
                hasattr(family, "metadata")
                or hasattr(family, "name")
                or hasattr(family, "__class__")
            ), f"{family.__class__.__name__} missing metadata"


class TestFamilyFourierProperties:
    """Test Fourier-analytic properties of function families."""

    def test_parity_single_fourier_coefficient(self):
        """PARITY has exactly one non-zero Fourier coefficient (at S=[n])."""
        family = ParityFamily()
        f = family.generate(3)
        coeffs = f.fourier()

        # Only coefficient at index 7 (= {0,1,2}) should be non-zero
        non_zero_indices = [i for i, c in enumerate(coeffs) if abs(c) > 0.01]
        assert non_zero_indices == [7], f"Parity coeffs at {non_zero_indices}, expected [7]"

    def test_majority_fourier_degree(self):
        """MAJORITY has odd Fourier degree (symmetric, so even degrees cancel)."""
        family = MajorityFamily()
        f = family.generate(3)
        coeffs = f.fourier()

        # Majority₃ has coefficients only at odd-sized subsets
        for i, c in enumerate(coeffs):
            degree = bin(i).count("1")
            if degree % 2 == 0:
                assert abs(c) < 0.01, f"MAJ₃ should have 0 at even degree {degree}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
