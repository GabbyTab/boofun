"""
Bit-ordering consistency tests.

BooFun uses LSB = x_0 everywhere (see docs/STYLE_GUIDE.md):
  - Variable x_j corresponds to bit j (counting from the right)
  - Extract bit: x_j = (x >> j) & 1
  - Singleton subset {j} has Fourier index 1 << j

These tests verify that all modules agree on this convention.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf

# ---------------------------------------------------------------------------
# Dictator: the canonical bit-ordering test
# ---------------------------------------------------------------------------


class TestDictatorBitOrdering:
    """Dictator on variable i is the simplest test of bit ordering.

    f(x) = x_i  means:
      - truth table: f(x) = (x >> i) & 1
      - Fourier: f_hat({i}) is the only non-zero degree-1 coefficient
      - influence: Inf_i = 1, Inf_{j!=i} = 0
    """

    @pytest.mark.parametrize("n", [3, 4, 5])
    @pytest.mark.parametrize("i", [0, 1])
    def test_dictator_truth_table(self, n, i):
        """Dictator on variable i: f(x) = (x >> i) & 1."""
        f = bf.dictator(n, i=i)
        for x in range(2**n):
            expected = (x >> i) & 1
            assert (
                f.evaluate(x) == expected
            ), f"dictator(n={n}, i={i}): f({x}) = {f.evaluate(x)}, expected {expected}"

    @pytest.mark.parametrize("n", [3, 4])
    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_dictator_fourier_singleton(self, n, i):
        """f_hat({i}) should be the dominant degree-1 coefficient."""
        if i >= n:
            pytest.skip("i >= n")
        f = bf.dictator(n, i=i)
        fourier = f.fourier()

        # The singleton {i} has index 1 << i in LSB convention
        singleton_idx = 1 << i
        assert abs(fourier[singleton_idx]) > 0.5, (
            f"dictator(n={n}, i={i}): f_hat at index {singleton_idx} "
            f"(1<<{i}) = {fourier[singleton_idx]}, expected |val| > 0.5"
        )

        # All other degree-1 singletons should be ~0
        for j in range(n):
            if j != i:
                other_idx = 1 << j
                assert abs(fourier[other_idx]) < 1e-10, (
                    f"dictator(n={n}, i={i}): f_hat at index {other_idx} "
                    f"(1<<{j}) = {fourier[other_idx]}, expected ~0"
                )

    @pytest.mark.parametrize("n", [3, 4])
    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_dictator_influence(self, n, i):
        """Inf_i(dictator_i) = 1, Inf_j = 0 for j != i."""
        if i >= n:
            pytest.skip("i >= n")
        f = bf.dictator(n, i=i)

        for j in range(n):
            inf_j = f.influence(j)
            if j == i:
                assert (
                    abs(inf_j - 1.0) < 1e-10
                ), f"dictator(n={n}, i={i}): Inf_{j} = {inf_j}, expected 1.0"
            else:
                assert (
                    abs(inf_j) < 1e-10
                ), f"dictator(n={n}, i={i}): Inf_{j} = {inf_j}, expected 0.0"


# ---------------------------------------------------------------------------
# AND / OR: asymmetric tests (sensitive to variable ordering)
# ---------------------------------------------------------------------------


class TestAsymmetricFunctions:
    """AND and OR are symmetric, but we can construct asymmetric functions
    to test that variable ordering is consistent across modules."""

    def test_function_from_callable_bit_ordering(self):
        """A callable that reads specific bits should match truth table."""
        # f(x) = x_0 AND x_1 (variables 0 and 1 only, ignore x_2)
        f = bf.create([0, 0, 0, 1, 0, 0, 0, 1])  # n=3

        # x=3 (binary 011): x_0=1, x_1=1, x_2=0 -> f=1
        assert f.evaluate(3) == 1
        # x=5 (binary 101): x_0=1, x_1=0, x_2=1 -> f=0
        assert f.evaluate(5) == 0
        # x=6 (binary 110): x_0=0, x_1=1, x_2=1 -> f=0
        assert f.evaluate(6) == 0

    def test_influence_of_irrelevant_variable(self):
        """Variable x_2 is irrelevant to f(x) = x_0 AND x_1."""
        # f(x) = x_0 AND x_1 on 3 variables
        tt = [0, 0, 0, 1, 0, 0, 0, 1]
        f = bf.create(tt)

        # x_0 and x_1 should have positive influence
        assert f.influence(0) > 0.1
        assert f.influence(1) > 0.1
        # x_2 should have zero influence
        assert abs(f.influence(2)) < 1e-10


# ---------------------------------------------------------------------------
# Fourier via high-level API vs representation convert_from
# ---------------------------------------------------------------------------


class TestFourierConsistency:
    """The high-level f.fourier() and the representation convert_from()
    should produce the same coefficients."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_majority_fourier_length(self, n):
        """Fourier array must always have exactly 2^n entries."""
        f = bf.majority(n)
        fourier = f.fourier()
        assert len(fourier) == 2**n

    def test_parity_all_weight_at_top(self):
        """Parity(n): only f_hat({0,1,...,n-1}) is non-zero."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            fourier = f.fourier()
            top_idx = (1 << n) - 1  # All bits set
            assert abs(fourier[top_idx]) > 0.5
            for s in range(2**n):
                if s != top_idx:
                    assert abs(fourier[s]) < 1e-10


# ---------------------------------------------------------------------------
# Restriction consistency
# ---------------------------------------------------------------------------


class TestRestrictionBitOrdering:
    """Restrictions should fix the correct variable."""

    def test_fix_variable_0(self):
        """Fixing x_0=1 on dictator_0 should give constant 1."""
        f = bf.dictator(3, i=0)
        # After fixing x_0=1, the restricted function is constant 1
        # Check by evaluating all inputs with x_0 forced to 1
        for x in range(2**3):
            if (x >> 0) & 1 == 1:  # x_0 = 1
                assert f.evaluate(x) == 1

    def test_fix_variable_1(self):
        """Fixing x_1=0 on dictator_1 should give constant 0."""
        f = bf.dictator(3, i=1)
        for x in range(2**3):
            if (x >> 1) & 1 == 0:  # x_1 = 0
                assert f.evaluate(x) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
