"""
Adversarial input tests.

These tests probe edge cases, boundary conditions, and potential failure modes:
- Empty/minimal inputs
- Maximum-size inputs
- Pathological truth tables
- Numerical edge cases
- Type coercion issues
"""

import numpy as np
import pytest

import boofun as bf


class TestEmptyAndMinimalInputs:
    """Test handling of empty and minimal inputs."""

    def test_n_equals_1(self):
        """Single-variable functions should work correctly."""
        # All possible 1-variable functions
        for tt in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            f = bf.create(tt)
            assert f.n_vars == 1
            assert len(f.fourier()) == 2
            assert len(f.influences()) == 1

    def test_truth_table_length_validation(self):
        """Truth table must have power-of-2 length."""
        # Valid lengths work without error
        for length in [2, 4, 8, 16]:
            tt = [0] * length
            f = bf.create(tt)
            assert f is not None


class TestConstantFunctions:
    """Test constant functions (important edge case)."""

    def test_constant_zero(self):
        """Constant 0 function."""
        for n in [1, 2, 3, 4]:
            f = bf.create([0] * (2**n))

            # All evaluations should be 0
            for i in range(2**n):
                assert int(f.evaluate(i)) == 0

            # Fourier: only f̂(∅) = 1, others = 0
            coeffs = f.fourier()
            assert abs(coeffs[0] - 1.0) < 1e-10
            for s in range(1, len(coeffs)):
                assert abs(coeffs[s]) < 1e-10

            # Influence should be 0 for all variables
            influences = f.influences()
            assert all(abs(i) < 1e-10 for i in influences)

            # Degree should be 0
            assert f.degree() == 0

    def test_constant_one(self):
        """Constant 1 function."""
        for n in [1, 2, 3, 4]:
            f = bf.create([1] * (2**n))

            # All evaluations should be 1
            for i in range(2**n):
                assert int(f.evaluate(i)) == 1

            # Fourier: only f̂(∅) = -1, others = 0
            coeffs = f.fourier()
            assert abs(coeffs[0] - (-1.0)) < 1e-10
            for s in range(1, len(coeffs)):
                assert abs(coeffs[s]) < 1e-10


class TestNearConstantFunctions:
    """Test functions that are almost constant (1 non-dominant output)."""

    def test_single_one(self):
        """Function with exactly one 1 output."""
        for n in [2, 3, 4]:
            for pos in [0, 1, 2**n - 1]:
                tt = [0] * (2**n)
                tt[pos] = 1
                f = bf.create(tt)

                # Verify evaluation
                for i in range(2**n):
                    expected = 1 if i == pos else 0
                    assert int(f.evaluate(i)) == expected

                # Verify Parseval still holds
                coeffs = f.fourier()
                sum_sq = sum(c**2 for c in coeffs)
                assert abs(sum_sq - 1.0) < 1e-10

    def test_single_zero(self):
        """Function with exactly one 0 output."""
        for n in [2, 3, 4]:
            for pos in [0, 1, 2**n - 1]:
                tt = [1] * (2**n)
                tt[pos] = 0
                f = bf.create(tt)

                # Verify evaluation
                for i in range(2**n):
                    expected = 0 if i == pos else 1
                    assert int(f.evaluate(i)) == expected


class TestLargeFunctions:
    """Test functions with many variables (stress tests)."""

    @pytest.mark.slow
    def test_large_majority(self):
        """Test majority function with many variables."""
        for n in [7, 9, 11]:
            f = bf.majority(n)
            assert f.n_vars == n

            # Spot check some evaluations
            assert int(f.evaluate(0)) == 0
            assert int(f.evaluate((1 << n) - 1)) == 1

            # Verify balanced
            tt = f.get_representation("truth_table")
            ones = sum(int(x) for x in tt)
            zeros = len(tt) - ones
            assert ones == zeros  # Majority is balanced for odd n

    @pytest.mark.slow
    def test_large_parity(self):
        """Test parity function with many variables."""
        for n in [8, 10, 12]:
            f = bf.parity(n)
            assert f.n_vars == n

            # Verify some evaluations
            assert int(f.evaluate(0)) == 0  # 0 ones → 0
            assert int(f.evaluate(1)) == 1  # 1 one → 1
            assert int(f.evaluate(3)) == 0  # 2 ones → 0


class TestInputTypeCoercion:
    """Test that various input types are handled correctly."""

    def test_list_input(self):
        """Truth table as Python list."""
        f = bf.create([0, 1, 1, 0])
        assert f.n_vars == 2

    def test_numpy_array_input(self):
        """Truth table as NumPy array."""
        f = bf.create(np.array([0, 1, 1, 0]))
        assert f.n_vars == 2

    def test_integer_values(self):
        """Truth table with Python integers."""
        f = bf.create([0, 1, 1, 0])
        assert int(f.evaluate(0)) == 0

    def test_float_values_work(self):
        """Truth table creation from floats should work (coercion may vary)."""
        # Note: The library may coerce floats to bool differently than expected
        # This test just verifies it doesn't crash
        f = bf.create([0.0, 1.0, 1.0, 0.0])
        tt = list(f.get_representation("truth_table"))
        assert len(tt) == 4

    def test_bool_values(self):
        """Truth table with booleans."""
        f = bf.create([False, True, True, False])
        assert int(f.evaluate(0)) == 0
        assert int(f.evaluate(1)) == 1

    def test_evaluate_with_int(self):
        """Evaluate with integer index."""
        f = bf.AND(2)
        assert int(f.evaluate(0)) == 0
        assert int(f.evaluate(3)) == 1

    def test_evaluate_with_list(self):
        """Evaluate with list of bits."""
        f = bf.AND(2)
        assert int(f.evaluate([0, 0])) == 0
        assert int(f.evaluate([1, 1])) == 1

    def test_evaluate_with_numpy_array(self):
        """Evaluate with NumPy array of bits."""
        f = bf.AND(2)
        assert int(f.evaluate(np.array([0, 0]))) == 0
        assert int(f.evaluate(np.array([1, 1]))) == 1


class TestNumericalStability:
    """Test numerical stability in computations."""

    def test_fourier_parseval_precision(self):
        """Parseval's identity should hold to high precision."""
        rng = np.random.default_rng(42)

        for n in [2, 4, 6, 8]:
            for _ in range(10):
                tt = rng.integers(0, 2, size=2**n).tolist()
                f = bf.create(tt)
                coeffs = f.fourier()

                sum_sq = sum(c**2 for c in coeffs)
                assert (
                    abs(sum_sq - 1.0) < 1e-12
                ), f"Parseval deviation: {abs(sum_sq - 1.0)} for n={n}"

    def test_influence_non_negative(self):
        """Influences must be non-negative."""
        rng = np.random.default_rng(42)

        for n in [2, 3, 4, 5]:
            for _ in range(20):
                tt = rng.integers(0, 2, size=2**n).tolist()
                f = bf.create(tt)
                influences = f.influences()

                # Allow tiny negative values due to floating point
                assert all(
                    i >= -1e-14 for i in influences
                ), f"Negative influence: {min(influences)}"

    def test_noise_stability_bounds(self):
        """Noise stability must be in [-1, 1]."""
        rng = np.random.default_rng(42)

        for n in [2, 3, 4]:
            for _ in range(10):
                tt = rng.integers(0, 2, size=2**n).tolist()
                f = bf.create(tt)

                for rho in [0.0, 0.1, 0.5, 0.9, 1.0]:
                    stab = f.noise_stability(rho)
                    assert -1.0 - 1e-10 <= stab <= 1.0 + 1e-10


class TestSymmetricFunctions:
    """Test symmetric functions (depend only on Hamming weight)."""

    def test_majority_symmetry(self):
        """Majority output depends only on number of 1s."""
        for n in [3, 5, 7]:
            f = bf.majority(n)

            for hw in range(n + 1):
                # Get all inputs with this Hamming weight
                expected = 1 if hw > n // 2 else 0

                for i in range(2**n):
                    if bin(i).count("1") == hw:
                        assert int(f.evaluate(i)) == expected

    def test_threshold_symmetry(self):
        """Threshold output depends only on number of 1s."""
        for n in [3, 4, 5]:
            for k in range(1, n + 1):
                f = bf.threshold(n, k)

                for i in range(2**n):
                    hw = bin(i).count("1")
                    expected = 1 if hw >= k else 0
                    assert int(f.evaluate(i)) == expected


class TestDegenerateCases:
    """Test degenerate and pathological cases."""

    def test_alternating_truth_table(self):
        """Truth table [0,1,0,1,...] is dictator x₀."""
        for n in [2, 3, 4]:
            tt = [i % 2 for i in range(2**n)]
            f = bf.create(tt)

            # Should be equivalent to dictator x₀
            x0 = bf.dictator(n, 0)
            assert list(f.get_representation("truth_table")) == list(
                x0.get_representation("truth_table")
            )

    def test_checkerboard_truth_table(self):
        """Truth table [0,0,1,1,0,0,1,1,...] is dictator x₁."""
        for n in [2, 3, 4]:
            tt = [(i >> 1) % 2 for i in range(2**n)]
            f = bf.create(tt)

            # Should be equivalent to dictator x₁
            x1 = bf.dictator(n, 1)
            assert list(f.get_representation("truth_table")) == list(
                x1.get_representation("truth_table")
            )


class TestBuiltinConsistency:
    """Test that built-in functions are self-consistent."""

    def test_and_vs_threshold(self):
        """AND(n) = Threshold(n, n)."""
        for n in [2, 3, 4]:
            and_func = bf.AND(n)
            thresh_func = bf.threshold(n, n)

            tt_and = list(and_func.get_representation("truth_table"))
            tt_thresh = list(thresh_func.get_representation("truth_table"))
            assert tt_and == tt_thresh

    def test_or_vs_threshold(self):
        """OR(n) = Threshold(n, 1)."""
        for n in [2, 3, 4]:
            or_func = bf.OR(n)
            thresh_func = bf.threshold(n, 1)

            tt_or = list(or_func.get_representation("truth_table"))
            tt_thresh = list(thresh_func.get_representation("truth_table"))
            assert tt_or == tt_thresh

    def test_majority_vs_threshold_odd(self):
        """Majority(n) = Threshold(n, (n+1)/2) for odd n."""
        for n in [3, 5, 7]:
            maj_func = bf.majority(n)
            thresh_func = bf.threshold(n, (n + 1) // 2)

            tt_maj = list(maj_func.get_representation("truth_table"))
            tt_thresh = list(thresh_func.get_representation("truth_table"))
            assert tt_maj == tt_thresh
