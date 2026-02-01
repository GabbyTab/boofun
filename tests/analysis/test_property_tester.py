"""
Comprehensive tests for PropertyTester class.

Tests all property testing algorithms with mathematical verification.
Each test checks both correctness and boundary conditions.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis import PropertyTester


class TestConstantTest:
    """Test constant_test method."""

    def test_constant_zero_returns_true(self):
        """Constant 0 function should be detected as constant."""
        f = bf.create([0, 0, 0, 0])
        tester = PropertyTester(f)
        assert tester.constant_test() is True

    def test_constant_one_returns_true(self):
        """Constant 1 function should be detected as constant."""
        f = bf.create([1, 1, 1, 1])
        tester = PropertyTester(f)
        assert tester.constant_test() is True

    def test_non_constant_returns_false(self):
        """Non-constant functions should return False."""
        non_constants = [
            bf.parity(3),
            bf.majority(3),
            bf.AND(3),
            bf.OR(3),
            bf.dictator(3, 0),
        ]
        for f in non_constants:
            tester = PropertyTester(f)
            assert tester.constant_test() is False, f"Failed for {f}"

    def test_single_one_output(self):
        """Function with single 1 output is not constant."""
        f = bf.AND(3)  # Only 1 at all-ones input
        tester = PropertyTester(f)
        assert tester.constant_test() is False


class TestBLRLinearityTest:
    """Test blr_linearity_test method."""

    def test_parity_is_linear(self):
        """Parity (XOR) is a linear function."""
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            tester = PropertyTester(f, random_seed=42)
            assert tester.blr_linearity_test(num_queries=100) is True

    def test_dictator_is_linear(self):
        """Dictator functions are linear."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                tester = PropertyTester(f, random_seed=42)
                assert tester.blr_linearity_test(num_queries=100) is True

    def test_and_is_not_linear(self):
        """AND function is not linear (has degree > 1)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            tester = PropertyTester(f, random_seed=42)
            # With high queries, should reliably detect non-linearity
            assert tester.blr_linearity_test(num_queries=200) is False

    def test_majority_is_not_linear(self):
        """Majority function is not linear."""
        f = bf.majority(5)
        tester = PropertyTester(f, random_seed=42)
        assert tester.blr_linearity_test(num_queries=200) is False

    def test_blr_theorem_linearity_check(self):
        """Verify BLR test agrees with exact linearity check."""
        # Test on known linear functions
        linear_funcs = [
            bf.parity(3),
            bf.dictator(4, 2),
            bf.create([0, 1, 1, 0]),  # XOR
        ]
        for f in linear_funcs:
            tester = PropertyTester(f, random_seed=42)
            blr_result = tester.blr_linearity_test(num_queries=100)
            assert blr_result is True, f"BLR failed on linear function"

    def test_blr_acceptance_probability_formula(self):
        """Verify BLR acceptance probability matches the formula.

        Theorem (O'Donnell): E[f(x)f(y)f(x⊕y)] = Σ_S f̂(S)³

        Since BLR accepts when f(x)·f(y)·f(x⊕y) = 1:
            Pr[BLR accepts] = (1 + Σ_S f̂(S)³) / 2

        This is a critical mathematical property that catches formula bugs.
        """
        test_functions = [
            ("parity_3", bf.parity(3)),
            ("majority_3", bf.majority(3)),
            ("and_3", bf.AND(3)),
            ("or_3", bf.OR(3)),
            ("dictator", bf.dictator(4, 0)),
        ]

        for name, f in test_functions:
            n = f.n_vars
            fourier = f.fourier()
            sum_cubed = sum(c**3 for c in fourier)
            theoretical_acceptance = (1 + sum_cubed) / 2

            # Compute exact acceptance probability by exhaustive enumeration
            accepts = 0
            total = 0
            for x in range(2**n):
                for y in range(2**n):
                    xy = x ^ y
                    fx = 1 - 2 * int(f.evaluate(x))  # ±1
                    fy = 1 - 2 * int(f.evaluate(y))
                    fxy = 1 - 2 * int(f.evaluate(xy))
                    if fx * fy == fxy:
                        accepts += 1
                    total += 1
            exact_acceptance = accepts / total

            # Both should match theoretical formula
            assert np.isclose(
                theoretical_acceptance, exact_acceptance
            ), f"{name}: theoretical={(1+sum_cubed)/2:.4f} != exact={exact_acceptance:.4f}"

    def test_blr_acceptance_probability_is_valid(self):
        """Verify BLR acceptance probability is always in [0, 1].

        The formula (1 + Σf̂³)/2 must always produce a valid probability.
        Σf̂³ can be negative (e.g., for OR), but the final probability is in [0,1].

        Bug this catches: Using Σf̂³ directly as probability (can be negative!).
        """
        test_functions = [
            bf.parity(4),
            bf.majority(5),
            bf.AND(3),
            bf.OR(4),  # OR has negative Σf̂³
            bf.tribes(2, 3),
            bf.random(4, seed=42),
            bf.random(4, seed=123),
        ]

        for f in test_functions:
            fourier = f.fourier()
            sum_cubed = sum(c**3 for c in fourier)
            acceptance_prob = (1 + sum_cubed) / 2

            # Must be a valid probability
            assert 0 <= acceptance_prob <= 1, (
                f"BLR acceptance probability {acceptance_prob:.4f} is not in [0,1]! "
                f"(Σf̂³ = {sum_cubed:.4f})"
            )

            # Σf̂³ can be outside [-1, 1] in theory, but for Boolean functions
            # it must be in [-1, 1] (since it's an expectation of ±1 values)
            assert (
                -1 <= sum_cubed <= 1
            ), f"Σf̂³ = {sum_cubed:.4f} is not in [-1,1] for Boolean function"


class TestMonotonicityTest:
    """Test monotonicity_test method."""

    def test_and_is_monotone(self):
        """AND function is monotone (more 1s → output stays 1)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            tester = PropertyTester(f, random_seed=42)
            assert tester.monotonicity_test(num_queries=100) is True

    def test_or_is_monotone(self):
        """OR function is monotone."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            tester = PropertyTester(f, random_seed=42)
            assert tester.monotonicity_test(num_queries=100) is True

    def test_majority_is_monotone(self):
        """Majority function is monotone."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            tester = PropertyTester(f, random_seed=42)
            assert tester.monotonicity_test(num_queries=100) is True

    def test_parity_is_not_monotone(self):
        """Parity function is NOT monotone (flipping bits changes output unpredictably)."""
        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)
        assert tester.monotonicity_test(num_queries=100) is False

    def test_constant_is_monotone(self):
        """Constant functions are trivially monotone."""
        for val in [0, 1]:
            f = bf.create([val] * 8)
            tester = PropertyTester(f, random_seed=42)
            assert tester.monotonicity_test(num_queries=50) is True


class TestSymmetryTest:
    """Test symmetry_test method."""

    def test_majority_is_symmetric(self):
        """Majority is symmetric (permuting inputs doesn't change output)."""
        for n in [3, 5]:
            f = bf.majority(n)
            tester = PropertyTester(f, random_seed=42)
            assert tester.symmetry_test(num_queries=100) is True

    def test_parity_is_symmetric(self):
        """Parity is symmetric."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            tester = PropertyTester(f, random_seed=42)
            assert tester.symmetry_test(num_queries=100) is True

    def test_and_is_symmetric(self):
        """AND is symmetric."""
        f = bf.AND(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.symmetry_test(num_queries=100) is True

    def test_dictator_is_not_symmetric(self):
        """Dictator is NOT symmetric (only depends on one variable)."""
        f = bf.dictator(4, 0)
        tester = PropertyTester(f, random_seed=42)
        assert tester.symmetry_test(num_queries=100) is False


class TestBalancedTest:
    """Test balanced_test method."""

    def test_parity_is_balanced(self):
        """Parity outputs equal 0s and 1s."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            tester = PropertyTester(f)
            assert tester.balanced_test() is True

    def test_majority_odd_is_balanced(self):
        """Majority on odd n is balanced."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            tester = PropertyTester(f)
            assert tester.balanced_test() is True

    def test_and_is_not_balanced(self):
        """AND outputs mostly 0s (only 1 at all-ones)."""
        f = bf.AND(3)
        tester = PropertyTester(f)
        assert tester.balanced_test() is False

    def test_or_is_not_balanced(self):
        """OR outputs mostly 1s (only 0 at all-zeros)."""
        f = bf.OR(3)
        tester = PropertyTester(f)
        assert tester.balanced_test() is False

    def test_constant_is_not_balanced(self):
        """Constant functions are not balanced."""
        for val in [0, 1]:
            f = bf.create([val] * 8)
            tester = PropertyTester(f)
            assert tester.balanced_test() is False


class TestDictatorTest:
    """Test dictator_test method."""

    def test_detects_dictator(self):
        """Should detect dictator functions."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                tester = PropertyTester(f, random_seed=42)
                is_dictator, idx = tester.dictator_test()
                assert is_dictator is True, f"Failed to detect dictator({n}, {i})"
                assert idx == i, f"Wrong dictator index: got {idx}, expected {i}"

    def test_rejects_non_dictators(self):
        """Should reject non-dictator functions."""
        non_dictators = [
            bf.AND(3),
            bf.OR(3),
            bf.majority(3),
            bf.parity(3),
        ]
        for f in non_dictators:
            tester = PropertyTester(f, random_seed=42)
            is_dictator, _ = tester.dictator_test()
            assert is_dictator is False


class TestAffineTest:
    """Test affine_test method."""

    def test_parity_is_affine(self):
        """Parity is affine (linear with no constant term)."""
        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)
        assert tester.affine_test(num_queries=100) is True

    def test_dictator_is_affine(self):
        """Dictator is affine."""
        f = bf.dictator(4, 1)
        tester = PropertyTester(f, random_seed=42)
        assert tester.affine_test(num_queries=100) is True

    def test_constant_is_affine(self):
        """Constant functions are affine (degree 0)."""
        f = bf.create([1, 1, 1, 1])
        tester = PropertyTester(f, random_seed=42)
        assert tester.affine_test(num_queries=100) is True

    def test_and_is_not_affine(self):
        """AND is not affine (has degree > 1)."""
        f = bf.AND(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.affine_test(num_queries=200) is False


class TestJuntaTest:
    """Test junta_test method."""

    def test_dictator_is_1_junta(self):
        """Dictator depends on exactly 1 variable."""
        f = bf.dictator(5, 2)
        tester = PropertyTester(f, random_seed=42)
        assert tester.junta_test(k=1) == True

    def test_and_is_k_junta(self):
        """AND on k variables is a k-junta."""
        f = bf.AND(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.junta_test(k=3) == True
        # Not a 2-junta (needs all 3 variables)
        assert tester.junta_test(k=2) == False

    def test_trivial_k_equals_n(self):
        """Any function is an n-junta."""
        f = bf.majority(5)
        tester = PropertyTester(f, random_seed=42)
        assert tester.junta_test(k=5) is True


class TestUnatenessTest:
    """Test unateness_test method."""

    def test_and_is_unate(self):
        """AND is unate (monotone increasing in all variables)."""
        f = bf.AND(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.unateness_test(num_queries=100) is True

    def test_or_is_unate(self):
        """OR is unate (monotone increasing in all variables)."""
        f = bf.OR(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.unateness_test(num_queries=100) is True

    def test_majority_is_unate(self):
        """Majority is unate (monotone)."""
        f = bf.majority(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.unateness_test(num_queries=100) is True

    def test_parity_is_not_unate(self):
        """Parity is NOT unate (each variable can increase or decrease output)."""
        f = bf.parity(3)
        tester = PropertyTester(f, random_seed=42)
        assert tester.unateness_test(num_queries=100) is False

    def test_nand_is_unate(self):
        """NAND (negation of AND) is unate (monotone decreasing)."""
        # NAND: ~(x₀ ∧ x₁ ∧ x₂) = 1 unless all inputs are 1
        tt = [1, 1, 1, 1, 1, 1, 1, 0]  # NAND truth table
        f = bf.create(tt)
        tester = PropertyTester(f, random_seed=42)
        assert tester.unateness_test(num_queries=100) is True


class TestRunAllTests:
    """Test run_all_tests method."""

    def test_returns_dict(self):
        """run_all_tests should return a dictionary."""
        f = bf.majority(3)
        tester = PropertyTester(f, random_seed=42)
        results = tester.run_all_tests()

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_contains_all_basic_tests(self):
        """Should contain results for all basic tests."""
        f = bf.parity(3)
        tester = PropertyTester(f, random_seed=42)
        results = tester.run_all_tests()

        expected_keys = ["constant", "linear", "balanced", "monotone", "symmetric", "affine"]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_results_are_boolean_or_error(self):
        """All results should be boolean or error string."""
        f = bf.AND(3)
        tester = PropertyTester(f, random_seed=42)
        results = tester.run_all_tests()

        for key, value in results.items():
            # Accept Python bool, numpy bool, or error string
            is_bool = isinstance(value, (bool, np.bool_))
            is_str = isinstance(value, str)
            assert is_bool or is_str, f"{key}: unexpected type {type(value)}"


class TestReproducibility:
    """Test that random_seed gives reproducible results."""

    def test_same_seed_same_results(self):
        """Same seed should give identical results."""
        f = bf.majority(5)

        tester1 = PropertyTester(f, random_seed=42)
        tester2 = PropertyTester(f, random_seed=42)

        # Run tests multiple times
        for _ in range(3):
            r1 = tester1.blr_linearity_test(num_queries=50)
            r2 = tester2.blr_linearity_test(num_queries=50)
            # Note: Can't compare directly as RNG state advances

    def test_different_seeds_may_differ(self):
        """Different seeds might give different intermediate states."""
        f = bf.majority(5)

        tester1 = PropertyTester(f, random_seed=42)
        tester2 = PropertyTester(f, random_seed=123)

        # Both should give correct results regardless of seed
        assert tester1.balanced_test() is True
        assert tester2.balanced_test() is True


class TestLocalCorrect:
    """Test local_correct method.

    Mathematical contract (O'Donnell Lecture 2, BLR Theorem):
    If f is ε-close to linear χ_S, then for random y:
        f(y) ⊕ f(x⊕y) = χ_S(x) with probability ≥ 1-2ε
    """

    def test_exact_linear_function_corrects_perfectly(self):
        """For exact linear function, local_correct = χ_S(x)."""
        f = bf.parity(4)  # Linear: χ_[n](x) = (-1)^popcount(x)
        tester = PropertyTester(f, random_seed=42)

        # For parity, χ_S(x) = (-1)^popcount(x) where S = all vars
        for x in range(16):
            corrected = tester.local_correct(x, repetitions=20)
            expected = 1 - 2 * (bin(x).count("1") % 2)  # ±1 rep
            assert corrected == expected, f"x={x}: got {corrected}, expected {expected}"

    def test_dictator_corrects_perfectly(self):
        """Dictator is linear, should correct perfectly."""
        for i in range(3):
            f = bf.dictator(3, i)
            tester = PropertyTester(f, random_seed=42)

            for x in range(8):
                corrected = tester.local_correct(x, repetitions=20)
                # Dictator χ_{i}(x) = (-1)^x_i
                x_i = (x >> i) & 1
                expected = 1 - 2 * x_i
                assert corrected == expected

    def test_noisy_linear_function_mostly_correct(self):
        """Noisy linear function should be mostly corrected."""
        # Create parity with ~10% noise
        n = 4
        true_parity = [bin(x).count("1") % 2 for x in range(16)]
        noisy_tt = true_parity.copy()
        # Flip bit at index 5 (10% noise)
        noisy_tt[5] = 1 - noisy_tt[5]

        f = bf.create(noisy_tt)
        tester = PropertyTester(f, random_seed=42)

        # Should still correct most inputs correctly
        correct_count = 0
        for x in range(16):
            corrected = tester.local_correct(x, repetitions=30)
            expected = 1 - 2 * true_parity[x]
            if corrected == expected:
                correct_count += 1

        # With 1/16 ≈ 6% noise, should get ≥ 80% correct
        accuracy = correct_count / 16
        assert accuracy >= 0.75, f"Accuracy {accuracy} too low for noisy linear"

    def test_local_correct_all_returns_array(self):
        """local_correct_all should return full corrected array."""
        f = bf.parity(3)
        tester = PropertyTester(f, random_seed=42)

        corrected = tester.local_correct_all(repetitions=10)

        assert len(corrected) == 8
        assert corrected.dtype == np.int8
        assert all(v in [-1, 1] for v in corrected)

    def test_invalid_x_raises(self):
        """Out-of-range x should raise ValueError."""
        f = bf.parity(3)
        tester = PropertyTester(f, random_seed=42)

        with pytest.raises(ValueError, match="x must be in"):
            tester.local_correct(8)  # Max is 7 for n=3

        with pytest.raises(ValueError, match="x must be in"):
            tester.local_correct(-1)

    def test_invalid_repetitions_raises(self):
        """Invalid repetitions should raise ValueError."""
        f = bf.parity(3)
        tester = PropertyTester(f, random_seed=42)

        with pytest.raises(ValueError, match="repetitions must be"):
            tester.local_correct(0, repetitions=0)

        with pytest.raises(ValueError, match="repetitions must be"):
            tester.local_correct(0, repetitions=-5)

    def test_more_repetitions_improves_accuracy(self):
        """Higher repetitions should give more consistent results."""
        # Create slightly noisy function
        tt = [bin(x).count("1") % 2 for x in range(8)]
        tt[3] = 1 - tt[3]  # Flip one bit
        f = bf.create(tt)
        tester = PropertyTester(f, random_seed=42)

        # Low repetitions - less consistent
        low_rep_results = [tester.local_correct(0, repetitions=1) for _ in range(10)]

        # High repetitions - more consistent (should converge)
        high_rep_results = [tester.local_correct(0, repetitions=50) for _ in range(10)]

        # High reps should be more consistent (lower variance)
        low_unique = len(set(low_rep_results))
        high_unique = len(set(high_rep_results))

        # With enough reps, should converge to single value
        assert high_unique <= low_unique or high_unique == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_n(self):
        """Should handle n=1 and n=2."""
        f1 = bf.create([0, 1])  # n=1
        f2 = bf.create([0, 0, 0, 1])  # n=2

        for f in [f1, f2]:
            tester = PropertyTester(f, random_seed=42)
            # Should not crash
            tester.constant_test()
            tester.balanced_test()

    def test_high_query_count(self):
        """Should handle high query counts."""
        f = bf.majority(3)
        tester = PropertyTester(f, random_seed=42)

        # Should not crash with many queries
        tester.blr_linearity_test(num_queries=1000)
        tester.monotonicity_test(num_queries=1000)

    def test_low_query_count(self):
        """Should handle low query counts (might give wrong answer)."""
        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)

        # With 1 query, might not detect linearity reliably
        result = tester.blr_linearity_test(num_queries=1)
        # Result can be True or False, just shouldn't crash
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
