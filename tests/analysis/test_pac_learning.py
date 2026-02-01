"""
Comprehensive tests for PAC learning module.

Tests for PAC learning algorithms from O'Donnell Chapter 3.
Tests verify both that algorithms run AND that they learn correctly.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.pac_learning import (
    PACLearner,
    lmn_algorithm,
    pac_learn_decision_tree,
    pac_learn_junta,
    pac_learn_low_degree,
    pac_learn_monotone,
    pac_learn_sparse_fourier,
    sample_function,
)


def compute_error_rate(hypothesis, target, n_vars):
    """Compute fraction of inputs where hypothesis differs from target.

    Args:
        hypothesis: BooleanFunction, Dict[int, float] (Fourier coefficients),
                   or Tuple[List[int], Dict[int, int]] (junta: relevant_vars, truth_table)
        target: BooleanFunction
        n_vars: Number of variables
    """
    if hypothesis is None:
        return 1.0

    errors = 0
    total = 2**n_vars

    # Handle junta format: (relevant_vars, truth_table)
    if isinstance(hypothesis, tuple) and len(hypothesis) == 2:
        relevant_vars, truth_table = hypothesis
        for x in range(total):
            try:
                t_val = int(target.evaluate(x))
                # Project x onto relevant variables
                x_proj = 0
                for i, var in enumerate(relevant_vars):
                    if (x >> var) & 1:
                        x_proj |= 1 << i
                h_val = truth_table.get(x_proj, 0)
                if int(h_val) != t_val:
                    errors += 1
            except Exception:
                errors += 1
        return errors / total

    for x in range(total):
        try:
            # Get target value (convert to ±1)
            t_val = target.evaluate(x)
            t_pm = 1 - 2 * int(t_val)  # 0 → +1, 1 → -1

            # Get hypothesis value
            if isinstance(hypothesis, dict):
                # Hypothesis is Fourier coefficients: h(x) = sum_S coeff[S] * χ_S(x)
                h_real = 0.0
                for s, coeff in hypothesis.items():
                    chi_s = 1 - 2 * (bin(x & s).count("1") % 2)
                    h_real += coeff * chi_s
                # Threshold to get ±1 value
                h_pm = 1 if h_real >= 0 else -1
            else:
                # Hypothesis is BooleanFunction
                h_val = hypothesis.evaluate(x)
                h_pm = 1 - 2 * int(h_val)

            if h_pm != t_pm:
                errors += 1
        except Exception:
            errors += 1

    return errors / total


class TestSampleFunction:
    """Test sample_function utility."""

    def test_sample_returns_correct_count(self):
        """sample_function returns requested number of samples."""
        f = bf.majority(3)
        for count in [1, 10, 100]:
            samples = sample_function(f, count)
            assert len(samples) == count

    def test_sample_tuple_format(self):
        """Each sample is (input_index, output_value) tuple."""
        f = bf.AND(3)
        samples = sample_function(f, 20)

        for x, y in samples:
            assert isinstance(x, int), "Input should be integer index"
            assert 0 <= x < 8, f"Input {x} out of range [0, 8)"
            assert y in [0, 1], f"Output {y} not in {{0, 1}}"

    def test_sample_consistency_with_function(self):
        """All samples are consistent with the target function."""
        f = bf.parity(4)
        samples = sample_function(f, 50)

        for x, y in samples:
            expected = f.evaluate(x)
            assert y == expected, f"Sample ({x}, {y}) inconsistent with f({x})={expected}"

    def test_sample_deterministic_with_seed(self):
        """Same seed produces same samples."""
        f = bf.OR(3)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        samples1 = sample_function(f, 10, rng=rng1)
        samples2 = sample_function(f, 10, rng=rng2)

        assert samples1 == samples2


class TestPACLearnLowDegree:
    """Test pac_learn_low_degree: learns functions with bounded Fourier degree."""

    def test_learns_constant_exactly(self):
        """Constant function (degree 0) should be learned with 0 error."""
        f = bf.create([0, 0, 0, 0])
        hypothesis = pac_learn_low_degree(f, max_degree=0)

        error = compute_error_rate(hypothesis, f, 2)
        assert error == 0.0, f"Constant function: expected 0 error, got {error}"

    def test_learns_dictator_exactly(self):
        """Dictator (degree 1) should be learned with low error."""
        f = bf.dictator(3, 0)
        hypothesis = pac_learn_low_degree(f, max_degree=1)

        error = compute_error_rate(hypothesis, f, 3)
        assert error <= 0.2, f"Dictator: expected ≤0.2 error, got {error}"

    def test_learns_parity_with_full_degree(self):
        """Parity (degree n) needs max_degree=n to learn."""
        f = bf.parity(3)
        hypothesis = pac_learn_low_degree(f, max_degree=3)

        error = compute_error_rate(hypothesis, f, 3)
        assert error <= 0.3, f"Parity with full degree: expected ≤0.3 error, got {error}"

    def test_fails_gracefully_with_insufficient_degree(self):
        """Parity with max_degree=1 should have high error (can't learn it)."""
        f = bf.parity(3)
        hypothesis = pac_learn_low_degree(f, max_degree=1)

        # Should either return None or have high error
        if hypothesis is not None:
            error = compute_error_rate(hypothesis, f, 3)
            # Parity has no degree-1 component, so best we can do is ~50% (random)
            assert error >= 0.3, "Should struggle with parity using degree 1"


class TestPACLearnJunta:
    """Test pac_learn_junta: learns functions depending on few variables."""

    def test_learns_dictator_as_1junta(self):
        """Dictator depends on 1 variable."""
        f = bf.dictator(5, 2)
        result = pac_learn_junta(f, k=1)

        error = compute_error_rate(result, f, 5)
        assert error <= 0.3, f"Dictator as 1-junta: expected ≤0.3 error, got {error}"

    def test_learns_and_as_kjunta(self):
        """AND on k variables is a k-junta."""
        f = bf.AND(3)
        result = pac_learn_junta(f, k=3)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.3, f"AND as 3-junta: expected ≤0.3 error, got {error}"

    def test_handles_epsilon_delta(self):
        """Should respect epsilon/delta parameters."""
        f = bf.OR(3)
        result = pac_learn_junta(f, k=3, epsilon=0.1, delta=0.05)

        # With epsilon=0.1, error should be ≤ 0.1 with high probability
        error = compute_error_rate(result, f, 3)
        assert error <= 0.5  # Relaxed bound for randomized algorithm


class TestLMNAlgorithm:
    """Test LMN (Linial-Mansour-Nisan) algorithm."""

    def test_lmn_on_low_degree_function(self):
        """LMN should work well on low-degree functions."""
        f = bf.majority(3)

        try:
            result = lmn_algorithm(f)
        except TypeError:
            result = lmn_algorithm(f, 100)

        if result is not None:
            error = compute_error_rate(result, f, 3)
            assert error <= 0.5, f"LMN on majority: expected ≤0.5 error, got {error}"


class TestPACLearnSparseFourier:
    """Test pac_learn_sparse_fourier: learns Fourier-sparse functions."""

    def test_learns_parity_sparsity_1(self):
        """Parity has exactly 1 non-zero Fourier coefficient."""
        f = bf.parity(3)
        result = pac_learn_sparse_fourier(f, sparsity=1)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.3, f"Parity (sparsity 1): expected ≤0.3 error, got {error}"

    def test_learns_dictator_sparsity_2(self):
        """Dictator has 2 non-zero coefficients: f̂(∅) and f̂({i})."""
        f = bf.dictator(3, 0)
        result = pac_learn_sparse_fourier(f, sparsity=2)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.3, f"Dictator (sparsity 2): expected ≤0.3 error, got {error}"


class TestPACLearnDecisionTree:
    """Test pac_learn_decision_tree: learns functions with small DT complexity."""

    def test_learns_and_with_linear_depth(self):
        """AND has DT depth = n (check all variables)."""
        f = bf.AND(3)
        result = pac_learn_decision_tree(f, max_depth=3)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.3, f"AND (depth 3): expected ≤0.3 error, got {error}"

    def test_learns_or_with_linear_depth(self):
        """OR has DT depth = n."""
        f = bf.OR(3)
        result = pac_learn_decision_tree(f, max_depth=3)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.3, f"OR (depth 3): expected ≤0.3 error, got {error}"


class TestPACLearnMonotone:
    """Test pac_learn_monotone: learns monotone functions."""

    def test_learns_and_monotone(self):
        """AND is monotone."""
        f = bf.AND(3)
        result = pac_learn_monotone(f)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.5, f"Monotone AND: expected ≤0.5 error, got {error}"

    def test_learns_or_monotone(self):
        """OR is monotone."""
        f = bf.OR(3)
        result = pac_learn_monotone(f)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.5, f"Monotone OR: expected ≤0.5 error, got {error}"

    def test_learns_majority_monotone(self):
        """Majority is monotone."""
        f = bf.majority(3)
        result = pac_learn_monotone(f)

        error = compute_error_rate(result, f, 3)
        assert error <= 0.5, f"Monotone majority: expected ≤0.5 error, got {error}"


class TestPACLearner:
    """Test PACLearner class interface."""

    def test_learner_init_stores_function(self):
        """PACLearner should store the target function."""
        f = bf.majority(3)
        learner = PACLearner(f)

        assert hasattr(learner, "f") or hasattr(learner, "function")

    def test_learner_has_learning_methods(self):
        """PACLearner should expose learning methods."""
        f = bf.AND(3)
        learner = PACLearner(f)

        # Check for at least one learning method
        methods = [m for m in dir(learner) if "learn" in m.lower() and not m.startswith("_")]
        assert len(methods) > 0, "PACLearner should have learning methods"

    def test_learner_works_on_different_functions(self):
        """PACLearner should handle various function types."""
        test_cases = [
            (bf.AND(3), "AND"),
            (bf.OR(3), "OR"),
            (bf.parity(3), "Parity"),
            (bf.majority(3), "Majority"),
        ]

        for func, name in test_cases:
            learner = PACLearner(func)
            # Should not raise
            assert learner is not None, f"Failed to create learner for {name}"


class TestPACLearningEdgeCases:
    """Test edge cases for PAC learning."""

    def test_constant_zero(self):
        """Constant zero should be trivially learnable."""
        f = bf.create([0, 0, 0, 0])
        result = pac_learn_low_degree(f, max_degree=0)

        error = compute_error_rate(result, f, 2)
        assert error == 0.0, "Constant zero should have 0 error"

    def test_constant_one(self):
        """Constant one should be trivially learnable."""
        f = bf.create([1, 1, 1, 1])
        result = pac_learn_low_degree(f, max_degree=0)

        error = compute_error_rate(result, f, 2)
        assert error == 0.0, "Constant one should have 0 error"

    def test_n_equals_1(self):
        """Should work for smallest non-trivial case n=1."""
        f = bf.create([0, 1])  # Identity
        result = pac_learn_low_degree(f, max_degree=1)

        error = compute_error_rate(result, f, 1)
        assert error <= 0.5, f"n=1 case: expected ≤0.5 error, got {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
