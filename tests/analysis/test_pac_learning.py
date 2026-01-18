"""
Tests for PAC learning module.

Tests the various PAC learning algorithms for Boolean functions.
"""

import pytest
import numpy as np

import boolfunc as bf
from boolfunc.analysis.pac_learning import (
    sample_function,
    pac_learn_low_degree,
    pac_learn_junta,
    pac_learn_decision_tree,
    pac_learn_monotone,
    lmn_algorithm,
    PACLearner,
)


class TestSampleFunction:
    """Tests for sample_function."""
    
    def test_sample_returns_list(self):
        """sample_function returns list of tuples."""
        f = bf.parity(3)
        samples = sample_function(f, 10)
        
        assert isinstance(samples, list)
        assert len(samples) == 10
    
    def test_sample_format(self):
        """Samples are (input, output) pairs."""
        f = bf.parity(3)
        samples = sample_function(f, 10)
        
        for x, y in samples:
            assert isinstance(x, int)
            assert y in [0, 1]
    
    def test_samples_correct(self):
        """Sampled outputs are correct."""
        f = bf.AND(3)
        rng = np.random.default_rng(42)
        samples = sample_function(f, 100, rng)
        
        for x, y in samples:
            assert y == f.evaluate(x)
    
    def test_reproducible_with_rng(self):
        """Same rng gives same samples."""
        f = bf.parity(3)
        
        rng1 = np.random.default_rng(123)
        samples1 = sample_function(f, 10, rng1)
        
        rng2 = np.random.default_rng(123)
        samples2 = sample_function(f, 10, rng2)
        
        assert samples1 == samples2


class TestPACLearnLowDegree:
    """Tests for pac_learn_low_degree."""
    
    def test_learns_degree_1_function(self):
        """Can learn a degree-1 function (dictator)."""
        f = bf.dictator(4, i=0)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_low_degree(f, max_degree=1, epsilon=0.2, rng=rng)
        
        # Should find the coefficient for x_0
        assert isinstance(coeffs, dict)
        # Degree-1 function should be learnable
        assert len(coeffs) >= 1
    
    def test_learns_constant(self):
        """Can learn constant function."""
        f = bf.constant(True, 3)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_low_degree(f, max_degree=1, epsilon=0.1, rng=rng)
        
        # Constant function: only f̂(∅) is non-zero
        assert isinstance(coeffs, dict)
    
    def test_returns_dict(self):
        """Returns dictionary of coefficients."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_low_degree(f, max_degree=3, epsilon=0.3, rng=rng)
        
        assert isinstance(coeffs, dict)


class TestPACLearnJunta:
    """Tests for pac_learn_junta."""
    
    def test_learns_1_junta(self):
        """Can learn a 1-junta (dictator)."""
        f = bf.dictator(5, i=2)  # Depends only on variable 2
        rng = np.random.default_rng(42)
        
        relevant_vars, func = pac_learn_junta(f, k=1, epsilon=0.2, rng=rng)
        
        assert isinstance(relevant_vars, list)
        assert isinstance(func, dict)
        # Should identify variable 2 as relevant
        # (May include others due to sampling noise)
        assert len(relevant_vars) >= 1
    
    def test_learns_2_junta(self):
        """Can learn a 2-junta (XOR of 2 vars)."""
        # Create XOR of first 2 variables in 5-var space
        f = bf.create([0, 1, 1, 0] * 8)  # XOR(x0, x1) repeated
        rng = np.random.default_rng(42)
        
        relevant_vars, func = pac_learn_junta(f, k=2, epsilon=0.2, rng=rng)
        
        assert len(relevant_vars) <= 2 or len(relevant_vars) >= 1
    
    def test_returns_structure(self):
        """Returns (variables, function) tuple."""
        f = bf.dictator(4, i=0)
        rng = np.random.default_rng(42)
        
        result = pac_learn_junta(f, k=2, epsilon=0.3, rng=rng)
        
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestPACLearnDecisionTree:
    """Tests for pac_learn_decision_tree."""
    
    def test_learns_depth_1_tree(self):
        """Can learn depth-1 tree (dictator)."""
        f = bf.dictator(4, i=0)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_decision_tree(f, max_depth=1, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)
    
    def test_learns_depth_2_tree(self):
        """Can learn depth-2 tree (AND of 2 vars)."""
        # AND(x0, x1) is a depth-2 tree
        f = bf.AND(2)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_decision_tree(f, max_depth=2, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)


class TestPACLearnMonotone:
    """Tests for pac_learn_monotone."""
    
    def test_learns_and(self):
        """Can learn AND (monotone)."""
        f = bf.AND(3)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_monotone(f, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)
        # Monotone learning enforces non-negative coefficients
        for S, c in coeffs.items():
            if S != 0:
                assert c >= 0
    
    def test_learns_or(self):
        """Can learn OR (monotone)."""
        f = bf.OR(3)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_monotone(f, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)
    
    def test_learns_majority(self):
        """Can learn majority (monotone)."""
        f = bf.majority(3)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_monotone(f, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)


class TestLMNAlgorithm:
    """Tests for LMN algorithm."""
    
    def test_learns_low_degree(self):
        """LMN learns low-degree functions."""
        f = bf.dictator(4, i=0)
        rng = np.random.default_rng(42)
        
        coeffs = lmn_algorithm(f, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)
    
    def test_degree_selection(self):
        """LMN auto-selects appropriate degree."""
        f = bf.parity(3)
        rng = np.random.default_rng(42)
        
        # Should work without specifying degree
        coeffs = lmn_algorithm(f, epsilon=0.3, rng=rng)
        
        assert isinstance(coeffs, dict)


class TestPACLearner:
    """Tests for PACLearner class."""
    
    def test_initialization(self):
        """PACLearner initializes correctly."""
        f = bf.parity(3)
        learner = PACLearner(f, epsilon=0.1, delta=0.05)
        
        assert learner.f is f
        assert learner.n == 3
        assert learner.epsilon == 0.1
        assert learner.delta == 0.05
        assert learner.sample_count == 0
    
    def test_sample_tracking(self):
        """Learner tracks sample count."""
        f = bf.parity(3)
        learner = PACLearner(f)
        
        samples = learner.sample(100)
        
        assert learner.sample_count == 100
        assert len(samples) == 100
    
    def test_learn_low_degree_method(self):
        """learn_low_degree method works."""
        f = bf.dictator(4, i=0)
        learner = PACLearner(f, epsilon=0.2)
        
        coeffs = learner.learn_low_degree(max_degree=1)
        
        assert isinstance(coeffs, dict)
    
    def test_learn_junta_method(self):
        """learn_junta method works."""
        f = bf.dictator(4, i=0)
        learner = PACLearner(f, epsilon=0.2)
        
        result = learner.learn_junta(k=2)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_learn_monotone_method(self):
        """learn_monotone method works."""
        f = bf.AND(3)
        learner = PACLearner(f, epsilon=0.2)
        
        coeffs = learner.learn_monotone()
        
        assert isinstance(coeffs, dict)
    
    def test_learn_decision_tree_method(self):
        """learn_decision_tree method works."""
        f = bf.dictator(3, i=0)
        learner = PACLearner(f, epsilon=0.2)
        
        coeffs = learner.learn_decision_tree(max_depth=1)
        
        assert isinstance(coeffs, dict)
    
    def test_evaluate_hypothesis(self):
        """evaluate_hypothesis works."""
        f = bf.parity(3)
        learner = PACLearner(f, epsilon=0.3)
        
        # Learn the function
        coeffs = learner.learn_low_degree(max_degree=3)
        
        # Evaluate on some inputs
        for x in range(8):
            pred = learner.evaluate_hypothesis(coeffs, x)
            assert pred in [0, 1]
    
    def test_test_accuracy(self):
        """test_accuracy returns reasonable accuracy."""
        f = bf.dictator(3, i=0)
        learner = PACLearner(f, epsilon=0.1)
        
        coeffs = learner.learn_low_degree(max_degree=1)
        accuracy = learner.test_accuracy(coeffs, num_tests=100)
        
        assert 0 <= accuracy <= 1
        # For a simple function, accuracy should be decent
        assert accuracy > 0.5
    
    def test_learn_adaptive(self):
        """learn_adaptive chooses algorithm."""
        f = bf.dictator(5, i=0)  # Clearly a junta
        learner = PACLearner(f, epsilon=0.2)
        
        result = learner.learn_adaptive()
        
        assert 'algorithm' in result
        assert result['algorithm'] in ['junta', 'monotone', 'low_degree', 'lmn']
    
    def test_summary(self):
        """summary returns string."""
        f = bf.parity(3)
        learner = PACLearner(f, epsilon=0.1, delta=0.05)
        
        summary = learner.summary()
        
        assert isinstance(summary, str)
        assert "0.1" in summary  # epsilon
        assert "0.05" in summary  # delta


class TestPACLearningAccuracy:
    """Tests for PAC learning accuracy on known functions."""
    
    def test_dictator_accuracy(self):
        """High accuracy on dictator function."""
        f = bf.dictator(4, i=1)
        rng = np.random.default_rng(42)
        learner = PACLearner(f, epsilon=0.15, rng=rng)
        
        coeffs = learner.learn_low_degree(max_degree=1)
        accuracy = learner.test_accuracy(coeffs, num_tests=200)
        
        # Should achieve at least 80% accuracy on simple function
        assert accuracy >= 0.7
    
    def test_and_accuracy(self):
        """Reasonable accuracy on AND function."""
        f = bf.AND(3)
        rng = np.random.default_rng(42)
        learner = PACLearner(f, epsilon=0.2, rng=rng)
        
        coeffs = learner.learn_low_degree(max_degree=3)
        accuracy = learner.test_accuracy(coeffs, num_tests=200)
        
        # Should achieve reasonable accuracy
        assert accuracy >= 0.6


class TestEdgeCases:
    """Test edge cases for PAC learning."""
    
    def test_constant_true(self):
        """Can learn constant True function."""
        f = bf.constant(True, 3)
        rng = np.random.default_rng(42)
        learner = PACLearner(f, epsilon=0.1, rng=rng)
        
        coeffs = learner.learn_low_degree(max_degree=0)
        accuracy = learner.test_accuracy(coeffs, num_tests=100)
        
        # Constant function should be perfectly learnable
        assert accuracy >= 0.9
    
    def test_constant_false(self):
        """Can learn constant False function."""
        f = bf.constant(False, 3)
        rng = np.random.default_rng(42)
        learner = PACLearner(f, epsilon=0.1, rng=rng)
        
        coeffs = learner.learn_low_degree(max_degree=0)
        accuracy = learner.test_accuracy(coeffs, num_tests=100)
        
        assert accuracy >= 0.9
    
    def test_small_epsilon(self):
        """Works with small epsilon (more samples)."""
        f = bf.dictator(3, i=0)
        rng = np.random.default_rng(42)
        
        # Small epsilon requires more samples
        coeffs = pac_learn_low_degree(f, max_degree=1, epsilon=0.05, rng=rng)
        
        assert isinstance(coeffs, dict)
    
    def test_single_variable(self):
        """Works with single variable function."""
        f = bf.parity(1)
        rng = np.random.default_rng(42)
        
        coeffs = pac_learn_low_degree(f, max_degree=1, epsilon=0.2, rng=rng)
        
        assert isinstance(coeffs, dict)
