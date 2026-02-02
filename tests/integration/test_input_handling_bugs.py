"""
Tests for input handling bugs identified in notebooks.

These tests verify:
1. Lambda/callable function creation and evaluation
2. Symbolic expression parsing and evaluation  
3. Large n handling (overflow prevention)
4. Representation conversions work end-to-end

Following TEST_GUIDELINES.md:
- Tests verify mathematical correctness, not just existence
- Edge cases are covered
- Expected failures are documented with xfail until fixed
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

import boofun as bf


# =============================================================================
# Bug 1: Lambda/Callable Handling
# =============================================================================

class TestLambdaCallableHandling:
    """
    Tests for bf.create(lambda x: ..., n=N) functionality.
    
    The issue: When a user creates a function from a lambda, the evaluation
    may fail due to input format mismatches. The lambda might expect:
    - A list/array of bits [0, 1, 0, 1]
    - An integer index 5
    - Different conventions ({0,1} vs {-1,+1})
    
    See: flexible_inputs_and_oracles.ipynb errors
    """

    def test_lambda_with_list_input_basic_xor(self):
        """Lambda that expects list input and returns XOR."""
        # User expectation: lambda receives [x0, x1] as a list
        xor_lambda = lambda x: x[0] ^ x[1]
        
        f = bf.create(xor_lambda, n=2)
        
        # Verify the function was created
        assert f is not None
        assert f.n_vars == 2
        
        # Verify evaluation matches XOR truth table
        # XOR: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
        assert f.evaluate([0, 0]) == False
        assert f.evaluate([0, 1]) == True  
        assert f.evaluate([1, 0]) == True
        assert f.evaluate([1, 1]) == False

    def test_lambda_with_sum_threshold(self):
        """Lambda that uses sum() - common pattern that fails."""
        # This is a common pattern users write
        threshold_lambda = lambda x: sum(x) >= 3
        
        f = bf.create(threshold_lambda, n=5)
        
        # Verify evaluation
        assert f.evaluate([1, 1, 1, 0, 0]) == True   # sum=3
        assert f.evaluate([1, 1, 0, 0, 0]) == False  # sum=2
        assert f.evaluate([1, 1, 1, 1, 1]) == True   # sum=5

    def test_lambda_with_len_access(self):
        """Lambda that accesses len(x)."""
        majority_lambda = lambda x: sum(x) > len(x) / 2
        
        f = bf.create(majority_lambda, n=5)
        
        # Verify it computes majority correctly
        assert f.evaluate([1, 1, 1, 0, 0]) == True   # 3/5 > 2.5
        assert f.evaluate([1, 1, 0, 0, 0]) == False  # 2/5 < 2.5

    def test_lambda_with_all_builtin(self):
        """Lambda using all() - requires iterable."""
        and_lambda = lambda x: all(x[:3])  # AND of first 3 vars
        
        f = bf.create(and_lambda, n=5)
        
        assert f.evaluate([1, 1, 1, 0, 0]) == True
        assert f.evaluate([1, 1, 0, 0, 0]) == False

    def test_lambda_with_any_builtin(self):
        """Lambda using any() - requires iterable."""
        or_lambda = lambda x: any(x[:2])  # OR of first 2 vars
        
        f = bf.create(or_lambda, n=3)
        
        assert f.evaluate([0, 0, 1]) == False
        assert f.evaluate([0, 1, 0]) == True
        assert f.evaluate([1, 0, 0]) == True

    def test_lambda_created_function_can_get_truth_table(self):
        """Verify lambda functions can be converted to truth table."""
        xor_lambda = lambda x: x[0] ^ x[1]
        f = bf.create(xor_lambda, n=2)
        
        # This should work - convert to truth table
        tt = f.get_representation("truth_table")
        
        # Verify truth table is correct for XOR
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(tt, expected)

    def test_lambda_created_function_fourier(self):
        """Verify lambda functions can compute Fourier coefficients."""
        xor_lambda = lambda x: x[0] ^ x[1]
        f = bf.create(xor_lambda, n=2)
        
        # Get Fourier coefficients
        fourier = f.fourier()
        
        # XOR has only one nonzero coefficient at S={0,1}
        # In standard ordering, index 3 = {0,1}
        assert len(fourier) == 4
        # The coefficient at {0,1} should be ±1
        assert abs(abs(fourier[3]) - 1.0) < 0.01

    def test_named_function_instead_of_lambda(self):
        """Named functions should work the same as lambdas."""
        def my_and(x):
            return x[0] and x[1]
        
        f = bf.create(my_and, n=2)
        
        assert f.evaluate([0, 0]) == False
        assert f.evaluate([0, 1]) == False
        assert f.evaluate([1, 0]) == False
        assert f.evaluate([1, 1]) == True


# =============================================================================
# Bug 2: Symbolic Expression Handling
# =============================================================================

class TestSymbolicExpressionHandling:
    """
    Tests for bf.create("x0 & x1", n=2) functionality.
    
    The issue: Symbolic expressions fail during conversion to truth table
    because convert_from returns empty tuple, making funcs=None.
    
    See: flexible_inputs_and_oracles.ipynb errors
    """

    def test_symbolic_and_expression(self):
        """Basic AND expression."""
        f = bf.create("x0 & x1", n=2)
        
        assert f is not None
        assert f.n_vars == 2
        
        # Evaluate - should match AND
        assert f.evaluate([0, 0]) == False
        assert f.evaluate([0, 1]) == False
        assert f.evaluate([1, 0]) == False
        assert f.evaluate([1, 1]) == True

    def test_symbolic_or_expression(self):
        """Basic OR expression."""
        f = bf.create("x0 | x1", n=2)
        
        assert f.evaluate([0, 0]) == False
        assert f.evaluate([0, 1]) == True
        assert f.evaluate([1, 0]) == True
        assert f.evaluate([1, 1]) == True

    def test_symbolic_xor_expression(self):
        """XOR expression."""
        f = bf.create("x0 ^ x1", n=2)
        
        assert f.evaluate([0, 0]) == False
        assert f.evaluate([0, 1]) == True
        assert f.evaluate([1, 0]) == True
        assert f.evaluate([1, 1]) == False

    def test_symbolic_not_expression(self):
        """NOT expression."""
        f = bf.create("not x0", n=1)
        
        assert f.evaluate([0]) == True
        assert f.evaluate([1]) == False

    def test_symbolic_complex_expression(self):
        """Complex expression with multiple operators."""
        f = bf.create("(x0 | x1) & (not x2)", n=3)
        
        # OR of x0,x1 must be true, AND x2 must be false
        assert f.evaluate([0, 0, 0]) == False  # OR is false
        assert f.evaluate([1, 0, 0]) == True   # OR=1, NOT x2=1
        assert f.evaluate([1, 0, 1]) == False  # OR=1, NOT x2=0
        assert f.evaluate([0, 1, 0]) == True   # OR=1, NOT x2=1

    def test_symbolic_get_truth_table(self):
        """Symbolic expression should convert to truth table."""
        f = bf.create("x0 & x1", n=2)
        
        tt = f.get_representation("truth_table")
        
        # AND truth table: [0,0,0,1]
        expected = np.array([False, False, False, True])
        np.testing.assert_array_equal(tt, expected)

    def test_symbolic_with_variables_kwarg(self):
        """Using explicit variables parameter."""
        f = bf.create("a and b", variables=["a", "b"])
        
        assert f.evaluate([0, 0]) == False
        assert f.evaluate([1, 1]) == True

    def test_symbolic_parity_3(self):
        """3-variable parity via symbolic."""
        f = bf.create("x0 ^ x1 ^ x2", n=3)
        
        # Parity truth table
        for i in range(8):
            bits = [(i >> j) & 1 for j in range(3)]
            expected = sum(bits) % 2 == 1
            assert f.evaluate(bits) == expected, f"Failed for input {bits}"


# =============================================================================
# Bug 3: Large n / Integer Overflow
# =============================================================================

class TestLargeNHandling:
    """
    Tests for functions with large n values.
    
    The issue: estimate_fourier_coefficient uses rng.integers(0, 1 << n)
    which overflows for n > 63.
    
    See: flexible_inputs_and_oracles.ipynb overflow errors
    """

    def test_create_large_n_builtin(self):
        """Built-in functions should work for large n."""
        # These should work for large n
        f = bf.majority(31)  # n=31 still works
        assert f.n_vars == 31
        
        f = bf.parity(30)
        assert f.n_vars == 30

    def test_evaluate_large_n_lazy(self):
        """Lazy evaluation should work for large n without materializing."""
        # For n=30, we can't materialize but should be able to evaluate
        f = bf.majority(30)
        
        # Evaluate at a specific point
        input_bits = [1] * 16 + [0] * 14  # 16 ones, 14 zeros
        result = f.evaluate(input_bits)
        assert result == True  # 16 > 15

    def test_sampling_works_for_moderate_n(self):
        """Sampling algorithms should work for n < 64."""
        from boofun.analysis.learning import estimate_fourier_coefficient
        
        # n=30 should work fine (2^30 fits in int64)
        f = bf.majority(30)
        
        # This should not overflow
        result, stderr = estimate_fourier_coefficient(f, S=0, num_samples=100)
        assert isinstance(result, float)
        assert isinstance(stderr, float)

    def test_sampling_error_for_very_large_n(self):
        """For n >= 64, sampling should raise clear error."""
        from boofun.analysis.sampling import sample_uniform
        
        # n >= 64 should raise ValueError with helpful message
        with pytest.raises(ValueError, match="n < 64"):
            sample_uniform(n=64, n_samples=10)
        
        with pytest.raises(ValueError, match="n < 64"):
            sample_uniform(n=100, n_samples=10)

    def test_sample_uniform_bits_works_for_any_n(self):
        """sample_uniform_bits should work for any n (no overflow)."""
        from boofun.analysis.sampling import sample_uniform_bits
        
        # Small n
        samples = sample_uniform_bits(n=5, n_samples=10)
        assert samples.shape == (10, 5)
        assert np.all((samples >= 0) & (samples <= 1))
        
        # Large n (would overflow with integers)
        samples = sample_uniform_bits(n=100, n_samples=10)
        assert samples.shape == (10, 100)
        assert np.all((samples >= 0) & (samples <= 1))


# =============================================================================
# Bug 4: Representation Conversion Round-trips
# =============================================================================

class TestRepresentationConversions:
    """
    Tests for converting between representations.
    
    These verify that round-trip conversions preserve semantics.
    """

    def test_truth_table_to_fourier_round_trip(self):
        """Truth table -> Fourier -> evaluate should match."""
        f = bf.AND(3)
        
        # Get Fourier representation
        fourier = f.fourier()
        
        # The function should still evaluate correctly
        assert f.evaluate([1, 1, 1]) == True
        assert f.evaluate([0, 0, 0]) == False
        
        # Parseval: sum of squared coefficients = 1
        sum_sq = sum(c**2 for c in fourier)
        assert abs(sum_sq - 1.0) < 0.01

    def test_builtin_to_truth_table_correctness(self):
        """Built-in functions should have correct truth tables."""
        maj3 = bf.majority(3)
        tt = maj3.get_representation("truth_table")
        
        # Majority-3: 1 iff at least 2 bits are 1
        expected = [
            False,  # 000 -> 0
            False,  # 001 -> 0
            False,  # 010 -> 0
            True,   # 011 -> 1
            False,  # 100 -> 0
            True,   # 101 -> 1
            True,   # 110 -> 1
            True,   # 111 -> 1
        ]
        np.testing.assert_array_equal(tt, expected)

    def test_function_to_truth_table_conversion(self):
        """Lambda function -> truth table conversion."""
        xor = lambda x: x[0] ^ x[1]
        f = bf.create(xor, n=2)
        
        # Convert to truth table
        tt = f.get_representation("truth_table")
        
        # Verify XOR truth table
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(tt, expected)

    def test_symbolic_to_truth_table_conversion(self):
        """Symbolic expression -> truth table conversion."""
        f = bf.create("x0 & x1", n=2)
        
        # Convert to truth table  
        tt = f.get_representation("truth_table")
        
        # Verify AND truth table
        expected = np.array([False, False, False, True])
        np.testing.assert_array_equal(tt, expected)


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases that might reveal related bugs."""

    def test_n_equals_1(self):
        """Single variable functions."""
        f = bf.dictator(1, 0)
        assert f.evaluate([0]) == False
        assert f.evaluate([1]) == True

    def test_constant_functions(self):
        """Constant True and False functions."""
        true_fn = bf.constant(True, 3)
        false_fn = bf.constant(False, 3)
        
        for i in range(8):
            bits = [(i >> j) & 1 for j in range(3)]
            assert true_fn.evaluate(bits) == True
            assert false_fn.evaluate(bits) == False

    def test_lambda_returning_int(self):
        """Lambda that returns int instead of bool."""
        f = bf.create(lambda x: 1 if x[0] else 0, n=1)
        
        # Should still work
        assert f.evaluate([0]) == False
        assert f.evaluate([1]) == True

    def test_lambda_with_numpy_operations(self):
        """Lambda using numpy operations."""
        f = bf.create(lambda x: np.sum(x) >= 2, n=3)
        
        assert f.evaluate([0, 0, 0]) == False
        assert f.evaluate([1, 1, 0]) == True
        assert f.evaluate([1, 1, 1]) == True


# =============================================================================
# API Naming Consistency
# =============================================================================

class TestAPIConsistency:
    """Test API naming is consistent and documented."""

    def test_hypercontractivity_function_names(self):
        """Verify hypercontractivity function names match imports."""
        from boofun.analysis.hypercontractivity import (
            bonami_lemma_bound,
            kkl_lower_bound,
        )
        
        f = bf.majority(5)
        
        # These should be callable
        assert callable(bonami_lemma_bound)
        assert callable(kkl_lower_bound)
        
        # Test they work
        result = kkl_lower_bound(f.total_influence(), f.n_vars)
        assert isinstance(result, float)

    def test_dictator_argument_order(self):
        """Verify dictator(n, i) not dictator(i, n)."""
        # dictator(n, i) - n variables, i-th is the dictator
        f = bf.dictator(5, 2)  # 5 variables, x_2 is dictator
        
        assert f.n_vars == 5
        
        # Only x_2 should matter
        influences = f.influences()
        assert influences[2] == 1.0
        for i in [0, 1, 3, 4]:
            assert influences[i] == 0.0

    def test_tribes_argument_order(self):
        """Verify tribes(k, n) where k is tribe size, n is total vars."""
        # tribes(2, 4) = (x0 OR x1) AND (x2 OR x3)
        f = bf.tribes(2, 4)
        
        assert f.n_vars == 4
        
        # Both tribes must be satisfied
        assert f.evaluate([1, 0, 1, 0]) == True   # Both tribes have a 1
        assert f.evaluate([0, 0, 1, 0]) == False  # First tribe is 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
