import sys

sys.path.insert(0, "src")
"""
Integration tests for basic BooFun functionality.

These tests verify that the core components work together correctly,
focusing on real-world usage patterns and end-to-end functionality.
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis import SpectralAnalyzer
from boofun.core.builtins import BooleanFunctionBuiltins


class TestBasicFunctionCreation:
    """Test creation of Boolean functions from different representations."""

    def test_create_from_truth_table(self):
        """Test creating function from truth table."""
        # XOR function: [0, 1, 1, 0]
        truth_table = [False, True, True, False]
        f = bf.create(truth_table)

        assert f.n_vars == 2
        assert f.has_rep("truth_table")

        # Test evaluation
        assert f.evaluate(np.array(0)) == False  # 00 -> 0
        assert f.evaluate(np.array(1)) == True  # 01 -> 1
        assert f.evaluate(np.array(2)) == True  # 10 -> 1
        assert f.evaluate(np.array(3)) == False  # 11 -> 0

    def test_create_from_function(self):
        """Test creating function from Python callable."""

        def xor_func(x):
            """XOR of two bits"""
            return x[0] ^ x[1]

        f = bf.create(xor_func, n=2)
        assert f.n_vars == 2
        assert f.has_rep("function")

    def test_create_from_symbolic(self):
        """Test creating function from symbolic expression."""
        expr = "x0 and not x1"
        f = bf.create(expr, variables=["x0", "x1"], n=2)

        assert f.n_vars == 2
        assert f.has_rep("symbolic")


class TestBuiltinFunctions:
    """Test built-in Boolean functions."""

    def test_majority_function(self):
        """Test majority function creation and evaluation."""
        # 3-variable majority
        maj3 = BooleanFunctionBuiltins.majority(3)

        assert maj3.n_vars == 3

        # Test all inputs
        expected = [False, False, False, True, False, True, True, True]
        for i, expected_val in enumerate(expected):
            result = maj3.evaluate(np.array(i))
            assert result == expected_val, f"Majority failed at input {i:03b}"

    def test_dictator_function(self):
        """Test dictator function creation and evaluation."""
        # 3-variable dictator on variable 1 (middle bit)
        dict_1 = BooleanFunctionBuiltins.dictator(3, 1)

        assert dict_1.n_vars == 3

        # Test all inputs - should return middle bit
        for i in range(8):
            expected = bool((i >> 1) & 1)  # Extract middle bit
            result = dict_1.evaluate(np.array(i))
            assert result == expected, f"Dictator failed at input {i:03b}"

    def test_parity_function(self):
        """Test parity function creation and evaluation."""
        parity2 = BooleanFunctionBuiltins.parity(2)

        assert parity2.n_vars == 2

        # XOR truth table: [0, 1, 1, 0]
        expected = [False, True, True, False]
        for i, expected_val in enumerate(expected):
            result = parity2.evaluate(np.array(i))
            assert result == expected_val, f"Parity failed at input {i:02b}"

    def test_constant_function(self):
        """Test constant function creation and evaluation."""
        const_true = BooleanFunctionBuiltins.constant(True, 2)
        const_false = BooleanFunctionBuiltins.constant(False, 2)

        # Test all inputs return constant value
        for i in range(4):
            assert const_true.evaluate(np.array(i)) == True
            assert const_false.evaluate(np.array(i)) == False

    def test_tribes_function(self):
        """Test tribes function creation and basic properties."""
        # 4-variable tribes with k=2: (x0 ∨ x1) ∧ (x2 ∨ x3)
        tribes_2_4 = BooleanFunctionBuiltins.tribes(k=2, n=4)

        assert tribes_2_4.n_vars == 4

        # Test specific cases
        assert tribes_2_4.evaluate(np.array(0b0000)) == False  # All false
        assert tribes_2_4.evaluate(np.array(0b0001)) == False  # Only x3 true
        assert tribes_2_4.evaluate(np.array(0b0011)) == False  # Only second tribe
        assert tribes_2_4.evaluate(np.array(0b1100)) == False  # Only first tribe
        assert tribes_2_4.evaluate(np.array(0b1111)) == True  # All true
        assert tribes_2_4.evaluate(np.array(0b1001)) == True  # One from each tribe


class TestSpectralAnalysis:
    """Test spectral analysis functionality."""

    def test_influences_computation(self):
        """Test influence computation for known functions."""
        # Dictator function - only one variable has influence
        dict_0 = BooleanFunctionBuiltins.dictator(3, 0)
        analyzer = SpectralAnalyzer(dict_0)

        influences = analyzer.influences()

        # Only variable 0 should have influence 1, others should be 0
        assert abs(influences[0] - 1.0) < 1e-10
        assert abs(influences[1] - 0.0) < 1e-10
        assert abs(influences[2] - 0.0) < 1e-10

        # Total influence should be 1
        assert abs(analyzer.total_influence() - 1.0) < 1e-10

    def test_parity_influences(self):
        """Test influences for parity function."""
        # Parity function - all variables should have equal influence
        parity3 = BooleanFunctionBuiltins.parity(3)
        analyzer = SpectralAnalyzer(parity3)

        influences = analyzer.influences()

        # All influences should be 1.0 for parity (flipping any bit changes output)
        for inf in influences:
            assert abs(inf - 1.0) < 1e-10

        # Total influence should be 3.0 for 3-variable parity
        assert abs(analyzer.total_influence() - 3.0) < 1e-10

    def test_constant_function_analysis(self):
        """Test analysis of constant functions."""
        const_func = BooleanFunctionBuiltins.constant(True, 3)
        analyzer = SpectralAnalyzer(const_func)

        influences = analyzer.influences()

        # Constant function should have zero influence for all variables
        for inf in influences:
            assert abs(inf) < 1e-10

        assert abs(analyzer.total_influence()) < 1e-10

    def test_fourier_expansion(self):
        """Test Fourier expansion computation."""
        # Simple 2-variable function
        f = bf.create([False, True, True, False])  # XOR
        analyzer = SpectralAnalyzer(f)

        fourier_coeffs = analyzer.fourier_expansion()

        # Should have 4 coefficients for 2-variable function
        assert len(fourier_coeffs) == 4

        # Parseval's theorem: sum of squared coefficients should equal 1
        parseval_sum = np.sum(fourier_coeffs**2)
        assert abs(parseval_sum - 1.0) < 1e-10

    def test_noise_stability(self):
        """Test noise stability computation."""
        # Constant function should have noise stability 1 for any rho
        const_func = BooleanFunctionBuiltins.constant(True, 2)
        analyzer = SpectralAnalyzer(const_func)

        stability_09 = analyzer.noise_stability(0.9)
        stability_05 = analyzer.noise_stability(0.5)

        assert abs(stability_09 - 1.0) < 1e-10
        assert abs(stability_05 - 1.0) < 1e-10

    def test_spectral_concentration(self):
        """Test spectral concentration measurement."""
        # Dictator function has all weight on degree-1 coefficients
        dict_func = BooleanFunctionBuiltins.dictator(3, 0)
        analyzer = SpectralAnalyzer(dict_func)

        conc_0 = analyzer.spectral_concentration(0)
        conc_1 = analyzer.spectral_concentration(1)
        conc_2 = analyzer.spectral_concentration(2)

        # Dictator should have no degree-0 weight, all weight at degree-1
        assert abs(conc_0 - 0.0) < 1e-10
        assert abs(conc_1 - 1.0) < 1e-10
        assert abs(conc_2 - 1.0) < 1e-10


class TestFunctionOperations:
    """Test Boolean function operations and compositions."""

    def test_function_evaluation_formats(self):
        """Test different input formats for evaluation."""
        f = bf.create([False, True, True, False])  # XOR

        # Test integer index
        assert f.evaluate(np.array(0)) == False
        assert f.evaluate(np.array(3)) == False

        # Test binary vector
        assert f.evaluate(np.array([0, 1])) == True
        assert f.evaluate(np.array([1, 0])) == True

        # Test batch evaluation
        batch_indices = np.array([0, 1, 2, 3])
        results = f.evaluate(batch_indices)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(results, expected)

    def test_numpy_array_conversion(self):
        """Test conversion to NumPy array."""
        f = bf.create([False, True, True, False])

        arr = np.array(f)
        expected = np.array([False, True, True, False])
        np.testing.assert_array_equal(arr, expected)

        # Test with dtype specification
        bool_arr = np.array(f, dtype=bool)
        np.testing.assert_array_equal(bool_arr, expected)

    def test_function_call_syntax(self):
        """Test using functions with call syntax."""
        f = bf.create([False, True, True, False])

        # Should work like evaluate()
        assert f(np.array(0)) == False
        assert f(np.array(1)) == True
        assert f(np.array(2)) == True
        assert f(np.array(3)) == False


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_truth_table_size(self):
        """Test error on invalid truth table sizes."""
        with pytest.raises(ValueError):
            BooleanFunctionBuiltins.majority(0)  # Zero variables

        with pytest.raises(ValueError):
            BooleanFunctionBuiltins.dictator(-1, 3)  # Invalid index

        with pytest.raises(ValueError):
            BooleanFunctionBuiltins.dictator(3, 3)  # Index out of range (i=3 >= n=3)

    def test_evaluation_out_of_bounds(self):
        """Test evaluation with invalid indices."""
        f = bf.create([False, True, True, False])  # 2-variable function

        with pytest.raises(IndexError):
            f.evaluate(np.array(4))  # Index too large

        with pytest.raises(IndexError):
            f.evaluate(np.array(-1))  # Negative index

    def test_spectral_analyzer_invalid_function(self):
        """Test spectral analyzer with invalid function."""
        f = bf.BooleanFunction(n=None)  # No variables defined

        with pytest.raises(ValueError):
            SpectralAnalyzer(f)

    def test_noise_stability_invalid_rho(self):
        """Test noise stability with invalid correlation."""
        f = bf.create([False, True, True, False])
        analyzer = SpectralAnalyzer(f)

        with pytest.raises(ValueError):
            analyzer.noise_stability(2.0)  # rho > 1

        with pytest.raises(ValueError):
            analyzer.noise_stability(-2.0)  # rho < -1


class TestStringRepresentations:
    """Test string representations and debugging output."""

    def test_function_string_representation(self):
        """Test __str__ and __repr__ methods."""
        f = bf.create([False, True, True, False])

        str_repr = str(f)
        assert "BooleanFunction" in str_repr
        assert "vars=2" in str_repr

        repr_str = repr(f)
        assert "BooleanFunction" in repr_str
        assert "n_vars=2" in repr_str


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
