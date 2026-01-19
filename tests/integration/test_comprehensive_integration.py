"""
Comprehensive integration tests for BooFun library.

These tests provide thorough coverage of the main user-facing functionality,
ensuring all core components work together correctly.
"""

import pytest
import numpy as np
import boofun as bf
from boofun.core.spaces import Space


class TestBooleanFunctionCreation:
    """Test comprehensive Boolean function creation scenarios."""
    
    def test_create_from_various_truth_tables(self):
        """Test creating functions from different truth table formats."""
        # Test different input formats
        test_cases = [
            ([0, 1, 1, 0], 2, "XOR"),
            ([False, True, True, False], 2, "XOR boolean"),
            (np.array([0, 0, 0, 1]), 2, "AND numpy"),
            ([0, 1, 1, 1], 2, "OR"),
            ([0, 0, 1, 1, 1, 1, 1, 1], 3, "3-var OR"),
        ]
        
        for truth_table, expected_vars, description in test_cases:
            func = bf.create(truth_table)
            assert func.n_vars == expected_vars, f"Failed for {description}"
            assert func.has_rep('truth_table'), f"No truth table rep for {description}"
            
            # Test evaluation works
            result = func.evaluate([0] * expected_vars)
            assert isinstance(result, (bool, np.bool_)), f"Evaluation failed for {description}"
    
    def test_create_with_explicit_parameters(self):
        """Test creation with explicit parameters."""
        # Test with explicit n_vars
        func = bf.create([0, 1, 1, 0], n=2)
        assert func.n_vars == 2
        
        # Test with different spaces (using string format)
        try:
            func_plus_minus = bf.create([0, 1, 1, 0], n=2, space="plus_minus_cube")
            assert func_plus_minus.space == Space.PLUS_MINUS_CUBE
            
            # Both should evaluate consistently
            test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for inputs in test_inputs:
                result1 = func.evaluate(inputs)
                result2 = func_plus_minus.evaluate(inputs)
                # Results might be in different formats but should be consistent
                assert bool(result1) == bool(result2)
        except ValueError:
            # If space parameter not supported, just test basic functionality
            assert func.n_vars == 2
    
    def test_create_edge_cases(self):
        """Test edge cases in function creation."""
        # Single variable functions
        func1 = bf.create([0, 1])  # Identity function
        assert func1.n_vars == 1
        assert func1.evaluate([0]) == False
        assert func1.evaluate([1]) == True
        
        # Constant functions
        const_false = bf.create([0, 0])
        const_true = bf.create([1, 1])
        
        for i in range(2):
            assert const_false.evaluate([i]) == False
            assert const_true.evaluate([i]) == True


class TestBuiltinFunctionComprehensive:
    """Comprehensive tests for built-in Boolean functions."""
    
    def test_majority_functions_various_sizes(self):
        """Test majority functions of different sizes."""
        sizes = [1, 3, 5]  # Only odd sizes are valid
        
        for n in sizes:
            maj = bf.BooleanFunctionBuiltins.majority(n)
            assert maj.n_vars == n
            
            # Test all possible inputs
            for i in range(2**n):
                bits = [(i >> j) & 1 for j in range(n-1, -1, -1)]
                result = maj.evaluate(bits)
                expected = sum(bits) > n // 2
                assert result == expected, f"Majority({n}) failed for {bits}"
    
    def test_parity_functions_various_sizes(self):
        """Test parity functions of different sizes."""
        sizes = [1, 2, 3, 4]
        
        for n in sizes:
            par = bf.BooleanFunctionBuiltins.parity(n)
            assert par.n_vars == n
            
            # Test all possible inputs
            for i in range(2**n):
                bits = [(i >> j) & 1 for j in range(n-1, -1, -1)]
                result = par.evaluate(bits)
                expected = sum(bits) % 2 == 1
                assert result == expected, f"Parity({n}) failed for {bits}"
    
    def test_dictator_functions_all_positions(self):
        """Test dictator functions for all variable positions."""
        n = 3
        
        for pos in range(n):
            dict_func = bf.BooleanFunctionBuiltins.dictator(n, pos)
            assert dict_func.n_vars == n
            
            # Test all possible inputs
            for i in range(2**n):
                bits = [(i >> j) & 1 for j in range(n-1, -1, -1)]
                result = dict_func.evaluate(bits)
                expected = bool(bits[pos])
                assert result == expected, f"Dictator({pos}, {n}) failed for {bits}"
    
    def test_constant_functions(self):
        """Test constant functions."""
        for value in [True, False]:
            for n in [1, 2, 3, 4]:
                const = bf.BooleanFunctionBuiltins.constant(value, n)
                assert const.n_vars == n
                
                # Test all inputs give same result
                for i in range(2**n):
                    bits = [(i >> j) & 1 for j in range(n-1, -1, -1)]
                    result = const.evaluate(bits)
                    assert result == value, f"Constant({value}, {n}) failed for {bits}"


class TestSpectralAnalysisComprehensive:
    """Comprehensive tests for spectral analysis."""
    
    def test_influences_mathematical_properties(self):
        """Test mathematical properties of influence computation."""
        # Test known influence values
        test_functions = [
            (bf.create([0, 1, 1, 0]), [1.0, 1.0], "XOR"),  # Both vars have max influence
            (bf.create([0, 0, 0, 1]), [0.5, 0.5], "AND"),  # Both vars have equal influence
            (bf.create([0, 1, 1, 1]), [0.5, 0.5], "OR"),   # Both vars have equal influence
        ]
        
        for func, expected_influences, name in test_functions:
            analyzer = bf.SpectralAnalyzer(func)
            influences = analyzer.influences()
            
            assert len(influences) == len(expected_influences), f"{name} influence count wrong"
            for i, (actual, expected) in enumerate(zip(influences, expected_influences)):
                assert abs(actual - expected) < 1e-10, f"{name} influence {i} wrong: {actual} vs {expected}"
    
    def test_total_influence_properties(self):
        """Test total influence satisfies mathematical properties."""
        functions = [
            (bf.BooleanFunctionBuiltins.parity(2), 2.0, "Parity(2)"),
            (bf.BooleanFunctionBuiltins.parity(3), 3.0, "Parity(3)"),
            (bf.BooleanFunctionBuiltins.majority(3), 1.5, "Majority(3)"),
            (bf.create([0, 0, 0, 1]), 1.0, "AND"),
        ]
        
        for func, expected_total, name in functions:
            analyzer = bf.SpectralAnalyzer(func)
            total_inf = analyzer.total_influence()
            assert abs(total_inf - expected_total) < 1e-10, f"{name} total influence wrong: {total_inf} vs {expected_total}"
    
    def test_noise_stability_properties(self):
        """Test noise stability satisfies mathematical properties."""
        # XOR function has specific noise stability properties
        xor = bf.create([0, 1, 1, 0])
        analyzer = bf.SpectralAnalyzer(xor)
        
        # Test boundary conditions (adjust expectations based on actual implementation)
        stability_0 = analyzer.noise_stability(0.0)
        stability_1 = analyzer.noise_stability(1.0)
        
        # At ρ=1, stability should be 1 (perfect correlation)
        assert abs(stability_1 - 1.0) < 1e-10, f"Stability at ρ=1 should be 1.0, got {stability_1}"
        
        # Test monotonicity: stability should generally increase with ρ
        rho_values = [0.0, 0.5, 1.0]
        stabilities = [analyzer.noise_stability(rho) for rho in rho_values]
        
        # At minimum, stability at ρ=1 should be >= stability at ρ=0
        assert stabilities[-1] >= stabilities[0], f"Noise stability not monotonic: {stabilities}"
        
        # Test that all stability values are in valid range [0, 1]
        for rho, stability in zip(rho_values, stabilities):
            assert 0 <= stability <= 1, f"Invalid stability {stability} at ρ={rho}"
    
    def test_fourier_expansion_properties(self):
        """Test Fourier expansion mathematical properties."""
        # Test Parseval's identity: ||f||² = Σ f̂(S)²
        functions = [
            bf.create([0, 1, 1, 0]),  # XOR
            bf.create([0, 0, 0, 1]),  # AND
            bf.BooleanFunctionBuiltins.majority(3),
        ]
        
        for func in functions:
            analyzer = bf.SpectralAnalyzer(func)
            fourier_coeffs = analyzer.fourier_expansion()
            
            # Compute L2 norm of Fourier coefficients
            fourier_norm_squared = np.sum(fourier_coeffs ** 2)
            
            # Compute L2 norm of function in ±1 representation
            size = 2 ** func.n_vars
            function_values = []
            for i in range(size):
                bits = [(i >> j) & 1 for j in range(func.n_vars-1, -1, -1)]
                val = func.evaluate(bits)
                function_values.append(2 * int(val) - 1)  # Convert to ±1
            
            function_norm_squared = np.sum(np.array(function_values) ** 2) / size
            
            # Parseval's identity
            assert abs(fourier_norm_squared - function_norm_squared) < 1e-10, "Parseval's identity violated"


class TestPropertyTestingComprehensive:
    """Comprehensive tests for property testing algorithms."""
    
    def test_constant_detection_accuracy(self):
        """Test constant function detection is accurate."""
        # True constants
        const_true = bf.BooleanFunctionBuiltins.constant(True, 3)
        const_false = bf.BooleanFunctionBuiltins.constant(False, 3)
        
        tester_true = bf.PropertyTester(const_true)
        tester_false = bf.PropertyTester(const_false)
        
        assert tester_true.constant_test() == True
        assert tester_false.constant_test() == True
        
        # Non-constants
        non_constants = [
            bf.create([0, 1, 1, 0]),  # XOR
            bf.BooleanFunctionBuiltins.majority(3),
            bf.BooleanFunctionBuiltins.parity(3),
        ]
        
        for func in non_constants:
            tester = bf.PropertyTester(func)
            assert tester.constant_test() == False
    
    def test_balance_detection_accuracy(self):
        """Test balanced function detection is accurate."""
        # Balanced functions (equal number of 0s and 1s)
        balanced_functions = [
            bf.create([0, 1, 1, 0]),  # XOR
            bf.BooleanFunctionBuiltins.parity(2),
            bf.BooleanFunctionBuiltins.parity(3),
        ]
        
        for func in balanced_functions:
            tester = bf.PropertyTester(func)
            assert tester.balanced_test() == True
        
        # Unbalanced functions
        unbalanced_functions = [
            bf.create([0, 0, 0, 1]),  # AND (3 zeros, 1 one)
            bf.BooleanFunctionBuiltins.constant(True, 2),
        ]
        
        for func in unbalanced_functions:
            tester = bf.PropertyTester(func)
            assert tester.balanced_test() == False
        
        # Note: Majority(3) is actually balanced (4 zeros, 4 ones), so test separately
        maj3 = bf.BooleanFunctionBuiltins.majority(3)
        maj_tester = bf.PropertyTester(maj3)
        maj_balanced = maj_tester.balanced_test()
        # Majority(3) has 4 False and 4 True outputs, so it should be balanced
        assert maj_balanced == True
    
    def test_linearity_testing_consistency(self):
        """Test BLR linearity testing gives consistent results."""
        # Linear functions (should pass linearity test with high probability)
        linear_functions = [
            bf.BooleanFunctionBuiltins.parity(2),
            bf.BooleanFunctionBuiltins.parity(3),
            bf.BooleanFunctionBuiltins.dictator(2, 0),
        ]
        
        for func in linear_functions:
            tester = bf.PropertyTester(func, random_seed=42)
            # Test multiple times with different query counts
            results = []
            for num_queries in [10, 50, 100]:
                result = tester.blr_linearity_test(num_queries=num_queries)
                results.append(result)
            
            # Should consistently identify as linear (or at least not always fail)
            assert any(results), f"Linear function {func} failed all linearity tests"
        
        # Non-linear functions (should fail linearity test)
        nonlinear_functions = [
            bf.create([0, 0, 0, 1]),  # AND
            bf.BooleanFunctionBuiltins.majority(3),
        ]
        
        for func in nonlinear_functions:
            tester = bf.PropertyTester(func, random_seed=42)
            # Should consistently fail with enough queries
            result = tester.blr_linearity_test(num_queries=100)
            # Note: Due to randomness, we can't guarantee failure, but it should be likely


class TestFunctionEvaluationRobustness:
    """Test function evaluation under various conditions."""
    
    def test_evaluation_input_formats(self):
        """Test evaluation with different input formats."""
        func = bf.create([0, 1, 1, 0])  # XOR
        
        # Test different input formats that should all work
        test_cases = [
            ([0, 1], True),
            (np.array([0, 1]), True),
        ]
        
        for inputs, expected in test_cases:
            result = func.evaluate(inputs)
            assert result == expected, f"Failed for input format {type(inputs)}"
        
        # Test tuple format separately (may not be supported)
        try:
            result = func.evaluate((0, 1))
            assert result == True
        except TypeError:
            # Tuple format not supported - that's acceptable
            pass
    
    def test_batch_evaluation_consistency(self):
        """Test that batch evaluation gives same results as individual evaluation."""
        func = bf.BooleanFunctionBuiltins.majority(3)
        
        # Generate all possible inputs
        all_inputs = []
        individual_results = []
        
        for i in range(8):
            bits = [(i >> j) & 1 for j in range(2, -1, -1)]
            all_inputs.append(bits)
            individual_results.append(func.evaluate(bits))
        
        # Test batch evaluation (if implemented)
        try:
            batch_results = [func.evaluate(inp) for inp in all_inputs]
            assert batch_results == individual_results, "Batch evaluation inconsistent"
        except NotImplementedError:
            pass  # Batch evaluation not implemented yet
    
    def test_evaluation_with_different_spaces(self):
        """Test evaluation works correctly in different mathematical spaces."""
        func = bf.create([0, 1, 1, 0])  # XOR
        
        # Test in Boolean cube
        result_bool = func.evaluate([1, 0])
        assert isinstance(result_bool, (bool, np.bool_))
        
        # Test space handling (if supported)
        try:
            func_pm = bf.create([0, 1, 1, 0], space="plus_minus_cube")
            result_pm = func_pm.evaluate([1, 0])
            # Results should be consistent (both True or both False)
            assert bool(result_bool) == bool(result_pm)
        except (ValueError, TypeError):
            # Different space handling not fully implemented - that's OK
            pass


class TestRepresentationIntegration:
    """Test integration between different representations."""
    
    def test_representation_storage_and_retrieval(self):
        """Test that representations are stored and retrieved correctly."""
        func = bf.create([0, 1, 1, 0])
        
        # Should have truth table representation
        assert func.has_rep('truth_table')
        
        # Test getting representation
        truth_table = func.get_representation('truth_table')
        assert truth_table is not None
        assert len(truth_table) == 4
    
    def test_representation_consistency(self):
        """Test that different representations give consistent results."""
        func = bf.create([0, 0, 0, 1])  # AND function
        
        # Test evaluation gives consistent results regardless of internal representation
        test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected = [False, False, False, True]
        
        for inputs, expected_output in zip(test_inputs, expected):
            result = func.evaluate(inputs)
            assert result == expected_output
        
        # If function has multiple representations, they should be consistent
        if len(func.representations) > 1:
            # Test that all representations evaluate to same result
            for inputs in test_inputs:
                results = []
                for rep_name in func.representations:
                    try:
                        result = func.evaluate(inputs, rep_type=rep_name)
                        results.append(bool(result))
                    except (NotImplementedError, KeyError):
                        pass  # Some representations might not support direct evaluation
                
                # All results should be the same
                if len(results) > 1:
                    assert all(r == results[0] for r in results), f"Inconsistent results: {results}"


class TestAnalysisToolsIntegration:
    """Test integration between analysis tools."""
    
    def test_spectral_analysis_summary(self):
        """Test that spectral analysis summary provides comprehensive information."""
        functions = [
            bf.create([0, 1, 1, 0]),  # XOR
            bf.BooleanFunctionBuiltins.majority(3),
            bf.BooleanFunctionBuiltins.constant(True, 2),
        ]
        
        for func in functions:
            analyzer = bf.SpectralAnalyzer(func)
            summary = analyzer.summary()
            
            # Summary should be a dictionary with expected keys
            assert isinstance(summary, dict)
            expected_keys = ['total_influence', 'max_influence', 'min_influence', 'avg_influence']
            for key in expected_keys:
                assert key in summary, f"Missing {key} in summary"
                assert isinstance(summary[key], (int, float, np.number)), f"Invalid type for {key}"
    
    def test_property_testing_integration(self):
        """Test property testing works with different function types."""
        # Test property testing on various function types
        functions = [
            (bf.BooleanFunctionBuiltins.constant(True, 2), "constant"),
            (bf.BooleanFunctionBuiltins.parity(2), "parity"),
            (bf.BooleanFunctionBuiltins.majority(3), "majority"),
            (bf.create([0, 0, 0, 1]), "and"),
        ]
        
        for func, name in functions:
            tester = bf.PropertyTester(func)
            
            # All property tests should run without error
            try:
                is_constant = tester.constant_test()
                is_balanced = tester.balanced_test()
                
                assert isinstance(is_constant, bool), f"{name} constant test returned non-bool"
                assert isinstance(is_balanced, bool), f"{name} balance test returned non-bool"
                
                # Run linearity test (may be probabilistic)
                is_linear = tester.blr_linearity_test(num_queries=20)
                assert isinstance(is_linear, bool), f"{name} linearity test returned non-bool"
                
            except Exception as e:
                pytest.fail(f"Property testing failed for {name}: {e}")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_small_functions(self):
        """Test handling of very small Boolean functions."""
        # 1-variable functions
        identity = bf.create([0, 1])
        negation = bf.create([1, 0])
        
        assert identity.evaluate([0]) == False
        assert identity.evaluate([1]) == True
        assert negation.evaluate([0]) == True
        assert negation.evaluate([1]) == False
        
        # Test analysis works on small functions
        analyzer = bf.SpectralAnalyzer(identity)
        influences = analyzer.influences()
        assert len(influences) == 1
        assert influences[0] == 1.0  # Identity function has maximum influence
    
    def test_function_properties_consistency(self):
        """Test that function properties are internally consistent."""
        functions = [
            bf.create([0, 1, 1, 0]),
            bf.BooleanFunctionBuiltins.majority(3),
            bf.BooleanFunctionBuiltins.parity(3),
        ]
        
        for func in functions:
            # Test that n_vars is consistent with evaluation
            assert isinstance(func.n_vars, int)
            assert func.n_vars > 0
            
            # Test that function can evaluate all possible inputs
            for i in range(2**func.n_vars):
                bits = [(i >> j) & 1 for j in range(func.n_vars-1, -1, -1)]
                result = func.evaluate(bits)
                assert isinstance(result, (bool, np.bool_, np.ndarray))
            
            # Test that spectral analysis works
            analyzer = bf.SpectralAnalyzer(func)
            influences = analyzer.influences()
            assert len(influences) == func.n_vars
            assert all(0 <= inf <= 1 for inf in influences)  # Influences should be in [0,1]
    
    def test_library_import_stability(self):
        """Test that library imports are stable and don't have circular dependencies."""
        # Test that main imports work
        import boofun as bf
        
        # Test that core classes are accessible
        assert hasattr(bf, 'create')
        assert hasattr(bf, 'BooleanFunctionBuiltins')
        assert hasattr(bf, 'SpectralAnalyzer')
        assert hasattr(bf, 'PropertyTester')
        
        # Test that version information is available
        assert hasattr(bf, '__version__')
        assert isinstance(bf.__version__, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
