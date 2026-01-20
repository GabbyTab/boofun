"""
Comprehensive tests for advanced BooFun features.

Tests the new advanced features including conversion graph, batch processing,
GPU acceleration, Numba optimizations, ANF representation, and testing framework.
"""

import os
import sys

import numpy as np
import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boofun as bf
from boofun.core.batch_processing import (
    BatchProcessorManager,
    get_batch_processor_stats,
)
from boofun.core.conversion_graph import (
    ConversionCost,
    find_conversion_path,
    get_conversion_options,
)
from boofun.core.gpu_acceleration import get_gpu_info, is_gpu_available, should_use_gpu
from boofun.core.numba_optimizations import get_numba_stats, is_numba_available


class TestANFRepresentation:
    """Test ANF (Algebraic Normal Form) representation."""

    def test_anf_creation(self):
        """Test ANF representation creation."""
        # Create XOR function
        xor = bf.create([False, True, True, False])

        # Get ANF representation
        anf_data = xor.get_representation("anf")

        assert isinstance(anf_data, dict)
        assert len(anf_data) > 0

        # XOR should have linear terms
        linear_terms = [mono for mono in anf_data.keys() if len(mono) == 1 and anf_data[mono] != 0]
        assert len(linear_terms) == 2  # x0 and x1

    def test_anf_evaluation(self):
        """Test ANF evaluation."""
        # Create simple function: f(x0, x1) = x0
        identity = bf.create([False, False, True, True])
        anf_data = identity.get_representation("anf")

        # Test evaluation
        from boofun.core.representations.anf_form import ANFRepresentation

        anf_repr = ANFRepresentation()

        # Test all inputs
        for i in range(4):
            expected = identity.evaluate(i)
            result = anf_repr.evaluate(i, anf_data, bf.Space.BOOLEAN_CUBE, 2)
            assert result == expected

    def test_anf_properties(self):
        """Test ANF analysis methods."""
        # Create majority function
        majority = bf.create([False, False, False, True, False, True, True, True])  # 3-var majority
        anf_data = majority.get_representation("anf")

        from boofun.core.representations.anf_form import ANFRepresentation

        anf_repr = ANFRepresentation()

        # Test degree computation
        degree = anf_repr._get_degree(anf_data)
        assert degree >= 2  # Majority has degree 2 or 3

        # Test linearity check
        assert not anf_repr.is_linear(anf_data)  # Majority is not linear

        # Test quadratic check (may be true for majority)
        is_quad = anf_repr.is_quadratic(anf_data)
        assert isinstance(is_quad, bool)


class TestConversionGraph:
    """Test conversion graph system."""

    def test_conversion_path_finding(self):
        """Test finding conversion paths."""
        path = find_conversion_path("truth_table", "anf", n_vars=3)

        assert path is not None
        assert path.source == "truth_table"
        assert path.target == "anf"
        assert len(path.edges) >= 1
        assert path.total_cost.total_cost > 0

    def test_conversion_options(self):
        """Test getting conversion options."""
        options = get_conversion_options("truth_table")

        assert isinstance(options, dict)
        assert "anf" in options
        assert "fourier_expansion" in options

        # Check path quality
        anf_path = options["anf"]
        assert anf_path.source == "truth_table"
        assert anf_path.target == "anf"

    def test_conversion_cost(self):
        """Test conversion cost calculations."""
        cost1 = ConversionCost(10, 5, 0.0, True)
        cost2 = ConversionCost(20, 10, 0.1, True)

        combined = cost1 + cost2
        assert combined.time_complexity == 30
        assert combined.space_complexity == 10  # Max of the two
        assert combined.accuracy_loss == 0.1
        assert combined.is_exact == True

    def test_boolean_function_conversion_methods(self):
        """Test BooleanFunction conversion methods."""
        xor = bf.create([False, True, True, False])

        # Test conversion options
        options = xor.get_conversion_options()
        assert isinstance(options, dict)
        assert len(options) > 0

        # Test cost estimation
        cost = xor.estimate_conversion_cost("anf")
        assert cost is not None


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_batch_processor_manager(self):
        """Test batch processor manager."""
        manager = BatchProcessorManager()
        stats = manager.get_processor_stats()

        assert "available_processors" in stats
        assert "numba_available" in stats
        assert "cpu_count" in stats
        assert "thresholds" in stats

    def test_truth_table_batch_processing(self):
        """Test batch processing for truth tables."""
        # Create XOR function
        xor = bf.create([False, True, True, False])

        # Large batch of inputs
        batch_inputs = np.arange(100) % 4  # Repeat 0,1,2,3

        # Process batch
        results = xor.evaluate(batch_inputs)

        assert len(results) == 100
        assert isinstance(results, np.ndarray)

        # Check correctness
        expected = np.array([xor.evaluate(i % 4) for i in range(100)])
        np.testing.assert_array_equal(results, expected)

    def test_batch_processing_stats(self):
        """Test batch processing statistics."""
        stats = get_batch_processor_stats()

        assert isinstance(stats, dict)
        assert "available_processors" in stats
        assert "numba_available" in stats

    def test_automatic_batch_selection(self):
        """Test automatic batch processing selection."""
        # Create function
        majority = bf.create([False, False, False, True, False, True, True, True])

        # Small batch (should use standard evaluation)
        small_batch = np.array([0, 1, 2, 3])
        small_results = majority.evaluate(small_batch)

        # Large batch (should use batch processing)
        large_batch = np.arange(1000) % 8
        large_results = majority.evaluate(large_batch)

        assert len(small_results) == 4
        assert len(large_results) == 1000


class TestGPUAcceleration:
    """Test GPU acceleration (if available)."""

    def test_gpu_availability_check(self):
        """Test GPU availability detection."""
        available = is_gpu_available()
        assert isinstance(available, bool)

        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "gpu_available" in info
        assert "active_backend" in info

    def test_gpu_usage_heuristics(self):
        """Test GPU usage decision heuristics."""
        # Small data should not use GPU
        small_decision = should_use_gpu("truth_table", 100, 4)
        assert isinstance(small_decision, bool)

        # Large data might use GPU (if available)
        large_decision = should_use_gpu("truth_table", 100000, 10)
        assert isinstance(large_decision, bool)

    @pytest.mark.skipif(not is_gpu_available(), reason="GPU not available")
    def test_gpu_acceleration_integration(self):
        """Test GPU acceleration integration (only if GPU available)."""
        # Create large function
        large_func = bf.create([i % 2 for i in range(256)])  # 8-variable function

        # Large batch evaluation (should trigger GPU if beneficial)
        large_batch = np.arange(10000) % 256
        results = large_func.evaluate(large_batch)

        assert len(results) == 10000
        assert isinstance(results, np.ndarray)


class TestNumbaOptimizations:
    """Test Numba JIT optimizations."""

    def test_numba_availability(self):
        """Test Numba availability detection."""
        available = is_numba_available()
        assert isinstance(available, bool)

        stats = get_numba_stats()
        assert isinstance(stats, dict)
        assert "numba_available" in stats

    @pytest.mark.skipif(not is_numba_available(), reason="Numba not available")
    def test_numba_optimized_analysis(self):
        """Test Numba-optimized analysis functions."""
        # Create function for analysis
        xor = bf.create([False, True, True, False])

        # Test optimized influences computation
        analyzer = bf.SpectralAnalyzer(xor)
        influences = analyzer.influences()

        assert len(influences) == 2
        assert np.allclose(influences, [1.0, 1.0])  # XOR has maximum influence for both variables

        # Test optimized noise stability
        stability = analyzer.noise_stability(0.9)
        assert isinstance(stability, float)
        assert 0 <= stability <= 1


class TestTestingFramework:
    """Test the testing framework itself."""

    def test_boolean_function_validator(self):
        """Test BooleanFunction validator."""
        xor = bf.create([False, True, True, False])

        # Quick validation
        is_valid = bf.quick_validate(xor, verbose=False)
        assert isinstance(is_valid, bool)

        # Detailed validation
        validator = bf.BooleanFunctionValidator(xor)
        results = validator.validate_all()

        assert isinstance(results, dict)
        assert "overall_status" in results
        assert "basic_properties" in results

    def validate_representation_tester(self):
        """Test representation tester."""
        from boofun.core.representations.truth_table import TruthTableRepresentation

        repr_tester = bf.validate_representation(TruthTableRepresentation())

        assert isinstance(repr_tester, dict)
        assert "overall_passed" in repr_tester
        assert "interface_compliance" in repr_tester


class TestErrorIntegration:
    """Test error model integration."""

    def test_error_model_creation(self):
        """Test creating functions with different error models."""
        # Exact error model
        exact_func = bf.create([False, True, True, False], error_model=bf.ExactErrorModel())
        assert exact_func.error_model.get_confidence(True) == 1.0

        # PAC error model
        pac_func = bf.create([False, True, True, False], error_model=bf.PACErrorModel(0.1, 0.1))
        confidence = pac_func.error_model.get_confidence(True)
        assert 0 < confidence <= 1

        # Noise error model
        noise_func = bf.create([False, True, True, False], error_model=bf.NoiseErrorModel(0.01))
        assert noise_func.error_model.noise_rate == 0.01

    def test_error_model_evaluation(self):
        """Test error model effects on evaluation."""
        # Create function with noise
        noisy_func = bf.create([False, True, True, False], error_model=bf.NoiseErrorModel(0.1))

        # Multiple evaluations might give different results due to noise
        results = [noisy_func.evaluate(1) for _ in range(10)]

        # Extract boolean values from results (may be dicts due to error model)
        bool_results = []
        for result in results:
            if isinstance(result, dict) and "value" in result:
                bool_results.append(bool(result["value"]))
            else:
                bool_results.append(bool(result))

        # Most should be correct (True for XOR(1) = True)
        correct_count = sum(bool_results)
        assert correct_count >= 7  # Allow for some noise


class TestSpaceHandling:
    """Test consistent space handling."""

    def test_space_translation(self):
        """Test space translation utilities."""
        # Test boolean to plus-minus conversion
        bool_val = np.array([0, 1])
        pm_val = bf.Space.translate(bool_val, bf.Space.BOOLEAN_CUBE, bf.Space.PLUS_MINUS_CUBE)
        expected = np.array([-1, 1])
        np.testing.assert_array_equal(pm_val, expected)

        # Test plus-minus to boolean conversion
        pm_val = np.array([-1, 1])
        bool_val = bf.Space.translate(pm_val, bf.Space.PLUS_MINUS_CUBE, bf.Space.BOOLEAN_CUBE)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(bool_val, expected)

    def test_function_space_consistency(self):
        """Test that functions handle different spaces consistently."""
        # Create function in boolean space
        bool_func = bf.create([False, True, True, False], space="boolean_cube")

        # Create same function in plus-minus space
        pm_func = bf.create([False, True, True, False], space="plus_minus_cube")

        # Both should give same logical results
        bool_result = bool_func.evaluate([0, 1])
        pm_result = pm_func.evaluate([-1, 1])

        assert bool_result == pm_result


class TestAdapters:
    """Test adapter system."""

    def test_callable_adapter(self):
        """Test adapting Python callables."""
        # Define XOR as lambda
        xor_lambda = lambda x: x[0] ^ x[1]

        # Adapt to BooleanFunction
        adapted = bf.adapt_callable(xor_lambda, n_vars=2)

        # Test evaluation
        assert adapted.evaluate([0, 0]) == False
        assert adapted.evaluate([0, 1]) == True
        assert adapted.evaluate([1, 0]) == True
        assert adapted.evaluate([1, 1]) == False

    def test_legacy_adapter(self):
        """Test legacy function adapter."""
        from boofun.core.adapters import LegacyAdapter

        # Mock legacy function
        class LegacyFunction:
            def legacy_evaluate(self, inputs):
                return inputs[0] and inputs[1]  # AND function

        legacy_func = LegacyFunction()

        # Adapt using legacy adapter
        adapter = LegacyAdapter(evaluation_method="legacy_evaluate")
        adapted = adapter.adapt(legacy_func)

        # Test evaluation
        assert adapted.evaluate([0, 0]) == False
        assert adapted.evaluate([0, 1]) == False
        assert adapted.evaluate([1, 0]) == False
        assert adapted.evaluate([1, 1]) == True


class TestQuantumIntegration:
    """Test quantum module integration."""

    def test_quantum_function_creation(self):
        """Test creating quantum Boolean functions."""
        # Create classical function
        xor = bf.create([False, True, True, False])

        # Create quantum version
        quantum_xor = bf.create_quantum_boolean_function(xor)

        assert quantum_xor.function == xor
        assert quantum_xor.n_vars == 2

    def test_quantum_analysis(self):
        """Test quantum analysis methods."""
        xor = bf.create([False, True, True, False])
        quantum_xor = bf.create_quantum_boolean_function(xor)

        # Test quantum Fourier analysis
        fourier_results = quantum_xor.quantum_fourier_analysis()

        assert isinstance(fourier_results, dict)
        assert "method" in fourier_results
        assert "fourier_coefficients" in fourier_results

    def test_quantum_property_testing(self):
        """Test quantum property testing."""
        xor = bf.create([False, True, True, False])
        quantum_xor = bf.create_quantum_boolean_function(xor)

        # Test quantum linearity test
        linearity_result = quantum_xor.quantum_property_testing("linearity")

        assert isinstance(linearity_result, dict)
        assert "property" in linearity_result
        assert "is_linear" in linearity_result


# Integration tests
class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_complete_workflow(self):
        """Test complete Boolean function analysis workflow."""
        # Create function
        majority = bf.create([False, False, False, True, False, True, True, True])

        # Add a second representation for validation consistency check
        _ = majority.get_representation("anf")

        # Validate function
        assert bf.quick_validate(majority)

        # Get multiple representations
        truth_table = majority.get_representation("truth_table")
        anf_data = majority.get_representation("anf")
        fourier_coeffs = majority.get_representation("fourier_expansion")

        assert len(truth_table) == 8
        assert isinstance(anf_data, dict)
        assert len(fourier_coeffs) >= 4  # Fourier coeffs may be sparse or dense

        # Analyze properties
        analyzer = bf.SpectralAnalyzer(majority)
        influences = analyzer.influences()
        noise_stability = analyzer.noise_stability(0.9)

        assert len(influences) == 3
        assert 0 <= noise_stability <= 1

        # Test property detection
        tester = bf.PropertyTester(majority, random_seed=42)
        properties = tester.run_all_tests()

        assert isinstance(properties, dict)
        assert "monotone" in properties
        assert properties["monotone"] == True  # Majority is monotone

    def test_performance_features(self):
        """Test performance features work together."""
        # Create larger function for performance testing
        large_func = bf.create([i % 2 == 0 for i in range(64)])  # 6-variable function

        # Test batch processing
        batch_inputs = np.arange(1000) % 64
        batch_results = large_func.evaluate(batch_inputs)
        assert len(batch_results) == 1000

        # Test conversion graph
        options = large_func.get_conversion_options()
        assert len(options) > 0

        # Test analysis with optimizations
        analyzer = bf.SpectralAnalyzer(large_func)
        influences = analyzer.influences()
        assert len(influences) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
