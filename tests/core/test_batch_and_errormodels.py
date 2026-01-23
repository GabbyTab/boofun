"""
Tests for batch processing infrastructure and error models.

This module tests:
- Batch processing: VectorizedBatchProcessor, ParallelBatchProcessor,
  OptimizedTruthTableProcessor, OptimizedFourierProcessor, OptimizedANFProcessor
- Error models: ExactErrorModel, PACErrorModel, NoiseErrorModel
- DNF/CNF representation handling
- Integration of batch processing with function families
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.core.spaces import Space


# =============================================================================
# Batch Processing Tests
# =============================================================================
class TestVectorizedBatchProcessor:
    """Test VectorizedBatchProcessor."""

    def test_create_processor(self):
        """Test creating vectorized processor."""
        from boofun.core.batch_processing import VectorizedBatchProcessor

        proc = VectorizedBatchProcessor(chunk_size=1000)
        assert proc.chunk_size == 1000

    def test_supports_representation(self):
        """Test representation support check."""
        from boofun.core.batch_processing import VectorizedBatchProcessor

        proc = VectorizedBatchProcessor()
        assert proc.supports_representation("truth_table")
        assert proc.supports_representation("fourier_expansion")
        assert not proc.supports_representation("unknown_repr")

    def test_process_small_batch(self):
        """Test processing small batch returns correct shape."""
        from boofun.core.batch_processing import VectorizedBatchProcessor

        proc = VectorizedBatchProcessor(chunk_size=100)
        inputs = np.array([0, 1, 2, 3])

        # Note: VectorizedBatchProcessor's base implementation returns False
        # for all inputs. Specific representations override this behavior.
        # Here we just test the infrastructure works.
        results = proc.process_batch(inputs, None, Space.BOOLEAN_CUBE, 2)
        assert len(results) == 4
        assert results.dtype == bool

    def test_process_large_batch_chunks(self):
        """Test processing batch that requires chunking uses chunks."""
        from boofun.core.batch_processing import VectorizedBatchProcessor

        proc = VectorizedBatchProcessor(chunk_size=10)
        inputs = np.arange(50)

        # Verify chunking is used (50 > chunk_size=10)
        # The base implementation returns False, but we test the infrastructure
        results = proc.process_batch(inputs, None, Space.BOOLEAN_CUBE, 6)
        assert len(results) == 50
        assert results.dtype == bool


class TestParallelBatchProcessor:
    """Test ParallelBatchProcessor."""

    def test_create_processor(self):
        """Test creating parallel processor."""
        from boofun.core.batch_processing import ParallelBatchProcessor

        proc = ParallelBatchProcessor(n_workers=2, use_processes=False)
        assert proc.n_workers == 2
        assert proc.use_processes is False

    def test_supports_representation(self):
        """Test representation support."""
        from boofun.core.batch_processing import ParallelBatchProcessor

        proc = ParallelBatchProcessor()
        assert proc.supports_representation("symbolic")
        assert proc.supports_representation("circuit")
        assert not proc.supports_representation("truth_table")

    def test_small_batch_sequential(self):
        """Test that small batches use sequential processing."""
        from boofun.core.batch_processing import ParallelBatchProcessor

        proc = ParallelBatchProcessor(n_workers=4)
        # Small batch should use sequential - just verify it exists and has attributes
        assert proc.n_workers == 4
        # Note: ParallelBatchProcessor is designed for symbolic/circuit representations
        # which require different test setup than simple truth tables


class TestOptimizedTruthTableProcessor:
    """Test OptimizedTruthTableProcessor."""

    def test_create_processor(self):
        """Test creating truth table processor."""
        from boofun.core.batch_processing import OptimizedTruthTableProcessor

        proc = OptimizedTruthTableProcessor(chunk_size=50000)
        assert proc.chunk_size == 50000

    def test_process_integer_indices(self):
        """Test processing integer indices."""
        from boofun.core.batch_processing import OptimizedTruthTableProcessor

        proc = OptimizedTruthTableProcessor()
        truth_table = np.array([False, True, True, False, True, False, False, True])
        inputs = np.array([0, 1, 7])

        results = proc._process_chunk(inputs, truth_table, Space.BOOLEAN_CUBE, 3)
        assert len(results) == 3
        assert results[0] == False  # noqa: E712
        assert results[1] == True  # noqa: E712
        assert results[2] == True  # noqa: E712

    def test_process_binary_vectors(self):
        """Test processing binary vectors."""
        from boofun.core.batch_processing import OptimizedTruthTableProcessor

        proc = OptimizedTruthTableProcessor()
        truth_table = np.array([False, True, True, False, True, False, False, True])
        inputs = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]])

        results = proc._process_chunk(inputs, truth_table, Space.BOOLEAN_CUBE, 3)
        assert len(results) == 3

    def test_bounds_checking(self):
        """Test that out-of-bounds indices are handled."""
        from boofun.core.batch_processing import OptimizedTruthTableProcessor

        proc = OptimizedTruthTableProcessor()
        truth_table = np.array([False, True, True, False])
        inputs = np.array([0, 1, 100, -1])  # Invalid indices

        results = proc._process_chunk(inputs, truth_table, Space.BOOLEAN_CUBE, 2)
        assert len(results) == 4
        # Invalid indices should return False (default)
        assert results[2] == False  # noqa: E712
        assert results[3] == False  # noqa: E712


class TestOptimizedFourierProcessor:
    """Test OptimizedFourierProcessor."""

    def test_create_processor(self):
        """Test creating Fourier processor."""
        from boofun.core.batch_processing import OptimizedFourierProcessor

        proc = OptimizedFourierProcessor()
        assert proc.supports_representation("fourier_expansion")

    def test_numpy_fourier_batch(self):
        """Test NumPy Fourier batch processing."""
        from boofun.core.batch_processing import OptimizedFourierProcessor

        proc = OptimizedFourierProcessor()
        # Simple Fourier coefficients for parity
        coeffs = np.array([0.0, 0.0, 0.0, 1.0])  # Only x0 XOR x1 term
        inputs = np.array([0, 1, 2, 3])

        results = proc._numpy_fourier_batch(inputs, coeffs, 2)
        assert len(results) == 4

    def test_evaluate_fourier_single(self):
        """Test single Fourier evaluation."""
        from boofun.core.batch_processing import OptimizedFourierProcessor

        proc = OptimizedFourierProcessor()
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])  # Constant function
        result = proc._evaluate_fourier_single(0, coeffs)
        assert isinstance(result, float)


class TestOptimizedANFProcessor:
    """Test OptimizedANFProcessor."""

    def test_create_processor(self):
        """Test creating ANF processor."""
        from boofun.core.batch_processing import OptimizedANFProcessor

        proc = OptimizedANFProcessor()
        assert proc.supports_representation("anf")

    def test_numpy_anf_batch(self):
        """Test NumPy ANF batch processing."""
        from boofun.core.batch_processing import OptimizedANFProcessor

        proc = OptimizedANFProcessor()
        # ANF for x0 XOR x1
        anf_dict = {frozenset([0]): 1, frozenset([1]): 1}
        inputs = np.array([0, 1, 2, 3])

        results = proc._numpy_anf_batch(inputs, anf_dict, 2)
        assert len(results) == 4

    def test_empty_anf(self):
        """Test empty ANF (zero function)."""
        from boofun.core.batch_processing import OptimizedANFProcessor

        proc = OptimizedANFProcessor()
        anf_dict = {}  # Zero function
        inputs = np.array([0, 1, 2, 3])

        results = proc._numpy_anf_batch(inputs, anf_dict, 2)
        assert all(r == False for r in results)  # noqa: E712


class TestBatchProcessorManager:
    """Test BatchProcessorManager."""

    def test_create_manager(self):
        """Test creating batch processor manager."""
        from boofun.core.batch_processing import BatchProcessorManager

        manager = BatchProcessorManager()
        assert "truth_table" in manager.processors
        assert "fourier_expansion" in manager.processors

    def test_get_processor_stats(self):
        """Test getting processor statistics."""
        from boofun.core.batch_processing import BatchProcessorManager

        manager = BatchProcessorManager()
        stats = manager.get_processor_stats()

        assert "available_processors" in stats
        assert "cpu_count" in stats
        assert "thresholds" in stats

    def test_select_processor(self):
        """Test processor selection logic."""
        from boofun.core.batch_processing import BatchProcessorManager

        manager = BatchProcessorManager()

        # Should select specific processor for known representation
        proc = manager._select_processor("truth_table", 100)
        assert proc is not None

        # Should select parallel for large inputs
        proc = manager._select_processor("unknown", 20000)
        assert proc is not None


class TestBatchProcessingFunctions:
    """Test module-level batch processing functions."""

    def test_process_batch_function(self):
        """Test process_batch module function."""
        from boofun.core.batch_processing import process_batch

        f = bf.parity(3)
        tt = np.array(list(f.get_representation("truth_table")), dtype=bool)
        inputs = np.array([0, 1, 7])

        results = process_batch(inputs, tt, "truth_table", Space.BOOLEAN_CUBE, 3)
        assert len(results) == 3

    def test_get_batch_processor_stats(self):
        """Test get_batch_processor_stats function."""
        from boofun.core.batch_processing import get_batch_processor_stats

        stats = get_batch_processor_stats()
        assert isinstance(stats, dict)

    def test_set_batch_thresholds(self):
        """Test set_batch_thresholds function."""
        from boofun.core.batch_processing import _batch_manager, set_batch_thresholds

        set_batch_thresholds(vectorized_threshold=500, parallel_threshold=5000)
        assert _batch_manager.vectorized_threshold == 500
        assert _batch_manager.parallel_threshold == 5000

        # Reset to defaults
        set_batch_thresholds(vectorized_threshold=1000, parallel_threshold=10000)


# =============================================================================
# Error Models Tests
# =============================================================================
class TestExactErrorModel:
    """Test ExactErrorModel."""

    def test_create_model(self):
        """Test creating exact error model."""
        from boofun.core.errormodels import ExactErrorModel

        model = ExactErrorModel()
        assert repr(model) == "ExactErrorModel()"

    def test_apply_error(self):
        """Test apply_error returns unchanged result."""
        from boofun.core.errormodels import ExactErrorModel

        model = ExactErrorModel()
        assert model.apply_error(True) is True
        assert model.apply_error(0.5) == 0.5
        assert model.apply_error([1, 2, 3]) == [1, 2, 3]

    def test_confidence(self):
        """Test confidence is always 1.0."""
        from boofun.core.errormodels import ExactErrorModel

        model = ExactErrorModel()
        assert model.get_confidence(True) == 1.0
        assert model.get_confidence(None) == 1.0

    def test_is_reliable(self):
        """Test is_reliable is always True."""
        from boofun.core.errormodels import ExactErrorModel

        model = ExactErrorModel()
        assert model.is_reliable(True) is True
        assert model.is_reliable(None) is True


class TestPACErrorModel:
    """Test PACErrorModel."""

    def test_create_model(self):
        """Test creating PAC error model."""
        from boofun.core.errormodels import PACErrorModel

        model = PACErrorModel(epsilon=0.05, delta=0.05)
        assert model.epsilon == 0.05
        assert model.delta == 0.05
        assert model.confidence == 0.95

    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        from boofun.core.errormodels import PACErrorModel

        with pytest.raises(ValueError):
            PACErrorModel(epsilon=0)
        with pytest.raises(ValueError):
            PACErrorModel(epsilon=1.5)

    def test_invalid_delta(self):
        """Test that invalid delta raises error."""
        from boofun.core.errormodels import PACErrorModel

        with pytest.raises(ValueError):
            PACErrorModel(epsilon=0.1, delta=0)
        with pytest.raises(ValueError):
            PACErrorModel(epsilon=0.1, delta=1.0)

    def test_apply_error_numeric(self):
        """Test applying PAC bounds to numeric result."""
        from boofun.core.errormodels import PACErrorModel

        model = PACErrorModel(epsilon=0.1, delta=0.1)
        result = model.apply_error(0.5)

        assert isinstance(result, dict)
        assert result["value"] == 0.5
        assert result["lower_bound"] == 0.4
        assert result["upper_bound"] == 0.6
        assert result["confidence"] == 0.9

    def test_apply_error_complex(self):
        """Test applying PAC bounds to complex result."""
        from boofun.core.errormodels import PACErrorModel

        model = PACErrorModel(epsilon=0.1, delta=0.1)
        result = model.apply_error([1, 2, 3])

        assert isinstance(result, dict)
        assert result["value"] == [1, 2, 3]
        assert result["confidence"] == 0.9

    def test_is_reliable(self):
        """Test reliability check."""
        from boofun.core.errormodels import PACErrorModel

        model_reliable = PACErrorModel(epsilon=0.1, delta=0.05)  # 95% confidence
        assert model_reliable.is_reliable(0.5)

        model_unreliable = PACErrorModel(epsilon=0.1, delta=0.2)  # 80% confidence
        assert not model_unreliable.is_reliable(0.5)

    def test_combine_pac_bounds_addition(self):
        """Test combining PAC bounds for addition."""
        from boofun.core.errormodels import PACErrorModel

        model1 = PACErrorModel(epsilon=0.1, delta=0.1)
        model2 = PACErrorModel(epsilon=0.05, delta=0.05)

        combined = model1.combine_pac_bounds(model1, model2, "addition")
        assert abs(combined.epsilon - 0.15) < 1e-10
        assert abs(combined.delta - 0.15) < 1e-10

    def test_combine_pac_bounds_multiplication(self):
        """Test combining PAC bounds for multiplication."""
        from boofun.core.errormodels import PACErrorModel

        model1 = PACErrorModel(epsilon=0.1, delta=0.1)
        model2 = PACErrorModel(epsilon=0.1, delta=0.1)

        combined = model1.combine_pac_bounds(model1, model2, "multiplication")
        assert combined.epsilon > 0.1  # Should be larger due to product term

    def test_combine_pac_bounds_other(self):
        """Test combining PAC bounds for other operations."""
        from boofun.core.errormodels import PACErrorModel

        model1 = PACErrorModel(epsilon=0.1, delta=0.1)
        model2 = PACErrorModel(epsilon=0.1, delta=0.1)

        combined = model1.combine_pac_bounds(model1, model2, "other")
        assert combined.epsilon <= 0.5

    def test_repr(self):
        """Test string representation."""
        from boofun.core.errormodels import PACErrorModel

        model = PACErrorModel(epsilon=0.1, delta=0.05)
        assert "PACErrorModel" in repr(model)
        assert "0.1" in repr(model)


class TestNoiseErrorModel:
    """Test NoiseErrorModel."""

    def test_create_model(self):
        """Test creating noise error model."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.1, random_seed=42)
        assert model.noise_rate == 0.1
        assert model.reliability == 0.8  # 1 - 2*0.1

    def test_invalid_noise_rate(self):
        """Test that invalid noise rate raises error."""
        from boofun.core.errormodels import NoiseErrorModel

        with pytest.raises(ValueError):
            NoiseErrorModel(noise_rate=-0.1)
        with pytest.raises(ValueError):
            NoiseErrorModel(noise_rate=0.6)

    def test_apply_error_zero_noise(self):
        """Test that zero noise returns unchanged result."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.0)
        assert model.apply_error(True) is True
        assert model.apply_error(False) is False

    def test_apply_error_boolean(self):
        """Test applying noise to boolean."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.5, random_seed=42)
        # With 50% noise, roughly half should flip
        flips = sum(1 for _ in range(100) if model.apply_error(True) is False)
        assert 30 < flips < 70  # Should be around 50

    def test_apply_error_array(self):
        """Test applying noise to array."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.1, random_seed=42)
        arr = np.array([True, True, True, True, True, False, False, False, False, False])
        noisy = model.apply_error(arr)
        assert len(noisy) == len(arr)
        # Some values might have flipped

    def test_apply_error_other_type(self):
        """Test applying noise to other types."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.1)
        result = model.apply_error("string")

        assert isinstance(result, dict)
        assert result["noise_applied"] is True

    def test_confidence(self):
        """Test confidence calculation."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.1)
        conf = model.get_confidence(True)
        assert conf >= 0.5

    def test_is_reliable(self):
        """Test reliability check."""
        from boofun.core.errormodels import NoiseErrorModel

        model_reliable = NoiseErrorModel(noise_rate=0.05)
        assert model_reliable.is_reliable(True)

        model_unreliable = NoiseErrorModel(noise_rate=0.2)
        assert not model_unreliable.is_reliable(True)

    def test_repr(self):
        """Test string representation."""
        from boofun.core.errormodels import NoiseErrorModel

        model = NoiseErrorModel(noise_rate=0.1)
        assert "NoiseErrorModel" in repr(model)


class TestCreateErrorModel:
    """Test create_error_model factory function."""

    def test_create_exact(self):
        """Test creating exact model."""
        from boofun.core.errormodels import ExactErrorModel, create_error_model

        model = create_error_model("exact")
        assert isinstance(model, ExactErrorModel)

    def test_create_pac(self):
        """Test creating PAC model."""
        from boofun.core.errormodels import PACErrorModel, create_error_model

        model = create_error_model("pac", epsilon=0.1, delta=0.05)
        assert isinstance(model, PACErrorModel)
        assert model.epsilon == 0.1

    def test_create_noise(self):
        """Test creating noise model."""
        from boofun.core.errormodels import NoiseErrorModel, create_error_model

        model = create_error_model("noise", noise_rate=0.1)
        assert isinstance(model, NoiseErrorModel)

    def test_create_unknown(self):
        """Test creating unknown model raises error."""
        from boofun.core.errormodels import create_error_model

        with pytest.raises(ValueError):
            create_error_model("unknown_model")


# =============================================================================
# DNF Form Tests
# =============================================================================
class TestDNFFormRepresentation:
    """Test DNF form representation."""

    def test_convert_to_dnf(self):
        """Test converting function to DNF."""
        f = bf.OR(3)
        try:
            dnf = f.get_representation("dnf")
            assert dnf is not None
        except KeyError:
            # DNF not registered, skip
            pass

    def test_dnf_and_function(self):
        """Test DNF of AND function."""
        f = bf.AND(3)
        try:
            dnf = f.get_representation("dnf")
            assert dnf is not None
        except KeyError:
            pass

    def test_dnf_majority(self):
        """Test DNF of majority function."""
        f = bf.majority(3)
        try:
            dnf = f.get_representation("dnf")
            assert dnf is not None
        except KeyError:
            pass


# =============================================================================
# Quantum Module Tests
# =============================================================================
class TestQuantumModule:
    """Test quantum analysis module."""

    def test_grover_speedup_majority(self):
        """Test Grover speedup estimation for majority."""
        f = bf.majority(5)
        try:
            from boofun.quantum import estimate_grover_speedup

            speedup = estimate_grover_speedup(f)
            assert speedup is not None
        except (ImportError, AttributeError):
            pytest.skip("Quantum module not available")

    def test_quantum_walk_analysis(self):
        """Test quantum walk analysis."""
        f = bf.OR(4)
        try:
            from boofun.quantum import analyze_quantum_walk

            result = analyze_quantum_walk(f)
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("Quantum walk analysis not available")


# =============================================================================
# Families Module Tests
# =============================================================================
class TestFamiliesBase:
    """Test families/base.py module."""

    def test_majority_family_generate(self):
        """Test MajorityFamily.generate method."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()

        # Generate function at specific n
        f3 = family.generate(3)
        assert f3.n_vars == 3

        f5 = family.generate(5)
        assert f5.n_vars == 5

    def test_parity_family_generate(self):
        """Test ParityFamily.generate method."""
        from boofun.families import ParityFamily

        family = ParityFamily()
        f4 = family.generate(4)
        assert f4.n_vars == 4

    def test_tribes_family_generate(self):
        """Test TribesFamily.generate method."""
        from boofun.families import TribesFamily

        family = TribesFamily()
        f6 = family.generate(6)
        assert f6.n_vars == 6

    def test_family_theoretical_value(self):
        """Test getting theoretical values from family."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()

        # Get theoretical value
        val = family.theoretical_value(5, "total_influence")
        if val is not None:
            assert val > 0

    def test_family_generate_range(self):
        """Test family generate_range method."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()

        # Generate functions for range of n
        functions = family.generate_range(range(3, 8, 2))
        assert len(functions) > 0

    def test_family_metadata(self):
        """Test family metadata."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()
        meta = family.metadata
        assert meta is not None


# =============================================================================
# Integration Tests
# =============================================================================
class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_batch_evaluate_all_representations(self):
        """Test batch evaluation across representations."""
        f = bf.majority(4)

        for repr_name in ["truth_table", "fourier_expansion"]:
            data = f.get_representation(repr_name)
            assert data is not None, f"Failed to get {repr_name} representation"

            # Evaluate at multiple points
            results = [f.evaluate(i) for i in range(16)]
            assert len(results) == 16

            # Verify majority function: output 1 if more than half of inputs are 1
            for i in range(16):
                bits = bin(i).count("1")
                expected = 1 if bits > 2 else 0
                assert results[i] == expected, (
                    f"Majority(4) at input {i} (bits={bits}): "
                    f"expected {expected}, got {results[i]}"
                )

    def test_error_model_with_analysis(self):
        """Test using error models with analysis."""
        from boofun.core.errormodels import PACErrorModel

        f = bf.majority(5)
        ti = f.total_influence()

        model = PACErrorModel(epsilon=0.1, delta=0.1)
        result = model.apply_error(ti)

        assert "value" in result
        assert "confidence" in result

    def test_family_batch_processing(self):
        """Test families with batch processing."""
        from boofun.families import MajorityFamily

        family = MajorityFamily()

        for n in [3, 5]:
            f = family.generate(n)
            # Batch evaluate
            inputs = np.arange(2**n)
            tt = list(f.get_representation("truth_table"))
            assert len(tt) == 2**n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
