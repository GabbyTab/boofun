"""
Tests for batch processing module.

Tests for:
- VectorizedBatchProcessor
- ParallelBatchProcessor
- OptimizedTruthTableProcessor
- OptimizedFourierProcessor
- BatchProcessorManager
- Module functions
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from boofun.core.batch_processing import (
    VectorizedBatchProcessor,
    ParallelBatchProcessor,
    OptimizedTruthTableProcessor,
    OptimizedFourierProcessor,
    OptimizedANFProcessor,
    BatchProcessorManager,
    process_batch,
    get_batch_processor_stats,
    set_batch_thresholds,
)
from boofun.core.spaces import Space


class TestVectorizedBatchProcessor:
    """Tests for VectorizedBatchProcessor."""
    
    def test_initialization(self):
        """Processor initializes with default chunk size."""
        processor = VectorizedBatchProcessor()
        
        assert processor.chunk_size == 10000
        assert 'truth_table' in processor.supported_representations
    
    def test_custom_chunk_size(self):
        """Can set custom chunk size."""
        processor = VectorizedBatchProcessor(chunk_size=5000)
        
        assert processor.chunk_size == 5000
    
    def test_supports_representation(self):
        """Check representation support."""
        processor = VectorizedBatchProcessor()
        
        assert processor.supports_representation('truth_table')
        assert processor.supports_representation('fourier_expansion')
        assert not processor.supports_representation('unknown_rep')
    
    def test_process_batch_small(self):
        """Process small batch."""
        processor = VectorizedBatchProcessor()
        
        inputs = np.array([0, 1, 2, 3])
        truth_table = np.array([False, True, True, False])
        
        # This uses the generic fallback
        results = processor.process_batch(inputs, truth_table, None, 2)
        
        assert len(results) == 4


class TestParallelBatchProcessor:
    """Tests for ParallelBatchProcessor."""
    
    def test_initialization(self):
        """Processor initializes with defaults."""
        processor = ParallelBatchProcessor()
        
        assert processor.n_workers >= 1
        assert processor.use_processes == True
    
    def test_custom_workers(self):
        """Can set custom worker count."""
        processor = ParallelBatchProcessor(n_workers=2, use_processes=False)
        
        assert processor.n_workers == 2
        assert processor.use_processes == False
    
    def test_supports_representation(self):
        """Check representation support."""
        processor = ParallelBatchProcessor()
        
        assert processor.supports_representation('circuit')
        assert processor.supports_representation('bdd')
    
    def test_process_batch_small_sequential(self):
        """Small batch uses sequential processing."""
        processor = ParallelBatchProcessor(n_workers=2)
        
        # Small batch - should use sequential
        inputs = np.array([0, 1])
        truth_table = np.array([False, True, True, False])
        
        results = processor.process_batch(inputs, truth_table, None, 2)
        
        assert len(results) == 2


class TestOptimizedTruthTableProcessor:
    """Tests for OptimizedTruthTableProcessor."""
    
    def test_initialization(self):
        """Processor initializes correctly."""
        processor = OptimizedTruthTableProcessor()
        
        assert processor.chunk_size == 100000
        assert processor.supports_representation('truth_table')
    
    def test_process_integer_indices(self):
        """Process integer indices correctly."""
        processor = OptimizedTruthTableProcessor()
        
        inputs = np.array([0, 1, 2, 3])
        truth_table = np.array([False, True, True, False])
        
        results = processor._process_chunk(inputs, truth_table, None, 2)
        
        assert np.array_equal(results, truth_table)
    
    def test_process_out_of_bounds(self):
        """Handle out of bounds indices."""
        processor = OptimizedTruthTableProcessor()
        
        inputs = np.array([0, 1, 10, -1])  # 10 and -1 are out of bounds
        truth_table = np.array([False, True, True, False])
        
        results = processor._process_chunk(inputs, truth_table, None, 2)
        
        assert results[0] == False
        assert results[1] == True
        # Out of bounds should be False
        assert results[2] == False
    
    def test_process_binary_vectors(self):
        """Process binary vectors correctly."""
        processor = OptimizedTruthTableProcessor()
        
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        truth_table = np.array([False, True, True, False])
        
        results = processor._process_chunk(inputs, truth_table, None, 2)
        
        # Binary vectors [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
        assert np.array_equal(results, truth_table)
    
    def test_binary_vectors_to_indices(self):
        """Convert binary vectors to indices."""
        processor = OptimizedTruthTableProcessor()
        
        vectors = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        indices = processor._binary_vectors_to_indices(vectors)
        
        assert np.array_equal(indices, [0, 1, 2, 3])


class TestOptimizedFourierProcessor:
    """Tests for OptimizedFourierProcessor."""
    
    def test_initialization(self):
        """Processor initializes correctly."""
        processor = OptimizedFourierProcessor()
        
        assert processor.chunk_size == 50000
        assert processor.supports_representation('fourier_expansion')
    
    def test_evaluate_fourier_single(self):
        """Evaluate single Fourier expansion."""
        processor = OptimizedFourierProcessor()
        
        # Constant function: f̂(∅) = 1, others = 0
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        
        result = processor._evaluate_fourier_single(0, coeffs)
        
        assert result == 1.0
    
    def test_numpy_fourier_batch(self):
        """NumPy Fourier batch evaluation."""
        processor = OptimizedFourierProcessor()
        
        # Constant function
        coeffs = np.array([1.0, 0.0, 0.0, 0.0])
        inputs = np.array([0, 1, 2, 3])
        
        results = processor._numpy_fourier_batch(inputs, coeffs, 2)
        
        # All should be True (positive)
        assert all(results)


class TestOptimizedANFProcessor:
    """Tests for OptimizedANFProcessor."""
    
    def test_initialization(self):
        """Processor initializes correctly."""
        processor = OptimizedANFProcessor()
        
        assert processor.supports_representation('anf')
    
    def test_numpy_anf_batch_empty(self):
        """Empty ANF gives all False."""
        processor = OptimizedANFProcessor()
        
        anf_dict = {}  # Empty
        inputs = np.array([0, 1, 2, 3])
        
        results = processor._numpy_anf_batch(inputs, anf_dict, 2)
        
        assert all(r == False for r in results)


class TestBatchProcessorManager:
    """Tests for BatchProcessorManager."""
    
    def test_initialization(self):
        """Manager initializes with processors."""
        manager = BatchProcessorManager()
        
        assert 'truth_table' in manager.processors
        assert 'fourier_expansion' in manager.processors
        assert 'vectorized' in manager.processors
        assert 'parallel' in manager.processors
    
    def test_get_processor_stats(self):
        """Get processor statistics."""
        manager = BatchProcessorManager()
        
        stats = manager.get_processor_stats()
        
        assert 'available_processors' in stats
        assert 'numba_available' in stats
        assert 'cpu_count' in stats
        assert 'thresholds' in stats
    
    def test_process_batch_truth_table(self):
        """Process with truth table."""
        manager = BatchProcessorManager()
        
        inputs = np.array([0, 1, 2, 3])
        truth_table = np.array([False, True, True, False])
        
        results = manager.process_batch(
            inputs, truth_table, 'truth_table', Space.BOOLEAN_CUBE, 2
        )
        
        assert len(results) == 4
    
    def test_select_processor_by_representation(self):
        """Select processor based on representation."""
        manager = BatchProcessorManager()
        
        processor = manager._select_processor('truth_table', 100)
        assert isinstance(processor, OptimizedTruthTableProcessor)
        
        processor = manager._select_processor('fourier_expansion', 100)
        assert isinstance(processor, OptimizedFourierProcessor)
    
    def test_select_processor_by_size(self):
        """Select processor based on input size."""
        manager = BatchProcessorManager()
        
        # Large input should use parallel
        processor = manager._select_processor('unknown', 20000)
        assert isinstance(processor, ParallelBatchProcessor)


class TestModuleFunctions:
    """Tests for module-level functions."""
    
    def test_process_batch(self):
        """process_batch function works."""
        inputs = np.array([0, 1, 2, 3])
        truth_table = np.array([False, True, True, False])
        
        results = process_batch(
            inputs, truth_table, 'truth_table', Space.BOOLEAN_CUBE, 2
        )
        
        assert len(results) == 4
    
    def test_get_batch_processor_stats(self):
        """get_batch_processor_stats works."""
        stats = get_batch_processor_stats()
        
        assert isinstance(stats, dict)
        assert 'available_processors' in stats
    
    def test_set_batch_thresholds(self):
        """set_batch_thresholds works."""
        # Set custom thresholds
        set_batch_thresholds(vectorized_threshold=500, parallel_threshold=5000)
        
        stats = get_batch_processor_stats()
        assert stats['thresholds']['vectorized'] == 500
        assert stats['thresholds']['parallel'] == 5000
        
        # Reset to defaults
        set_batch_thresholds(vectorized_threshold=1000, parallel_threshold=10000)


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_input(self):
        """Handle empty input array."""
        manager = BatchProcessorManager()
        
        inputs = np.array([])
        truth_table = np.array([False, True, True, False])
        
        results = manager.process_batch(
            inputs, truth_table, 'truth_table', Space.BOOLEAN_CUBE, 2
        )
        
        assert len(results) == 0
    
    def test_single_input(self):
        """Handle single input."""
        manager = BatchProcessorManager()
        
        inputs = np.array([1])
        truth_table = np.array([False, True, True, False])
        
        results = manager.process_batch(
            inputs, truth_table, 'truth_table', Space.BOOLEAN_CUBE, 2
        )
        
        assert len(results) == 1
        assert results[0] == True
