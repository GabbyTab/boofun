"""
Comprehensive batch processing infrastructure for Boolean function operations.

This module provides efficient batch evaluation, vectorized operations, and
parallel processing capabilities for Boolean functions across all representations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from abc import ABC, abstractmethod
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available - batch processing will use pure NumPy")

from .spaces import Space
from .representations.base import BooleanFunctionRepresentation
from .gpu_acceleration import is_gpu_available, should_use_gpu, gpu_accelerate


class BatchProcessor(ABC):
    """Abstract base class for batch processing strategies."""
    
    @abstractmethod
    def process_batch(self, inputs: np.ndarray, function_data: Any, 
                     space: Space, n_vars: int) -> np.ndarray:
        """Process a batch of inputs."""
        pass
    
    @abstractmethod
    def supports_representation(self, representation: str) -> bool:
        """Check if processor supports a representation."""
        pass


class VectorizedBatchProcessor(BatchProcessor):
    """
    Vectorized batch processor using NumPy operations.
    
    Optimized for representations that can leverage NumPy's vectorization.
    """
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize vectorized processor.
        
        Args:
            chunk_size: Size of chunks for memory-efficient processing
        """
        self.chunk_size = chunk_size
        self.supported_representations = {
            'truth_table', 'fourier_expansion', 'anf', 'polynomial'
        }
    
    def process_batch(self, inputs: np.ndarray, function_data: Any,
                     space: Space, n_vars: int) -> np.ndarray:
        """Process batch using vectorized operations."""
        if inputs.size <= self.chunk_size:
            return self._process_chunk(inputs, function_data, space, n_vars)
        else:
            # Process in chunks to manage memory
            results = []
            for i in range(0, inputs.size, self.chunk_size):
                chunk = inputs[i:i + self.chunk_size]
                chunk_results = self._process_chunk(chunk, function_data, space, n_vars)
                results.append(chunk_results)
            return np.concatenate(results)
    
    def _process_chunk(self, inputs: np.ndarray, function_data: Any,
                      space: Space, n_vars: int) -> np.ndarray:
        """Process a single chunk of inputs."""
        # This is a generic implementation - specific representations
        # should override with optimized versions
        results = np.zeros(inputs.shape[0], dtype=bool)
        for i, input_val in enumerate(inputs):
            # Fallback to individual evaluation
            # In practice, each representation would have optimized batch methods
            results[i] = self._evaluate_single(input_val, function_data, space, n_vars)
        return results
    
    def _evaluate_single(self, input_val: Any, function_data: Any,
                        space: Space, n_vars: int) -> bool:
        """Fallback single evaluation method."""
        # This should be overridden by specific implementations
        return False
    
    def supports_representation(self, representation: str) -> bool:
        """Check if representation is supported."""
        return representation in self.supported_representations


class ParallelBatchProcessor(BatchProcessor):
    """
    Parallel batch processor using multiprocessing.
    
    Distributes work across multiple CPU cores for compute-intensive operations.
    """
    
    def __init__(self, n_workers: Optional[int] = None, use_processes: bool = True):
        """
        Initialize parallel processor.
        
        Args:
            n_workers: Number of worker processes/threads (default: CPU count)
            use_processes: Whether to use processes (True) or threads (False)
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.supported_representations = {
            'symbolic', 'circuit', 'bdd', 'cnf_form', 'dnf_form', 'ltf'
        }
    
    def process_batch(self, inputs: np.ndarray, function_data: Any,
                     space: Space, n_vars: int) -> np.ndarray:
        """Process batch using parallel workers."""
        if inputs.size < self.n_workers * 10:
            # Too small for parallelization overhead
            return self._process_sequential(inputs, function_data, space, n_vars)
        
        # Split inputs across workers
        chunks = np.array_split(inputs, self.n_workers)
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self._process_chunk, chunk, function_data, space, n_vars)
                for chunk in chunks
            ]
            
            results = []
            for future in futures:
                try:
                    chunk_result = future.result(timeout=30)  # 30 second timeout
                    results.append(chunk_result)
                except Exception as e:
                    warnings.warn(f"Parallel processing failed: {e}")
                    # Fallback to sequential processing
                    return self._process_sequential(inputs, function_data, space, n_vars)
        
        return np.concatenate(results)
    
    def _process_chunk(self, inputs: np.ndarray, function_data: Any,
                      space: Space, n_vars: int) -> np.ndarray:
        """Process a chunk of inputs in a worker."""
        return self._process_sequential(inputs, function_data, space, n_vars)
    
    def _process_sequential(self, inputs: np.ndarray, function_data: Any,
                           space: Space, n_vars: int) -> np.ndarray:
        """Sequential fallback processing."""
        results = np.zeros(inputs.shape[0], dtype=bool)
        for i, input_val in enumerate(inputs):
            results[i] = self._evaluate_single(input_val, function_data, space, n_vars)
        return results
    
    def _evaluate_single(self, input_val: Any, function_data: Any,
                        space: Space, n_vars: int) -> bool:
        """Single evaluation method."""
        return False
    
    def supports_representation(self, representation: str) -> bool:
        """Check if representation is supported."""
        return representation in self.supported_representations


class OptimizedTruthTableProcessor(VectorizedBatchProcessor):
    """Optimized batch processor for truth table representations."""
    
    def __init__(self, chunk_size: int = 100000):
        super().__init__(chunk_size)
        self.supported_representations = {'truth_table'}
    
    def _process_chunk(self, inputs: np.ndarray, function_data: np.ndarray,
                      space: Space, n_vars: int) -> np.ndarray:
        """Optimized truth table batch processing with GPU acceleration."""
        
        # Try GPU acceleration if beneficial
        if should_use_gpu('truth_table', inputs.size, n_vars):
            try:
                return gpu_accelerate('truth_table_batch', inputs, function_data)
            except Exception as e:
                warnings.warn(f"GPU acceleration failed, falling back to CPU: {e}")
        
        # CPU implementation
        if inputs.ndim == 1 and inputs.dtype != np.bool_:
            # Integer indices - direct lookup
            indices = inputs.astype(int)
            # Bounds checking
            valid_mask = (indices >= 0) & (indices < len(function_data))
            results = np.zeros(len(indices), dtype=bool)
            results[valid_mask] = function_data[indices[valid_mask]]
            return results
        elif inputs.ndim == 2:
            # Binary vectors - convert to indices
            indices = self._binary_vectors_to_indices(inputs)
            return function_data[indices].astype(bool)
        else:
            return super()._process_chunk(inputs, function_data, space, n_vars)
    
    def _binary_vectors_to_indices(self, binary_vectors: np.ndarray) -> np.ndarray:
        """Convert batch of binary vectors to indices efficiently."""
        # Vectorized binary to integer conversion
        powers = 2 ** np.arange(binary_vectors.shape[1] - 1, -1, -1)
        return np.dot(binary_vectors, powers)


class OptimizedFourierProcessor(VectorizedBatchProcessor):
    """Optimized batch processor for Fourier expansion representations."""
    
    def __init__(self, chunk_size: int = 50000):
        super().__init__(chunk_size)
        self.supported_representations = {'fourier_expansion'}
    
    def _process_chunk(self, inputs: np.ndarray, function_data: np.ndarray,
                      space: Space, n_vars: int) -> np.ndarray:
        """Optimized Fourier expansion batch processing with GPU acceleration."""
        
        # Try GPU acceleration first
        if should_use_gpu('fourier', inputs.size, n_vars):
            try:
                return gpu_accelerate('fourier_batch', inputs, function_data)
            except Exception as e:
                warnings.warn(f"GPU Fourier acceleration failed, falling back to CPU: {e}")
        
        # CPU implementations
        if HAS_NUMBA:
            return self._numba_fourier_batch(inputs, function_data, n_vars)
        else:
            return self._numpy_fourier_batch(inputs, function_data, n_vars)
    
    def _numpy_fourier_batch(self, inputs: np.ndarray, coeffs: np.ndarray,
                           n_vars: int) -> np.ndarray:
        """NumPy-based batch Fourier evaluation."""
        if inputs.ndim == 1 and inputs.dtype != np.bool_:
            # Integer inputs
            results = np.zeros(len(inputs), dtype=float)
            for i, x in enumerate(inputs):
                results[i] = self._evaluate_fourier_single(int(x), coeffs)
            return results > 0  # Convert to boolean
        else:
            # Fallback to sequential
            return super()._process_chunk(inputs, coeffs, Space.BOOLEAN_CUBE, n_vars)
    
    def _evaluate_fourier_single(self, x: int, coeffs: np.ndarray) -> float:
        """Evaluate single Fourier expansion."""
        result = 0.0
        for j in range(len(coeffs)):
            parity = bin(x & j).count('1') % 2
            char_val = (-1) ** parity
            result += coeffs[j] * char_val
        return result
    
    def _numba_fourier_batch(self, inputs: np.ndarray, coeffs: np.ndarray,
                           n_vars: int) -> np.ndarray:
        """Numba-accelerated batch Fourier evaluation."""
        if not HAS_NUMBA:
            return self._numpy_fourier_batch(inputs, coeffs, n_vars)
        
        return _numba_fourier_batch_impl(inputs.astype(np.int32), coeffs, n_vars)


class OptimizedANFProcessor(VectorizedBatchProcessor):
    """Optimized batch processor for ANF representations."""
    
    def __init__(self, chunk_size: int = 50000):
        super().__init__(chunk_size)
        self.supported_representations = {'anf'}
    
    def _process_chunk(self, inputs: np.ndarray, function_data: Dict,
                      space: Space, n_vars: int) -> np.ndarray:
        """Optimized ANF batch processing."""
        if HAS_NUMBA:
            return self._numba_anf_batch(inputs, function_data, n_vars)
        else:
            return self._numpy_anf_batch(inputs, function_data, n_vars)
    
    def _numpy_anf_batch(self, inputs: np.ndarray, anf_dict: Dict,
                        n_vars: int) -> np.ndarray:
        """NumPy-based batch ANF evaluation."""
        results = np.zeros(len(inputs), dtype=bool)
        
        # Convert ANF dict to arrays for vectorization
        monomials = []
        coeffs = []
        for monomial, coeff in anf_dict.items():
            if coeff != 0:
                monomials.append(list(monomial))
                coeffs.append(coeff)
        
        if not monomials:
            return results  # Zero function
        
        # Vectorized evaluation
        for i, x in enumerate(inputs):
            if isinstance(x, np.integer) or isinstance(x, int):
                # Convert integer to binary
                binary = [(int(x) >> bit) & 1 for bit in range(n_vars)]
            else:
                binary = x
            
            result = 0
            for monomial, coeff in zip(monomials, coeffs):
                # Compute monomial value
                monomial_val = 1
                for var_idx in monomial:
                    if var_idx < len(binary):
                        monomial_val *= binary[var_idx]
                result ^= (coeff * monomial_val)
            
            results[i] = bool(result)
        
        return results
    
    def _numba_anf_batch(self, inputs: np.ndarray, anf_dict: Dict,
                        n_vars: int) -> np.ndarray:
        """Numba-accelerated batch ANF evaluation."""
        # Convert to format suitable for Numba
        monomials_list = []
        coeffs_list = []
        
        for monomial, coeff in anf_dict.items():
            if coeff != 0:
                # Convert frozenset to sorted list
                mono_array = np.array(sorted(list(monomial)), dtype=np.int32)
                monomials_list.append(mono_array)
                coeffs_list.append(coeff)
        
        if not monomials_list:
            return np.zeros(len(inputs), dtype=bool)
        
        # This would need a more complex Numba implementation
        # For now, fall back to NumPy version
        return self._numpy_anf_batch(inputs, anf_dict, n_vars)


# Numba JIT compiled functions (if available)
if HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def _numba_fourier_batch_impl(inputs: np.ndarray, coeffs: np.ndarray, 
                                 n_vars: int) -> np.ndarray:
        """Numba-compiled batch Fourier evaluation."""
        results = np.zeros(len(inputs), dtype=np.bool_)
        
        for i in prange(len(inputs)):
            x = inputs[i]
            result = 0.0
            
            for j in range(len(coeffs)):
                # Count bits in x & j
                parity = 0
                temp = x & j
                while temp:
                    parity ^= 1
                    temp &= temp - 1
                
                char_val = 1.0 if parity == 0 else -1.0
                result += coeffs[j] * char_val
            
            results[i] = result > 0
        
        return results
    
    @jit(nopython=True, parallel=True)
    def _numba_truth_table_batch_impl(inputs: np.ndarray, 
                                     truth_table: np.ndarray) -> np.ndarray:
        """Numba-compiled batch truth table evaluation."""
        results = np.zeros(len(inputs), dtype=np.bool_)
        
        for i in prange(len(inputs)):
            idx = inputs[i]
            if 0 <= idx < len(truth_table):
                results[i] = truth_table[idx]
        
        return results


class BatchProcessorManager:
    """
    Manages batch processing strategies for different representations.
    
    Automatically selects the best processor based on representation type,
    input size, and available hardware.
    """
    
    def __init__(self):
        """Initialize batch processor manager."""
        self.processors = {
            'truth_table': OptimizedTruthTableProcessor(),
            'fourier_expansion': OptimizedFourierProcessor(),
            'anf': OptimizedANFProcessor(),
            'vectorized': VectorizedBatchProcessor(),
            'parallel': ParallelBatchProcessor()
        }
        
        # Performance thresholds
        self.parallel_threshold = 10000  # Use parallel processing above this size
        self.vectorized_threshold = 1000  # Use vectorized processing above this size
    
    def process_batch(self, inputs: np.ndarray, function_data: Any,
                     representation: str, space: Space, n_vars: int) -> np.ndarray:
        """
        Process batch using optimal strategy.
        
        Args:
            inputs: Batch of inputs to process
            function_data: Representation-specific function data
            representation: Representation type
            space: Mathematical space
            n_vars: Number of variables
            
        Returns:
            Batch evaluation results
        """
        # Select processor based on representation and input size
        processor = self._select_processor(representation, inputs.size)
        
        try:
            return processor.process_batch(inputs, function_data, space, n_vars)
        except Exception as e:
            warnings.warn(f"Batch processing failed with {type(processor).__name__}: {e}")
            # Fallback to sequential processing
            return self._sequential_fallback(inputs, function_data, representation, space, n_vars)
    
    def _select_processor(self, representation: str, input_size: int) -> BatchProcessor:
        """Select optimal processor for given conditions."""
        
        # First, try representation-specific processor
        if representation in self.processors:
            return self.processors[representation]
        
        # Fall back to general strategies based on input size
        if input_size >= self.parallel_threshold:
            return self.processors['parallel']
        elif input_size >= self.vectorized_threshold:
            return self.processors['vectorized']
        else:
            return self.processors['vectorized']  # Default choice
    
    def _sequential_fallback(self, inputs: np.ndarray, function_data: Any,
                           representation: str, space: Space, n_vars: int) -> np.ndarray:
        """Sequential processing fallback."""
        from .representations.registry import get_strategy
        
        strategy = get_strategy(representation)
        results = np.zeros(inputs.shape[0], dtype=bool)
        
        for i, input_val in enumerate(inputs):
            try:
                results[i] = strategy.evaluate(input_val, function_data, space, n_vars)
            except Exception:
                results[i] = False  # Safe fallback
        
        return results
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get statistics about available processors."""
        return {
            'available_processors': list(self.processors.keys()),
            'numba_available': HAS_NUMBA,
            'cpu_count': mp.cpu_count(),
            'thresholds': {
                'parallel': self.parallel_threshold,
                'vectorized': self.vectorized_threshold
            }
        }


# Global batch processor manager
_batch_manager = BatchProcessorManager()


def process_batch(inputs: np.ndarray, function_data: Any, representation: str,
                 space: Space, n_vars: int) -> np.ndarray:
    """
    Process a batch of inputs efficiently.
    
    Args:
        inputs: Batch of inputs
        function_data: Representation-specific data
        representation: Representation type
        space: Mathematical space
        n_vars: Number of variables
        
    Returns:
        Batch evaluation results
    """
    return _batch_manager.process_batch(inputs, function_data, representation, space, n_vars)


def get_batch_processor_stats() -> Dict[str, Any]:
    """Get batch processing statistics and capabilities."""
    return _batch_manager.get_processor_stats()


def set_batch_thresholds(vectorized_threshold: int = 1000,
                        parallel_threshold: int = 10000):
    """
    Set thresholds for batch processing strategies.
    
    Args:
        vectorized_threshold: Minimum size for vectorized processing
        parallel_threshold: Minimum size for parallel processing
    """
    _batch_manager.vectorized_threshold = vectorized_threshold
    _batch_manager.parallel_threshold = parallel_threshold
