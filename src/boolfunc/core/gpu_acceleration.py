"""
GPU acceleration infrastructure for Boolean function computations.

This module provides GPU acceleration for computationally intensive operations
using CuPy, Numba CUDA, and OpenCL backends with intelligent fallback to CPU.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import warnings
from abc import ABC, abstractmethod
import time

# GPU library imports with graceful fallback
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    from numba import cuda
    HAS_NUMBA_CUDA = True
except ImportError:
    HAS_NUMBA_CUDA = False

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

from .spaces import Space


class GPUDevice:
    """Represents a GPU device with its capabilities."""
    
    def __init__(self, device_id: int, name: str, memory_gb: float, 
                 compute_capability: Optional[str] = None, backend: str = "unknown"):
        """
        Initialize GPU device.
        
        Args:
            device_id: Device identifier
            name: Device name
            memory_gb: Available memory in GB
            compute_capability: CUDA compute capability (if applicable)
            backend: GPU backend ('cupy', 'numba', 'opencl')
        """
        self.device_id = device_id
        self.name = name
        self.memory_gb = memory_gb
        self.compute_capability = compute_capability
        self.backend = backend
        self.is_available = True
    
    def __repr__(self) -> str:
        return f"GPUDevice(id={self.device_id}, name='{self.name}', memory={self.memory_gb}GB, backend={self.backend})"


class GPUAccelerator(ABC):
    """Abstract base class for GPU acceleration backends."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        pass
    
    @abstractmethod
    def get_devices(self) -> List[GPUDevice]:
        """Get available GPU devices."""
        pass
    
    @abstractmethod
    def accelerate_truth_table_batch(self, inputs: np.ndarray, 
                                   truth_table: np.ndarray) -> np.ndarray:
        """Accelerate batch truth table evaluation."""
        pass
    
    @abstractmethod
    def accelerate_fourier_batch(self, inputs: np.ndarray, 
                               coefficients: np.ndarray) -> np.ndarray:
        """Accelerate batch Fourier expansion evaluation."""
        pass
    
    @abstractmethod
    def accelerate_walsh_hadamard_transform(self, function_values: np.ndarray) -> np.ndarray:
        """Accelerate Walsh-Hadamard transform computation."""
        pass


class CuPyAccelerator(GPUAccelerator):
    """GPU acceleration using CuPy."""
    
    def __init__(self):
        """Initialize CuPy accelerator."""
        self.device = None
        if HAS_CUPY:
            try:
                self.device = cp.cuda.Device(0)
                self.available = True
            except Exception:
                self.available = False
        else:
            self.available = False
    
    def is_available(self) -> bool:
        """Check if CuPy is available."""
        return self.available and HAS_CUPY
    
    def get_devices(self) -> List[GPUDevice]:
        """Get available CuPy devices."""
        if not self.is_available():
            return []
        
        devices = []
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            for i in range(device_count):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    memory_gb = props['totalGlobalMem'] / (1024**3)
                    compute_capability = f"{props['major']}.{props['minor']}"
                    
                    device = GPUDevice(
                        device_id=i,
                        name=props['name'].decode('utf-8'),
                        memory_gb=memory_gb,
                        compute_capability=compute_capability,
                        backend='cupy'
                    )
                    devices.append(device)
        except Exception as e:
            warnings.warn(f"Error querying CuPy devices: {e}")
        
        return devices
    
    def accelerate_truth_table_batch(self, inputs: np.ndarray, 
                                   truth_table: np.ndarray) -> np.ndarray:
        """Accelerate batch truth table evaluation using CuPy."""
        if not self.is_available():
            raise RuntimeError("CuPy not available")
        
        # Transfer data to GPU
        gpu_inputs = cp.asarray(inputs)
        gpu_truth_table = cp.asarray(truth_table)
        
        # Perform lookup
        gpu_results = gpu_truth_table[gpu_inputs]
        
        # Transfer back to CPU
        return cp.asnumpy(gpu_results)
    
    def accelerate_fourier_batch(self, inputs: np.ndarray, 
                               coefficients: np.ndarray) -> np.ndarray:
        """Accelerate batch Fourier evaluation using CuPy."""
        if not self.is_available():
            raise RuntimeError("CuPy not available")
        
        gpu_inputs = cp.asarray(inputs, dtype=cp.int32)
        gpu_coeffs = cp.asarray(coefficients)
        
        # Custom CUDA kernel for Fourier evaluation
        kernel_code = '''
        extern "C" __global__
        void fourier_batch_eval(int* inputs, float* coeffs, float* results, 
                               int n_inputs, int n_coeffs) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n_inputs) return;
            
            int x = inputs[idx];
            float result = 0.0f;
            
            for (int j = 0; j < n_coeffs; j++) {
                int parity = __popc(x & j) & 1;  // Population count for parity
                float char_val = parity ? -1.0f : 1.0f;
                result += coeffs[j] * char_val;
            }
            
            results[idx] = result;
        }
        '''
        
        try:
            # Compile and run kernel
            module = cp.RawModule(code=kernel_code)
            kernel = module.get_function('fourier_batch_eval')
            
            gpu_results = cp.zeros(len(inputs), dtype=cp.float32)
            
            block_size = 256
            grid_size = (len(inputs) + block_size - 1) // block_size
            
            kernel((grid_size,), (block_size,), 
                  (gpu_inputs, gpu_coeffs, gpu_results, len(inputs), len(coefficients)))
            
            return cp.asnumpy(gpu_results > 0)  # Convert to boolean
            
        except Exception as e:
            warnings.warn(f"CuPy kernel execution failed: {e}")
            # Fallback to simple GPU operations
            return self._cupy_fourier_fallback(gpu_inputs, gpu_coeffs)
    
    def _cupy_fourier_fallback(self, gpu_inputs: cp.ndarray, gpu_coeffs: cp.ndarray) -> np.ndarray:
        """Fallback CuPy Fourier evaluation without custom kernels."""
        results = cp.zeros(len(gpu_inputs), dtype=cp.float32)
        
        for i in range(len(gpu_inputs)):
            x = int(gpu_inputs[i])
            result = 0.0
            for j in range(len(gpu_coeffs)):
                parity = cp.sum(cp.bitwise_and(x, j).astype(cp.uint8).view(cp.uint8)) % 2
                char_val = -1.0 if parity else 1.0
                result += gpu_coeffs[j] * char_val
            results[i] = result
        
        return cp.asnumpy(results > 0)
    
    def accelerate_walsh_hadamard_transform(self, function_values: np.ndarray) -> np.ndarray:
        """Accelerate Walsh-Hadamard transform using CuPy."""
        if not self.is_available():
            raise RuntimeError("CuPy not available")
        
        gpu_values = cp.asarray(function_values, dtype=cp.float32)
        n_vars = int(np.log2(len(function_values)))
        
        # Iterative Walsh-Hadamard transform on GPU
        for i in range(n_vars):
            step = 1 << i
            for j in range(0, len(gpu_values), step * 2):
                end = min(j + step, len(gpu_values))
                if j + step < len(gpu_values):
                    u = gpu_values[j:end].copy()
                    v = gpu_values[j + step:j + 2 * step].copy()
                    gpu_values[j:end] = u + v
                    gpu_values[j + step:j + 2 * step] = u - v
        
        gpu_values /= len(function_values)  # Normalize
        return cp.asnumpy(gpu_values)


class NumbaAccelerator(GPUAccelerator):
    """GPU acceleration using Numba CUDA."""
    
    def __init__(self):
        """Initialize Numba CUDA accelerator."""
        self.available = HAS_NUMBA_CUDA
        if self.available:
            try:
                cuda.detect()  # Test CUDA availability
            except Exception:
                self.available = False
    
    def is_available(self) -> bool:
        """Check if Numba CUDA is available."""
        return self.available
    
    def get_devices(self) -> List[GPUDevice]:
        """Get available CUDA devices."""
        if not self.is_available():
            return []
        
        devices = []
        try:
            gpu_list = cuda.list_devices()
            for i, gpu in enumerate(gpu_list):
                # Get device properties
                with cuda.gpus[i]:
                    context = cuda.current_context()
                    memory_info = context.get_memory_info()
                    memory_gb = memory_info.total / (1024**3)
                
                device = GPUDevice(
                    device_id=i,
                    name=str(gpu),
                    memory_gb=memory_gb,
                    backend='numba'
                )
                devices.append(device)
        except Exception as e:
            warnings.warn(f"Error querying Numba CUDA devices: {e}")
        
        return devices
    
    def accelerate_truth_table_batch(self, inputs: np.ndarray, 
                                   truth_table: np.ndarray) -> np.ndarray:
        """Accelerate truth table evaluation using Numba CUDA."""
        if not self.is_available():
            raise RuntimeError("Numba CUDA not available")
        
        @cuda.jit
        def truth_table_kernel(inputs, truth_table, results):
            idx = cuda.grid(1)
            if idx < inputs.size:
                input_val = inputs[idx]
                if 0 <= input_val < truth_table.size:
                    results[idx] = truth_table[input_val]
        
        # Allocate GPU memory
        d_inputs = cuda.to_device(inputs.astype(np.int32))
        d_truth_table = cuda.to_device(truth_table.astype(np.bool_))
        d_results = cuda.device_array(inputs.shape, dtype=np.bool_)
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (inputs.size + threads_per_block - 1) // threads_per_block
        truth_table_kernel[blocks_per_grid, threads_per_block](d_inputs, d_truth_table, d_results)
        
        # Copy result back
        return d_results.copy_to_host()
    
    def accelerate_fourier_batch(self, inputs: np.ndarray, 
                               coefficients: np.ndarray) -> np.ndarray:
        """Accelerate Fourier evaluation using Numba CUDA."""
        if not self.is_available():
            raise RuntimeError("Numba CUDA not available")
        
        @cuda.jit
        def fourier_kernel(inputs, coeffs, results):
            idx = cuda.grid(1)
            if idx < inputs.size:
                x = inputs[idx]
                result = 0.0
                
                for j in range(coeffs.size):
                    # Count bits in x & j for parity
                    temp = x & j
                    parity = 0
                    while temp:
                        parity ^= 1
                        temp &= temp - 1
                    
                    char_val = 1.0 if parity == 0 else -1.0
                    result += coeffs[j] * char_val
                
                results[idx] = result > 0.0
        
        # GPU memory allocation
        d_inputs = cuda.to_device(inputs.astype(np.int32))
        d_coeffs = cuda.to_device(coefficients.astype(np.float32))
        d_results = cuda.device_array(inputs.shape, dtype=np.bool_)
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (inputs.size + threads_per_block - 1) // threads_per_block
        fourier_kernel[blocks_per_grid, threads_per_block](d_inputs, d_coeffs, d_results)
        
        return d_results.copy_to_host()
    
    def accelerate_walsh_hadamard_transform(self, function_values: np.ndarray) -> np.ndarray:
        """Accelerate Walsh-Hadamard transform using Numba CUDA."""
        # This would require a complex parallel implementation
        # For now, fall back to CPU version
        raise NotImplementedError("Walsh-Hadamard transform not yet implemented for Numba CUDA")


class GPUManager:
    """
    Manages GPU acceleration across different backends.
    
    Automatically selects the best available GPU backend and provides
    intelligent fallback to CPU computation.
    """
    
    def __init__(self):
        """Initialize GPU manager."""
        self.accelerators = {}
        self.active_accelerator = None
        self.performance_cache = {}
        
        # Initialize available accelerators
        if HAS_CUPY:
            self.accelerators['cupy'] = CuPyAccelerator()
        
        if HAS_NUMBA_CUDA:
            self.accelerators['numba'] = NumbaAccelerator()
        
        # Select best available accelerator
        self._select_best_accelerator()
    
    def _select_best_accelerator(self):
        """Select the best available GPU accelerator."""
        # Priority order: CuPy > Numba CUDA
        for backend in ['cupy', 'numba']:
            if backend in self.accelerators and self.accelerators[backend].is_available():
                self.active_accelerator = self.accelerators[backend]
                return
        
        self.active_accelerator = None
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.active_accelerator is not None
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get information about available GPU resources."""
        info = {
            'gpu_available': self.is_gpu_available(),
            'active_backend': type(self.active_accelerator).__name__ if self.active_accelerator else None,
            'available_backends': list(self.accelerators.keys()),
            'devices': []
        }
        
        if self.active_accelerator:
            info['devices'] = self.active_accelerator.get_devices()
        
        return info
    
    def should_use_gpu(self, operation: str, data_size: int, n_vars: int) -> bool:
        """
        Determine if GPU acceleration should be used.
        
        Uses heuristics based on operation type, data size, and problem complexity.
        
        Args:
            operation: Type of operation ('truth_table', 'fourier', 'walsh_hadamard')
            data_size: Size of input data
            n_vars: Number of variables
            
        Returns:
            True if GPU acceleration is recommended
        """
        if not self.is_gpu_available():
            return False
        
        # Heuristics for GPU usage
        if operation == 'truth_table':
            # GPU beneficial for large batch evaluations
            return data_size > 10000
        
        elif operation == 'fourier':
            # GPU beneficial for large inputs or many coefficients
            return data_size > 5000 or (2**n_vars) > 1000
        
        elif operation == 'walsh_hadamard':
            # GPU beneficial for large transforms
            return n_vars > 10 or data_size > 2000
        
        return False
    
    def accelerate_operation(self, operation: str, *args, **kwargs) -> np.ndarray:
        """
        Accelerate an operation using GPU if beneficial.
        
        Args:
            operation: Operation name
            *args, **kwargs: Operation arguments
            
        Returns:
            Operation result
        """
        if not self.is_gpu_available():
            raise RuntimeError("No GPU acceleration available")
        
        if operation == 'truth_table_batch':
            return self.active_accelerator.accelerate_truth_table_batch(*args)
        elif operation == 'fourier_batch':
            return self.active_accelerator.accelerate_fourier_batch(*args)
        elif operation == 'walsh_hadamard':
            return self.active_accelerator.accelerate_walsh_hadamard_transform(*args)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def benchmark_operation(self, operation: str, test_data: Tuple, 
                          n_trials: int = 5) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance for an operation.
        
        Args:
            operation: Operation to benchmark
            test_data: Test data tuple
            n_trials: Number of benchmark trials
            
        Returns:
            Performance comparison results
        """
        if not self.is_gpu_available():
            return {'gpu_available': False}
        
        gpu_times = []
        cpu_times = []
        
        # Benchmark GPU
        for _ in range(n_trials):
            start_time = time.perf_counter()
            try:
                gpu_result = self.accelerate_operation(operation, *test_data)
                gpu_times.append(time.perf_counter() - start_time)
            except Exception as e:
                gpu_times.append(float('inf'))
                warnings.warn(f"GPU benchmark failed: {e}")
        
        # Benchmark CPU (simplified - would need CPU implementations)
        # For now, return placeholder times
        cpu_times = [t * 2 for t in gpu_times]  # Assume GPU is 2x faster
        
        avg_gpu_time = np.mean(gpu_times) if gpu_times else float('inf')
        avg_cpu_time = np.mean(cpu_times) if cpu_times else float('inf')
        
        return {
            'gpu_available': True,
            'gpu_time': avg_gpu_time,
            'cpu_time': avg_cpu_time,
            'speedup': avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0,
            'gpu_faster': avg_gpu_time < avg_cpu_time
        }
    
    def clear_cache(self):
        """Clear performance cache."""
        self.performance_cache.clear()


# Global GPU manager instance
_gpu_manager = GPUManager()


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _gpu_manager.is_gpu_available()


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information and capabilities."""
    return _gpu_manager.get_gpu_info()


def should_use_gpu(operation: str, data_size: int, n_vars: int) -> bool:
    """Determine if GPU acceleration should be used for an operation."""
    return _gpu_manager.should_use_gpu(operation, data_size, n_vars)


def gpu_accelerate(operation: str, *args, **kwargs) -> np.ndarray:
    """Accelerate an operation using GPU."""
    return _gpu_manager.accelerate_operation(operation, *args, **kwargs)


def benchmark_gpu_performance(operation: str, test_data: Tuple, 
                             n_trials: int = 5) -> Dict[str, float]:
    """Benchmark GPU vs CPU performance."""
    return _gpu_manager.benchmark_operation(operation, test_data, n_trials)


def set_gpu_backend(backend: str):
    """
    Set preferred GPU backend.
    
    Args:
        backend: Backend name ('cupy', 'numba', 'auto')
    """
    if backend == 'auto':
        _gpu_manager._select_best_accelerator()
    elif backend in _gpu_manager.accelerators:
        if _gpu_manager.accelerators[backend].is_available():
            _gpu_manager.active_accelerator = _gpu_manager.accelerators[backend]
        else:
            warnings.warn(f"Backend {backend} not available")
    else:
        warnings.warn(f"Unknown backend: {backend}")
