"""
Comprehensive tests for core/gpu module.

Tests GPU acceleration features including fallback to CPU.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.gpu import (
    is_gpu_available,
    is_gpu_enabled,
    enable_gpu,
    get_array_module,
    to_gpu,
    to_cpu,
    gpu_walsh_hadamard,
)


class TestGPUAvailability:
    """Test GPU availability checking."""
    
    def test_is_gpu_available_returns_bool(self):
        """is_gpu_available should return boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)
    
    def test_is_gpu_enabled_returns_bool(self):
        """is_gpu_enabled should return boolean."""
        result = is_gpu_enabled()
        assert isinstance(result, bool)
    
    def test_enable_gpu_without_gpu(self):
        """enable_gpu should handle case when GPU not available."""
        original = is_gpu_enabled()
        
        # Try to enable - should not crash
        enable_gpu(True)
        
        # Try to disable
        enable_gpu(False)
        
        # Restore
        enable_gpu(original)


class TestArrayOperations:
    """Test array transfer operations."""
    
    def test_get_array_module_numpy(self):
        """get_array_module should return numpy for numpy arrays."""
        arr = np.array([1, 2, 3])
        module = get_array_module(arr)
        
        assert module is np
    
    def test_to_gpu_returns_array(self):
        """to_gpu should return an array."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = to_gpu(arr)
        
        assert result is not None
        assert len(result) == 4
    
    def test_to_cpu_numpy_passthrough(self):
        """to_cpu should pass through numpy arrays unchanged."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_cpu(arr)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
    
    def test_roundtrip_gpu_cpu(self):
        """Array should survive GPU->CPU roundtrip."""
        original = np.array([1.0, 2.0, 3.0, 4.0])
        
        gpu_arr = to_gpu(original)
        cpu_arr = to_cpu(gpu_arr)
        
        assert np.allclose(cpu_arr, original)


class TestGPUWalshHadamard:
    """Test GPU-accelerated Walsh-Hadamard Transform."""
    
    def test_wht_constant_function(self):
        """WHT of constant +1 should have only constant term."""
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = gpu_walsh_hadamard(values)
        
        # Convert to CPU if needed
        result = to_cpu(result)
        
        assert abs(result[0] - 1.0) < 1e-10
        assert all(abs(r) < 1e-10 for r in result[1:])
    
    def test_wht_alternating(self):
        """WHT of alternating values."""
        # XOR function: +1, -1, -1, +1
        values = np.array([1.0, -1.0, -1.0, 1.0])
        result = gpu_walsh_hadamard(values)
        result = to_cpu(result)
        
        # Should have weight only on degree 2 (index 3)
        assert abs(result[-1]) > 0.5
    
    def test_wht_size_2(self):
        """WHT should work for n=1 (size 2)."""
        values = np.array([1.0, -1.0])  # Dictator
        result = gpu_walsh_hadamard(values)
        result = to_cpu(result)
        
        assert len(result) == 2
    
    def test_wht_size_8(self):
        """WHT should work for n=3 (size 8)."""
        values = np.ones(8)
        result = gpu_walsh_hadamard(values)
        result = to_cpu(result)
        
        assert len(result) == 8
    
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_wht_different_sizes(self, n):
        """WHT should work for various sizes."""
        values = np.ones(2**n)
        result = gpu_walsh_hadamard(values)
        result = to_cpu(result)
        
        assert len(result) == 2**n
        assert np.isfinite(result).all()
    
    def test_wht_parseval(self):
        """WHT should satisfy Parseval's identity."""
        # Random values on the hypercube
        np.random.seed(42)
        values = np.random.choice([-1.0, 1.0], size=16)
        
        result = gpu_walsh_hadamard(values)
        result = to_cpu(result)
        
        # Parseval: sum of squared coefficients = 1 (for normalized WHT)
        total_weight = np.sum(result ** 2)
        assert abs(total_weight - 1.0) < 1e-10


class TestGPUWithBooleanFunctions:
    """Test GPU operations with actual Boolean functions."""
    
    def test_gpu_wht_matches_bf_fourier(self):
        """GPU WHT should match BooleanFunction.fourier()."""
        f = bf.majority(3)
        
        # Get truth table in ±1 form
        tt = list(f.get_representation("truth_table"))
        pm_values = np.array([1.0 - 2.0 * v for v in tt])  # O'Donnell convention
        
        # GPU WHT
        gpu_result = gpu_walsh_hadamard(pm_values)
        gpu_fourier = to_cpu(gpu_result)
        
        # Standard Fourier
        bf_fourier = np.array(f.fourier())
        
        # Should match
        assert np.allclose(gpu_fourier, bf_fourier, atol=1e-10)
    
    def test_gpu_wht_parity(self):
        """GPU WHT for parity should have single non-zero coefficient."""
        f = bf.parity(3)
        
        tt = list(f.get_representation("truth_table"))
        pm_values = np.array([1.0 - 2.0 * v for v in tt])
        
        result = gpu_walsh_hadamard(pm_values)
        result = to_cpu(result)
        
        # Parity has coefficient only at index 7 (all variables)
        non_zero = np.where(np.abs(result) > 1e-10)[0]
        assert len(non_zero) == 1
        assert non_zero[0] == 7  # {0,1,2}
    
    @pytest.mark.parametrize("func_name,n", [
        ("AND", 3),
        ("OR", 3),
        ("majority", 3),
        ("parity", 4),
    ])
    def test_gpu_wht_various_functions(self, func_name, n):
        """GPU WHT should work for various Boolean functions."""
        if func_name == "AND":
            f = bf.AND(n)
        elif func_name == "OR":
            f = bf.OR(n)
        elif func_name == "majority":
            f = bf.majority(n)
        else:
            f = bf.parity(n)
        
        tt = list(f.get_representation("truth_table"))
        pm_values = np.array([1.0 - 2.0 * v for v in tt])
        
        result = gpu_walsh_hadamard(pm_values)
        result = to_cpu(result)
        
        # Basic sanity checks
        assert len(result) == 2**n
        assert np.isfinite(result).all()
        assert abs(np.sum(result**2) - 1.0) < 1e-10  # Parseval


class TestGPUEdgeCases:
    """Test edge cases for GPU operations."""
    
    def test_empty_enable_disable_cycle(self):
        """Multiple enable/disable cycles should not crash."""
        for _ in range(5):
            enable_gpu(True)
            enable_gpu(False)
    
    def test_wht_single_element(self):
        """WHT should handle single element (n=0)."""
        values = np.array([1.0])
        result = gpu_walsh_hadamard(values)
        result = to_cpu(result)
        
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
