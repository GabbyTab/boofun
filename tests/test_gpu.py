"""
Tests for GPU acceleration module.

These tests verify that:
1. GPU functions fall back to CPU correctly when CuPy is unavailable
2. GPU functions produce correct results (when available)
3. The enable/disable mechanism works
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from boolfunc.core import gpu
import boolfunc as bf


class TestGPUAvailability:
    """Test GPU availability detection."""
    
    def test_is_gpu_available_returns_bool(self):
        """is_gpu_available should return a boolean."""
        result = gpu.is_gpu_available()
        assert isinstance(result, bool)
    
    def test_is_gpu_enabled_returns_bool(self):
        """is_gpu_enabled should return a boolean."""
        result = gpu.is_gpu_enabled()
        assert isinstance(result, bool)
    
    def test_enable_gpu_without_cupy(self):
        """enable_gpu should warn if CuPy not available."""
        if not gpu.CUPY_AVAILABLE:
            with pytest.warns(UserWarning, match="CuPy not available"):
                gpu.enable_gpu(True)


class TestCPUFallback:
    """Test that GPU functions fall back to CPU correctly."""
    
    def test_gpu_wht_fallback(self):
        """gpu_walsh_hadamard should work without GPU."""
        # Create simple test case
        n = 4
        values = np.array([1, -1, -1, 1, -1, 1, 1, -1,
                          -1, 1, 1, -1, 1, -1, -1, 1], dtype=float)
        
        result = gpu.gpu_walsh_hadamard(values)
        
        # Should produce valid WHT
        assert len(result) == 16
        assert np.isfinite(result).all()
    
    def test_gpu_influences_fallback(self):
        """gpu_influences should work without GPU."""
        # Use a known function
        f = bf.parity(4)
        fourier = f.fourier()
        
        influences = gpu.gpu_influences(fourier)
        
        # Parity: all influences = 1
        assert len(influences) == 4
        assert np.allclose(influences, 1.0)
    
    def test_gpu_noise_stability_fallback(self):
        """gpu_noise_stability should work without GPU."""
        f = bf.dictator(4, 0)
        fourier = f.fourier()
        
        stab = gpu.gpu_noise_stability(fourier, 0.9)
        
        # Dictator: Stab_rho = rho
        assert abs(stab - 0.9) < 0.01
    
    def test_gpu_spectral_weights_fallback(self):
        """gpu_spectral_weight_by_degree should work without GPU."""
        f = bf.parity(4)
        fourier = f.fourier()
        
        weights = gpu.gpu_spectral_weight_by_degree(fourier)
        
        # Parity: all weight at degree n
        assert len(weights) == 5  # degrees 0, 1, 2, 3, 4
        assert weights[4] > 0.99  # All weight at degree 4


class TestGPUBooleanFunctionOps:
    """Test the GPUBooleanFunctionOps class."""
    
    def test_init(self):
        """Should initialize correctly."""
        tt = [0, 0, 0, 1]  # AND of 2 vars
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        assert ops.n == 2
        assert len(ops.truth_table) == 4
    
    def test_pm_values(self):
        """pm_values should convert to ±1 correctly.
        
        Convention: True (1) → +1, False (0) → -1
        """
        tt = [0, 1, 1, 0]
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        pm = ops.pm_values
        
        # 0 → -1, 1 → +1
        assert np.array_equal(pm, [-1, 1, 1, -1])
    
    def test_fourier(self):
        """fourier() should compute coefficients correctly."""
        # XOR function
        tt = [0, 1, 1, 0]  # x0 XOR x1
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        fourier = ops.fourier()
        
        # XOR has all weight at degree 2 (coefficient at some degree-2 subset)
        # The specific index depends on bit ordering
        assert np.sum(fourier ** 2) > 0.9  # Parseval: sum = 1
    
    def test_fourier_caching(self):
        """fourier() should cache results."""
        tt = [0, 1, 1, 0]
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        fourier1 = ops.fourier()
        fourier2 = ops.fourier()
        
        assert fourier1 is fourier2  # Same object
    
    def test_influences(self):
        """influences() should compute correctly."""
        # Dictator on first variable
        tt = [0, 1, 0, 1]  # x0
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        inf = ops.influences()
        
        # First variable has influence 1, second has 0
        # (exact values depend on bit ordering)
        assert max(inf) > 0.9
    
    def test_noise_stability(self):
        """noise_stability() should compute correctly."""
        # Use parity for predictable degree
        f = bf.parity(3)
        tt = list(f.get_representation("truth_table"))
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        stab = ops.noise_stability(0.5)
        
        # Parity on 3 vars has degree 3, so Stab_0.5 = 0.5^3 = 0.125
        assert abs(stab - 0.125) < 0.01
    
    def test_spectral_weights(self):
        """spectral_weights() should compute correctly."""
        # Use parity for predictable weights
        f = bf.parity(3)
        tt = list(f.get_representation("truth_table"))
        ops = gpu.GPUBooleanFunctionOps(tt)
        
        weights = ops.spectral_weights()
        
        # Parity has all weight at degree n=3
        assert weights[3] > 0.99


class TestArrayTransfer:
    """Test GPU/CPU array transfer functions."""
    
    def test_to_gpu_without_cupy(self):
        """to_gpu should return input when GPU not available."""
        arr = np.array([1, 2, 3])
        result = gpu.to_gpu(arr)
        
        assert isinstance(result, np.ndarray)
    
    def test_to_cpu_with_numpy(self):
        """to_cpu should handle numpy arrays."""
        arr = np.array([1, 2, 3])
        result = gpu.to_cpu(arr)
        
        assert np.array_equal(result, arr)


class TestCorrectness:
    """Test that GPU results match CPU results."""
    
    def test_wht_matches_direct_computation(self):
        """WHT should match direct coefficient computation."""
        f = bf.majority(5)
        
        # Get Fourier via library
        fourier_lib = f.fourier()
        
        # Get Fourier via GPU module
        ops = gpu.GPUBooleanFunctionOps(
            list(f.get_representation("truth_table"))
        )
        fourier_gpu = ops.fourier()
        
        # Should match
        assert np.allclose(fourier_lib, fourier_gpu)
    
    def test_influences_match(self):
        """GPU influences should match library influences."""
        f = bf.majority(5)
        
        lib_inf = f.influences()
        
        ops = gpu.GPUBooleanFunctionOps(
            list(f.get_representation("truth_table"))
        )
        gpu_inf = ops.influences()
        
        assert np.allclose(lib_inf, gpu_inf)
    
    def test_noise_stability_matches(self):
        """GPU noise stability should match library."""
        f = bf.majority(5)
        
        lib_stab = f.noise_stability(0.7)
        
        ops = gpu.GPUBooleanFunctionOps(
            list(f.get_representation("truth_table"))
        )
        gpu_stab = ops.noise_stability(0.7)
        
        assert abs(lib_stab - gpu_stab) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
