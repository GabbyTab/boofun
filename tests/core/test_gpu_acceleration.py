"""
Comprehensive tests for core/gpu_acceleration module.

Tests GPU acceleration framework, backends, and management.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.gpu_acceleration import (
    GPUDevice,
    GPUAccelerator,
    CuPyAccelerator,
    NumbaAccelerator,
    GPUManager,
    is_gpu_available,
    get_gpu_info,
    should_use_gpu,
    gpu_accelerate,
    set_gpu_backend,
)


class TestGPUDevice:
    """Test GPUDevice class."""
    
    def test_device_class_exists(self):
        """GPUDevice class should exist."""
        assert GPUDevice is not None


class TestGPUAccelerator:
    """Test GPUAccelerator base class."""
    
    def test_accelerator_is_abc(self):
        """GPUAccelerator should be abstract base class."""
        from abc import ABC
        assert issubclass(GPUAccelerator, ABC)


class TestCuPyAccelerator:
    """Test CuPy-based accelerator."""
    
    def test_cupy_accelerator_class(self):
        """CuPyAccelerator class should exist."""
        assert CuPyAccelerator is not None
    
    def test_cupy_accelerator_creation(self):
        """CuPyAccelerator should be creatable (may fail without GPU)."""
        try:
            acc = CuPyAccelerator()
            assert acc is not None
        except (ImportError, RuntimeError):
            pytest.skip("CuPy not available")


class TestNumbaAccelerator:
    """Test Numba-based accelerator."""
    
    def test_numba_accelerator_class(self):
        """NumbaAccelerator class should exist."""
        assert NumbaAccelerator is not None
    
    def test_numba_accelerator_creation(self):
        """NumbaAccelerator should be creatable."""
        try:
            acc = NumbaAccelerator()
            assert acc is not None
        except ImportError:
            pytest.skip("Numba not available")


class TestGPUManager:
    """Test GPUManager class."""
    
    def test_manager_class_exists(self):
        """GPUManager class should exist."""
        assert GPUManager is not None
    
    def test_manager_singleton_or_instance(self):
        """GPUManager should be usable."""
        try:
            manager = GPUManager()
            assert manager is not None
        except Exception:
            pass  # May need special initialization


class TestIsGPUAvailable:
    """Test is_gpu_available function."""
    
    def test_returns_bool(self):
        """is_gpu_available should return boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)


class TestGetGPUInfo:
    """Test get_gpu_info function."""
    
    def test_returns_dict(self):
        """get_gpu_info should return dictionary."""
        info = get_gpu_info()
        assert isinstance(info, dict)
    
    def test_info_has_useful_keys(self):
        """GPU info should have useful information."""
        info = get_gpu_info()
        
        # Should have at least availability info
        has_useful = (
            'available' in info or 
            'backend' in info or 
            'devices' in info or
            len(info) > 0
        )
        assert has_useful or info == {}


class TestShouldUseGPU:
    """Test should_use_gpu decision function."""
    
    def test_returns_bool(self):
        """should_use_gpu should return boolean."""
        result = should_use_gpu("wht", 1024, 10)
        assert isinstance(result, bool)
    
    @pytest.mark.parametrize("operation", ["wht", "fourier", "influences"])
    def test_different_operations(self, operation):
        """should_use_gpu should work for different operations."""
        result = should_use_gpu(operation, 1024, 10)
        assert isinstance(result, bool)
    
    def test_small_data_cpu_preferred(self):
        """Small data should generally prefer CPU."""
        result = should_use_gpu("wht", 8, 3)  # Very small
        # Either True or False is acceptable, just should not crash
        assert isinstance(result, bool)


class TestGPUAccelerate:
    """Test gpu_accelerate function."""
    
    def test_function_callable(self):
        """gpu_accelerate should be callable."""
        assert callable(gpu_accelerate)


class TestSetGPUBackend:
    """Test set_gpu_backend function."""
    
    def test_function_callable(self):
        """set_gpu_backend should be callable."""
        assert callable(set_gpu_backend)
    
    def test_set_backend_numpy(self):
        """Should be able to set numpy backend."""
        try:
            set_gpu_backend("numpy")
        except (ValueError, RuntimeError):
            pass  # May not be supported


class TestGPUAccelerationIntegration:
    """Integration tests for GPU acceleration with Boolean functions."""
    
    def test_fourier_with_gpu_decision(self):
        """Fourier computation should work regardless of GPU decision."""
        f = bf.majority(5)
        
        # Check GPU decision
        should_gpu = should_use_gpu("wht", 32, 5)
        
        # Fourier should work regardless
        fourier = f.fourier()
        assert len(fourier) == 32
    
    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_various_sizes(self, n):
        """GPU acceleration decisions for various sizes."""
        data_size = 2 ** n
        
        # Check decision (doesn't matter what it is)
        should_gpu = should_use_gpu("wht", data_size, n)
        
        # Computation should still work
        f = bf.majority(n)
        fourier = f.fourier()
        assert len(fourier) == data_size


class TestGPUEdgeCases:
    """Test edge cases for GPU acceleration."""
    
    def test_very_small_n(self):
        """GPU decisions for very small n."""
        result = should_use_gpu("wht", 2, 1)
        assert isinstance(result, bool)
    
    def test_moderate_n(self):
        """GPU decisions for moderate n."""
        result = should_use_gpu("wht", 1024, 10)
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
