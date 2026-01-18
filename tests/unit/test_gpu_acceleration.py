"""
Tests for GPU acceleration module.

Tests the GPU infrastructure without requiring actual GPU hardware.
Tests focus on:
- Device detection and management
- Heuristics for when to use GPU
- Graceful fallback behavior
- Module-level API functions
"""

import pytest
import numpy as np

from boolfunc.core.gpu_acceleration import (
    GPUDevice,
    GPUManager,
    is_gpu_available,
    get_gpu_info,
    should_use_gpu,
    set_gpu_backend,
    HAS_CUPY,
    HAS_NUMBA_CUDA,
)


class TestGPUDevice:
    """Tests for GPUDevice class."""
    
    def test_device_creation(self):
        """Can create a GPUDevice instance."""
        device = GPUDevice(
            device_id=0,
            name="Test GPU",
            memory_gb=8.0,
            compute_capability="7.5",
            backend="test"
        )
        
        assert device.device_id == 0
        assert device.name == "Test GPU"
        assert device.memory_gb == 8.0
        assert device.compute_capability == "7.5"
        assert device.backend == "test"
        assert device.is_available is True
    
    def test_device_repr(self):
        """GPUDevice has readable repr."""
        device = GPUDevice(
            device_id=0,
            name="TestGPU",
            memory_gb=4.0,
            backend="cupy"
        )
        
        repr_str = repr(device)
        assert "TestGPU" in repr_str
        assert "4.0GB" in repr_str
        assert "cupy" in repr_str


class TestGPUManager:
    """Tests for GPUManager class."""
    
    def test_manager_initialization(self):
        """GPUManager initializes without error."""
        manager = GPUManager()
        # Should not raise even without GPU
        assert isinstance(manager.accelerators, dict)
    
    def test_is_gpu_available_returns_bool(self):
        """is_gpu_available returns boolean."""
        manager = GPUManager()
        result = manager.is_gpu_available()
        assert isinstance(result, bool)
    
    def test_get_gpu_info_structure(self):
        """get_gpu_info returns expected structure."""
        manager = GPUManager()
        info = manager.get_gpu_info()
        
        assert "gpu_available" in info
        assert "active_backend" in info
        assert "available_backends" in info
        assert "devices" in info
        
        assert isinstance(info["gpu_available"], bool)
        assert isinstance(info["available_backends"], list)
        assert isinstance(info["devices"], list)
    
    def test_should_use_gpu_truth_table(self):
        """should_use_gpu returns correct heuristics for truth tables."""
        manager = GPUManager()
        
        # Small data - should not use GPU
        result_small = manager.should_use_gpu("truth_table", data_size=100, n_vars=5)
        
        # Large data - depends on GPU availability
        result_large = manager.should_use_gpu("truth_table", data_size=100000, n_vars=10)
        
        # If no GPU, both should be False
        if not manager.is_gpu_available():
            assert result_small is False
            assert result_large is False
        else:
            # If GPU available, small should still be False
            assert result_small is False
    
    def test_should_use_gpu_fourier(self):
        """should_use_gpu returns correct heuristics for Fourier."""
        manager = GPUManager()
        
        # Small data
        result_small = manager.should_use_gpu("fourier", data_size=100, n_vars=3)
        
        # If no GPU, should be False
        if not manager.is_gpu_available():
            assert result_small is False
    
    def test_should_use_gpu_walsh_hadamard(self):
        """should_use_gpu returns correct heuristics for Walsh-Hadamard."""
        manager = GPUManager()
        
        # Small n
        result_small = manager.should_use_gpu("walsh_hadamard", data_size=256, n_vars=8)
        
        # If no GPU, should be False
        if not manager.is_gpu_available():
            assert result_small is False
    
    def test_clear_cache(self):
        """clear_cache works without error."""
        manager = GPUManager()
        manager.performance_cache["test"] = "value"
        manager.clear_cache()
        assert len(manager.performance_cache) == 0


class TestModuleFunctions:
    """Tests for module-level API functions."""
    
    def test_is_gpu_available_function(self):
        """Module-level is_gpu_available works."""
        result = is_gpu_available()
        assert isinstance(result, bool)
    
    def test_get_gpu_info_function(self):
        """Module-level get_gpu_info works."""
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "gpu_available" in info
    
    def test_should_use_gpu_function(self):
        """Module-level should_use_gpu works."""
        result = should_use_gpu("truth_table", data_size=100, n_vars=5)
        assert isinstance(result, bool)
    
    def test_set_gpu_backend_auto(self):
        """set_gpu_backend('auto') works without error."""
        # Should not raise
        set_gpu_backend('auto')
    
    def test_set_gpu_backend_unknown(self):
        """set_gpu_backend with unknown backend warns."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_gpu_backend('nonexistent_backend')
            # Should produce a warning
            assert len(w) >= 1


class TestGPUHeuristics:
    """Tests for GPU usage heuristics."""
    
    def test_heuristics_consistency(self):
        """Heuristics are consistent across calls."""
        manager = GPUManager()
        
        # Same inputs should give same results
        result1 = manager.should_use_gpu("truth_table", 50000, 10)
        result2 = manager.should_use_gpu("truth_table", 50000, 10)
        
        assert result1 == result2
    
    def test_heuristics_data_size_threshold(self):
        """Larger data size increases likelihood of GPU use."""
        manager = GPUManager()
        
        if manager.is_gpu_available():
            # Large data should trigger GPU for truth_table (threshold is 10000)
            result_large = manager.should_use_gpu("truth_table", 20000, 5)
            result_small = manager.should_use_gpu("truth_table", 100, 5)
            
            # Large should be True, small should be False
            assert result_large is True
            assert result_small is False


class TestGPUAcceleratorInterface:
    """Tests for GPU accelerator interface."""
    
    def test_cupy_availability_flag(self):
        """HAS_CUPY flag is boolean."""
        assert isinstance(HAS_CUPY, bool)
    
    def test_numba_availability_flag(self):
        """HAS_NUMBA_CUDA flag is boolean."""
        assert isinstance(HAS_NUMBA_CUDA, bool)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestCuPyAccelerator:
    """Tests for CuPy accelerator (only run if CuPy available)."""
    
    def test_cupy_accelerator_available(self):
        """CuPy accelerator reports availability correctly."""
        from boolfunc.core.gpu_acceleration import CuPyAccelerator
        
        accel = CuPyAccelerator()
        # If we got here, CuPy is installed
        assert isinstance(accel.is_available(), bool)
    
    def test_cupy_devices_list(self):
        """CuPy can list devices."""
        from boolfunc.core.gpu_acceleration import CuPyAccelerator
        
        accel = CuPyAccelerator()
        if accel.is_available():
            devices = accel.get_devices()
            assert isinstance(devices, list)


@pytest.mark.skipif(not HAS_NUMBA_CUDA, reason="Numba CUDA not available")
class TestNumbaAccelerator:
    """Tests for Numba CUDA accelerator (only run if available)."""
    
    def test_numba_accelerator_available(self):
        """Numba accelerator reports availability correctly."""
        from boolfunc.core.gpu_acceleration import NumbaAccelerator
        
        accel = NumbaAccelerator()
        assert isinstance(accel.is_available(), bool)


class TestCPUFallback:
    """Tests for CPU fallback behavior."""
    
    def test_no_gpu_returns_false(self):
        """When no GPU available, should_use_gpu returns False."""
        # Create a manager and temporarily disable all accelerators
        manager = GPUManager()
        original = manager.active_accelerator
        manager.active_accelerator = None
        
        result = manager.should_use_gpu("truth_table", 100000, 10)
        assert result is False
        
        # Restore
        manager.active_accelerator = original
    
    def test_accelerate_without_gpu_raises(self):
        """accelerate_operation raises when no GPU."""
        manager = GPUManager()
        original = manager.active_accelerator
        manager.active_accelerator = None
        
        with pytest.raises(RuntimeError, match="No GPU"):
            manager.accelerate_operation("truth_table_batch", np.array([1, 2, 3]))
        
        # Restore
        manager.active_accelerator = original
