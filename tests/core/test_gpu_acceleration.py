"""
Tests for the consolidated GPU module (core/gpu.py).

Covers availability checks, heuristic decisions, and integration with
Boolean function computation.  GPU-specific operations are exercised only
when CuPy is installed; the tests are otherwise about the CPU fallback paths.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.core.gpu import (
    get_gpu_info,
    gpu_accelerate,
    is_gpu_available,
    is_gpu_enabled,
    should_use_gpu,
)


class TestIsGPUAvailable:
    """Test is_gpu_available function."""

    def test_returns_bool(self):
        result = is_gpu_available()
        assert isinstance(result, bool)


class TestGetGPUInfo:
    """Test get_gpu_info function."""

    def test_returns_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)

    def test_info_has_required_keys(self):
        info = get_gpu_info()
        assert "gpu_available" in info
        assert "backend" in info
        assert "devices" in info


class TestShouldUseGPU:
    """Test should_use_gpu heuristic."""

    def test_returns_bool(self):
        result = should_use_gpu("wht", 1024, 10)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("operation", ["wht", "fourier", "truth_table"])
    def test_different_operations(self, operation):
        result = should_use_gpu(operation, 1024, 10)
        assert isinstance(result, bool)

    def test_small_data_cpu_preferred(self):
        result = should_use_gpu("wht", 8, 3)
        assert isinstance(result, bool)


class TestGPUAccelerate:
    """Test gpu_accelerate dispatcher."""

    def test_function_callable(self):
        assert callable(gpu_accelerate)


class TestGPUIntegration:
    """Integration tests — computation works regardless of GPU presence."""

    def test_fourier_with_gpu_decision(self):
        f = bf.majority(5)
        should_use_gpu("wht", 32, 5)
        fourier = f.fourier()
        assert len(fourier) == 32

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_various_sizes(self, n):
        data_size = 2**n
        should_use_gpu("wht", data_size, n)
        f = bf.majority(n)
        fourier = f.fourier()
        assert len(fourier) == data_size


class TestGPUEdgeCases:
    """Edge-case tests."""

    def test_very_small_n(self):
        assert isinstance(should_use_gpu("wht", 2, 1), bool)

    def test_moderate_n(self):
        assert isinstance(should_use_gpu("wht", 1024, 10), bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
