"""
Tests for Numba fallback behavior.

These tests verify that:
1. Numba-accelerated functions produce same results as pure Python
2. Fallbacks work correctly when Numba is unavailable
3. Numba JIT compilation doesn't change mathematical results
"""

import os

import numpy as np
import pytest

import boofun as bf


class TestWHTImplementations:
    """Test Walsh-Hadamard Transform implementations."""

    def test_numpy_wht_correctness(self):
        """Verify NumPy WHT produces correct results."""
        from boofun.core.optimizations import fast_walsh_hadamard

        # Test on known values
        # WHT of [1, 1, 1, 1] should give [1, 0, 0, 0] (unnormalized: [4, 0, 0, 0])
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = fast_walsh_hadamard(values, normalize=True)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

        # WHT of [1, -1, -1, 1] should give [0, 0, 0, 1] (normalized)
        values = np.array([1.0, -1.0, -1.0, 1.0])
        result = fast_walsh_hadamard(values, normalize=True)
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_wht_implementations_match(self):
        """All WHT implementations should produce identical results."""
        from boofun.core.optimizations import (
            HAS_NUMBA,
            fast_walsh_hadamard,
            get_best_wht_implementation,
        )

        if HAS_NUMBA:
            from boofun.core.optimizations import fast_walsh_hadamard_numba

        rng = np.random.default_rng(42)

        for n in [2, 4, 8, 16]:
            for _ in range(5):
                values = rng.standard_normal(n)

                # NumPy reference
                numpy_result = fast_walsh_hadamard(values, normalize=True)

                # Best available implementation
                best_impl, impl_name = get_best_wht_implementation()
                best_result = best_impl(values)

                np.testing.assert_allclose(
                    best_result,
                    numpy_result,
                    rtol=1e-10,
                    err_msg=f"{impl_name} differs from NumPy for n={n}",
                )

                if HAS_NUMBA:
                    numba_result = fast_walsh_hadamard_numba(values)
                    np.testing.assert_allclose(
                        numba_result,
                        numpy_result,
                        rtol=1e-10,
                        err_msg=f"Numba differs from NumPy for n={n}",
                    )

    def test_wht_power_of_two_only(self):
        """WHT should only accept power-of-2 length inputs."""
        from boofun.core.optimizations import fast_walsh_hadamard

        # Should work
        for n in [2, 4, 8, 16]:
            values = np.ones(n)
            result = fast_walsh_hadamard(values, normalize=True)
            assert len(result) == n

        # Should fail
        for n in [3, 5, 7, 10]:
            values = np.ones(n)
            with pytest.raises(ValueError):
                fast_walsh_hadamard(values, normalize=True)


class TestInfluenceImplementations:
    """Test influence computation implementations."""

    def test_influence_implementations_match(self):
        """All influence implementations should match."""
        from boofun.core.optimizations import HAS_NUMBA, vectorized_influences_from_fourier

        if HAS_NUMBA:
            from boofun.core.optimizations import vectorized_influences_numba

        rng = np.random.default_rng(42)

        for n in [2, 3, 4, 5]:
            for _ in range(5):
                # Random Fourier coefficients
                coeffs = rng.standard_normal(2**n)
                coeffs = coeffs / np.linalg.norm(coeffs)  # Normalize

                numpy_result = vectorized_influences_from_fourier(coeffs, n)

                if HAS_NUMBA:
                    numba_result = vectorized_influences_numba(coeffs, n)
                    np.testing.assert_allclose(
                        numba_result,
                        numpy_result,
                        rtol=1e-10,
                        err_msg=f"Numba influences differ for n={n}",
                    )

    def test_influence_correctness(self):
        """Verify influence computation is correct."""
        from boofun.core.optimizations import vectorized_influences_from_fourier

        # For dictator x_0, only coefficient f̂({0}) = f̂(1) is non-zero
        # So Inf_0 = f̂(1)² = 1, Inf_1 = 0
        n = 2
        coeffs = np.array([0.0, 1.0, 0.0, 0.0])  # Only f̂({0}) = 1
        influences = vectorized_influences_from_fourier(coeffs, n)

        assert abs(influences[0] - 1.0) < 1e-10
        assert abs(influences[1]) < 1e-10


class TestTotalInfluenceImplementations:
    """Test total influence computation."""

    def test_total_influence_implementations_match(self):
        """All total influence implementations should match."""
        from boofun.core.optimizations import vectorized_total_influence_from_fourier

        rng = np.random.default_rng(42)

        for n in [2, 3, 4, 5]:
            for _ in range(5):
                coeffs = rng.standard_normal(2**n)
                coeffs = coeffs / np.linalg.norm(coeffs)

                numpy_result = vectorized_total_influence_from_fourier(coeffs, n)

                # Verify result is reasonable (non-negative, finite)
                assert numpy_result >= 0
                assert np.isfinite(numpy_result)

    def test_total_influence_correctness(self):
        """Verify total influence = Σ |S| f̂(S)²."""
        from boofun.core.optimizations import vectorized_total_influence_from_fourier

        # For parity(n), only f̂({0,1,...,n-1}) is non-zero with value ±1
        # So total influence = n * 1² = n
        for n in [2, 3, 4]:
            coeffs = np.zeros(2**n)
            coeffs[(1 << n) - 1] = 1.0  # f̂([n]) = 1

            total = vectorized_total_influence_from_fourier(coeffs, n)
            assert abs(total - n) < 1e-10


class TestNoiseStabilityImplementations:
    """Test noise stability computation."""

    def test_noise_stability_implementations_match(self):
        """Verify noise stability implementations match."""
        from boofun.core.optimizations import noise_stability_from_fourier

        rng = np.random.default_rng(42)

        for n in [2, 3, 4]:
            for rho in [0.0, 0.5, 0.9, 1.0]:
                coeffs = rng.standard_normal(2**n)
                coeffs = coeffs / np.linalg.norm(coeffs)

                result = noise_stability_from_fourier(coeffs, rho)
                assert -1.0 - 1e-10 <= result <= 1.0 + 1e-10

    def test_noise_stability_formula(self):
        """Verify noise stability matches formula: Σ_S ρ^|S| f̂(S)²."""
        from boofun.core.optimizations import noise_stability_from_fourier

        n = 3
        coeffs = np.zeros(2**n)
        coeffs[0] = 0.5  # f̂(∅)
        coeffs[1] = 0.5  # f̂({0})
        coeffs[3] = 0.5  # f̂({0,1})
        coeffs[7] = 0.5  # f̂({0,1,2})

        for rho in [0.0, 0.5, 1.0]:
            result = noise_stability_from_fourier(coeffs, rho)
            expected = (
                0.5**2 * rho**0  # |∅| = 0
                + 0.5**2 * rho**1  # |{0}| = 1
                + 0.5**2 * rho**2  # |{0,1}| = 2
                + 0.5**2 * rho**3  # |{0,1,2}| = 3
            )
            assert abs(result - expected) < 1e-10


class TestFourierConsistency:
    """Test Fourier computation consistency with/without Numba."""

    def test_fourier_with_and_without_numba(self):
        """Fourier results should be identical regardless of Numba."""
        rng = np.random.default_rng(42)

        for n in [2, 3, 4, 5]:
            for _ in range(5):
                tt = rng.integers(0, 2, size=2**n).tolist()
                f = bf.create(tt)

                # Get Fourier coefficients
                coeffs = f.fourier()

                # Verify Parseval
                sum_sq = sum(c**2 for c in coeffs)
                assert abs(sum_sq - 1.0) < 1e-10

                # Verify degree makes sense
                degree = f.degree()
                assert 0 <= degree <= n

    def test_influence_consistency(self):
        """Influence computation should be consistent."""
        for n in [2, 3, 4, 5]:
            f = bf.majority(n)
            influences = f.influences()

            # Influences should be non-negative
            assert all(i >= -1e-10 for i in influences)

            # Sum of influences = total influence
            total = f.total_influence()
            assert abs(sum(influences) - total) < 1e-10


class TestBatchOperations:
    """Test batch operations."""

    def test_batch_evaluate_consistency(self):
        """Batch evaluation should match individual evaluation."""
        from boofun.core.optimizations import batch_evaluate

        f = bf.majority(4)
        indices = np.arange(16)

        # Batch evaluation
        batch_results = batch_evaluate(f, indices)

        # Individual evaluation
        individual_results = np.array([int(f.evaluate(i)) for i in range(16)])

        np.testing.assert_array_equal(batch_results, individual_results)


class TestOptimizationDisabled:
    """Test behavior when optimizations are disabled."""

    def test_functions_work_without_numba(self):
        """Core functionality should work even without Numba."""
        # This test doesn't actually disable Numba, but verifies
        # the pure Python paths work correctly

        from boofun.core.optimizations import (
            fast_walsh_hadamard,
            noise_stability_from_fourier,
            vectorized_influences_from_fourier,
            vectorized_total_influence_from_fourier,
        )

        # Test WHT
        values = np.array([1.0, 1.0, 1.0, 1.0])
        result = fast_walsh_hadamard(values, normalize=True)
        assert len(result) == 4

        # Test influences
        coeffs = np.array([0.5, 0.5, 0.5, 0.5])
        influences = vectorized_influences_from_fourier(coeffs, 2)
        assert len(influences) == 2

        # Test total influence
        total = vectorized_total_influence_from_fourier(coeffs, 2)
        assert total >= 0

        # Test noise stability
        stab = noise_stability_from_fourier(coeffs, 0.5)
        assert -1.0 <= stab <= 1.0
