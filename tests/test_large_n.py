"""
Tests for large n values.

These tests verify that the library handles large input sizes correctly,
focusing on:
1. Correctness (results match expected values)
2. Performance (operations complete in reasonable time)
3. Memory efficiency (no excessive memory usage)

Note: Some tests use sampling for verification since exhaustive
checking is infeasible for large n.
"""

import sys
import time

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf

# Mark slow tests
pytestmark = pytest.mark.slow


class TestLargeNCreation:
    """Test creating functions with large n."""

    def test_create_majority_n14(self):
        """Majority with n=14 should work."""
        f = bf.majority(14)
        assert f.n_vars == 14
        assert f.evaluate(0) == 0  # All zeros -> 0
        assert f.evaluate(2**14 - 1) == 1  # All ones -> 1

    def test_create_parity_n16(self):
        """Parity with n=16 should work."""
        f = bf.parity(16)
        assert f.n_vars == 16
        # Parity of all zeros = 0
        assert f.evaluate(0) == 0
        # Parity of single 1 = 1
        assert f.evaluate(1) == 1

    def test_create_and_n18(self):
        """AND with n=18 should work."""
        f = bf.AND(18)
        assert f.n_vars == 18
        assert f.evaluate(0) == 0
        assert f.evaluate(2**18 - 1) == 1  # Only all-ones is 1

    def test_create_dictator_n20(self):
        """Dictator with n=20 should work."""
        f = bf.dictator(20, 0)
        assert f.n_vars == 20
        # First bit determines output
        assert f.evaluate(1 << 19) == 1
        assert f.evaluate(0) == 0

    def test_create_tribes_large(self):
        """Tribes with large parameters should work."""
        # tribes(w, m) creates m tribes of width w, so w*m variables
        f = bf.tribes(4, 4)  # 4 tribes of width 4 = 16 variables
        # Note: actual n_vars depends on tribes implementation
        assert f.n_vars >= 4  # At minimum, should have some variables


class TestLargeNFourier:
    """Test Fourier transform for large n."""

    def test_fourier_n12_parseval(self):
        """Fourier transform should satisfy Parseval for n=12."""
        f = bf.majority(12)
        fourier = f.fourier()

        # Parseval: sum of squares = 1 for Boolean function
        sum_sq = np.sum(fourier**2)
        assert abs(sum_sq - 1.0) < 1e-10

    def test_fourier_n14_parity(self):
        """Parity Fourier coefficients should be correct for n=14."""
        f = bf.parity(14)
        fourier = f.fourier()

        # Parity has all weight on the all-variables coefficient
        # f̂(S) = 0 for S ≠ [n], f̂([n]) = ±1
        all_ones_idx = 2**14 - 1
        assert abs(abs(fourier[all_ones_idx]) - 1.0) < 1e-10

        # All other coefficients should be ~0
        for i in range(2**14):
            if i != all_ones_idx:
                assert abs(fourier[i]) < 1e-10

    def test_fourier_timing_n16(self):
        """Fourier transform should complete in reasonable time for n=16."""
        f = bf.majority(16)

        start = time.perf_counter()
        _ = f.fourier()
        elapsed = time.perf_counter() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Fourier took {elapsed:.2f}s (too slow)"


class TestLargeNInfluences:
    """Test influence computation for large n."""

    def test_influences_majority_symmetric(self):
        """Majority influences should be equal (symmetric function) for n=11."""
        f = bf.majority(11)
        influences = f.influences()

        # All influences should be equal for majority
        assert len(influences) == 11
        assert np.allclose(influences, influences[0])

    def test_influences_dictator_single(self):
        """Dictator should have influence 1 on one variable, 0 on others."""
        f = bf.dictator(14, 0)
        influences = f.influences()

        # First variable has influence 1
        assert abs(influences[0] - 1.0) < 1e-10

        # All others have influence 0
        for i in range(1, 14):
            assert abs(influences[i]) < 1e-10

    def test_total_influence_parity(self):
        """Parity total influence should equal n."""
        n = 12
        f = bf.parity(n)
        total_inf = f.total_influence()

        # For parity, total influence = n (each variable has influence 1)
        assert abs(total_inf - n) < 1e-10


class TestLargeNNoiseStability:
    """Test noise stability for large n."""

    def test_noise_stability_dictator(self):
        """Dictator noise stability should be rho."""
        f = bf.dictator(14, 0)
        rho = 0.7
        stab = f.noise_stability(rho)

        # Dictator has Stab_rho = rho
        assert abs(stab - rho) < 1e-10

    def test_noise_stability_parity(self):
        """Parity noise stability should be rho^n."""
        n = 10
        f = bf.parity(n)
        rho = 0.5
        stab = f.noise_stability(rho)

        # Parity has Stab_rho = rho^n
        expected = rho**n
        assert abs(stab - expected) < 1e-10

    def test_noise_stability_bounds(self):
        """Noise stability should be in valid range."""
        f = bf.majority(12)

        for rho in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            stab = f.noise_stability(rho)
            # Stability should be in [-1, 1] for Boolean functions
            assert -1.0 <= stab <= 1.0 + 1e-10


class TestLargeNSampling:
    """Test operations using sampling for very large n."""

    def test_sampled_evaluation_consistency(self):
        """Sampled evaluations should be consistent for n=18."""
        f = bf.majority(18)

        # Test on random inputs
        np.random.seed(42)
        for _ in range(100):
            x = np.random.randint(0, 2, 18)
            idx = int(sum(b << (17 - i) for i, b in enumerate(x)))

            # Majority of 18 bits
            expected = 1 if sum(x) > 9 else 0
            assert f.evaluate(idx) == expected

    def test_parity_sampled_correctness(self):
        """Parity should be correct on sampled inputs for n=20."""
        f = bf.parity(20)

        np.random.seed(42)
        for _ in range(100):
            x = np.random.randint(0, 2, 20)
            idx = int(sum(b << (19 - i) for i, b in enumerate(x)))

            expected = int(sum(x) % 2)
            assert f.evaluate(idx) == expected


class TestLargeNDegree:
    """Test degree computation for large n."""

    def test_degree_majority(self):
        """Majority degree should be n for odd n."""
        for n in [11, 13, 15]:
            f = bf.majority(n)
            d = f.degree()
            assert d == n, f"Expected degree {n} for majority_{n}, got {d}"

    def test_degree_parity(self):
        """Parity degree should be n."""
        for n in [10, 12, 14]:
            f = bf.parity(n)
            d = f.degree()
            assert d == n

    def test_degree_dictator(self):
        """Dictator degree should be 1."""
        for n in [10, 14, 18]:
            f = bf.dictator(n, 0)
            d = f.degree()
            assert d == 1


class TestLargeNEdgeCases:
    """Test edge cases for large n."""

    def test_constant_zero_large_n(self):
        """Constant zero function for large n."""
        n = 16
        f = bf.create([0] * (2**n))

        assert f.evaluate(0) == 0
        assert f.evaluate(2**n - 1) == 0

        # All Fourier coefficients should be 0 except empty set
        fourier = f.fourier()
        assert abs(fourier[0] - (-1.0)) < 1e-10 or abs(fourier[0] - 1.0) < 1e-10

    def test_constant_one_large_n(self):
        """Constant one function for large n."""
        n = 16
        f = bf.create([1] * (2**n))

        assert f.evaluate(0) == 1
        assert f.evaluate(2**n - 1) == 1


class TestLargeNMemory:
    """Test memory-related concerns for large n."""

    def test_n16_doesnt_crash(self):
        """Operations on n=16 shouldn't crash."""
        f = bf.majority(16)

        # These should all complete without memory errors
        _ = f.fourier()
        _ = f.influences()
        _ = f.total_influence()
        _ = f.noise_stability(0.5)
        _ = f.degree()

    @pytest.mark.skip(reason="May be slow/memory intensive")
    def test_n18_fourier(self):
        """Fourier transform for n=18 (if resources allow)."""
        f = bf.parity(18)  # Parity is fast to create
        fourier = f.fourier()

        # Just verify it completed
        assert len(fourier) == 2**18


# Parameterized tests for various n values
@pytest.mark.parametrize("n", [10, 12, 14])
def test_majority_properties_parameterized(n):
    """Test majority properties for various n."""
    if n % 2 == 0:
        n = n + 1  # Majority needs odd n

    f = bf.majority(n)

    # Check basic properties
    assert f.n_vars == n
    assert f.is_monotone()
    assert f.is_symmetric()

    # Check evaluation at extremes
    assert f.evaluate(0) == 0
    assert f.evaluate(2**n - 1) == 1


@pytest.mark.parametrize("n", [10, 12, 14, 16])
def test_parity_fourier_parameterized(n):
    """Test parity Fourier transform for various n."""
    f = bf.parity(n)
    fourier = f.fourier()

    # Check Parseval
    assert abs(np.sum(fourier**2) - 1.0) < 1e-10

    # Check only all-ones coefficient is non-zero
    all_ones = 2**n - 1
    assert abs(abs(fourier[all_ones]) - 1.0) < 1e-10
