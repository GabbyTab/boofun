"""
Tests for numerical precision and floating-point edge cases.

These tests verify that the library handles floating-point arithmetic
correctly and maintains precision in edge cases.
"""

import numpy as np
import pytest

import boofun as bf


class TestFourierPrecision:
    """Test precision of Fourier computations."""

    def test_parseval_identity_exact(self):
        """Parseval's identity: sum of squared coefficients = 1."""
        for n in [3, 4, 5, 6]:
            f = bf.random(n, seed=42 + n)
            coeffs = f.fourier()
            parseval_sum = np.sum(coeffs**2)

            # Should be exactly 1.0 (within machine epsilon)
            assert abs(parseval_sum - 1.0) < 1e-14, (
                f"Parseval failed for n={n}: sum={parseval_sum}, error={abs(parseval_sum - 1.0)}"
            )

    def test_parseval_constant_function(self):
        """Constant function has f̂(∅) = ±1, all others 0."""
        for val in [0, 1]:
            f = bf.create([val] * 8)  # Constant function on 3 vars
            coeffs = f.fourier()

            # f̂(∅) should be ±1
            expected_empty = 1.0 if val == 0 else -1.0  # In ±1 convention
            assert abs(coeffs[0] - expected_empty) < 1e-14

            # All other coefficients should be exactly 0
            for i in range(1, len(coeffs)):
                assert abs(coeffs[i]) < 1e-14, f"Non-zero coefficient at {i}: {coeffs[i]}"

    def test_parity_single_nonzero_coefficient(self):
        """Parity has only one non-zero coefficient (the full set)."""
        for n in [3, 4, 5, 6]:
            f = bf.parity(n)
            coeffs = f.fourier()

            full_set = (1 << n) - 1  # All bits set
            for i in range(len(coeffs)):
                if i == full_set:
                    assert abs(abs(coeffs[i]) - 1.0) < 1e-14, (
                        f"Parity({n}) f̂(full_set) should be ±1, got {coeffs[i]}"
                    )
                else:
                    assert abs(coeffs[i]) < 1e-14, (
                        f"Parity({n}) f̂({i}) should be 0, got {coeffs[i]}"
                    )

    def test_dictator_single_nonzero_coefficient(self):
        """Dictator has only f̂({i}) = 1."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                coeffs = f.fourier()

                expected_idx = 1 << i
                for j in range(len(coeffs)):
                    if j == expected_idx:
                        assert abs(coeffs[j] - 1.0) < 1e-14, (
                            f"Dictator({n}, {i}) f̂({j}) should be 1, got {coeffs[j]}"
                        )
                    else:
                        assert abs(coeffs[j]) < 1e-14, (
                            f"Dictator({n}, {i}) f̂({j}) should be 0, got {coeffs[j]}"
                        )


class TestInfluencePrecision:
    """Test precision of influence computations."""

    def test_total_influence_equals_sum(self):
        """Total influence should equal sum of individual influences exactly."""
        for n in [3, 4, 5, 6]:
            f = bf.random(n, seed=100 + n)
            infs = f.influences()
            total = f.total_influence()

            assert abs(np.sum(infs) - total) < 1e-14, (
                f"Influence sum mismatch for n={n}: sum={np.sum(infs)}, total={total}"
            )

    def test_dictator_influence_exact(self):
        """Dictator has Inf_i = 1 and Inf_j = 0 for j ≠ i."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                infs = f.influences()

                for j in range(n):
                    expected = 1.0 if j == i else 0.0
                    assert abs(infs[j] - expected) < 1e-14, (
                        f"Dictator({n}, {i}) Inf[{j}] should be {expected}, got {infs[j]}"
                    )

    def test_parity_uniform_influence(self):
        """Parity has Inf_i = 1 for all i."""
        for n in [3, 4, 5]:
            f = bf.parity(n)
            infs = f.influences()

            for i in range(n):
                assert abs(infs[i] - 1.0) < 1e-14, (
                    f"Parity({n}) Inf[{i}] should be 1, got {infs[i]}"
                )


class TestNoiseStabilityPrecision:
    """Test precision of noise stability computations."""

    def test_noise_stability_at_zero(self):
        """Stab_0[f] = E[f]^2 for any f."""
        for n in [3, 4, 5]:
            f = bf.random(n, seed=200 + n)
            stability = f.noise_stability(0.0)
            # f̂(∅) = E[f] in the ±1 convention
            expectation = f.fourier()[0]

            expected = expectation**2
            assert abs(stability - expected) < 1e-14, (
                f"Stab_0 should be E[f]^2 = {expected}, got {stability}"
            )

    def test_noise_stability_at_one(self):
        """Stab_1[f] = 1 for any f."""
        for n in [3, 4, 5]:
            f = bf.random(n, seed=300 + n)
            stability = f.noise_stability(1.0)

            assert abs(stability - 1.0) < 1e-14, (
                f"Stab_1 should be 1, got {stability}"
            )

    def test_noise_stability_symmetry(self):
        """Stab_rho is a polynomial in rho, test at several points."""
        f = bf.majority(5)

        # Stability should be smooth
        rhos = [0.0, 0.25, 0.5, 0.75, 1.0]
        stabs = [f.noise_stability(rho) for rho in rhos]

        # All values should be in [-1, 1]
        for rho, stab in zip(rhos, stabs):
            assert -1.0 - 1e-14 <= stab <= 1.0 + 1e-14, (
                f"Stab_{rho} = {stab} out of range"
            )

        # Should be monotonically increasing (for most functions)
        # This is true for majority
        for i in range(len(stabs) - 1):
            assert stabs[i] <= stabs[i + 1] + 1e-10, (
                f"Stability not monotonic: Stab_{rhos[i]} = {stabs[i]} > Stab_{rhos[i+1]} = {stabs[i+1]}"
            )


class TestSubtractiveCancellation:
    """Test that subtractive cancellation doesn't cause precision loss."""

    def test_nearly_balanced_function(self):
        """Test function that's almost balanced (potential cancellation)."""
        # Create a function with 8 ones and 8 zeros (balanced)
        tt = [0, 1] * 8
        f = bf.create(tt)

        # f̂(∅) = E[f] should be exactly 0 for balanced in ±1 convention
        exp = f.fourier()[0]
        assert abs(exp) < 1e-14, f"Balanced function expectation should be 0, got {exp}"

    def test_slightly_unbalanced(self):
        """Test slightly unbalanced function."""
        # 9 ones, 7 zeros out of 16
        tt = [0, 1] * 8
        tt[0] = 1  # Now 9 ones

        f = bf.create(tt)

        # E[f] = (7*(+1) + 9*(-1)) / 16 = -2/16 = -0.125 in ±1 convention
        # (0 maps to +1, 1 maps to -1)
        exp = f.fourier()[0]

        # Verify it's computed correctly (not lost to cancellation)
        assert abs(exp - (-0.125)) < 1e-14, (
            f"Expectation should be -0.125, got {exp}"
        )


class TestLargeFunctions:
    """Test precision with larger n (potential accumulation of errors)."""

    def test_fourier_parseval_large(self):
        """Parseval still holds for larger functions."""
        for n in [8, 10, 12]:
            f = bf.random(n, seed=n)
            coeffs = f.fourier()
            parseval_sum = np.sum(coeffs**2)

            # Tolerance scales slightly with problem size due to accumulated error
            tol = n * 1e-14
            assert abs(parseval_sum - 1.0) < tol, (
                f"Parseval failed for n={n}: sum={parseval_sum}, error={abs(parseval_sum - 1.0)}"
            )

    def test_influence_sum_large(self):
        """Influence sum equals total for larger functions."""
        for n in [8, 10, 12]:
            f = bf.random(n, seed=n + 100)
            infs = f.influences()
            total = f.total_influence()

            tol = n * 1e-14
            assert abs(np.sum(infs) - total) < tol, (
                f"Influence sum mismatch for n={n}: error={abs(np.sum(infs) - total)}"
            )


class TestEdgeCaseInputs:
    """Test edge cases in inputs."""

    def test_single_variable(self):
        """n=1 functions should work correctly."""
        # Two possible functions: constant 0, constant 1, x, not x
        f0 = bf.create([0, 0])  # Constant 0
        f1 = bf.create([1, 1])  # Constant 1
        fx = bf.create([0, 1])  # Identity (x)
        fn = bf.create([1, 0])  # Negation (not x)

        # Verify Fourier coefficients
        assert abs(f0.fourier()[0] - 1.0) < 1e-14  # f̂(∅) = 1
        assert abs(f1.fourier()[0] + 1.0) < 1e-14  # f̂(∅) = -1
        assert abs(fx.fourier()[1] - 1.0) < 1e-14  # f̂({0}) = 1
        assert abs(fn.fourier()[1] + 1.0) < 1e-14  # f̂({0}) = -1

    def test_all_zeros(self):
        """All-zeros truth table."""
        for n in [2, 3, 4]:
            tt = [0] * (1 << n)
            f = bf.create(tt)

            coeffs = f.fourier()
            # Only f̂(∅) should be non-zero (= 1)
            assert abs(coeffs[0] - 1.0) < 1e-14
            for i in range(1, len(coeffs)):
                assert abs(coeffs[i]) < 1e-14

    def test_all_ones(self):
        """All-ones truth table."""
        for n in [2, 3, 4]:
            tt = [1] * (1 << n)
            f = bf.create(tt)

            coeffs = f.fourier()
            # Only f̂(∅) should be non-zero (= -1 in ±1 convention)
            assert abs(coeffs[0] + 1.0) < 1e-14
            for i in range(1, len(coeffs)):
                assert abs(coeffs[i]) < 1e-14


class TestFloatingPointSpecialValues:
    """Test handling of special floating-point values."""

    def test_no_nan_in_fourier(self):
        """Fourier coefficients should never be NaN."""
        for n in [3, 4, 5]:
            f = bf.random(n, seed=n * 7)
            coeffs = f.fourier()

            assert not np.any(np.isnan(coeffs)), f"NaN found in Fourier coefficients for n={n}"

    def test_no_inf_in_fourier(self):
        """Fourier coefficients should never be infinite."""
        for n in [3, 4, 5]:
            f = bf.random(n, seed=n * 11)
            coeffs = f.fourier()

            assert not np.any(np.isinf(coeffs)), f"Inf found in Fourier coefficients for n={n}"

    def test_no_nan_in_influences(self):
        """Influences should never be NaN."""
        for n in [3, 4, 5]:
            f = bf.random(n, seed=n * 13)
            infs = f.influences()

            assert not np.any(np.isnan(infs)), f"NaN found in influences for n={n}"

    def test_influences_non_negative(self):
        """Influences should always be non-negative."""
        for n in [3, 4, 5, 6]:
            f = bf.random(n, seed=n * 17)
            infs = f.influences()

            # Allow tiny negative values due to floating point
            assert np.all(infs >= -1e-14), (
                f"Negative influence for n={n}: min={np.min(infs)}"
            )


class TestDegreeComputations:
    """Test precision of degree computations."""

    def test_constant_has_degree_zero(self):
        """Constant functions have degree 0."""
        f0 = bf.create([0, 0, 0, 0])
        f1 = bf.create([1, 1, 1, 1])

        assert f0.degree() == 0
        assert f1.degree() == 0

    def test_dictator_has_degree_one(self):
        """Dictator functions have degree 1."""
        for n in [2, 3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                assert f.degree() == 1, f"Dictator({n}, {i}) should have degree 1"

    def test_parity_has_full_degree(self):
        """Parity has degree n."""
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            assert f.degree() == n, f"Parity({n}) should have degree {n}"

    def test_and_has_full_degree(self):
        """AND has degree n."""
        for n in [2, 3, 4, 5]:
            f = bf.AND(n)
            assert f.degree() == n, f"AND({n}) should have degree {n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
