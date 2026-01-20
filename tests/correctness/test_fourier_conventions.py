"""
Tests for Fourier transform conventions and edge cases.

BooFun uses the O'Donnell convention:
- Boolean domain {0, 1} maps to {+1, -1} via χ(x) = (-1)^x
- Fourier basis: χ_S(x) = ∏_{i∈S} (-1)^{x_i}
- Coefficients satisfy Parseval: Σ_S f̂(S)² = 𝔼[f²] = 𝔼[f] for Boolean f

These tests verify mathematical correctness and edge cases.
"""

import numpy as np
import pytest

import boofun as bf


class TestParsevalsIdentity:
    """Test Parseval's identity: Σ f̂(S)² = 𝔼[f²]."""

    def test_parseval_balanced_functions(self):
        """Balanced functions should have Σf̂(S)² = 1."""
        # XOR is balanced
        f = bf.parity(3)
        coeffs = f.fourier()
        sum_sq = sum(c**2 for c in coeffs)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval for parity(3): {sum_sq}"

        # Majority(3) is balanced
        f = bf.majority(3)
        coeffs = f.fourier()
        sum_sq = sum(c**2 for c in coeffs)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval for majority(3): {sum_sq}"

    def test_parseval_constant_functions(self):
        """Constant functions should have Σf̂(S)² = 1."""
        # Constant 0: f̂(∅) = 1
        f = bf.create([0, 0, 0, 0])
        coeffs = f.fourier()
        sum_sq = sum(c**2 for c in coeffs)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval for constant 0: {sum_sq}"

        # Constant 1: f̂(∅) = -1
        f = bf.create([1, 1, 1, 1])
        coeffs = f.fourier()
        sum_sq = sum(c**2 for c in coeffs)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval for constant 1: {sum_sq}"

    def test_parseval_dictators(self):
        """Dictator functions should have Σf̂(S)² = 1."""
        for n in [2, 3, 4]:
            for i in range(n):
                f = bf.dictator(n, i)
                coeffs = f.fourier()
                sum_sq = sum(c**2 for c in coeffs)
                assert abs(sum_sq - 1.0) < 1e-10, f"Parseval for dictator({n}, {i}): {sum_sq}"

    def test_parseval_random_functions(self):
        """Random functions should satisfy Parseval."""
        rng = np.random.default_rng(42)
        for n in [2, 3, 4, 5]:
            for _ in range(5):
                tt = rng.integers(0, 2, size=2**n).tolist()
                f = bf.create(tt)
                coeffs = f.fourier()
                sum_sq = sum(c**2 for c in coeffs)

                # E[f²] = E[f] for Boolean functions
                expected = sum(tt) / len(tt)
                expected_pm = 1 - 2 * expected  # Map to ±1 convention
                # In ±1 domain, E[f²] = 1 always
                assert abs(sum_sq - 1.0) < 1e-10, f"Parseval for random n={n}: {sum_sq}"


class TestFourierCoefficients:
    """Test specific Fourier coefficient values."""

    def test_constant_zero_coefficients(self):
        """Constant 0 function: f̂(∅)=1, others=0."""
        f = bf.create([0, 0, 0, 0])
        coeffs = f.fourier()

        assert abs(coeffs[0] - 1.0) < 1e-10
        for s in range(1, 4):
            assert abs(coeffs[s]) < 1e-10

    def test_constant_one_coefficients(self):
        """Constant 1 function: f̂(∅)=-1, others=0."""
        f = bf.create([1, 1, 1, 1])
        coeffs = f.fourier()

        assert abs(coeffs[0] - (-1.0)) < 1e-10
        for s in range(1, 4):
            assert abs(coeffs[s]) < 1e-10

    def test_dictator_coefficients(self):
        """Dictator x_i: f̂({i})=1, others=0."""
        for n in [2, 3]:
            for i in range(n):
                f = bf.dictator(n, i)
                coeffs = f.fourier()

                expected_idx = 1 << i
                for s in range(len(coeffs)):
                    if s == expected_idx:
                        assert abs(coeffs[s] - 1.0) < 1e-10
                    else:
                        assert abs(coeffs[s]) < 1e-10

    def test_parity_coefficients(self):
        """Parity: f̂([n])=±1, others=0."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            coeffs = f.fourier()

            full_set = (1 << n) - 1
            for s in range(len(coeffs)):
                if s == full_set:
                    # Could be +1 or -1 depending on sign convention
                    assert abs(abs(coeffs[s]) - 1.0) < 1e-10
                else:
                    assert abs(coeffs[s]) < 1e-10


class TestFourierEdgeCases:
    """Test edge cases in Fourier computation."""

    def test_n_equals_1(self):
        """Single-variable functions."""
        # Constant 0
        f0 = bf.create([0, 0])
        c0 = f0.fourier()
        assert len(c0) == 2
        assert abs(c0[0] - 1.0) < 1e-10

        # Constant 1
        f1 = bf.create([1, 1])
        c1 = f1.fourier()
        assert abs(c1[0] - (-1.0)) < 1e-10

        # Identity (x)
        fx = bf.create([0, 1])
        cx = fx.fourier()
        assert abs(cx[0]) < 1e-10
        assert abs(cx[1] - 1.0) < 1e-10

        # Negation (¬x)
        fnx = bf.create([1, 0])
        cnx = fnx.fourier()
        assert abs(cnx[0]) < 1e-10
        assert abs(cnx[1] - (-1.0)) < 1e-10

    def test_large_n_correctness(self):
        """Verify Fourier is correct for larger n."""
        n = 6
        f = bf.parity(n)
        coeffs = f.fourier()

        # Only coefficient at 2^n - 1 should be non-zero
        full_set = (1 << n) - 1
        non_zero_count = sum(1 for c in coeffs if abs(c) > 1e-10)
        assert non_zero_count == 1

        assert abs(abs(coeffs[full_set]) - 1.0) < 1e-10

    def test_fourier_inverse(self):
        """Fourier transform should be invertible."""
        for n in [2, 3, 4]:
            rng = np.random.default_rng(123 + n)
            tt = rng.integers(0, 2, size=2**n).tolist()
            f = bf.create(tt)

            # Get Fourier coefficients
            coeffs = f.fourier()

            # Manually reconstruct truth table
            reconstructed = []
            for x in range(2**n):
                val = 0.0
                for s in range(2**n):
                    chi = 1.0
                    for i in range(n):
                        if (s >> i) & 1:
                            chi *= 1 - 2 * ((x >> i) & 1)
                    val += coeffs[s] * chi
                # Convert from ±1 to {0,1}
                reconstructed.append(int(round((1 - val) / 2)))

            assert reconstructed == tt, f"n={n}: inverse failed"


class TestInfluenceFourierRelation:
    """Test the influence-Fourier relationship: Inf_i(f) = Σ_{S∋i} f̂(S)²."""

    def test_influence_from_fourier(self):
        """Influence should match sum of squared coefficients."""
        for n in [2, 3, 4]:
            rng = np.random.default_rng(42 + n)
            tt = rng.integers(0, 2, size=2**n).tolist()
            f = bf.create(tt)

            influences = f.influences()
            coeffs = f.fourier()

            for i in range(n):
                # Compute influence from Fourier
                influence_from_fourier = sum(
                    coeffs[s] ** 2 for s in range(len(coeffs)) if (s >> i) & 1
                )
                assert (
                    abs(influences[i] - influence_from_fourier) < 1e-10
                ), f"n={n}, i={i}: {influences[i]} vs {influence_from_fourier}"

    def test_total_influence_spectral(self):
        """Total influence = Σ_i Inf_i = Σ_S |S| f̂(S)²."""
        for n in [2, 3, 4]:
            f = bf.majority(n)
            influences = f.influences()
            total = sum(influences)

            coeffs = f.fourier()
            spectral_total = sum(bin(s).count("1") * coeffs[s] ** 2 for s in range(len(coeffs)))

            assert abs(total - spectral_total) < 1e-10


class TestNoiseStability:
    """Test noise stability computations."""

    def test_noise_stability_bounds(self):
        """Noise stability should be in [-1, 1]."""
        for n in [2, 3, 4]:
            for _ in range(5):
                rng = np.random.default_rng()
                tt = rng.integers(0, 2, size=2**n).tolist()
                f = bf.create(tt)

                for rho in [0.0, 0.5, 0.9, 1.0]:
                    stab = f.noise_stability(rho)
                    assert (
                        -1.0 - 1e-10 <= stab <= 1.0 + 1e-10
                    ), f"Noise stability {stab} out of bounds for rho={rho}"

    def test_noise_stability_rho_0(self):
        """At ρ=0, noise stability = 𝔼[f]² = f̂(∅)²."""
        for n in [2, 3]:
            f = bf.majority(n)
            stab = f.noise_stability(0.0)
            coeffs = f.fourier()
            expected = coeffs[0] ** 2
            assert abs(stab - expected) < 1e-10

    def test_noise_stability_rho_1(self):
        """At ρ=1, noise stability = Σ f̂(S)² = 1."""
        for n in [2, 3, 4]:
            f = bf.majority(n)
            stab = f.noise_stability(1.0)
            assert abs(stab - 1.0) < 1e-10


class TestDegreeBounds:
    """Test Fourier degree bounds."""

    def test_constant_degree_zero(self):
        """Constant functions have degree 0."""
        f0 = bf.create([0, 0, 0, 0])
        f1 = bf.create([1, 1, 1, 1])

        assert f0.degree() == 0
        assert f1.degree() == 0

    def test_dictator_degree_one(self):
        """Dictators have degree 1."""
        for n in [2, 3, 4]:
            for i in range(n):
                f = bf.dictator(n, i)
                assert f.degree() == 1

    def test_parity_degree_n(self):
        """Parity has degree n."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            assert f.degree() == n

    def test_and_degree_n(self):
        """AND has degree n (highest monomial is x₁x₂...xₙ)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            assert f.degree() == n

    def test_majority_degree(self):
        """Majority has degree n (all variables matter)."""
        for n in [3, 5]:
            f = bf.majority(n)
            # Majority has degree n but only odd-degree terms
            assert f.degree() == n
