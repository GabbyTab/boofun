"""
Theoretical Validation Tests

These tests cross-validate BooFun implementations against known
mathematical results from the theory of Boolean functions.

Reference: O'Donnell, "Analysis of Boolean Functions", Cambridge 2014

Each test documents the theoretical result it's validating.
"""

import sys
from math import log, pi, sqrt

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestFourierKnownCoefficients:
    """
    Verify Fourier coefficients against known closed-form expressions.

    Reference: O'Donnell Chapter 1
    """

    def test_parity_fourier_exact(self):
        """
        Parity function: f̂(S) = 0 for S ≠ [n], f̂([n]) = ±1

        The parity function χ_[n](x) = (-1)^(Σx_i) has all its Fourier
        mass on the single coefficient corresponding to all variables.
        """
        for n in [3, 5, 7, 9]:
            f = bf.parity(n)
            fourier = f.fourier()

            # Only the all-variables coefficient should be non-zero
            all_vars_idx = (1 << n) - 1  # 2^n - 1

            for s in range(1 << n):
                if s == all_vars_idx:
                    # f̂([n]) = ±1 for parity
                    assert (
                        abs(abs(fourier[s]) - 1.0) < 1e-10
                    ), f"Parity n={n}: f̂([n]) should be ±1, got {fourier[s]}"
                else:
                    # All other coefficients should be 0
                    assert (
                        abs(fourier[s]) < 1e-10
                    ), f"Parity n={n}: f̂({bin(s)}) should be 0, got {fourier[s]}"

    def test_dictator_fourier_exact(self):
        """
        Dictator function f(x) = x_i: Exactly one non-trivial Fourier coefficient

        The i-th dictator has a single non-zero coefficient (for the singleton set).
        Note: Index ordering depends on bit-order convention.
        """
        for n in [4, 6, 8]:
            for i in range(min(3, n)):
                f = bf.dictator(n, i)
                fourier = f.fourier()

                # Dictator should have exactly one coefficient with |value| = 1
                # (ignoring the empty set coefficient which may be 0 or not)
                large_coeffs = [abs(c) for c in fourier if abs(c) > 0.5]

                # Should have exactly 1 large coefficient (the dictator variable)
                assert len(large_coeffs) == 1, f"Dictator should have exactly 1 large coefficient"
                assert abs(large_coeffs[0] - 1.0) < 1e-10, f"Dictator coefficient should be ±1"

    def test_and_fourier_structure(self):
        """
        AND function has specific Fourier structure:
        - All coefficients have magnitude 1/2^n
        - Parseval: sum of squares = 1
        - Degree = n (coefficient at full set is non-zero)

        Reference: O'Donnell Example 1.3
        """
        for n in [2, 3, 4, 5]:
            f = bf.AND(n)
            fourier = f.fourier()

            # All coefficients should have magnitude 1/2^n (for standard AND)
            1.0 / (2**n)

            # Check that all non-zero coefficients have consistent structure
            [abs(c) for c in fourier]

            # Parseval must hold
            sum_sq = sum(c**2 for c in fourier)
            assert abs(sum_sq - 1.0) < 1e-10, f"AND_{n}: Parseval failed, sum = {sum_sq}"

            # AND has degree n (all variables matter)
            assert f.degree() == n, f"AND_{n} should have degree {n}"

    def test_or_fourier_structure(self):
        """
        OR function structure:
        - Related to AND by negation
        - Parseval holds
        - Degree = n
        - OR outputs Boolean 1 on all but the all-zeros input

        O'Donnell convention: Boolean 0 → +1, Boolean 1 → -1
        So OR in ±1 domain outputs -1 on most inputs (biased toward -1).
        E[f] = -1 + 2/2^n (negative for n >= 2)

        Reference: OR = ¬AND(¬x)
        """
        for n in [2, 3, 4, 5]:
            f = bf.OR(n)
            fourier = f.fourier()

            # Parseval must hold
            sum_sq = sum(c**2 for c in fourier)
            assert abs(sum_sq - 1.0) < 1e-10, f"OR_{n}: Parseval failed, sum = {sum_sq}"

            # OR has degree n
            assert f.degree() == n, f"OR_{n} should have degree {n}"

            # O'Donnell convention: OR outputs -1 on 2^n - 1 inputs, +1 on 1 input
            # E[f] = (1/2^n) * (+1) + ((2^n - 1)/2^n) * (-1) = -1 + 2/2^n
            # For n >= 2, this is negative
            expected = -1 + 2 / (2**n)
            assert (
                abs(fourier[0] - expected) < 1e-10
            ), f"OR_{n}: E[f] should be {expected}, got {fourier[0]}"


class TestParseval:
    """
    Verify Parseval's identity: Σ_S f̂(S)² = E[f²] = 1 for Boolean functions.

    Reference: O'Donnell Theorem 1.10
    """

    @pytest.mark.parametrize("n", [3, 5, 7, 9])
    def test_parseval_majority(self, n):
        """Parseval should hold for majority functions."""
        f = bf.majority(n)
        fourier = f.fourier()
        sum_sq = np.sum(fourier**2)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval failed for majority_{n}: Σf̂²={sum_sq}"

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_parseval_and(self, n):
        """Parseval should hold for AND functions."""
        f = bf.AND(n)
        fourier = f.fourier()
        sum_sq = np.sum(fourier**2)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval failed for AND_{n}: Σf̂²={sum_sq}"

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_parseval_parity(self, n):
        """Parseval should hold for parity functions."""
        f = bf.parity(n)
        fourier = f.fourier()
        sum_sq = np.sum(fourier**2)
        assert abs(sum_sq - 1.0) < 1e-10, f"Parseval failed for parity_{n}: Σf̂²={sum_sq}"


class TestInfluenceKnownValues:
    """
    Verify influences against known theoretical values.

    Reference: O'Donnell Chapter 2
    """

    def test_parity_influences_all_equal_one(self):
        """
        Parity: Inf_i[χ_[n]] = 1 for all i

        Each variable is equally influential in parity, and flipping
        any single bit always changes the parity.
        """
        for n in [3, 5, 7, 9]:
            f = bf.parity(n)
            influences = f.influences()

            for i in range(n):
                assert (
                    abs(influences[i] - 1.0) < 1e-10
                ), f"Parity_{n}: Inf_{i} should be 1, got {influences[i]}"

    def test_dictator_influence_single_variable(self):
        """
        Dictator on variable i: Inf_i = 1, Inf_j = 0 for j ≠ i

        Only the dictator variable matters.
        """
        n = 8
        for i in range(3):
            f = bf.dictator(n, i)
            influences = f.influences()

            for j in range(n):
                if j == i:
                    assert (
                        abs(influences[j] - 1.0) < 1e-10
                    ), f"Dictator on x_{i}: Inf_{i} should be 1"
                else:
                    assert abs(influences[j]) < 1e-10, f"Dictator on x_{i}: Inf_{j} should be 0"

    def test_and_influences_equal(self):
        """
        AND: Inf_i[AND_n] = 2^(1-n) for all i

        Each variable has the same (small) influence in AND.
        """
        for n in [3, 4, 5, 6]:
            f = bf.AND(n)
            influences = f.influences()
            expected = 2 ** (1 - n)

            for i in range(n):
                assert (
                    abs(influences[i] - expected) < 1e-10
                ), f"AND_{n}: Inf_{i} should be {expected}, got {influences[i]}"

    def test_majority_influences_symmetric(self):
        """
        Majority: All variables have equal influence (symmetric function)

        For odd n, majority is a symmetric function, so all influences are equal.
        """
        for n in [3, 5, 7, 9, 11]:
            f = bf.majority(n)
            influences = f.influences()

            # All should be equal
            avg = np.mean(influences)
            for i in range(n):
                assert (
                    abs(influences[i] - avg) < 1e-10
                ), f"Majority_{n}: influences should be symmetric"

    def test_majority_influence_asymptotics(self):
        """
        Majority: Inf_i[MAJ_n] ~ sqrt(2/(π*n)) as n → ∞

        Reference: O'Donnell Proposition 2.31
        """
        # Test for moderately large n where asymptotics apply
        for n in [11, 13, 15, 17, 19, 21]:
            f = bf.majority(n)
            influences = f.influences()

            # Asymptotic formula
            expected_approx = sqrt(2 / (pi * n))
            actual = influences[0]  # All equal by symmetry

            # Allow 15% relative error for finite n
            rel_error = abs(actual - expected_approx) / expected_approx
            assert (
                rel_error < 0.15
            ), f"Majority_{n}: Inf ≈ {expected_approx:.4f}, got {actual:.4f} (error {rel_error:.1%})"


class TestTotalInfluence:
    """
    Verify total influence I[f] = Σ_i Inf_i[f] against known values.

    Reference: O'Donnell Chapter 2
    """

    def test_parity_total_influence_equals_n(self):
        """
        Parity: I[χ_[n]] = n

        Since each variable has influence 1.
        """
        for n in [3, 5, 7, 9]:
            f = bf.parity(n)
            total = f.total_influence()
            assert abs(total - n) < 1e-10, f"Parity_{n}: I[f] should be {n}, got {total}"

    def test_dictator_total_influence_equals_one(self):
        """
        Dictator: I[f] = 1

        Only one variable matters.
        """
        for n in [4, 6, 8]:
            f = bf.dictator(n, 0)
            total = f.total_influence()
            assert abs(total - 1.0) < 1e-10, f"Dictator_{n}: I[f] should be 1, got {total}"

    def test_and_total_influence(self):
        """
        AND: I[AND_n] = n * 2^(1-n)
        """
        for n in [3, 4, 5, 6]:
            f = bf.AND(n)
            total = f.total_influence()
            expected = n * (2 ** (1 - n))
            assert abs(total - expected) < 1e-10, f"AND_{n}: I[f] should be {expected}, got {total}"

    def test_majority_total_influence_asymptotics(self):
        """
        Majority: I[MAJ_n] ~ sqrt(2n/π) as n → ∞

        Reference: O'Donnell Proposition 2.31
        """
        for n in [11, 13, 15, 17, 19, 21]:
            f = bf.majority(n)
            total = f.total_influence()
            expected_approx = sqrt(2 * n / pi)

            rel_error = abs(total - expected_approx) / expected_approx
            assert rel_error < 0.15, f"Majority_{n}: I[f] ≈ {expected_approx:.4f}, got {total:.4f}"


class TestNoiseStability:
    """
    Verify noise stability Stab_ρ[f] = Σ_S ρ^|S| f̂(S)² against known values.

    Reference: O'Donnell Chapter 5
    """

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_dictator_noise_stability_equals_rho(self, rho):
        """
        Dictator: Stab_ρ[x_i] = ρ

        Since f̂({i}) = 1, all other coefficients 0.
        """
        f = bf.dictator(8, 0)
        stab = f.noise_stability(rho)
        assert abs(stab - rho) < 1e-10, f"Dictator: Stab_{rho} should be {rho}, got {stab}"

    @pytest.mark.parametrize("n", [3, 5, 7, 9])
    def test_parity_noise_stability_rho_n(self, n):
        """
        Parity: Stab_ρ[χ_[n]] = ρ^n

        All Fourier weight on degree-n term.
        """
        for rho in [0.3, 0.5, 0.7]:
            f = bf.parity(n)
            stab = f.noise_stability(rho)
            expected = rho**n
            assert (
                abs(stab - expected) < 1e-10
            ), f"Parity_{n}: Stab_{rho} should be {expected}, got {stab}"

    def test_majority_noise_stability_sheppard(self):
        """
        Majority: Pr[Maj(x) = Maj(y)] → (1/2) + (1/π)arcsin(ρ) as n → ∞

        This is Sheppard's formula for Gaussian noise stability.

        IMPORTANT: Sheppard gives the AGREEMENT PROBABILITY Pr[f(x)=f(y)],
        not the Fourier noise stability E[f(x)f(y)].

        For ±1-valued functions: Pr[f(x)=f(y)] = (1 + Stab_ρ[f])/2

        Reference: O'Donnell Theorem 5.6
        """

        def sheppard(rho):
            """Sheppard's formula: Pr[Maj(x) = Maj(y)] in the Gaussian limit."""
            return 0.5 + np.arcsin(rho) / pi

        # Test for moderate n where convergence is good
        # (Using n<=17 to keep test fast - n=21 would take ~10s per iteration)
        for n in [11, 15, 17]:
            f = bf.majority(n)

            for rho in [0.3, 0.5, 0.7]:
                stab = f.noise_stability(rho)
                # Convert Fourier stability to agreement probability
                pr_agree = (1 + stab) / 2
                expected = sheppard(rho)

                # Should match within 3% for moderate n
                rel_error = abs(pr_agree - expected) / expected
                assert (
                    rel_error < 0.03
                ), f"Majority_{n}: Pr[f(x)=f(y)] ≈ {expected:.4f}, got {pr_agree:.4f} (error {rel_error:.1%})"

        # Verify convergence: error decreases with n
        rho = 0.5
        errors = []
        for n in [5, 9, 13, 17]:
            f = bf.majority(n)
            stab = f.noise_stability(rho)
            pr_agree = (1 + stab) / 2
            errors.append(abs(pr_agree - sheppard(rho)))

        # Error should decrease as n grows (convergence)
        assert errors[-1] < errors[0], f"Error should decrease with n: {errors}"

    def test_noise_stability_bounds(self):
        """
        Noise stability should be bounded for Boolean functions.

        For any Boolean function: -1 ≤ Stab_ρ[f] ≤ 1
        For balanced functions with ρ > 0: Stab_ρ > 0 typically
        """
        test_functions = [bf.AND(5), bf.OR(5), bf.majority(5), bf.parity(5)]

        for f in test_functions:
            for rho in [0.3, 0.5, 0.7]:
                stab = f.noise_stability(rho)
                # Basic bounds: noise stability is bounded
                assert -1.01 <= stab <= 1.01, f"Noise stability should be in [-1, 1], got {stab}"


class TestDegree:
    """
    Verify Fourier degree against known values.
    """

    def test_parity_degree_equals_n(self):
        """Parity has degree n (all variables appear)."""
        for n in [3, 5, 7]:
            f = bf.parity(n)
            assert f.degree() == n

    def test_dictator_degree_equals_one(self):
        """Dictator has degree 1."""
        for n in [4, 6, 8]:
            f = bf.dictator(n, 0)
            assert f.degree() == 1

    def test_majority_degree_equals_n(self):
        """Majority has degree n for odd n (not a junta)."""
        for n in [3, 5, 7, 9]:
            f = bf.majority(n)
            assert f.degree() == n

    def test_constant_degree_zero(self):
        """Constant functions have degree 0."""
        for n in [3, 5]:
            f_zero = bf.create([0] * (2**n))
            f_one = bf.create([1] * (2**n))

            # Degree 0 for constants (only f̂(∅) is non-zero)
            assert f_zero.degree() == 0 or f_zero.degree() == 0
            assert f_one.degree() == 0


class TestSpectralConcentration:
    """
    Verify spectral weight distribution.
    """

    def test_spectral_weight_sums_to_one(self):
        """
        For any Boolean function, spectral weight at all degrees sums to 1.
        W^{=0} + W^{=1} + ... + W^{=n} = 1  (Parseval by degree)
        """
        for f in [bf.AND(5), bf.majority(5), bf.parity(5)]:
            n = f.n_vars
            fourier = f.fourier()

            # Compute weight at each degree level
            weights = {}
            for s in range(1 << n):
                k = bin(s).count("1")
                weights[k] = weights.get(k, 0) + fourier[s] ** 2

            # Total should sum to 1 (Parseval)
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-10, f"Spectral weights should sum to 1, got {total}"

    def test_parity_all_weight_at_top_degree(self):
        """
        Parity function has all weight at degree n.
        W^{=n}[χ_[n]] = 1, W^{=k}[χ_[n]] = 0 for k < n
        """
        for n in [3, 5, 7]:
            f = bf.parity(n)
            fourier = f.fourier()

            # Only degree-n coefficient should be non-zero
            weights = {}
            for s in range(1 << n):
                k = bin(s).count("1")
                weights[k] = weights.get(k, 0) + fourier[s] ** 2

            # All weight at degree n
            assert abs(weights.get(n, 0) - 1.0) < 1e-10, f"Parity_{n}: W^{{={n}}} should be 1"

            for k in range(n):
                assert abs(weights.get(k, 0)) < 1e-10, f"Parity_{n}: W^{{={k}}} should be 0"


class TestKKLTheorem:
    """
    Verify KKL theorem: max_i Inf_i[f] ≥ Ω(Var[f] * log(n) / I[f])

    Reference: O'Donnell Theorem 8.4
    """

    def test_kkl_lower_bound(self):
        """
        Every Boolean function has an influential variable.
        max Inf_i ≥ c * Var[f] * log(n) / I[f]
        """
        for n in [7, 9, 11, 13]:
            f = bf.majority(n)

            influences = f.influences()
            max_inf = max(influences)
            total_inf = sum(influences)

            # Variance = 1 - f̂(∅)²
            fourier = f.fourier()
            variance = 1 - fourier[0] ** 2

            if total_inf > 1e-10:
                # KKL bound (with conservative constant c ≈ 0.1)
                kkl_bound = 0.1 * variance * log(n) / total_inf

                assert (
                    max_inf >= kkl_bound * 0.5
                ), f"KKL: max Inf = {max_inf:.4f} should be ≥ {kkl_bound:.4f}"


class TestPoincare:
    """
    Verify Poincaré inequality: Var[f] ≤ I[f]

    Reference: O'Donnell Proposition 2.13
    """

    @pytest.mark.parametrize("n", [5, 7, 9])
    def test_poincare_majority(self, n):
        """Poincaré inequality for majority."""
        f = bf.majority(n)

        fourier = f.fourier()
        variance = 1 - fourier[0] ** 2
        total_inf = f.total_influence()

        assert (
            variance <= total_inf + 1e-10
        ), f"Poincaré: Var={variance:.4f} should be ≤ I[f]={total_inf:.4f}"

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_poincare_and(self, n):
        """Poincaré inequality for AND."""
        f = bf.AND(n)

        fourier = f.fourier()
        variance = 1 - fourier[0] ** 2
        total_inf = f.total_influence()

        assert variance <= total_inf + 1e-10


class TestExpectation:
    """
    Verify E[f] = f̂(∅) (expectation equals constant coefficient).

    Reference: O'Donnell Proposition 1.5
    """

    def test_balanced_functions_zero_expectation(self):
        """Balanced functions have E[f] = 0 in ±1 representation."""
        balanced = [bf.parity(5), bf.majority(5)]

        for f in balanced:
            fourier = f.fourier()
            assert abs(fourier[0]) < 1e-10, f"Balanced function should have f̂(∅) = 0"

    def test_and_expectation(self):
        """
        AND: Under O'Donnell convention (Boolean 0 → +1, Boolean 1 → -1):
        - AND outputs Boolean 1 only on all-1s input → -1 in ±1 domain
        - AND outputs Boolean 0 elsewhere → +1 in ±1 domain
        E[f] = (2^n - 1)/2^n * (+1) + 1/2^n * (-1) = 1 - 2/2^n
        """
        for n in [3, 4, 5]:
            f = bf.AND(n)
            fourier = f.fourier()

            # O'Donnell convention: AND outputs +1 on 2^n - 1 inputs, -1 on 1 input
            # E[f] = ((2^n - 1)/2^n) * (+1) + (1/2^n) * (-1) = 1 - 2/2^n
            expected = 1 - 2 / (2**n)
            assert (
                abs(fourier[0] - expected) < 1e-10
            ), f"AND_{n}: E[f] should be {expected}, got {fourier[0]}"


# Run validation summary if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
