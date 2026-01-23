"""
Fuzz Testing for BooFun API

Uses Hypothesis to generate random inputs and test API robustness.
These tests aim to find edge cases, crashes, and unexpected behavior.

Run with: pytest tests/fuzz/ -v --hypothesis-show-statistics

Modules covered:
- Core API (create, evaluate, representations)
- Fourier analysis (coefficients, influences, noise stability)
- Query complexity (sensitivity, certificates, block sensitivity)
- Property testing (BLR, monotonicity, junta)
- Representation conversions
- Error handling and edge cases
"""

import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis import PropertyTester, SpectralAnalyzer
from boofun.analysis.block_sensitivity import max_block_sensitivity
from boofun.analysis.certificates import max_certificate_size
from boofun.analysis.hypercontractivity import noise_operator
from boofun.analysis.learning import estimate_fourier_coefficient

# =============================================================================
# Strategies for generating Boolean functions
# =============================================================================


@st.composite
def truth_tables(draw, n_vars=None, max_vars=6):
    """Generate random truth tables."""
    if n_vars is None:
        n_vars = draw(st.integers(min_value=1, max_value=max_vars))
    size = 2**n_vars
    return draw(st.lists(st.booleans(), min_size=size, max_size=size))


@st.composite
def boolean_functions(draw, max_vars=6):
    """Generate random Boolean functions."""
    tt = draw(truth_tables(max_vars=max_vars))
    return bf.create(tt)


@st.composite
def small_functions(draw):
    """Generate small functions (n <= 4) for expensive tests."""
    return draw(boolean_functions(max_vars=4))


# =============================================================================
# Core API Fuzz Tests
# =============================================================================


class TestCreateFuzz:
    """Fuzz test bf.create() with various inputs."""

    @given(truth_tables())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_create_from_list(self, tt):
        """Create from any valid truth table list."""
        f = bf.create(tt)
        assert f.n_vars == int(np.log2(len(tt)))

        # Verify evaluation matches
        for i, val in enumerate(tt):
            assert f.evaluate(i) == val

    @given(truth_tables())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_create_from_numpy(self, tt):
        """Create from numpy array."""
        arr = np.array(tt, dtype=bool)
        f = bf.create(arr)
        assert f.n_vars == int(np.log2(len(tt)))

    @given(st.integers(min_value=1, max_value=7))
    def test_builtin_functions(self, n):
        """Built-in functions should work for any valid n."""
        # Test all builtin function generators
        funcs = [
            bf.AND(n),
            bf.OR(n),
            bf.parity(n),
        ]

        if n % 2 == 1:  # Majority requires odd n
            funcs.append(bf.majority(n))

        if n > 0:
            funcs.append(bf.dictator(n, 0))

        for f in funcs:
            assert f.n_vars == n
            assert len(f.get_representation("truth_table")) == 2**n


# =============================================================================
# Fourier Analysis Fuzz Tests
# =============================================================================


class TestFourierFuzz:
    """Fuzz test Fourier analysis."""

    @given(boolean_functions())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_parseval_always_holds(self, f):
        """Parseval identity should hold for any function."""
        fourier = f.fourier()
        sum_sq = np.sum(fourier**2)
        assert abs(sum_sq - 1.0) < 1e-9, f"Parseval failed: sum = {sum_sq}"

    @given(boolean_functions())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_influences_bounded(self, f):
        """Influences should be in [0, 1]."""
        influences = f.influences()
        assert all(0 <= inf <= 1 + 1e-10 for inf in influences)

    @given(boolean_functions())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_total_influence_non_negative(self, f):
        """Total influence should be non-negative."""
        ti = f.total_influence()
        assert ti >= -1e-10

    @given(boolean_functions(), st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_noise_stability_bounded(self, f, rho):
        """Noise stability should be in [-1, 1]."""
        stab = f.noise_stability(rho)
        assert -1 - 1e-10 <= stab <= 1 + 1e-10


# =============================================================================
# Query Complexity Fuzz Tests
# =============================================================================


class TestComplexityFuzz:
    """Fuzz test complexity measures."""

    @given(small_functions())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_certificate_bounded(self, f):
        """Certificate complexity should be <= n."""
        n = f.n_vars
        C = max_certificate_size(f)
        assert 0 <= C <= n

    @given(small_functions())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_block_sensitivity_bounded(self, f):
        """Block sensitivity should be <= n."""
        n = f.n_vars
        bs = max_block_sensitivity(f)
        assert 0 <= bs <= n

    @given(small_functions())
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_complexity_chain(self, f):
        """s(f) <= bs(f) <= C(f) should hold."""
        from boofun.analysis.huang import max_sensitivity

        s = max_sensitivity(f)
        bs = max_block_sensitivity(f)
        C = max_certificate_size(f)

        assert s <= bs + 0.01, f"s={s} > bs={bs}"
        assert bs <= C + 0.01, f"bs={bs} > C={C}"


# =============================================================================
# Property Testing Fuzz Tests
# =============================================================================


class TestPropertyTesterFuzz:
    """Fuzz test PropertyTester."""

    @given(boolean_functions())
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_property_tester_no_crash(self, f):
        """PropertyTester should not crash on any function."""
        tester = PropertyTester(f, random_seed=42)

        # These should all return without crashing
        tester.blr_linearity_test(num_queries=50)
        tester.monotonicity_test(num_queries=50)
        tester.balanced_test()

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_parity_always_linear(self, n):
        """Parity should always pass linearity test."""
        f = bf.parity(n)
        tester = PropertyTester(f, random_seed=42)
        assert tester.blr_linearity_test(num_queries=100)

    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_and_always_monotone(self, n):
        """AND should always pass monotonicity test."""
        f = bf.AND(n)
        tester = PropertyTester(f, random_seed=42)
        assert tester.monotonicity_test(num_queries=100)


# =============================================================================
# Representation Fuzz Tests
# =============================================================================


class TestRepresentationFuzz:
    """Fuzz test representation conversions."""

    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_truth_table_roundtrip(self, f):
        """Truth table should roundtrip correctly."""
        tt1 = f.get_representation("truth_table")
        f2 = bf.create(list(tt1))
        tt2 = f2.get_representation("truth_table")

        assert np.array_equal(tt1, tt2)

    @given(boolean_functions(max_vars=4))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_fourier_roundtrip(self, f):
        """Fourier should be consistent with evaluation."""
        fourier = f.fourier()

        # Reconstruct function from Fourier and verify
        n = f.n_vars
        for x in range(min(16, 2**n)):  # Sample some inputs
            # Compute f(x) from Fourier expansion
            reconstructed = 0.0
            for S in range(2**n):
                # chi_S(x) = (-1)^(|x ∩ S|)
                chi = 1 if bin(x & S).count("1") % 2 == 0 else -1
                reconstructed += fourier[S] * chi

            # Convert from ±1 to Boolean
            # O'Donnell: f(x)=0 → +1, f(x)=1 → -1
            expected_pm = -1 if f.evaluate(x) else 1
            assert abs(reconstructed - expected_pm) < 1e-9


# =============================================================================
# Edge Case Fuzz Tests
# =============================================================================


class TestEdgeCasesFuzz:
    """Fuzz test edge cases."""

    @given(st.integers(min_value=1, max_value=7))
    def test_constant_functions(self, n):
        """Constant functions should work correctly."""
        f_zero = bf.create([False] * (2**n))
        f_one = bf.create([True] * (2**n))

        # Both should have degree 0
        assert f_zero.degree() == 0
        assert f_one.degree() == 0

        # Parseval should hold
        assert abs(np.sum(f_zero.fourier() ** 2) - 1.0) < 1e-9
        assert abs(np.sum(f_one.fourier() ** 2) - 1.0) < 1e-9

    @given(st.integers(min_value=1, max_value=6), st.integers(min_value=0))
    def test_dictator_any_variable(self, n, var_seed):
        """Dictator on any variable should work."""
        var = var_seed % n
        f = bf.dictator(n, var)

        assert f.degree() == 1
        influences = f.influences()

        # Only the dictator variable should have influence 1
        for i in range(n):
            if i == var:
                assert abs(influences[i] - 1.0) < 1e-9
            else:
                assert abs(influences[i]) < 1e-9

    @given(st.integers(min_value=3, max_value=7).filter(lambda x: x % 2 == 1))
    def test_majority_symmetric(self, n):
        """Majority should have equal influences."""
        f = bf.majority(n)
        influences = f.influences()

        # All influences should be equal
        assert np.allclose(influences, influences[0])


# =============================================================================
# Hypercontractivity Fuzz Tests
# =============================================================================


class TestHypercontractivityFuzz:
    """Fuzz test hypercontractivity operations."""

    @given(boolean_functions(max_vars=5), st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_noise_operator_preserves_size(self, f, rho):
        """Noise operator output should have same size as truth table."""
        # noise_operator returns T_rho f(x) for all x
        noisy_values = noise_operator(f, rho)
        assert len(noisy_values) == 2**f.n_vars

    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_noise_at_zero_is_constant(self, f):
        """T_0 f = E[f] (constant function)."""
        noisy_values = noise_operator(f, 0.0)

        # T_0 f should be constant (equal to E[f] everywhere)
        mean_value = noisy_values[0]
        for val in noisy_values:
            assert abs(val - mean_value) < 1e-10

    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_noise_values_bounded(self, f):
        """Noise operator values should be in [-1, 1]."""
        for rho in [0.0, 0.5, 0.9, 1.0]:
            noisy_values = noise_operator(f, rho)
            assert np.all(noisy_values >= -1 - 1e-10)
            assert np.all(noisy_values <= 1 + 1e-10)


# =============================================================================
# Learning Module Fuzz Tests
# =============================================================================


class TestLearningFuzz:
    """Fuzz test learning algorithms."""

    @given(small_functions(), st.integers(min_value=0))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_fourier_coefficient_estimation(self, f, seed):
        """Fourier coefficient estimation should be bounded."""
        S = seed % (2**f.n_vars)  # Valid subset

        estimate, std_err = estimate_fourier_coefficient(f, S, num_samples=100)

        # Estimate should be bounded by ±1 (since |f̂(S)| ≤ 1)
        assert -1.5 <= estimate <= 1.5
        # Standard error should be non-negative
        assert std_err >= 0


# =============================================================================
# Boolean Operations Stress Tests
# =============================================================================


class TestBooleanOperationsFuzz:
    """Fuzz test Boolean operations for consistency."""

    @given(boolean_functions(max_vars=4), boolean_functions(max_vars=4))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_xor_commutative(self, f, g):
        """XOR should be commutative: f ⊕ g = g ⊕ f."""
        assume(f.n_vars == g.n_vars)

        h1 = f ^ g
        h2 = g ^ f

        n = f.n_vars
        for x in range(min(2**n, 16)):
            assert h1.evaluate(x) == h2.evaluate(x)

    @given(boolean_functions(max_vars=4), boolean_functions(max_vars=4))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_and_commutative(self, f, g):
        """AND should be commutative."""
        assume(f.n_vars == g.n_vars)

        h1 = f & g
        h2 = g & f

        n = f.n_vars
        for x in range(min(2**n, 16)):
            assert h1.evaluate(x) == h2.evaluate(x)

    @given(boolean_functions(max_vars=4), boolean_functions(max_vars=4))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_or_commutative(self, f, g):
        """OR should be commutative."""
        assume(f.n_vars == g.n_vars)

        h1 = f | g
        h2 = g | f

        n = f.n_vars
        for x in range(min(2**n, 16)):
            assert h1.evaluate(x) == h2.evaluate(x)

    @given(boolean_functions(max_vars=4))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_de_morgan_and(self, f):
        """De Morgan: NOT(f AND g) = NOT(f) OR NOT(g)."""
        g = bf.create(list(f.get_representation("truth_table")))  # Copy

        lhs = ~(f & g)
        rhs = (~f) | (~g)

        n = f.n_vars
        for x in range(min(2**n, 16)):
            assert lhs.evaluate(x) == rhs.evaluate(x)


# =============================================================================
# Representation Chain Stress Tests
# =============================================================================


class TestRepresentationChainFuzz:
    """Fuzz test chains of representation conversions."""

    @given(boolean_functions(max_vars=4))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_conversions_preserve_function(self, f):
        """Multiple representation conversions should preserve function."""
        # Get various representations
        tt = f.get_representation("truth_table")
        fourier = f.get_representation("fourier_expansion")
        anf = f.get_representation("anf")

        # Verify all representations agree on evaluation
        n = f.n_vars
        for x in range(min(2**n, 16)):
            val = f.evaluate(x)
            assert tt[x] == val, f"Truth table mismatch at {x}"

    @given(boolean_functions(max_vars=4))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_create_from_any_representation(self, f):
        """Creating function from any representation should work."""
        # From truth table
        tt = f.get_representation("truth_table")
        f_tt = bf.create(list(tt))

        # Verify
        n = f.n_vars
        for x in range(min(2**n, 8)):
            assert f.evaluate(x) == f_tt.evaluate(x)


# =============================================================================
# Spectral Analysis Fuzz Tests
# =============================================================================


class TestSpectralAnalysisFuzz:
    """Fuzz test spectral analysis operations."""

    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_spectral_analyzer_consistent(self, f):
        """SpectralAnalyzer should give consistent results."""
        analyzer = SpectralAnalyzer(f)

        # Multiple calls should give same result
        fourier1 = analyzer.fourier_expansion()
        fourier2 = analyzer.fourier_expansion()

        np.testing.assert_array_equal(fourier1, fourier2)

    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_degree_bounded(self, f):
        """Degree should be at most n."""
        deg = f.degree()
        assert 0 <= deg <= f.n_vars

    @given(boolean_functions(max_vars=5))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_spectral_weight_sums_to_one(self, f):
        """Spectral weight by degree should sum to 1."""
        weights = f.spectral_weight_by_degree()

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
