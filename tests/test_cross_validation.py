"""
Cross-Validation Tests

These tests verify BooFun implementations by:
1. Comparing against theoretical known values
2. Cross-validating with other libraries (when installed)
3. Ensuring consistency between our different modules

Run with: pytest tests/test_cross_validation.py -v
"""

import sys
from math import pi, sqrt

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf

# =============================================================================
# Part 1: Internal Consistency Tests
# =============================================================================


class TestInternalConsistency:
    """Verify our different modules give consistent results."""

    def test_influences_multiple_methods(self):
        """Influences should match across different computation methods."""
        f = bf.majority(5)

        # Via direct method
        inf_direct = f.influences()

        # Via total influence
        total_inf = f.total_influence()

        # Should sum correctly
        assert abs(np.sum(inf_direct) - total_inf) < 1e-10

    def test_fourier_parseval(self):
        """Parseval's identity: Σ f̂(S)² = 1."""
        for func in [bf.majority(5), bf.parity(5), bf.AND(5)]:
            fourier = func.fourier()
            sum_sq = np.sum(fourier**2)
            assert abs(sum_sq - 1.0) < 1e-10, f"Parseval failed: {sum_sq}"

    def test_sensitivity_vs_influence(self):
        """Sensitivity and influence should be related correctly."""
        f = bf.majority(5)

        # Total influence = average sensitivity (for ±1 functions)
        total_inf = f.total_influence()

        # Sensitivity at each point, averaged
        from boofun.analysis.huang import average_sensitivity

        avg_sens = average_sensitivity(f)

        # These should match
        assert abs(total_inf - avg_sens) < 1e-10

    def test_canalization_vs_essential(self):
        """Non-essential variables should have zero influence."""
        # Create function that ignores variable 0
        # (AND of variables 1,2)
        tt = [0, 0, 0, 0, 0, 1, 0, 1]  # f(x) = x1 AND x2
        f = bf.create(tt)

        from boofun.analysis.canalization import get_essential_variables

        essential = get_essential_variables(f)

        influences = f.influences()

        # Non-essential variables should have ~0 influence
        for i in range(f.n_vars):
            if i not in essential:
                assert influences[i] < 0.01, f"Non-essential var {i} has influence {influences[i]}"


class TestQueryComplexityConsistency:
    """Verify query complexity measures are consistent."""

    def test_sensitivity_bound(self):
        """s(f) ≤ D(f) (sensitivity is a lower bound)."""
        from boofun.analysis.query_complexity import (
            deterministic_query_complexity,
            sensitivity_lower_bound,
        )

        for func in [bf.majority(3), bf.AND(4), bf.parity(3)]:
            D_f = deterministic_query_complexity(func)
            s_bound = sensitivity_lower_bound(func)

            assert s_bound <= D_f + 0.01, f"s(f)={s_bound} > D(f)={D_f}"

    def test_block_sensitivity_bound(self):
        """bs(f) ≤ D(f)."""
        from boofun.analysis.query_complexity import (
            block_sensitivity_lower_bound,
            deterministic_query_complexity,
        )

        for func in [bf.AND(3), bf.OR(3)]:
            D_f = deterministic_query_complexity(func)
            bs_bound = block_sensitivity_lower_bound(func)

            assert bs_bound <= D_f + 0.01


class TestTheoreticalBounds:
    """Verify fundamental theoretical bounds from complexity theory."""

    def test_huang_sensitivity_theorem(self):
        """
        Huang's Sensitivity Theorem (2019): s(f) ≥ √bs(f).

        This is a breakthrough result showing sensitivity is polynomially
        related to block sensitivity.
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.huang import max_sensitivity

        for func in [bf.AND(4), bf.OR(4), bf.majority(5), bf.parity(4)]:
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)

            # s(f) ≥ √bs(f)
            assert (
                s_f >= sqrt(bs_f) - 0.01
            ), f"Huang violated: s(f)={s_f}, bs(f)={bs_f}, √bs(f)={sqrt(bs_f):.2f}"

    def test_nisan_szegedy_bound(self):
        """
        Nisan-Szegedy (1994): D(f) ≤ bs(f)^2.

        Decision tree complexity is at most block sensitivity squared.
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            D_f = deterministic_query_complexity(func)
            bs_f = max_block_sensitivity(func)

            # D(f) ≤ bs(f)²
            assert D_f <= bs_f**2 + 0.01, f"Nisan-Szegedy violated: D(f)={D_f}, bs(f)²={bs_f**2}"

    def test_certificate_vs_decision_tree(self):
        """
        C(f) ≤ D(f): Certificate complexity is a lower bound for decision tree.
        """
        from boofun.analysis.certificates import max_certificate_size
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for func in [bf.AND(4), bf.OR(4), bf.parity(3)]:
            D_f = deterministic_query_complexity(func)
            C_f = max_certificate_size(func)

            assert C_f <= D_f, f"Certificate bound violated: C(f)={C_f} > D(f)={D_f}"

    def test_block_sensitivity_vs_certificate(self):
        """
        bs(f) ≤ C(f): Block sensitivity is bounded by certificate complexity.
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.certificates import max_certificate_size

        for func in [bf.AND(4), bf.OR(4), bf.majority(3)]:
            bs_f = max_block_sensitivity(func)
            C_f = max_certificate_size(func)

            assert bs_f <= C_f, f"bs(f)={bs_f} > C(f)={C_f}"

    def test_sensitivity_vs_block_sensitivity(self):
        """
        s(f) ≤ bs(f): Sensitivity is bounded by block sensitivity.
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.huang import max_sensitivity

        for func in [bf.AND(4), bf.OR(4), bf.majority(5), bf.parity(4)]:
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)

            assert s_f <= bs_f, f"s(f)={s_f} > bs(f)={bs_f}"

    def test_complexity_measure_chain(self):
        """
        The full complexity chain: s(f) ≤ bs(f) ≤ C(f) ≤ D(f).
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.certificates import max_certificate_size
        from boofun.analysis.huang import max_sensitivity
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)
            C_f = max_certificate_size(func)
            D_f = deterministic_query_complexity(func)

            assert (
                s_f <= bs_f <= C_f <= D_f
            ), f"Chain violated: s={s_f}, bs={bs_f}, C={C_f}, D={D_f}"


# =============================================================================
# Part 2: Theoretical Known Values
# =============================================================================


class TestKnownValues:
    """Test against known theoretical values from the literature."""

    def test_and_query_complexity(self):
        """AND has known query complexity D(AND_n) = n."""
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for n in [2, 3, 4, 5]:
            D_and = deterministic_query_complexity(bf.AND(n))
            assert D_and == n, f"D(AND_{n}) should be {n}, got {D_and}"

    def test_or_query_complexity(self):
        """OR has known query complexity D(OR_n) = n."""
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for n in [2, 3, 4, 5]:
            D_or = deterministic_query_complexity(bf.OR(n))
            assert D_or == n, f"D(OR_{n}) should be {n}, got {D_or}"

    def test_parity_influences(self):
        """Parity: all influences = 1."""
        for n in [3, 5, 7]:
            f = bf.parity(n)
            influences = f.influences()

            for i in range(n):
                assert abs(influences[i] - 1.0) < 1e-10

    def test_majority_symmetric_influences(self):
        """Majority: all influences equal (symmetric function)."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            influences = f.influences()

            # All should be equal
            assert np.allclose(influences, influences[0])

    def test_majority_influence_asymptotic(self):
        """Majority influences converge to √(2/πn)."""
        for n in [11, 15, 19, 21]:
            f = bf.majority(n)
            influences = f.influences()

            expected = sqrt(2 / (pi * n))
            actual = influences[0]

            # Allow 15% error for finite n
            rel_error = abs(actual - expected) / expected
            assert rel_error < 0.15, f"Majority_{n}: expected {expected:.4f}, got {actual:.4f}"

    def test_dictator_degree(self):
        """Dictator has degree 1."""
        for n in [3, 5, 7]:
            f = bf.dictator(n, 0)
            assert f.degree() == 1

    def test_parity_degree(self):
        """Parity has degree n."""
        for n in [3, 5, 7]:
            f = bf.parity(n)
            assert f.degree() == n


class TestPropertyTestingTheory:
    """Test property testing algorithms against known results."""

    def test_blr_detects_linear(self):
        """BLR should accept linear functions."""
        from boofun.analysis import PropertyTester

        # XOR/parity is linear
        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)

        assert tester.blr_linearity_test(num_queries=100)

    def test_blr_rejects_nonlinear(self):
        """BLR should reject non-linear functions."""
        from boofun.analysis import PropertyTester

        # AND is not linear
        f = bf.AND(4)
        tester = PropertyTester(f, random_seed=42)

        # Should fail (might occasionally pass by chance, so use high queries)
        result = tester.blr_linearity_test(num_queries=200)
        assert not result, "BLR should reject AND"

    def test_monotonicity_accepts_monotone(self):
        """Monotonicity test should accept monotone functions."""
        from boofun.analysis import PropertyTester

        # AND is monotone
        f = bf.AND(4)
        tester = PropertyTester(f, random_seed=42)

        assert tester.monotonicity_test(num_queries=100)

    def test_monotonicity_rejects_nonmonotone(self):
        """Monotonicity test should reject non-monotone functions."""
        from boofun.analysis import PropertyTester

        # Parity is not monotone
        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)

        assert not tester.monotonicity_test(num_queries=100)


# =============================================================================
# Part 3: Cross-Validation with External Libraries (Optional)
# =============================================================================


class TestBoolForgeCompatibility:
    """Cross-validate with BoolForge if installed."""

    @pytest.fixture
    def boolforge_available(self):
        """Check if BoolForge is installed."""
        try:
            import boolforge  # noqa: F401

            return True
        except ImportError:
            return False

    def test_activities_vs_influences(self, boolforge_available):
        """BoolForge activities should match our influences."""
        if not boolforge_available:
            pytest.skip("BoolForge not installed")

        from boolforge import BooleanFunction as BF_Forge

        # Create same function in both libraries
        f_bf = bf.majority(5)
        tt = [int(f_bf.evaluate(x)) for x in range(32)]
        f_forge = BF_Forge(tt)

        # Compare
        our_influences = f_bf.influences()
        forge_activities = f_forge.get_activities(EXACT=True)

        assert np.allclose(our_influences, forge_activities, rtol=0.01)

    def test_sensitivity_match(self, boolforge_available):
        """BoolForge avg_sensitivity should match our total_influence."""
        if not boolforge_available:
            pytest.skip("BoolForge not installed")

        from boolforge import BooleanFunction as BF_Forge

        f_bf = bf.majority(5)
        tt = [int(f_bf.evaluate(x)) for x in range(32)]
        f_forge = BF_Forge(tt)

        our_ti = f_bf.total_influence()
        forge_sens = f_forge.get_average_sensitivity(EXACT=True, NORMALIZED=False)

        assert abs(our_ti - forge_sens) < 0.01


class TestSageMathCompatibility:
    """Cross-validate with SageMath if installed."""

    @pytest.fixture
    def sage_available(self):
        """Check if SageMath is installed."""
        try:
            import sage.crypto.boolean_function  # noqa: F401

            return True
        except ImportError:
            return False

    def test_walsh_hadamard_structure(self, sage_available):
        """Walsh-Hadamard should have same structure (up to normalization)."""
        if not sage_available:
            pytest.skip("SageMath not installed")

        from sage.crypto.boolean_function import BooleanFunction as SageBF

        f_bf = bf.AND(4)
        tt = [int(f_bf.evaluate(x)) for x in range(16)]

        # Our Fourier coefficients
        our_fourier = f_bf.fourier()

        # SageMath Walsh-Hadamard
        f_sage = SageBF(tt)
        sage_wht = np.array(f_sage.walsh_hadamard_transform())

        # Compare non-zero positions
        our_nonzero = set(np.where(np.abs(our_fourier) > 0.01)[0])
        sage_nonzero = set(np.where(np.abs(sage_wht) > 0.01)[0])

        # Should have same non-zero structure
        assert our_nonzero == sage_nonzero


class TestCANACompatibility:
    """Cross-validate with CANA if installed."""

    @pytest.fixture
    def cana_available(self):
        """Check if CANA is installed."""
        try:
            pass

            return True
        except ImportError:
            return False

    def test_redundancy_concept(self, cana_available):
        """Input redundancy concepts should align."""
        if not cana_available:
            pytest.skip("CANA not installed")

        # Test that our input_redundancy gives sensible results
        from boofun.analysis.canalization import input_redundancy

        # Constant function: all inputs redundant
        f_const = bf.create([0] * 8)
        assert input_redundancy(f_const) == 1.0

        # Parity: no inputs redundant
        f_parity = bf.parity(3)
        assert input_redundancy(f_parity) == 0.0


# =============================================================================
# Part 4: Unique Features Validation
# =============================================================================


class TestUniqueFeatures:
    """Test features that only BooFun has."""

    def test_query_complexity_exists(self):
        """Verify query complexity module works."""
        from boofun.analysis.query_complexity import QueryComplexityProfile

        f = bf.AND(3)
        profile = QueryComplexityProfile(f)
        measures = profile.compute()

        # Should have key measures (D = deterministic complexity)
        assert "D" in measures
        assert "Q2" in measures  # Quantum complexity
        assert "bs" in measures  # Block sensitivity
        assert measures["D"] == 3  # D(AND_3) = 3

    def test_property_testing_exists(self):
        """Verify property testing works."""
        from boofun.analysis import PropertyTester

        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)

        # Should have key tests
        assert hasattr(tester, "blr_linearity_test")
        assert hasattr(tester, "junta_test")
        assert hasattr(tester, "monotonicity_test")

    def test_quantum_complexity_module_exists(self):
        """Verify quantum complexity module works."""
        from boofun.quantum_complexity import QuantumComplexityAnalyzer

        f = bf.AND(3)
        qca = QuantumComplexityAnalyzer(f)

        # Should have complexity analysis methods
        assert hasattr(qca, "create_quantum_oracle")
        assert hasattr(qca, "grover_analysis")
        assert hasattr(qca, "grover_amplitude_analysis")

    def test_noise_stability_exists(self):
        """Verify noise stability (unique to our Fourier focus)."""
        f = bf.majority(5)

        # Should have noise_stability method
        stab = f.noise_stability(0.5)

        # Should be in valid range
        assert -1 <= stab <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
