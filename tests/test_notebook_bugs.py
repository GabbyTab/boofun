"""
Tests for bugs discovered during notebook auditing.

Each test class documents the bug it prevents, the notebook where it was
found, and the theoretical result that validates correctness.
"""

import sys
from math import pi, sqrt

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestLinearNonlinearityConsistency:
    """
    Bug: real_world_applications.ipynb showed "Is linear: True" and
    "Nonlinearity: 4" for the same function — a contradiction.

    Invariant: is_linear(f) == True  ⟹  nonlinearity(f) == 0
               nonlinearity(f) > 0  ⟹  is_linear(f) == False
    """

    def test_linear_implies_zero_nonlinearity(self):
        """Linear functions must have nonlinearity 0."""
        from boofun.analysis.cryptographic import nonlinearity

        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            assert f.is_linear(), f"parity({n}) should be linear"
            nl = nonlinearity(f)
            assert nl == 0, (
                f"parity({n}): is_linear=True but nonlinearity={nl} (should be 0)"
            )

    def test_positive_nonlinearity_implies_not_linear(self):
        """Positive nonlinearity means the function is not linear."""
        from boofun.analysis.cryptographic import nonlinearity

        test_functions = [
            ("majority(3)", bf.majority(3)),
            ("majority(5)", bf.majority(5)),
            ("AND(3)", bf.AND(3)),
            ("OR(4)", bf.OR(4)),
        ]
        for name, f in test_functions:
            nl = nonlinearity(f)
            if nl > 0:
                assert not f.is_linear(), (
                    f"{name}: nonlinearity={nl} but is_linear=True"
                )

    def test_bent_implies_not_linear(self):
        """Bent functions have maximum nonlinearity, hence not linear."""
        from boofun.analysis.cryptographic import is_bent, nonlinearity

        bent = bf.from_hex("ac90", n=4)
        assert is_bent(bent)
        assert not bent.is_linear()
        n = bent.n_vars
        expected_nl = 2 ** (n - 1) - 2 ** (n // 2 - 1)
        assert nonlinearity(bent) == expected_nl

    def test_dictator_is_linear(self):
        """Dictator functions are linear (they are characters)."""
        from boofun.analysis.cryptographic import nonlinearity

        for n in [3, 5]:
            f = bf.dictator(n, 0)
            assert f.is_linear()
            assert nonlinearity(f) == 0


class TestNonlinearityExactValues:
    """
    Bug: notebook computed nonlinearity with wrong formula (2^(n/2) vs 2^(n-1)).

    These tests verify exact nonlinearity values for known functions.
    """

    def test_parity_nonlinearity_zero(self):
        """Parity (XOR) is linear: nl = 0."""
        from boofun.analysis.cryptographic import nonlinearity

        for n in [2, 3, 4, 5]:
            assert nonlinearity(bf.parity(n)) == 0

    def test_bent_nonlinearity_maximum(self):
        """Bent function achieves nl = 2^(n-1) - 2^(n/2-1)."""
        from boofun.analysis.cryptographic import nonlinearity

        bent = bf.from_hex("ac90", n=4)
        # For n=4: max nl = 2^3 - 2^1 = 8 - 2 = 6
        assert nonlinearity(bent) == 6

    def test_majority3_nonlinearity(self):
        """Majority_3 = median: nl = 2 (distance 2 from nearest affine)."""
        from boofun.analysis.cryptographic import nonlinearity

        f = bf.majority(3)
        assert nonlinearity(f) == 2


class TestFindCriticalP:
    """
    Bug: find_critical_p was never tested. The tribes(3,3) confusion
    showed we need to verify thresholds against analytical values.

    Analytical thresholds:
    - AND_n: p_c = 0.5^(1/n)  (since Pr_p[AND=1] = p^n)
    - OR_n: p_c = 1 - 0.5^(1/n)  (dual of AND)
    - Majority_n (odd): p_c = 0.5  (by symmetry)
    """

    @pytest.fixture
    def find_critical_p(self):
        from boofun.analysis.global_hypercontractivity import find_critical_p

        return find_critical_p

    def test_and_critical_p(self, find_critical_p):
        """AND_n: Pr_p[AND=1] = p^n, so p_c = 0.5^(1/n)."""
        for n in [3, 5]:
            f = bf.AND(n)
            pc = find_critical_p(f, samples=5000, tolerance=0.02)
            expected = 0.5 ** (1.0 / n)
            assert abs(pc - expected) < 0.05, (
                f"AND({n}): p_c={pc:.4f}, expected={expected:.4f}"
            )

    def test_or_critical_p(self, find_critical_p):
        """OR_n: Pr_p[OR=1] = 1-(1-p)^n, so p_c = 1 - 0.5^(1/n)."""
        for n in [3, 5]:
            f = bf.OR(n)
            pc = find_critical_p(f, samples=5000, tolerance=0.02)
            expected = 1.0 - 0.5 ** (1.0 / n)
            assert abs(pc - expected) < 0.05, (
                f"OR({n}): p_c={pc:.4f}, expected={expected:.4f}"
            )

    def test_majority_critical_p(self, find_critical_p):
        """Majority_n (odd n): symmetric, so p_c = 0.5."""
        for n in [5, 7]:
            f = bf.majority(n)
            pc = find_critical_p(f, samples=5000, tolerance=0.02)
            assert abs(pc - 0.5) < 0.05, (
                f"Majority({n}): p_c={pc:.4f}, expected=0.5"
            )

    def test_and_or_duality(self, find_critical_p):
        """AND and OR are dual: p_c(AND_n) + p_c(OR_n) ≈ 1."""
        n = 5
        pc_and = find_critical_p(bf.AND(n), samples=5000, tolerance=0.02)
        pc_or = find_critical_p(bf.OR(n), samples=5000, tolerance=0.02)
        assert abs(pc_and + pc_or - 1.0) < 0.1, (
            f"AND/OR duality: p_c(AND)={pc_and:.4f} + p_c(OR)={pc_or:.4f} "
            f"= {pc_and + pc_or:.4f} (expected ≈ 1.0)"
        )


class TestPBiasedExpectationConvention:
    """
    Bug: MC p_biased_expectation (global_hypercontractivity) returns Pr[f=1]
    while exact p_biased_expectation (p_biased) returns E[f in ±1].
    No test verified they are consistent.

    Relationship: Pr[f=1] = (1 - E[f in ±1]) / 2
    """

    def test_mc_vs_exact_convention(self):
        """MC and exact p_biased_expectation must be related by (1-E)/2."""
        from boofun.analysis.global_hypercontractivity import (
            p_biased_expectation as mc_pbe,
        )
        from boofun.analysis.p_biased import (
            p_biased_expectation as exact_pbe,
        )

        np.random.seed(42)
        for name, f in [("AND(3)", bf.AND(3)), ("Majority(5)", bf.majority(5))]:
            for p in [0.3, 0.5, 0.7]:
                exact_pm1 = exact_pbe(f, p)
                mc_prob = mc_pbe(f, p, samples=50000)
                # Convert: Pr[f=1] = (1 - E[f in ±1]) / 2
                expected_prob = (1 - exact_pm1) / 2
                assert abs(mc_prob - expected_prob) < 0.03, (
                    f"{name} p={p}: MC={mc_prob:.4f}, "
                    f"expected={(1-exact_pm1)/2:.4f} "
                    f"(exact ±1={exact_pm1:.4f})"
                )

    def test_exact_pbe_known_values(self):
        """Verify exact p_biased_expectation against hand-computed values."""
        from boofun.analysis.p_biased import (
            p_biased_expectation as exact_pbe,
        )

        # AND_2 at p=0.3: Pr[f=1] = 0.3^2 = 0.09
        # E[f in ±1] = 1 - 2*0.09 = 0.82
        f = bf.AND(2)
        exp = exact_pbe(f, 0.3)
        assert abs(exp - 0.82) < 1e-10, f"AND(2) p=0.3: E={exp}, expected=0.82"

        # Parity_2 at p=0.5: E[f in ±1] = 0 (balanced)
        f = bf.parity(2)
        exp = exact_pbe(f, 0.5)
        assert abs(exp) < 1e-10, f"Parity(2) p=0.5: E={exp}, expected=0"


class TestTribesCorrectness:
    """
    Bug: tribes(3,3) was mistaken for a 9-variable function but is
    actually OR_3 on 3 variables.

    Verify tribes(k, n) has correct structure and truth table.
    """

    def test_tribes_n_vars(self):
        """tribes(k, n) should create a function on exactly n variables."""
        for k, n in [(2, 4), (2, 6), (3, 6), (3, 9), (3, 12)]:
            f = bf.tribes(k, n)
            assert f.n_vars == n, (
                f"tribes({k},{n}): n_vars={f.n_vars}, expected={n}"
            )

    def test_tribes_is_and_of_ors(self):
        """tribes(2, 4) = AND(OR(x0,x1), OR(x2,x3))."""
        f = bf.tribes(2, 4)
        tt = list(f.get_representation("truth_table"))

        # Manually compute AND(OR(x0,x1), OR(x2,x3))
        for x in range(16):
            x0, x1, x2, x3 = [(x >> j) & 1 for j in range(4)]
            expected = int((x0 or x1) and (x2 or x3))
            assert tt[x] == expected, (
                f"tribes(2,4) at x={x:04b}: got {tt[x]}, "
                f"expected {expected}"
            )

    def test_tribes_single_tribe_is_or(self):
        """tribes(n, n) = single tribe = OR_n."""
        for n in [3, 4, 5]:
            tribes_f = bf.tribes(n, n)
            or_f = bf.OR(n)
            tt_tribes = list(tribes_f.get_representation("truth_table"))
            tt_or = list(or_f.get_representation("truth_table"))
            assert tt_tribes == tt_or, (
                f"tribes({n},{n}) should equal OR({n})"
            )

    def test_tribes_balanced_approximation(self):
        """Large balanced tribes should have Pr[f=1] ≈ 0.5."""
        # tribes(k, n) with k = log(n) gives approximately balanced tribes
        # For tribes(2, 8): Pr[f=1] = (1-(1-p)^2)^4 at p=0.5
        # = (3/4)^4 = 0.3164
        f = bf.tribes(2, 8)
        tt = list(f.get_representation("truth_table"))
        pr_one = sum(tt) / len(tt)
        expected = (3 / 4) ** 4  # 0.3164
        assert abs(pr_one - expected) < 1e-10, (
            f"tribes(2,8): Pr[f=1]={pr_one}, expected={expected}"
        )

    def test_tribes_k_exceeds_n_raises(self):
        """tribes(k, n) with k > n should raise ValueError."""
        with pytest.raises(ValueError):
            bf.tribes(5, 3)


class TestMajorityTotalInfluenceFormula:
    """
    Bug: LTF notebook used sqrt(n/pi) instead of sqrt(2n/pi).

    Reference: O'Donnell Corollary 5.24:
        I[Maj_n] = n * C(n-1, (n-1)/2) / 2^(n-1)
        Asymptotically: I[Maj_n] ~ sqrt(2n/pi)
    """

    def test_exact_total_influence(self):
        """Verify exact I[Maj_n] against the closed-form formula."""
        from math import comb

        for n in [3, 5, 7, 9, 11]:
            f = bf.majority(n)
            exact = f.total_influence()
            # Closed form: n * C(n-1, (n-1)/2) / 2^(n-1)
            expected = n * comb(n - 1, (n - 1) // 2) / (2 ** (n - 1))
            assert abs(exact - expected) < 1e-10, (
                f"Majority({n}): I[f]={exact}, expected={expected}"
            )

    def test_asymptotic_formula_sqrt_2n_over_pi(self):
        """I[Maj_n] / sqrt(2n/pi) → 1 as n → ∞."""
        from math import comb

        for n in [11, 15, 21]:
            f = bf.majority(n)
            exact = f.total_influence()
            asymptotic = sqrt(2 * n / pi)
            ratio = exact / asymptotic
            assert abs(ratio - 1.0) < 0.1, (
                f"Majority({n}): I[f]/sqrt(2n/pi) = {ratio:.4f} "
                f"(should be ≈1.0)"
            )

    def test_wrong_formula_sqrt_n_over_pi_fails(self):
        """sqrt(n/pi) is NOT the correct asymptotic — it's off by sqrt(2)."""
        for n in [11, 15, 21]:
            f = bf.majority(n)
            exact = f.total_influence()
            wrong_asymptotic = sqrt(n / pi)
            ratio = exact / wrong_asymptotic
            # This ratio should be ≈ sqrt(2) ≈ 1.414, NOT ≈ 1.0
            assert ratio > 1.3, (
                f"Majority({n}): I[f]/sqrt(n/pi) = {ratio:.4f} "
                f"(should be ≈sqrt(2)=1.414, not ≈1.0)"
            )
