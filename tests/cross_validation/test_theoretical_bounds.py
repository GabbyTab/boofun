"""
Validation against published theorems and known complexity values.

Every test here checks BooFun's output against a result from the
literature (a theorem, a bound, or an exact known value), so a failure
means either a bug or a definitional mismatch with the standard
references.

References:
- Huang, "Induced subgraphs of hypercubes and a proof of the Sensitivity
  Conjecture", Annals of Mathematics 190 (2019): s(f) >= sqrt(bs(f)).
- Nisan & Szegedy, "On the degree of Boolean functions as real
  polynomials", Computational Complexity 4 (1994): D(f) <= bs(f)^2.
- Buhrman & de Wolf, "Complexity measures and decision tree complexity:
  a survey" (2002): the chain s(f) <= bs(f) <= C(f) <= D(f).
- O'Donnell, "Analysis of Boolean Functions" (2014) for the BLR and
  monotonicity property-testing guarantees.
"""

import sys
from math import sqrt

import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestQueryComplexityLowerBounds:
    """The *_lower_bound helpers must actually lower-bound D(f)."""

    def test_sensitivity_bound(self):
        """s(f) <= D(f) (sensitivity is a lower bound)."""
        from boofun.analysis.query_complexity import (
            deterministic_query_complexity,
            sensitivity_lower_bound,
        )

        for func in [bf.majority(3), bf.AND(4), bf.parity(3)]:
            D_f = deterministic_query_complexity(func)
            s_bound = sensitivity_lower_bound(func)

            assert s_bound <= D_f + 0.01, f"s(f)={s_bound} > D(f)={D_f}"

    def test_block_sensitivity_bound(self):
        """bs(f) <= D(f)."""
        from boofun.analysis.query_complexity import (
            block_sensitivity_lower_bound,
            deterministic_query_complexity,
        )

        for func in [bf.AND(3), bf.OR(3)]:
            D_f = deterministic_query_complexity(func)
            bs_bound = block_sensitivity_lower_bound(func)

            assert bs_bound <= D_f + 0.01


class TestTheoreticalBounds:
    """Fundamental theorems relating complexity measures."""

    def test_huang_sensitivity_theorem(self):
        """
        Huang's Sensitivity Theorem (2019): s(f) >= sqrt(bs(f)).

        This is a breakthrough result showing sensitivity is polynomially
        related to block sensitivity.
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.huang import max_sensitivity

        for func in [bf.AND(4), bf.OR(4), bf.majority(5), bf.parity(4)]:
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)

            # s(f) >= sqrt(bs(f))
            assert s_f >= sqrt(bs_f) - 0.01, (
                f"Huang violated: s(f)={s_f}, bs(f)={bs_f}, sqrt(bs(f))={sqrt(bs_f):.2f}"
            )

    def test_nisan_szegedy_bound(self):
        """
        Nisan-Szegedy (1994): D(f) <= bs(f)^2.

        Decision tree complexity is at most block sensitivity squared.
        """
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            D_f = deterministic_query_complexity(func)
            bs_f = max_block_sensitivity(func)

            # D(f) <= bs(f)^2
            assert D_f <= bs_f**2 + 0.01, f"Nisan-Szegedy violated: D(f)={D_f}, bs(f)²={bs_f**2}"

    def test_certificate_vs_decision_tree(self):
        """C(f) <= D(f): certificate complexity lower-bounds decision tree depth."""
        from boofun.analysis.certificates import max_certificate_size
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for func in [bf.AND(4), bf.OR(4), bf.parity(3)]:
            D_f = deterministic_query_complexity(func)
            C_f = max_certificate_size(func)

            assert C_f <= D_f, f"Certificate bound violated: C(f)={C_f} > D(f)={D_f}"

    def test_block_sensitivity_vs_certificate(self):
        """bs(f) <= C(f): block sensitivity is bounded by certificate complexity."""
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.certificates import max_certificate_size

        for func in [bf.AND(4), bf.OR(4), bf.majority(3)]:
            bs_f = max_block_sensitivity(func)
            C_f = max_certificate_size(func)

            assert bs_f <= C_f, f"bs(f)={bs_f} > C(f)={C_f}"

    def test_sensitivity_vs_block_sensitivity(self):
        """s(f) <= bs(f): sensitivity is bounded by block sensitivity."""
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.huang import max_sensitivity

        for func in [bf.AND(4), bf.OR(4), bf.majority(5), bf.parity(4)]:
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)

            assert s_f <= bs_f, f"s(f)={s_f} > bs(f)={bs_f}"

    def test_complexity_measure_chain(self):
        """The full complexity chain: s(f) <= bs(f) <= C(f) <= D(f)."""
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.certificates import max_certificate_size
        from boofun.analysis.huang import max_sensitivity
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            s_f = max_sensitivity(func)
            bs_f = max_block_sensitivity(func)
            C_f = max_certificate_size(func)
            D_f = deterministic_query_complexity(func)

            assert s_f <= bs_f <= C_f <= D_f, (
                f"Chain violated: s={s_f}, bs={bs_f}, C={C_f}, D={D_f}"
            )


class TestKnownComplexityValues:
    """Exact query-complexity and degree values from the literature."""

    def test_and_query_complexity(self):
        """AND has known query complexity D(AND_n) = n (evasive function)."""
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for n in [2, 3, 4, 5]:
            D_and = deterministic_query_complexity(bf.AND(n))
            assert D_and == n, f"D(AND_{n}) should be {n}, got {D_and}"

    def test_or_query_complexity(self):
        """OR has known query complexity D(OR_n) = n (evasive function)."""
        from boofun.analysis.query_complexity import deterministic_query_complexity

        for n in [2, 3, 4, 5]:
            D_or = deterministic_query_complexity(bf.OR(n))
            assert D_or == n, f"D(OR_{n}) should be {n}, got {D_or}"

    def test_dictator_degree(self):
        """Dictator has Fourier degree 1."""
        for n in [3, 5, 7]:
            f = bf.dictator(n, 0)
            assert f.degree() == 1

    def test_parity_degree(self):
        """Parity has Fourier degree n."""
        for n in [3, 5, 7]:
            f = bf.parity(n)
            assert f.degree() == n


class TestPropertyTestingTheory:
    """Property-testing algorithms behave as their guarantees promise.

    Seeds are pinned so accept/reject outcomes are deterministic.
    """

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
