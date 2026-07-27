"""
Exhaustive checks of the query-complexity guarantee tiers (issue #119).

The module docstring of ``boofun.analysis.query_complexity`` classifies
every function as exact, certified lower bound, or clamped estimate. This
suite enforces those claims exhaustively over all Boolean functions on
n = 2 and n = 3 variables (272 non-trivial functions), plus closed-form
anchors from the literature for larger named functions.
"""

from __future__ import annotations

from math import ceil, isclose, sqrt

import pytest

import boofun as bf
from boofun.analysis.complexity import (
    decision_tree_depth,
    max_certificate_complexity,
)
from boofun.analysis.fourier import fourier_degree
from boofun.analysis.query_complexity import (
    ambainis_complexity,
    approximate_degree,
    bounded_error_randomized_complexity,
    exact_quantum_complexity,
    general_adversary_bound,
    nondeterministic_complexity,
    nondeterministic_degree,
    one_sided_approximate_degree,
    polynomial_method_bound,
    quantum_query_complexity,
    spectral_adversary_bound,
    threshold_degree,
    zero_error_randomized_complexity,
)


def all_functions(n: int):
    for code in range(1 << (1 << n)):
        yield bf.create([(code >> x) & 1 for x in range(1 << n)])


@pytest.mark.parametrize("n", [2, 3])
class TestExhaustiveInvariants:
    def test_degree_hierarchy(self, n):
        """thrdeg <= deg2 <= deg and ndeg <= deg for every function."""
        for f in all_functions(n):
            deg = fourier_degree(f)
            deg2 = approximate_degree(f)
            thr = threshold_degree(f)
            assert thr <= deg2 <= deg, f.get_representation("truth_table")
            for side in (0, 1):
                assert nondeterministic_degree(f, side) <= deg

    def test_epsilon_monotonicity(self, n):
        """Stricter epsilon can only raise the approximate degree."""
        for f in all_functions(n):
            assert approximate_degree(f, 1 / 4) >= approximate_degree(f, 1 / 3)

    def test_estimates_respect_certified_windows(self, n):
        for f in all_functions(n):
            D = decision_tree_depth(f)
            if D == 0:
                continue  # constant functions return 0 everywhere
            deg = fourier_degree(f)
            deg2 = approximate_degree(f)
            q2 = quantum_query_complexity(f)
            qe = exact_quantum_complexity(f)
            assert deg2 / 2 - 1e-9 <= q2 <= D + 1e-9
            assert deg / 2 - 1e-9 <= qe <= D + 1e-9
            assert polynomial_method_bound(f) == pytest.approx(deg2 / 2)
            for r in (
                bounded_error_randomized_complexity(f),
                zero_error_randomized_complexity(f),
            ):
                assert 0 < r <= D + 1e-9

    def test_nondeterministic_complexity_is_certificate_complexity(self, n):
        for f in all_functions(n):
            if decision_tree_depth(f) == 0:
                continue
            for side in (0, 1):
                assert nondeterministic_complexity(f, side) == max_certificate_complexity(f, side)

    def test_adversary_bounds_below_positive_adv_cap(self, n):
        """Spalek-Szegedy: the positive adversary is at most sqrt(C0*C1),
        so our feasible witnesses must respect that cap too."""
        for f in all_functions(n):
            if decision_tree_depth(f) == 0:
                assert general_adversary_bound(f) == 0.0
                continue
            cap = sqrt(max_certificate_complexity(f, 0) * max_certificate_complexity(f, 1))
            assert ambainis_complexity(f) <= cap + 1e-9
            assert spectral_adversary_bound(f) <= cap + 1e-9

    def test_adversary_deterministic(self, n):
        """No hidden randomness: repeated evaluation is bit-identical."""
        for f in all_functions(n):
            assert ambainis_complexity(f) == ambainis_complexity(f)
            assert spectral_adversary_bound(f) == spectral_adversary_bound(f)


class TestClosedFormAnchors:
    """Literature values for named families."""

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_parity(self, n):
        f = bf.parity(n)
        # Any 1/3-approximation and any sign representation of parity
        # requires full degree (Minsky-Papert).
        assert approximate_degree(f) == n
        assert threshold_degree(f) == n
        # ndeg(PARITY_n) = ceil(n/2) (de Wolf 2003, Theorem 2.4).
        assert nondeterministic_degree(f, 1) == ceil(n / 2)
        # ADV(PARITY_n) = n; the sensitive-edge witness achieves it.
        assert isclose(general_adversary_bound(f), n)
        # Q2 estimate clamps to the certified polynomial-method bound n/2.
        assert quantum_query_complexity(f) >= n / 2

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_and_or(self, n):
        for f in (bf.AND(n), bf.OR(n)):
            # Threshold functions have threshold degree 1.
            assert threshold_degree(f) == 1
            # ADV(AND_n) = ADV(OR_n) = sqrt(n), achieved by the
            # sensitive-edge witness.
            assert isclose(general_adversary_bound(f), sqrt(n))
        # ndeg(OR_n, side=1) = 1 (p = sum of the variables);
        # ndeg(AND_n, side=1) = n (support is a single point).
        assert nondeterministic_degree(bf.OR(n), 1) == 1
        assert nondeterministic_degree(bf.AND(n), 1) == n

    def test_majority3(self):
        f = bf.majority(3)
        assert threshold_degree(f) == 1  # MAJ is a linear threshold function
        assert approximate_degree(f) == 1  # a degree-1 1/3-approximation exists
        assert isclose(general_adversary_bound(f), 2.0)  # ADV(MAJ3) = 2

    def test_dictator(self):
        f = bf.dictator(3, 0)
        assert approximate_degree(f) == 1
        assert threshold_degree(f) == 1
        assert isclose(general_adversary_bound(f), 1.0)

    def test_one_sided_degree_definition(self):
        # AND_3, side=1: p must be >= 2/3 only at the all-ones point and
        # <= 1/3 elsewhere; a linear polynomial suffices.
        assert one_sided_approximate_degree(bf.AND(3), side=1) == 1
        # PARITY needs full degree even one-sidedly.
        assert one_sided_approximate_degree(bf.parity(3), side=1) == 3

    def test_size_cap_raises(self):
        f = bf.parity(13)
        with pytest.raises(ValueError, match="n <= 12"):
            approximate_degree(f)
        with pytest.raises(ValueError, match="n <= 12"):
            threshold_degree(f)
        with pytest.raises(ValueError, match="n <= 12"):
            nondeterministic_degree(f)
        with pytest.raises(ValueError, match="n <= 12"):
            spectral_adversary_bound(f)

    @pytest.mark.parametrize("bad_epsilon", [-1.0, -1e-9, 0.5, 1.0, 2.0])
    def test_invalid_epsilon_rejected(self, bad_epsilon):
        """epsilon outside [0, 1/2) must raise, not return a plausible
        degree (epsilon = -1 previously returned n)."""
        f = bf.AND(3)
        with pytest.raises(ValueError, match="epsilon"):
            approximate_degree(f, epsilon=bad_epsilon)
        with pytest.raises(ValueError, match="epsilon"):
            one_sided_approximate_degree(f, epsilon=bad_epsilon)

    def test_epsilon_zero_is_exact_degree(self):
        """deg_0(f) = deg(f): only the exact representation has error 0."""
        for f in (bf.majority(3), bf.parity(3), bf.AND(3)):
            assert approximate_degree(f, epsilon=0.0) == fourier_degree(f)

    def test_qe_at_least_q2(self):
        """The estimate pair must respect the theorem Q2(f) <= QE(f)
        (MAJ3 previously printed Q2 = 1.73, QE = 1.50)."""
        for n in (2, 3):
            for f in all_functions(n):
                assert exact_quantum_complexity(f) >= quantum_query_complexity(f) - 1e-9


def _ndeg_exact_rational(truth_table: list[int], n: int, side: int) -> int:
    """Definition-literal nondeterministic degree in exact rational
    arithmetic (sympy): min degree d such that the degree-<=d polynomials
    vanishing on all off-side inputs (an exact nullspace over Q) contain,
    for every side input y, an element nonzero at y. Independent of the
    floating-point SVD implementation under test."""
    from itertools import combinations

    from sympy import Matrix

    side_rows = [x for x in range(1 << n) if truth_table[x] == side]
    off_rows = [x for x in range(1 << n) if truth_table[x] != side]
    if not off_rows or not side_rows:
        return 0

    for degree in range(n + 1):
        columns = [subset for d in range(degree + 1) for subset in combinations(range(n), d)]

        def chi(x: int, subset: tuple[int, ...]) -> int:
            return -1 if bin(x & sum(1 << i for i in subset)).count("1") % 2 else 1

        A_off = Matrix([[chi(x, s) for s in columns] for x in off_rows])
        null_basis = A_off.nullspace()
        if not null_basis:
            continue
        feasible = True
        for y in side_rows:
            row = Matrix([[chi(y, s) for s in columns]])
            if all((row * v)[0] == 0 for v in null_basis):
                feasible = False
                break
        if feasible:
            return degree
    return n


class TestNdegRationalCrossCheck:
    """Naive-check rung for nondeterministic_degree: the production
    implementation uses floating-point SVD rank with fixed tolerances;
    this compares it against exact rational-nullspace arithmetic."""

    def test_all_two_variable_functions_both_sides(self):
        pytest.importorskip("sympy")
        for f in all_functions(2):
            tt = [int(v) for v in f.get_representation("truth_table")]
            for side in (0, 1):
                assert nondeterministic_degree(f, side) == _ndeg_exact_rational(tt, 2, side), (
                    tt,
                    side,
                )

    def test_all_three_variable_functions_side_one(self):
        pytest.importorskip("sympy")
        for f in all_functions(3):
            tt = [int(v) for v in f.get_representation("truth_table")]
            assert nondeterministic_degree(f, 1) == _ndeg_exact_rational(tt, 3, 1), tt
