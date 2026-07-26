"""
Cross-validation tests comparing BooFun with BoolForge.

BoolForge (Kadelka & Coberly, 2025) is a Python library for Boolean
function and network analysis focused on systems biology.

These tests verify that BooFun and BoolForge agree on canalization,
basic function properties, and exact spectral quantities.

How these run: BoolForge is not a test dependency, so this module skips
under the regular per-PR matrix. The Cross-Validation (live) workflow
(.github/workflows/cross-validation.yml) installs BoolForge pinned to a
commit (SHA inlined in its install step) and runs this module un-skipped on main
pushes, weekly, and on demand, filing an issue on failure.

Seeds and tolerances: BoolForge estimates some measures by Monte Carlo
(nsim samples) unless ``exact=True`` is passed. We call every such API
with ``exact=True``, so all comparisons here are deterministic — integer
and boolean checks are exact, and float comparisons use atol=1e-10,
tight enough to fail if a BoolForge API ever silently falls back to
sampling. No RNG seeds are needed.
"""

from typing import Any, ClassVar

import pytest

# Try to import boolforge - skip if not installed
try:
    import boolforge

    HAS_BOOLFORGE = True
except ImportError:
    HAS_BOOLFORGE = False

import boofun as bf
from boofun.analysis.canalization import (
    get_canalizing_depth,
    get_essential_variables,
    get_symmetry_groups,
    is_canalizing,
)


@pytest.mark.skipif(not HAS_BOOLFORGE, reason="boolforge not installed")
class TestBoolForgeCrossValidation:
    """Cross-validation tests with BoolForge library."""

    # Test cases: (name, boofun_func, truth_table_list)
    TEST_CASES: ClassVar[list[tuple[str, Any, list[int]]]] = [
        ("AND(3)", bf.AND(3), [0, 0, 0, 0, 0, 0, 0, 1]),
        ("OR(3)", bf.OR(3), [0, 1, 1, 1, 1, 1, 1, 1]),
        ("PARITY(3)", bf.parity(3), [0, 1, 1, 0, 1, 0, 0, 1]),
        ("MAJ(3)", bf.majority(3), [0, 0, 0, 1, 0, 1, 1, 1]),
        ("AND(4)", bf.AND(4), [0] * 15 + [1]),
        ("OR(4)", bf.OR(4), [0] + [1] * 15),
        ("PARITY(4)", bf.parity(4), [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
    ]

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_is_canalizing(self, name, bf_func, tt):
        """Verify is_canalizing matches between libraries."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = is_canalizing(bf_func)
        boolforge_result = boolforge_func.is_canalizing()

        assert bf_result == boolforge_result, (
            f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"
        )

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_canalizing_depth(self, name, bf_func, tt):
        """Verify canalizing_depth matches between libraries."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = get_canalizing_depth(bf_func)
        boolforge_result = boolforge_func.get_canalizing_depth()

        assert bf_result == boolforge_result, (
            f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"
        )

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_essential_variables(self, name, bf_func, tt):
        """Verify number of essential variables matches."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = len(get_essential_variables(bf_func))
        boolforge_result = boolforge_func.get_number_of_essential_variables()

        assert bf_result == boolforge_result, (
            f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"
        )

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_is_monotonic(self, name, bf_func, tt):
        """Verify is_monotonic matches between libraries."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = bf_func.is_monotone()
        boolforge_result = boolforge_func.is_monotonic()

        assert bf_result == boolforge_result, (
            f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"
        )

    def test_symmetry_groups_majority(self):
        """Verify symmetry groups for symmetric functions."""
        bf_maj = bf.majority(3)
        boolforge_maj = boolforge.BooleanFunction([0, 0, 0, 1, 0, 1, 1, 1])

        bf_groups = get_symmetry_groups(bf_maj)
        boolforge_groups = boolforge_maj.get_symmetry_groups()

        # Both should show all variables in one group (majority is symmetric)
        assert len(bf_groups) == 1, "BooFun should have one symmetry group"
        assert len(boolforge_groups) == 1, "BoolForge should have one symmetry group"

        # Convert to comparable format
        bf_group_size = len(next(iter(bf_groups)))
        boolforge_group_size = len(boolforge_groups[0])

        assert bf_group_size == 3, f"BooFun group size: {bf_group_size}"
        assert boolforge_group_size == 3, f"BoolForge group size: {boolforge_group_size}"

    def test_nested_canalizing_and(self):
        """AND functions should be nested canalizing (depth = n)."""
        for n in [2, 3, 4, 5]:
            bf_func = bf.AND(n)
            tt = [0] * (2**n - 1) + [1]
            boolforge_func = boolforge.BooleanFunction(tt)

            bf_depth = get_canalizing_depth(bf_func)
            boolforge_depth = boolforge_func.get_canalizing_depth()

            # AND is fully nested canalizing (depth = n)
            assert bf_depth == n, f"AND({n}): BooFun depth={bf_depth}"
            assert boolforge_depth == n, f"AND({n}): BoolForge depth={boolforge_depth}"

    def test_parity_not_canalizing(self):
        """Parity functions should not be canalizing."""
        for n in [2, 3, 4]:
            bf_func = bf.parity(n)
            # Build parity truth table
            tt = [bin(x).count("1") % 2 for x in range(2**n)]
            boolforge_func = boolforge.BooleanFunction(tt)

            assert not is_canalizing(bf_func), f"PARITY({n}) should not be canalizing"
            assert not boolforge_func.is_canalizing(), (
                f"BoolForge PARITY({n}) should not be canalizing"
            )

            assert get_canalizing_depth(bf_func) == 0, f"PARITY({n}) depth should be 0"
            assert boolforge_func.get_canalizing_depth() == 0


@pytest.mark.skipif(not HAS_BOOLFORGE, reason="boolforge not installed")
class TestBoolForgeSpectral:
    """Cross-validate spectral quantities (influences, total influence).

    Both sides compute exact rational values (BoolForge in exact mode,
    BooFun by full enumeration), so the tolerance is a tight 1e-10 for
    floating-point representation only. Deliberately NOT looser: with a
    sloppy tolerance this test could not tell exact mode apart from
    BoolForge's default Monte Carlo estimate (nsim=10000 has sampling
    noise around 5e-3), which is exactly the regression it must catch.

    Where a closed form exists it is asserted too, making these three-way
    checks: BooFun == BoolForge == literature (O'Donnell 2014, Ch. 2).
    """

    TOL = 1e-10

    # (name, boofun function, closed-form per-variable influence or None)
    # parity(n): Inf_i = 1. AND(n): Inf_i = 2^(1-n). majority(5):
    # Inf_i = C(4,2)/2^4 = 6/16 = 0.375.
    SPECTRAL_CASES: ClassVar[list[tuple[str, Any, float | None]]] = [
        ("parity4", bf.parity(4), 1.0),
        ("and4", bf.AND(4), 2.0 ** (1 - 4)),
        ("majority5", bf.majority(5), 0.375),
    ]

    @pytest.mark.parametrize("name,f_bf,closed_form", SPECTRAL_CASES)
    def test_activities_vs_influences(self, name, f_bf, closed_form):
        """BoolForge activities (exact) match BooFun influences exactly."""
        import numpy as np

        n = f_bf.n_vars
        tt = [int(f_bf.evaluate(x)) for x in range(1 << n)]
        f_forge = boolforge.BooleanFunction(tt)

        our_influences = np.asarray(f_bf.influences())
        forge_activities = np.asarray(f_forge.get_activities(exact=True))

        assert np.allclose(our_influences, forge_activities, rtol=0.0, atol=self.TOL), (
            f"{name}: BooFun={our_influences}, BoolForge={forge_activities}"
        )
        if closed_form is not None:
            assert np.allclose(our_influences, closed_form, rtol=0.0, atol=self.TOL), (
                f"{name}: BooFun={our_influences}, closed form={closed_form}"
            )

    @pytest.mark.parametrize("name,f_bf,closed_form", SPECTRAL_CASES)
    def test_average_sensitivity_vs_total_influence(self, name, f_bf, closed_form):
        """BoolForge avg sensitivity (exact, unnormalized) == BooFun total influence."""
        n = f_bf.n_vars
        tt = [int(f_bf.evaluate(x)) for x in range(1 << n)]
        f_forge = boolforge.BooleanFunction(tt)

        our_ti = f_bf.total_influence()
        forge_sens = f_forge.get_average_sensitivity(exact=True, normalized=False)

        assert abs(our_ti - forge_sens) < self.TOL, (
            f"{name}: BooFun={our_ti}, BoolForge={forge_sens}"
        )
        if closed_form is not None:
            assert abs(our_ti - n * closed_form) < self.TOL, (
                f"{name}: BooFun TI={our_ti}, closed form={n * closed_form}"
            )


@pytest.mark.skipif(not HAS_BOOLFORGE, reason="boolforge not installed")
class TestBoolForgeEdgeCases:
    """Test edge cases and special functions."""

    def test_constant_function(self):
        """Constant functions have depth 0."""
        # All zeros
        tt_zero = [0, 0, 0, 0]
        boolforge_zero = boolforge.BooleanFunction(tt_zero)
        assert boolforge_zero.get_canalizing_depth() == 0

        # All ones
        tt_one = [1, 1, 1, 1]
        boolforge_one = boolforge.BooleanFunction(tt_one)
        assert boolforge_one.get_canalizing_depth() == 0

    def test_dictator_function(self):
        """Dictator functions are canalizing with depth 1."""
        # x0 (dictator on variable 0)
        tt_dict = [0, 1, 0, 1]  # f(x0, x1) = x0
        bf_dict = bf.dictator(2, 0)
        boolforge_dict = boolforge.BooleanFunction(tt_dict)

        bf_depth = get_canalizing_depth(bf_dict)
        boolforge_depth = boolforge_dict.get_canalizing_depth()

        assert bf_depth == 1, f"Dictator BooFun depth: {bf_depth}"
        assert boolforge_depth == 1, f"Dictator BoolForge depth: {boolforge_depth}"
