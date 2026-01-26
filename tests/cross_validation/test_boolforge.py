"""
Cross-validation tests comparing BooFun with BoolForge.

BoolForge (Kadelka & Coberly, 2025) is a Python library for Boolean
function and network analysis focused on systems biology.

These tests verify that BooFun and BoolForge agree on canalization
and basic function properties.

Note: BoolForge uses Monte Carlo for some measures (like average
sensitivity), so we only test exact measures here.
"""

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
    is_nested_canalizing,
)


@pytest.mark.skipif(not HAS_BOOLFORGE, reason="boolforge not installed")
class TestBoolForgeCrossValidation:
    """Cross-validation tests with BoolForge library."""

    # Test cases: (name, boofun_func, truth_table_list)
    TEST_CASES = [
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

        assert bf_result == boolforge_result, f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_canalizing_depth(self, name, bf_func, tt):
        """Verify canalizing_depth matches between libraries."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = get_canalizing_depth(bf_func)
        boolforge_result = boolforge_func.get_canalizing_depth()

        assert bf_result == boolforge_result, f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_essential_variables(self, name, bf_func, tt):
        """Verify number of essential variables matches."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = len(get_essential_variables(bf_func))
        boolforge_result = boolforge_func.get_number_of_essential_variables()

        assert bf_result == boolforge_result, f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"

    @pytest.mark.parametrize("name,bf_func,tt", TEST_CASES)
    def test_is_monotonic(self, name, bf_func, tt):
        """Verify is_monotonic matches between libraries."""
        boolforge_func = boolforge.BooleanFunction(tt)

        bf_result = bf_func.is_monotone()
        boolforge_result = boolforge_func.is_monotonic()

        assert bf_result == boolforge_result, f"{name}: BooFun={bf_result}, BoolForge={boolforge_result}"

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
        bf_group_size = len(list(bf_groups)[0])
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
            assert not boolforge_func.is_canalizing(), f"BoolForge PARITY({n}) should not be canalizing"

            assert get_canalizing_depth(bf_func) == 0, f"PARITY({n}) depth should be 0"
            assert boolforge_func.get_canalizing_depth() == 0


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
