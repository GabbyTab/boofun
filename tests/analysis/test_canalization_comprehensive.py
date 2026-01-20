"""
Comprehensive tests for canalization analysis module.

Tests for canalizing Boolean functions from systems biology perspective.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.canalization import (
    effective_degree,
    get_canalizing_depth,
    get_canalizing_variables,
    get_essential_variables,
    input_redundancy,
    is_canalizing,
)


class TestIsCanalizing:
    """Test is_canalizing function."""

    def test_and_is_canalizing(self):
        """AND function is canalizing (0 forces output to 0)."""
        for n in [2, 3, 4, 5]:
            f = bf.AND(n)
            assert is_canalizing(f), f"AND_{n} should be canalizing"

    def test_or_is_canalizing(self):
        """OR function is canalizing (1 forces output to 1)."""
        for n in [2, 3, 4, 5]:
            f = bf.OR(n)
            assert is_canalizing(f), f"OR_{n} should be canalizing"

    def test_parity_not_canalizing(self):
        """Parity/XOR is NOT canalizing."""
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            assert not is_canalizing(f), f"Parity_{n} should NOT be canalizing"

    def test_majority_not_canalizing(self):
        """Majority is NOT canalizing (no single variable dominates)."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            assert not is_canalizing(f), f"Majority_{n} should NOT be canalizing"

    def test_constant_is_canalizing(self):
        """Constant functions are trivially canalizing."""
        f_zero = bf.create([0, 0, 0, 0])
        f_one = bf.create([1, 1, 1, 1])

        assert is_canalizing(f_zero)
        assert is_canalizing(f_one)

    def test_dictator_is_canalizing(self):
        """Dictator functions are canalizing."""
        for n in [2, 3, 4]:
            for i in range(n):
                f = bf.dictator(n, i)
                assert is_canalizing(f), f"Dictator_{n}_{i} should be canalizing"


class TestGetCanalizingVariables:
    """Test get_canalizing_variables function."""

    def test_and_canalizing_info(self):
        """AND should have all variables canalizing on input 0."""
        f = bf.AND(3)
        canalizing_vars = get_canalizing_variables(f)

        # Each variable should canalize: when x_i=0, output=0
        assert len(canalizing_vars) >= 1

        # Check structure of returned data
        for cv in canalizing_vars:
            assert "variable" in cv
            assert "canalizing_input" in cv
            assert "canalized_output" in cv

    def test_or_canalizing_info(self):
        """OR should have all variables canalizing on input 1."""
        f = bf.OR(3)
        canalizing_vars = get_canalizing_variables(f)

        assert len(canalizing_vars) >= 1

        # For OR, when any x_i=1, output=1
        for cv in canalizing_vars:
            if cv["canalizing_input"] == 1:
                assert cv["canalized_output"] == 1

    def test_parity_no_canalizing_vars(self):
        """Parity should have no canalizing variables."""
        f = bf.parity(3)
        canalizing_vars = get_canalizing_variables(f)

        assert len(canalizing_vars) == 0


class TestCanalizingDepth:
    """Test get_canalizing_depth function."""

    def test_and_depth(self):
        """AND has canalizing depth n (all variables canalize in sequence)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            depth = get_canalizing_depth(f)
            assert depth == n, f"AND_{n} should have depth {n}, got {depth}"

    def test_or_depth(self):
        """OR has canalizing depth n."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            depth = get_canalizing_depth(f)
            assert depth == n, f"OR_{n} should have depth {n}, got {depth}"

    def test_parity_depth_zero(self):
        """Parity has canalizing depth 0."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            depth = get_canalizing_depth(f)
            assert depth == 0, f"Parity_{n} should have depth 0, got {depth}"

    def test_majority_low_depth(self):
        """Majority has low or zero canalizing depth."""
        f = bf.majority(3)
        depth = get_canalizing_depth(f)
        assert depth < 3  # Majority is not fully nested canalizing


class TestEssentialVariables:
    """Test get_essential_variables function."""

    def test_and_all_essential(self):
        """AND has all variables essential."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            essential = get_essential_variables(f)
            assert len(essential) == n
            assert set(essential) == set(range(n))

    def test_or_all_essential(self):
        """OR has all variables essential."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            essential = get_essential_variables(f)
            assert len(essential) == n

    def test_parity_all_essential(self):
        """Parity has all variables essential."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            essential = get_essential_variables(f)
            assert len(essential) == n

    def test_constant_no_essential(self):
        """Constant functions have no essential variables."""
        f = bf.create([0, 0, 0, 0])
        essential = get_essential_variables(f)
        assert len(essential) == 0

    def test_dictator_one_essential(self):
        """Dictator has exactly one essential variable."""
        for n in [3, 4, 5]:
            f = bf.dictator(n, 0)  # Test with dictator on variable 0
            essential = get_essential_variables(f)
            assert (
                len(essential) == 1
            ), f"Dictator should have 1 essential var, got {len(essential)}"

    def test_dummy_variable_not_essential(self):
        """Variables that don't affect output are not essential."""
        # f(x0, x1, x2) = x0 AND x1 (x2 is dummy)
        tt = [0, 0, 0, 1, 0, 0, 0, 1]
        f = bf.create(tt)
        essential = get_essential_variables(f)

        # Only x0 and x1 should be essential
        assert len(essential) == 2
        assert 0 in essential
        assert 1 in essential
        assert 2 not in essential


class TestInputRedundancy:
    """Test input_redundancy function."""

    def test_constant_full_redundancy(self):
        """Constant function has full input redundancy."""
        f = bf.create([0, 0, 0, 0])
        redundancy = input_redundancy(f)
        assert abs(redundancy - 1.0) < 1e-10

    def test_parity_no_redundancy(self):
        """Parity has no input redundancy."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            redundancy = input_redundancy(f)
            assert abs(redundancy) < 1e-10

    def test_redundancy_bounded(self):
        """Redundancy should be in [0, 1]."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3)]:
            redundancy = input_redundancy(func)
            assert 0 <= redundancy <= 1


class TestEffectiveDegree:
    """Test effective_degree function."""

    def test_degree_bounded(self):
        """Effective degree should be in [0, n]."""
        for n in [3, 4, 5]:
            for func in [bf.AND(n), bf.OR(n), bf.parity(n)]:
                degree = effective_degree(func)
                assert 0 <= degree <= n

    def test_parity_full_degree(self):
        """Parity has full effective degree (all variables matter)."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            degree = effective_degree(f)
            # Parity's effective degree should be close to n
            assert degree >= n - 1

    def test_constant_zero_degree(self):
        """Constant function has zero effective degree."""
        f = bf.create([0, 0, 0, 0])
        degree = effective_degree(f)
        assert abs(degree) < 1e-10


class TestCanalizationEdgeCases:
    """Test edge cases for canalization analysis."""

    def test_single_variable_function(self):
        """Single variable functions should work."""
        f_id = bf.create([0, 1])  # Identity
        f_not = bf.create([1, 0])  # NOT

        assert is_canalizing(f_id)
        assert is_canalizing(f_not)

    def test_two_variable_functions(self):
        """Two variable functions should work."""
        # AND, OR, XOR, NAND, NOR, etc.
        and_tt = [0, 0, 0, 1]
        or_tt = [0, 1, 1, 1]
        xor_tt = [0, 1, 1, 0]

        assert is_canalizing(bf.create(and_tt))
        assert is_canalizing(bf.create(or_tt))
        assert not is_canalizing(bf.create(xor_tt))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
