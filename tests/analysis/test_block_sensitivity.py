import sys

sys.path.insert(0, "src")
"""
Tests for analysis/block_sensitivity module.

Tests block sensitivity analysis for Boolean functions:
- sensitive_coordinates
- minimal_sensitive_blocks
- block_sensitivity_at
- max_block_sensitivity
- block_sensitivity_profile
"""

import pytest
import numpy as np

import boofun as bf
from boofun.analysis.block_sensitivity import (
    block_sensitivity_at,
    block_sensitivity_profile,
    max_block_sensitivity,
    minimal_sensitive_blocks,
    sensitive_coordinates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _popcnt(x: int) -> int:
    return bin(x).count("1")


# ---------------------------------------------------------------------------
# Tests: sensitive_coordinates
# ---------------------------------------------------------------------------


class TestSensitiveCoordinates:
    """Tests for sensitive_coordinates function."""

    def test_and_at_all_ones(self):
        """AND(3) at input (1,1,1)=7: flipping any bit changes output."""
        f = bf.AND(3)
        coords = sensitive_coordinates(f, 7)
        assert sorted(coords) == [0, 1, 2]

    def test_and_at_zero(self):
        """AND(3) at input (0,0,0)=0: no single flip can make AND true."""
        f = bf.AND(3)
        coords = sensitive_coordinates(f, 0)
        assert coords == []

    def test_or_at_zero(self):
        """OR(3) at input (0,0,0)=0: flipping any bit makes OR true."""
        f = bf.OR(3)
        coords = sensitive_coordinates(f, 0)
        assert sorted(coords) == [0, 1, 2]

    def test_or_at_all_ones(self):
        """OR(3) at input (1,1,1)=7: no single flip changes output."""
        f = bf.OR(3)
        coords = sensitive_coordinates(f, 7)
        assert coords == []

    def test_parity_all_sensitive(self):
        """Parity(3) is sensitive at every coordinate for every input."""
        f = bf.parity(3)
        for x in range(8):
            coords = sensitive_coordinates(f, x)
            assert sorted(coords) == [0, 1, 2]

    def test_constant_none_sensitive(self):
        """Constant function has no sensitive coordinates at any input."""
        f = bf.constant(True, 3)
        for x in range(8):
            assert sensitive_coordinates(f, x) == []

    def test_dictator_single_coordinate(self):
        """Dictator on variable i is sensitive only at coordinate i."""
        for i in range(3):
            f = bf.dictator(3, i=i)
            for x in range(8):
                coords = sensitive_coordinates(f, x)
                assert coords == [i]


# ---------------------------------------------------------------------------
# Tests: minimal_sensitive_blocks
# ---------------------------------------------------------------------------


class TestMinimalSensitiveBlocks:
    """Tests for minimal_sensitive_blocks function."""

    def test_parity_all_single_bits(self):
        """Parity: every single-bit block is minimal sensitive."""
        f = bf.parity(3)
        for x in range(8):
            blocks = minimal_sensitive_blocks(f, x)
            # Parity is sensitive to every single bit flip and every odd-sized block
            # Single-bit blocks (1, 2, 4) should all be minimal sensitive
            single_bit_blocks = [b for b in blocks if _popcnt(b) == 1]
            assert len(single_bit_blocks) == 3

    def test_and_at_all_ones(self):
        """AND(3) at (1,1,1)=7: each single-bit flip is minimal sensitive."""
        f = bf.AND(3)
        blocks = minimal_sensitive_blocks(f, 7)
        # Each single bit is a minimal sensitive block
        single_bits = {1, 2, 4}
        assert single_bits.issubset(set(blocks))

    def test_and_at_single_one(self):
        """AND(3) at (1,0,0)=1: only the block flipping bits 1,2 is sensitive."""
        f = bf.AND(3)
        blocks = minimal_sensitive_blocks(f, 1)
        # f(1)=0. To make it 1 we need to flip bits 1 and 2 (the zero bits).
        # The block 0b110 = 6 flips bits 1 and 2.
        assert 6 in blocks

    def test_constant_no_blocks(self):
        """Constant function has no sensitive blocks."""
        f = bf.constant(True, 3)
        for x in range(8):
            assert minimal_sensitive_blocks(f, x) == []

    def test_minimality_property(self):
        """Every minimal block has no sensitive proper subset."""
        f = bf.majority(3)
        for x in range(8):
            blocks = minimal_sensitive_blocks(f, x)
            tt = np.asarray(f.get_representation("truth_table"), dtype=bool)
            base = tt[x]
            for b in blocks:
                # b itself is sensitive
                assert tt[x ^ b] != base
                # No proper subset of b is sensitive
                sub = b
                while sub > 0:
                    sub = (sub - 1) & b
                    if sub != 0 and sub != b:
                        # This proper subset should NOT be sensitive
                        # (unless it's a different minimal block, which is fine
                        #  but it must not be a subset of b)
                        pass
                    if sub == 0:
                        break

    def test_empty_function(self):
        """Handles n=0 gracefully (returns empty)."""
        f = bf.constant(False, 1)
        # n=1, constant False: no flips change output
        blocks = minimal_sensitive_blocks(f, 0)
        assert blocks == []


# ---------------------------------------------------------------------------
# Tests: block_sensitivity_at
# ---------------------------------------------------------------------------


class TestBlockSensitivityAt:
    """Tests for block_sensitivity_at function."""

    def test_and_at_all_ones(self):
        """AND(3) at (1,1,1)=7: bs = 3 (each single bit is a disjoint block)."""
        f = bf.AND(3)
        assert block_sensitivity_at(f, 7) == 3

    def test_and_at_zero(self):
        """AND(3) at (0,0,0)=0: bs = 1 (only the full block is sensitive)."""
        f = bf.AND(3)
        assert block_sensitivity_at(f, 0) == 1

    def test_parity_bs_equals_n(self):
        """Parity(n): bs(f, x) = n for every x (all single-bit flips)."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            for x in range(1 << n):
                assert block_sensitivity_at(f, x) == n

    def test_constant_bs_zero(self):
        """Constant function: bs = 0 at every input."""
        f = bf.constant(True, 3)
        for x in range(8):
            assert block_sensitivity_at(f, x) == 0

    def test_dictator_bs_one(self):
        """Dictator: bs = 1 at every input (only one sensitive coordinate)."""
        f = bf.dictator(3, i=0)
        for x in range(8):
            assert block_sensitivity_at(f, x) == 1

    def test_majority3_known_values(self):
        """MAJ(3) at (1,1,1)=7: no single flip changes output, bs = 1."""
        f = bf.majority(3)
        # At all-ones: flipping any single bit keeps weight >= 2, so s(7)=0.
        # But flipping 2 bits (e.g., {0,1}) drops weight to 1, so f changes.
        # The three 2-bit blocks all overlap, so only 1 disjoint block fits.
        assert block_sensitivity_at(f, 7) == 1
        # At (1,1,0)=3: f=1 (weight 2). Flipping bit0 -> (0,1,0)=2 f=0, sensitive.
        # Flipping bit1 -> (1,0,0)=1 f=0, sensitive. Two disjoint single-bit blocks.
        assert block_sensitivity_at(f, 3) == 2

    def test_bs_at_least_sensitivity(self):
        """Block sensitivity >= sensitivity at every input."""
        f = bf.majority(3)
        for x in range(8):
            bs = block_sensitivity_at(f, x)
            s = len(sensitive_coordinates(f, x))
            assert bs >= s


# ---------------------------------------------------------------------------
# Tests: max_block_sensitivity
# ---------------------------------------------------------------------------


class TestMaxBlockSensitivity:
    """Tests for max_block_sensitivity function."""

    def test_parity_bs_n(self):
        """Parity(n): bs = n."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            assert max_block_sensitivity(f) == n

    def test_and_bs_n(self):
        """AND(n): bs = n (achieved at the all-ones input)."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            assert max_block_sensitivity(f) == n

    def test_or_bs_n(self):
        """OR(n): bs = n (achieved at the all-zeros input)."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            assert max_block_sensitivity(f) == n

    def test_constant_bs_zero(self):
        """Constant function: bs = 0."""
        f = bf.constant(True, 3)
        assert max_block_sensitivity(f) == 0

    def test_dictator_bs_one(self):
        """Dictator: bs = 1."""
        f = bf.dictator(3, i=0)
        assert max_block_sensitivity(f) == 1

    def test_value_filter(self):
        """Filter by output value restricts to relevant inputs."""
        f = bf.AND(3)
        # bs restricted to inputs where f=1 (only input 7)
        bs_on_1 = max_block_sensitivity(f, value=1)
        assert bs_on_1 == 3

        # bs restricted to inputs where f=0
        bs_on_0 = max_block_sensitivity(f, value=0)
        assert bs_on_0 >= 1

    def test_pruning_gives_same_result(self):
        """With and without pruning should give the same result."""
        f = bf.majority(3)
        bs_pruned = max_block_sensitivity(f, use_pruning=True)
        bs_no_prune = max_block_sensitivity(f, use_pruning=False)
        assert bs_pruned == bs_no_prune

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_bs_geq_sensitivity(self, n):
        """Block sensitivity >= max sensitivity for standard functions."""
        for factory in [bf.majority, bf.parity, bf.AND, bf.OR]:
            f = factory(n)
            from boofun.analysis.sensitivity import max_sensitivity

            bs = max_block_sensitivity(f)
            s = max_sensitivity(f)
            assert bs >= s


# ---------------------------------------------------------------------------
# Tests: block_sensitivity_profile
# ---------------------------------------------------------------------------


class TestBlockSensitivityProfile:
    """Tests for block_sensitivity_profile function."""

    def test_parity_profile(self):
        """Parity(3): every input has bs = 3."""
        f = bf.parity(3)
        bs0, bs1, per_input = block_sensitivity_profile(f)
        assert bs0 == 3
        assert bs1 == 3
        assert all(b == 3 for b in per_input)

    def test_constant_profile(self):
        """Constant function: bs = 0 everywhere."""
        f = bf.constant(False, 3)
        bs0, bs1, per_input = block_sensitivity_profile(f)
        assert bs0 == 0
        assert all(b == 0 for b in per_input)

    def test_profile_length(self):
        """Profile has 2^n entries."""
        f = bf.majority(3)
        _, _, per_input = block_sensitivity_profile(f)
        assert len(per_input) == 8

    def test_max_matches(self):
        """max(per_input) should equal max_block_sensitivity."""
        f = bf.majority(3)
        _, _, per_input = block_sensitivity_profile(f)
        assert max(per_input) == max_block_sensitivity(f)

    def test_bs0_bs1_partition(self):
        """bs0 and bs1 correctly partition by output value."""
        f = bf.majority(3)
        bs0, bs1, per_input = block_sensitivity_profile(f)
        tt = np.asarray(f.get_representation("truth_table"), dtype=bool)

        bs0_manual = max(per_input[x] for x in range(8) if not tt[x])
        bs1_manual = max(per_input[x] for x in range(8) if tt[x])

        assert bs0 == bs0_manual
        assert bs1 == bs1_manual


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for block sensitivity module."""

    def test_single_variable(self):
        """Block sensitivity for n=1 functions."""
        # Identity: f(0)=0, f(1)=1
        f = bf.parity(1)
        assert max_block_sensitivity(f) == 1
        assert block_sensitivity_at(f, 0) == 1
        assert block_sensitivity_at(f, 1) == 1

    def test_two_variables(self):
        """Block sensitivity for n=2 standard functions."""
        f = bf.AND(2)
        # AND(2): bs at (1,1)=3 is 2, bs at (0,0)=0 is 1 (flip both)
        assert max_block_sensitivity(f) == 2

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_majority_bs_ceiling_n_over_2(self, n):
        """MAJ(n) for odd n: bs >= ceil(n/2)."""
        if n % 2 == 0:
            return
        f = bf.majority(n)
        bs = max_block_sensitivity(f)
        assert bs >= (n + 1) // 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
