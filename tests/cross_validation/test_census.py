"""
Cross-validation against published exhaustive censuses (issue #114).

These tests enumerate *every* Boolean function of n variables (all 2**(2**n)
truth tables — truth tables, not NPN-equivalence classes) through BooFun's
public API and compare aggregate counts against published values. An
always-True predicate, an always-False predicate, or a systematic
convention error cannot match these counts.

Sources (all comparisons are exact integer equality):

- Monotone functions: Dedekind numbers, OEIS A000372
  (2, 3, 6, 20, 168 for n = 0..4).
- Unate ("bipolar") functions: OEIS A245079
  (2, 4, 14, 104, 2170 for n = 0..4).
- Canalizing functions: OEIS A102449 (2, 4, 14, 120, 3514 for n = 0..4).
- Bent functions: OEIS A004491 (8 for n = 2, 896 for n = 4).
- Canalizing-depth histogram for n = 4: He & Macauley (2016),
  "Stratification and enumeration of Boolean functions by canalizing
  depth", Physica D 314, https://doi.org/10.1016/j.physd.2015.09.016.

Conventions (do not collapse these two):

- ``is_canalizing`` treats constant functions as trivially canalizing,
  matching A102449 (e.g. a(1) = 4 counts all four one-variable functions).
- ``get_canalizing_depth`` assigns constant functions depth 0, matching
  He & Macauley. Hence for each n the depth-0 bucket equals
  (non-canalizing count) + 2, e.g. 62,022 + 2 = 62,024 for n = 4.

How these run: the n <= 3 censuses (at most 256 functions) run on every
PR. The full n = 4 census (65,536 functions, ~15 s locally) runs when
``BOOFUN_FULL_CENSUS=1``, which CI sets for non-PR events (pushes to
main, release tags, and manual runs) — the same cadence as the full test
matrix. Run it locally with ``BOOFUN_FULL_CENSUS=1 pytest
tests/cross_validation/test_census.py``.
"""

import os
from collections import Counter

import pytest

import boofun as bf
from boofun.analysis.basic_properties import is_monotone, is_unate
from boofun.analysis.canalization import get_canalizing_depth, is_canalizing
from boofun.analysis.cryptographic import is_bent

FULL_CENSUS = os.environ.get("BOOFUN_FULL_CENSUS") == "1"

# Published counts, indexed by n. Sources in the module docstring.
MONOTONE_COUNTS = {2: 6, 3: 20, 4: 168}  # OEIS A000372
UNATE_COUNTS = {2: 14, 3: 104, 4: 2170}  # OEIS A245079
CANALIZING_COUNTS = {2: 14, 3: 120, 4: 3514}  # OEIS A102449
BENT_COUNTS = {2: 8, 4: 896}  # OEIS A004491
DEPTH_HISTOGRAM_4 = {0: 62024, 1: 2184, 2: 336, 3: 256, 4: 736}  # He & Macauley (2016)


def _truth_table(bits: int, size: int) -> list[int]:
    """Truth table of function number ``bits``: entry x is bit x of ``bits``."""
    return [(bits >> x) & 1 for x in range(size)]


def _census(n: int) -> tuple[dict[str, int], dict[int, int]]:
    """Enumerate all 2**(2**n) functions; return (counts, depth histogram)."""
    size = 1 << n
    counts = {"monotone": 0, "unate": 0, "canalizing": 0, "bent": 0}
    depth_histogram: Counter[int] = Counter()

    for bits in range(1 << size):
        f = bf.create(_truth_table(bits, size))
        if is_monotone(f):
            counts["monotone"] += 1
        if is_unate(f)[0]:
            counts["unate"] += 1
        if is_canalizing(f):
            counts["canalizing"] += 1
        if n % 2 == 0 and is_bent(f):
            counts["bent"] += 1
        depth_histogram[get_canalizing_depth(f)] += 1

    return counts, dict(depth_histogram)


class TestSmallCensus:
    """Exhaustive n <= 3 censuses; cheap enough for every PR."""

    @pytest.mark.parametrize("n", [2, 3])
    def test_counts_match_published_values(self, n: int) -> None:
        counts, _ = _census(n)
        assert counts["monotone"] == MONOTONE_COUNTS[n], "monotone vs OEIS A000372"
        assert counts["unate"] == UNATE_COUNTS[n], "unate vs OEIS A245079"
        assert counts["canalizing"] == CANALIZING_COUNTS[n], "canalizing vs OEIS A102449"
        if n == 2:
            assert counts["bent"] == BENT_COUNTS[2], "bent vs OEIS A004491"

    @pytest.mark.parametrize("n", [2, 3])
    def test_depth_zero_bucket_convention(self, n: int) -> None:
        """Depth 0 = non-canalizing functions plus the two constants.

        This pins the two constant-function conventions against each other:
        constants are canalizing (A102449) but have depth 0 (He & Macauley).
        """
        _, histogram = _census(n)
        total = 1 << (1 << n)
        non_canalizing = total - CANALIZING_COUNTS[n]
        assert histogram[0] == non_canalizing + 2

    def test_per_function_spot_checks(self) -> None:
        """Per-function assertions in census enumeration order, so paired
        false positives/negatives cannot hide inside matching totals."""
        # Function number 2 on 2 variables is [0, 1, 0, 0] = b0 AND NOT b1:
        # unate but not monotone (the issue #114 counterexample).
        f = bf.create(_truth_table(2, 4))
        assert not is_monotone(f)
        assert is_unate(f)[0]

        # Function number 0x8000 on 4 variables is AND(4): monotone, unate,
        # canalizing with full nested depth.
        f = bf.create(_truth_table(0x8000, 16))
        assert is_monotone(f)
        assert is_unate(f)[0]
        assert is_canalizing(f)
        assert get_canalizing_depth(f) == 4

        # Function number 0x6996 on 4 variables is PARITY(4): none of the
        # monotone/unate/canalizing properties, but bent-adjacent checks
        # must still reject it (parity is affine on {0,1}^n, not bent).
        f = bf.create(_truth_table(0x6996, 16))
        assert not is_monotone(f)
        assert not is_unate(f)[0]
        assert not is_canalizing(f)
        assert get_canalizing_depth(f) == 0
        assert not is_bent(f)


@pytest.mark.skipif(
    not FULL_CENSUS,
    reason="full 4-variable census (~15 s locally) runs with BOOFUN_FULL_CENSUS=1 "
    "(set in CI for non-PR events)",
)
class TestFullFourVariableCensus:
    """All 65,536 four-variable functions against published counts."""

    def test_counts_match_published_values(self) -> None:
        counts, histogram = _census(4)
        assert counts["monotone"] == MONOTONE_COUNTS[4], "monotone vs OEIS A000372"
        assert counts["unate"] == UNATE_COUNTS[4], "unate vs OEIS A245079"
        assert counts["canalizing"] == CANALIZING_COUNTS[4], "canalizing vs OEIS A102449"
        assert counts["bent"] == BENT_COUNTS[4], "bent vs OEIS A004491"
        assert histogram == DEPTH_HISTOGRAM_4, "canalizing-depth histogram vs He & Macauley (2016)"
