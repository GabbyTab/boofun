import sys

sys.path.insert(0, "src")
"""
Tests for the api module.

Tests the main user-facing API functions:
- create (main factory)
- partial
- from_hex / to_hex
- Storage hints: auto, dense, packed, sparse, lazy
"""

import pytest
import numpy as np

import boofun as bf
from boofun.api import STORAGE_HINTS, create, from_hex, partial, to_hex
from boofun.core import BooleanFunction
from boofun.core.partial import PartialBooleanFunction


# ---------------------------------------------------------------------------
# Tests: create
# ---------------------------------------------------------------------------


class TestCreate:
    """Tests for the create() factory function."""

    def test_from_truth_table_list(self):
        """Create from a Python list truth table."""
        f = create([0, 1, 1, 0])
        assert isinstance(f, BooleanFunction)
        assert f.n_vars == 2
        assert bool(f.evaluate(0)) is False
        assert bool(f.evaluate(1)) is True

    def test_from_truth_table_numpy(self):
        """Create from a NumPy array truth table."""
        tt = np.array([0, 0, 0, 1], dtype=bool)
        f = create(tt)
        assert isinstance(f, BooleanFunction)
        assert f.n_vars == 2

    def test_from_callable(self):
        """Create from a callable function."""
        f = create(lambda x: sum(x) > 1, n=3)
        assert isinstance(f, BooleanFunction)
        assert f.n_vars == 3

    def test_with_n_override(self):
        """Explicit n parameter is respected."""
        tt = [0, 1, 1, 0]
        f = create(tt, n=2)
        assert f.n_vars == 2

    def test_invalid_storage_raises(self):
        """Invalid storage hint raises ValueError."""
        with pytest.raises(ValueError, match="Invalid storage hint"):
            create([0, 1, 1, 0], storage="invalid")

    def test_storage_dense(self):
        """Storage='dense' creates a function without error."""
        f = create([0, 1, 1, 0], storage="dense")
        assert isinstance(f, BooleanFunction)

    def test_storage_packed(self):
        """Storage='packed' creates a function without error."""
        f = create([0, 1, 1, 0], storage="packed")
        assert isinstance(f, BooleanFunction)

    def test_storage_sparse(self):
        """Storage='sparse' creates a function without error."""
        f = create([0, 1, 1, 0], storage="sparse")
        assert isinstance(f, BooleanFunction)

    def test_storage_lazy_with_callable(self):
        """Storage='lazy' with a callable creates a function."""
        f = create(lambda x: sum(x) > 1, n=3, storage="lazy")
        assert isinstance(f, BooleanFunction)

    def test_storage_lazy_requires_n(self):
        """Storage='lazy' without n raises ValueError."""
        with pytest.raises(ValueError, match="Must specify n"):
            create(lambda x: sum(x) > 1, storage="lazy")

    def test_storage_auto_small_n(self):
        """Storage='auto' for small n uses dense by default."""
        f = create([0, 1, 1, 0], storage="auto")
        assert isinstance(f, BooleanFunction)

    def test_all_storage_hints_valid(self):
        """All documented storage hints are in STORAGE_HINTS."""
        expected = {"auto", "dense", "packed", "sparse", "lazy"}
        assert STORAGE_HINTS == expected

    def test_create_preserves_truth_table(self):
        """Created function evaluates correctly at all inputs."""
        tt = [0, 1, 1, 0]
        f = create(tt)
        for i, expected in enumerate(tt):
            assert bool(f.evaluate(i)) == bool(expected)


# ---------------------------------------------------------------------------
# Tests: partial
# ---------------------------------------------------------------------------


class TestPartial:
    """Tests for the partial() function."""

    def test_creates_partial_function(self):
        """partial() returns a PartialBooleanFunction."""
        p = partial(n=3)
        assert isinstance(p, PartialBooleanFunction)

    def test_with_known_values(self):
        """Initial known values are stored correctly."""
        p = partial(n=3, known_values={0: True, 7: False})
        assert p.evaluate(0) is True
        assert p.evaluate(7) is False

    def test_with_name(self):
        """Name parameter is accepted."""
        p = partial(n=3, name="test_func")
        assert isinstance(p, PartialBooleanFunction)

    def test_empty_partial(self):
        """Empty partial function has no known values initially."""
        p = partial(n=3)
        assert p.num_known == 0


# ---------------------------------------------------------------------------
# Tests: from_hex / to_hex
# ---------------------------------------------------------------------------


class TestFromHex:
    """Tests for from_hex function."""

    def test_basic_conversion(self):
        """Create function from hex string."""
        f = from_hex("6", n=2)
        assert isinstance(f, BooleanFunction)
        assert f.n_vars == 2

    def test_with_0x_prefix(self):
        """Handles 0x prefix correctly."""
        f1 = from_hex("0x6", n=2)
        f2 = from_hex("6", n=2)
        tt1 = np.asarray(f1.get_representation("truth_table"))
        tt2 = np.asarray(f2.get_representation("truth_table"))
        assert np.array_equal(tt1, tt2)

    def test_roundtrip(self):
        """from_hex -> to_hex roundtrip preserves the function."""
        original_hex = "ac90"
        f = from_hex(original_hex, n=4)
        recovered = to_hex(f)
        assert recovered == original_hex

    def test_to_hex_format(self):
        """to_hex returns lowercase hex without prefix."""
        f = bf.AND(2)
        h = to_hex(f)
        assert h == h.lower()
        assert not h.startswith("0x")


class TestToHex:
    """Tests for to_hex function."""

    def test_parity2(self):
        """Parity(2) = XOR: truth table [0,1,1,0] = 0x6."""
        f = bf.parity(2)
        h = to_hex(f)
        assert h == "6"

    def test_and2(self):
        """AND(2): truth table [0,0,0,1] = 0x8."""
        f = bf.AND(2)
        h = to_hex(f)
        assert h == "8"

    def test_constant_true(self):
        """Constant True(2): truth table [1,1,1,1] = 0xf."""
        f = bf.constant(True, 2)
        h = to_hex(f)
        assert h == "f"

    def test_constant_false(self):
        """Constant False(2): truth table [0,0,0,0] = 0x0."""
        f = bf.constant(False, 2)
        h = to_hex(f)
        assert h == "0"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_hex_length(self, n):
        """Hex string has correct length = 2^n / 4."""
        f = bf.parity(n)
        h = to_hex(f)
        expected_len = (1 << n) // 4
        assert len(h) == expected_len


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the API module."""

    def test_single_variable(self):
        """Create function with 1 variable."""
        f = create([0, 1])
        assert f.n_vars == 1

    def test_from_hex_uppercase(self):
        """from_hex handles uppercase hex."""
        f = from_hex("AC90", n=4)
        assert isinstance(f, BooleanFunction)

    def test_from_hex_with_spaces(self):
        """from_hex strips spaces."""
        f = from_hex("ac 90", n=4)
        assert isinstance(f, BooleanFunction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
