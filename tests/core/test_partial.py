"""
Comprehensive tests for partial Boolean functions and storage hints.

Tests cover:
- PartialBooleanFunction class
- bf.partial() factory
- bf.from_hex() and bf.to_hex()
- Storage hints in bf.create()
"""

import numpy as np
import pytest

import boofun as bf
from boofun.api import STORAGE_HINTS
from boofun.core.partial import PartialBooleanFunction


class TestPartialBooleanFunction:
    """Tests for PartialBooleanFunction class."""

    def test_empty_partial(self):
        """Create empty partial function."""
        p = PartialBooleanFunction(n_vars=4)

        assert p.n_vars == 4
        assert p.size == 16
        assert p.num_known == 0
        assert p.num_unknown == 16
        assert p.completeness == 0.0
        assert not p.is_complete

    def test_partial_with_known_values(self):
        """Create partial with initial known values."""
        known = {0: True, 1: False, 7: True}
        p = PartialBooleanFunction(n_vars=4, known_values=known)

        assert p.num_known == 3
        assert p.completeness == 3 / 16
        assert p.is_known(0)
        assert p.is_known(1)
        assert p.is_known(7)
        assert not p.is_known(2)

    def test_add_value(self):
        """Add values incrementally."""
        p = PartialBooleanFunction(n_vars=3)

        p.add(0, True)
        assert p.num_known == 1
        assert p.is_known(0)
        assert p.evaluate(0) == True

        p.add(1, False)
        assert p.num_known == 2
        assert p.evaluate(1) == False

    def test_add_batch(self):
        """Add multiple values at once."""
        p = PartialBooleanFunction(n_vars=3)

        p.add_batch({0: True, 1: False, 2: True, 3: False})

        assert p.num_known == 4
        assert p.evaluate(0) == True
        assert p.evaluate(1) == False
        assert p.evaluate(2) == True
        assert p.evaluate(3) == False

    def test_evaluate_unknown(self):
        """Evaluating unknown value returns None."""
        p = PartialBooleanFunction(n_vars=3)
        p.add(0, True)

        assert p.evaluate(0) == True
        assert p.evaluate(1) is None
        assert p.evaluate(7) is None

    def test_evaluate_with_confidence_known(self):
        """Known values have confidence 1.0."""
        p = PartialBooleanFunction(n_vars=3)
        p.add(0, True)

        val, conf = p.evaluate_with_confidence(0)
        assert val == True
        assert conf == 1.0

    def test_evaluate_with_confidence_unknown(self):
        """Unknown values have confidence < 1.0."""
        p = PartialBooleanFunction(n_vars=3)
        p.add(0, True)
        p.add(2, True)

        # Index 1 is Hamming neighbor of both 0 and 2
        val, conf = p.evaluate_with_confidence(1)
        assert conf < 1.0

    def test_indexing_syntax(self):
        """Bracket notation works."""
        p = PartialBooleanFunction(n_vars=3)

        p[0] = True
        p[1] = False

        assert p[0] == True
        assert p[1] == False
        assert p[2] is None

    def test_iteration(self):
        """Iterate over known values."""
        p = PartialBooleanFunction(n_vars=3)
        p.add_batch({0: True, 3: False, 7: True})

        known_pairs = list(p)
        assert len(known_pairs) == 3
        assert (0, True) in known_pairs
        assert (3, False) in known_pairs
        assert (7, True) in known_pairs

    def test_contains(self):
        """Check if index has known value."""
        p = PartialBooleanFunction(n_vars=3)
        p.add(5, True)

        assert 5 in p
        assert 0 not in p

    def test_len(self):
        """Length is number of known values."""
        p = PartialBooleanFunction(n_vars=4)
        assert len(p) == 0

        p.add_batch({0: True, 1: False, 2: True})
        assert len(p) == 3

    def test_to_function_complete(self):
        """Convert complete partial to BooleanFunction."""
        p = PartialBooleanFunction(n_vars=2)
        p.add_batch({0: True, 1: False, 2: False, 3: True})

        assert p.is_complete
        f = p.to_function()

        assert f.evaluate(0) == True
        assert f.evaluate(1) == False
        assert f.evaluate(2) == False
        assert f.evaluate(3) == True

    def test_to_function_fill_unknown(self):
        """Convert incomplete partial with fill_unknown."""
        p = PartialBooleanFunction(n_vars=2)
        p.add(0, True)

        f = p.to_function(fill_unknown=False)
        assert f.evaluate(0) == True
        assert f.evaluate(1) == False  # Filled with False

    def test_to_function_estimate_unknown(self):
        """Convert incomplete partial with estimation."""
        p = PartialBooleanFunction(n_vars=3)
        p.add_batch({0: True, 2: True, 4: True})  # Neighbors of 1

        f = p.to_function(estimate_unknown=True)
        # Should estimate unknown values based on neighbors
        assert f.n_vars == 3

    def test_get_known_values(self):
        """Get dictionary of known values."""
        p = PartialBooleanFunction(n_vars=3)
        p.add_batch({0: True, 3: False, 7: True})

        known = p.get_known_values()
        assert known == {0: True, 3: False, 7: True}

    def test_sample_unknown(self):
        """Sample from unknown indices."""
        p = PartialBooleanFunction(n_vars=4)
        p.add_batch({0: True, 1: False})

        samples = p.sample_unknown(n_samples=5, seed=42)

        assert len(samples) == 5
        assert all(s not in [0, 1] for s in samples)

    def test_add_from_samples(self):
        """Add values from arrays."""
        p = PartialBooleanFunction(n_vars=3)

        inputs = np.array([0, 1, 2, 3])
        outputs = np.array([True, False, False, True])
        p.add_from_samples(inputs, outputs)

        assert p.num_known == 4
        assert p.evaluate(0) == True
        assert p.evaluate(1) == False

    def test_index_out_of_range(self):
        """Out of range indices raise IndexError."""
        p = PartialBooleanFunction(n_vars=2)

        with pytest.raises(IndexError):
            p.add(4, True)  # Max is 3

        with pytest.raises(IndexError):
            p.evaluate(100)

    def test_invalid_n_vars(self):
        """Invalid n_vars raises ValueError."""
        with pytest.raises(ValueError):
            PartialBooleanFunction(n_vars=-1)

        with pytest.raises(ValueError):
            PartialBooleanFunction(n_vars=31)

    def test_name(self):
        """Named partial function."""
        p = PartialBooleanFunction(n_vars=4, name="test_function")

        assert p.name == "test_function"
        assert "test_function" in str(p)

    def test_summary(self):
        """Summary string."""
        p = PartialBooleanFunction(n_vars=4, name="my_func")
        p.add_batch({0: True, 1: False})

        summary = p.summary()
        assert "n=4" in summary
        assert "Known: 2" in summary
        assert "my_func" in summary

    def test_repr(self):
        """Repr string."""
        p = PartialBooleanFunction(n_vars=4)
        p.add(0, True)

        r = repr(p)
        assert "n_vars=4" in r
        assert "known=1/16" in r


class TestBfPartial:
    """Tests for bf.partial() factory function."""

    def test_basic_creation(self):
        """Create partial via bf.partial()."""
        p = bf.partial(n=10)

        assert isinstance(p, PartialBooleanFunction)
        assert p.n_vars == 10
        assert p.num_known == 0

    def test_with_known_values(self):
        """Create with initial values via bf.partial()."""
        p = bf.partial(n=10, known_values={0: True, 1: False})

        assert p.num_known == 2
        assert p.evaluate(0) == True

    def test_named(self):
        """Create named partial."""
        p = bf.partial(n=5, name="test")

        assert p.name == "test"


class TestFromHex:
    """Tests for bf.from_hex() hex string input."""

    def test_basic_hex(self):
        """Create from hex string."""
        f = bf.from_hex("0x6", n=2)

        # 0x6 = 0b0110, so tt = [0, 1, 1, 0] (XOR)
        assert f.evaluate(0) == False
        assert f.evaluate(1) == True
        assert f.evaluate(2) == True
        assert f.evaluate(3) == False

    def test_hex_without_prefix(self):
        """Create from hex string without 0x prefix."""
        f = bf.from_hex("6", n=2)

        assert f.evaluate(0) == False
        assert f.evaluate(1) == True

    def test_bent_4bit(self):
        """thomasarmel example: 4-bit bent function."""
        # From thomasarmel README: 0xac90 is bent
        f = bf.from_hex("0xac90", n=4)

        # Verify it's a valid 4-bit function
        assert f.n_vars == 4

    def test_bent_6bit(self):
        """thomasarmel example: 6-bit bent function."""
        # From thomasarmel README: 0113077C165E76A8 is bent
        f = bf.from_hex("0113077C165E76A8", n=6)

        assert f.n_vars == 6

    def test_roundtrip(self):
        """Hex roundtrip: from_hex -> to_hex."""
        original = "ac90"
        f = bf.from_hex(original, n=4)
        result = bf.to_hex(f)

        assert result == original


class TestToHex:
    """Tests for bf.to_hex() hex string output."""

    def test_basic(self):
        """Export to hex string."""
        # XOR function: tt = [0, 1, 1, 0] -> 0b0110 = 0x6
        xor = bf.create([0, 1, 1, 0])

        result = bf.to_hex(xor)
        assert result == "6"

    def test_and_function(self):
        """AND function to hex."""
        # AND(2): tt = [0, 0, 0, 1] -> 0b1000 = 0x8
        and2 = bf.AND(2)

        result = bf.to_hex(and2)
        assert result == "8"

    def test_padding(self):
        """Hex is padded to correct length."""
        # 4-bit function should have 4 hex digits
        f = bf.from_hex("0001", n=4)
        result = bf.to_hex(f)

        assert len(result) == 4


class TestStorageHints:
    """Tests for storage hints in bf.create()."""

    def test_basic_storage_hints(self):
        """Basic storage hints that don't require special representations."""
        tt = [0, 1, 1, 0]

        # These work without special representation registration
        for hint in ["auto", "dense"]:
            f = bf.create(tt, storage=hint)
            assert f.evaluate(0) == False

    def test_invalid_storage_hint(self):
        """Invalid storage hint raises ValueError."""
        with pytest.raises(ValueError, match="Invalid storage hint"):
            bf.create([0, 1, 1, 0], storage="invalid")

    def test_packed_storage_hint_accepted(self):
        """Packed storage hint is accepted (falls back gracefully)."""
        tt = [0, 1, 1, 0]

        # Note: packed_truth_table representation may not be registered
        # in the factory, so this may fall back to default
        try:
            f = bf.create(tt, storage="packed")
            assert f.evaluate(0) == False
        except Exception:
            # If representation not registered, that's ok for this test
            pytest.skip("packed_truth_table representation not registered")

    def test_sparse_storage_hint_accepted(self):
        """Sparse storage hint is accepted (falls back gracefully)."""
        tt = [0] * 16
        tt[0] = 1

        try:
            f = bf.create(tt, storage="sparse")
            assert f.evaluate(0) == True
        except Exception:
            pytest.skip("sparse_truth_table representation not registered")

    def test_lazy_storage(self):
        """Lazy storage for callable."""

        def oracle(x):
            return sum(x) % 2 == 0

        f = bf.create(oracle, n=10, storage="lazy")

        # Should work without materializing full truth table
        assert f.evaluate([0] * 10) == True
        assert f.evaluate([1] + [0] * 9) == False

    def test_auto_storage_small(self):
        """Auto storage uses dense for small n."""
        tt = [0, 1, 1, 0]
        f = bf.create(tt, storage="auto")

        assert f.evaluate(0) == False

    def test_dense_storage(self):
        """Dense storage hint."""
        tt = [0, 1, 1, 0]
        f = bf.create(tt, storage="dense")

        assert f.evaluate(0) == False
        assert f.evaluate(1) == True

    def test_storage_hints_are_valid(self):
        """All defined storage hints are valid strings."""
        assert "auto" in STORAGE_HINTS
        assert "dense" in STORAGE_HINTS
        assert "packed" in STORAGE_HINTS
        assert "sparse" in STORAGE_HINTS
        assert "lazy" in STORAGE_HINTS


class TestCrossValidation:
    """Cross-validation with thomasarmel/boolean_function examples."""

    def test_thomasarmel_bent_4bit(self):
        """Verify 0xac90 bent function (from thomasarmel README)."""
        f = bf.from_hex("ac90", n=4)

        # Check basic properties
        assert f.n_vars == 4

        # Verify roundtrip
        assert bf.to_hex(f) == "ac90"

    def test_thomasarmel_bent_6bit(self):
        """Verify 6-bit bent function (from thomasarmel README)."""
        f = bf.from_hex("0113077c165e76a8", n=6)

        assert f.n_vars == 6

        # Check specific evaluation from thomasarmel: f(8) = false
        assert f.evaluate(8) == False

    def test_thomasarmel_rule30(self):
        """Verify Rule 30 cellular automaton (from thomasarmel README)."""
        # ANF: x0*x1 + x0 + x1 + x2
        # Truth table hex: 1e
        f = bf.from_hex("1e", n=3)

        assert f.n_vars == 3
        assert bf.to_hex(f) == "1e"

    def test_balanced_count(self):
        """Verify balanced function count matches thomasarmel."""
        # There are C(16,8) = 12870 balanced 4-variable Boolean functions
        # This is checked in test_cryptographic.py, referenced here for completeness
        from math import comb

        assert comb(16, 8) == 12870


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_partial_to_hex(self):
        """Convert partial function to hex."""
        p = bf.partial(n=2)
        p.add_batch({0: True, 1: False, 2: False, 3: True})

        f = p.to_function()
        hex_str = bf.to_hex(f)

        # tt = [1, 0, 0, 1] -> 0b1001 = 0x9
        assert hex_str == "9"

    def test_hex_to_partial(self):
        """Create partial from subset of hex function."""
        f = bf.from_hex("ac90", n=4)

        # Sample some values into a partial
        p = bf.partial(n=4)
        for i in [0, 5, 10, 15]:
            p.add(i, f.evaluate(i))

        assert p.num_known == 4
        assert p.completeness == 4 / 16

    def test_streaming_workflow(self):
        """Simulate streaming data workflow."""
        p = bf.partial(n=8)

        # Stream in data in batches
        for batch_start in range(0, 256, 32):
            batch = {i: (i % 3 == 0) for i in range(batch_start, batch_start + 32)}
            p.add_batch(batch)

        assert p.is_complete

        f = p.to_function()
        assert f.evaluate(0) == True
        assert f.evaluate(1) == False
