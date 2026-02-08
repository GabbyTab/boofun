import sys

sys.path.insert(0, "src")
"""
Tests for core/representations/symbolic module.

Tests SymbolicRepresentation:
- evaluate with variable-based expressions
- evaluate with composed BooleanFunction sub-expressions
- dump (serialization)
- convert_from / convert_to
- create_empty
- is_complete
- get_storage_requirements
"""

import pytest
import numpy as np

from boofun.core.representations.symbolic import SymbolicRepresentation
from boofun.core.spaces import Space


# ---------------------------------------------------------------------------
# Tests: basic methods
# ---------------------------------------------------------------------------


class TestCreateEmpty:
    """Tests for create_empty method."""

    def test_returns_empty_expr(self):
        """Empty representation has empty expression string."""
        rep = SymbolicRepresentation()
        expr, vars = rep.create_empty(3)
        assert expr == ""

    def test_correct_variable_names(self):
        """Variables are x0, x1, ..., x(n-1)."""
        rep = SymbolicRepresentation()
        expr, vars = rep.create_empty(4)
        assert vars == ["x0", "x1", "x2", "x3"]

    def test_single_variable(self):
        """Single variable case."""
        rep = SymbolicRepresentation()
        expr, vars = rep.create_empty(1)
        assert vars == ["x0"]

    def test_zero_variables(self):
        """Edge case: 0 variables."""
        rep = SymbolicRepresentation()
        expr, vars = rep.create_empty(0)
        assert vars == []
        assert expr == ""


class TestIsComplete:
    """Tests for is_complete method."""

    def test_empty_is_incomplete(self):
        """Empty expression is not complete."""
        rep = SymbolicRepresentation()
        assert not rep.is_complete(("", ["x0", "x1"]))

    def test_nonempty_is_complete(self):
        """Non-empty expression is complete."""
        rep = SymbolicRepresentation()
        assert rep.is_complete(("x0 and x1", ["x0", "x1"]))

    def test_whitespace_is_complete(self):
        """Whitespace-only expression counts as complete (truthy string)."""
        rep = SymbolicRepresentation()
        assert rep.is_complete((" ", ["x0"]))


class TestDump:
    """Tests for dump method."""

    def test_returns_dict(self):
        """Dump returns a dictionary with expression and variables."""
        rep = SymbolicRepresentation()
        result = rep.dump(("x0 and x1", ["x0", "x1"]))
        assert result == {"expression": "x0 and x1", "variables": ["x0", "x1"]}

    def test_empty_dump(self):
        """Dump of empty representation."""
        rep = SymbolicRepresentation()
        result = rep.dump(("", []))
        assert result == {"expression": "", "variables": []}


class TestGetStorageRequirements:
    """Tests for get_storage_requirements method."""

    def test_returns_dict(self):
        """Returns dictionary with expected keys."""
        rep = SymbolicRepresentation()
        req = rep.get_storage_requirements(3)
        assert "expression_chars" in req
        assert "variables" in req

    def test_variables_count(self):
        """Variables count matches n_vars."""
        rep = SymbolicRepresentation()
        req = rep.get_storage_requirements(5)
        assert req["variables"] == 5


# ---------------------------------------------------------------------------
# Tests: evaluate with variable expressions
# ---------------------------------------------------------------------------


class TestEvaluateVariableExpression:
    """Tests for evaluate with variable-based symbolic expressions."""

    def test_simple_and(self):
        """Evaluate 'x0 and x1' gives correct truth table."""
        rep = SymbolicRepresentation()
        data = ("x0 and x1", ["x0", "x1"])
        space = Space.BOOLEAN_CUBE

        # Evaluate at all 4 inputs
        for x in range(4):
            result = rep.evaluate(np.array(x), data, space, n_vars=2)
            x0 = bool(x & 1)
            x1 = bool((x >> 1) & 1)
            assert bool(result) == (x0 and x1)

    def test_simple_or(self):
        """Evaluate 'x0 or x1' gives correct truth table."""
        rep = SymbolicRepresentation()
        data = ("x0 or x1", ["x0", "x1"])
        space = Space.BOOLEAN_CUBE

        for x in range(4):
            result = rep.evaluate(np.array(x), data, space, n_vars=2)
            x0 = bool(x & 1)
            x1 = bool((x >> 1) & 1)
            assert bool(result) == (x0 or x1)

    def test_xor_expression(self):
        """Evaluate 'x0 != x1' (XOR)."""
        rep = SymbolicRepresentation()
        data = ("x0 != x1", ["x0", "x1"])
        space = Space.BOOLEAN_CUBE

        for x in range(4):
            result = rep.evaluate(np.array(x), data, space, n_vars=2)
            x0 = bool(x & 1)
            x1 = bool((x >> 1) & 1)
            assert bool(result) == (x0 != x1)

    def test_three_variable_majority(self):
        """Evaluate '(x0 and x1) or (x1 and x2) or (x0 and x2)' (majority)."""
        rep = SymbolicRepresentation()
        expr = "(x0 and x1) or (x1 and x2) or (x0 and x2)"
        data = (expr, ["x0", "x1", "x2"])
        space = Space.BOOLEAN_CUBE

        # Majority truth table
        maj_tt = [0, 0, 0, 1, 0, 1, 1, 1]
        for x in range(8):
            result = rep.evaluate(np.array(x), data, space, n_vars=3)
            assert bool(result) == bool(maj_tt[x])

    def test_batch_evaluation(self):
        """Evaluate on a batch of integer inputs."""
        rep = SymbolicRepresentation()
        data = ("x0 and x1", ["x0", "x1"])
        space = Space.BOOLEAN_CUBE

        inputs = np.array([0, 1, 2, 3])
        results = rep.evaluate(inputs, data, space, n_vars=2)
        expected = [False, False, False, True]
        assert list(results) == expected

    def test_scalar_input(self):
        """Evaluate on scalar input."""
        rep = SymbolicRepresentation()
        data = ("x0", ["x0"])
        space = Space.BOOLEAN_CUBE

        result = rep.evaluate(np.array(0), data, space, n_vars=1)
        assert bool(result) is False

        result = rep.evaluate(np.array(1), data, space, n_vars=1)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# Tests: convert_from and convert_to
# ---------------------------------------------------------------------------


class TestConversions:
    """Tests for convert_from and convert_to methods."""

    def test_convert_from_returns_tuple(self):
        """convert_from returns a (str, list) tuple."""
        rep = SymbolicRepresentation()
        from boofun.core.representations.truth_table import TruthTableRepresentation

        source = TruthTableRepresentation()
        result = rep.convert_from(
            source, np.array([0, 1, 1, 0], dtype=bool), Space.BOOLEAN_CUBE, 2
        )
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for SymbolicRepresentation."""

    def test_expression_with_not(self):
        """Evaluate 'not x0' (negation)."""
        rep = SymbolicRepresentation()
        data = ("not x0", ["x0"])
        space = Space.BOOLEAN_CUBE

        assert bool(rep.evaluate(np.array(0), data, space, n_vars=1)) is True
        assert bool(rep.evaluate(np.array(1), data, space, n_vars=1)) is False

    def test_empty_expression_returns_false(self):
        """Empty expression evaluation should not crash."""
        rep = SymbolicRepresentation()
        data = ("", ["x0"])
        space = Space.BOOLEAN_CUBE

        # Empty expression will evaluate to False (empty string is falsy)
        result = rep.evaluate(np.array(0), data, space, n_vars=1)
        assert bool(result) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
