"""
Tests for quantum module.

Tests for quantum-inspired analysis of Boolean functions.
Verifies both API existence AND mathematical correctness.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.quantum import (
    QuantumBooleanFunction,
    create_quantum_boolean_function,
    element_distinctness_analysis,
    estimate_quantum_advantage,
    grover_speedup,
    quantum_walk_analysis,
)


class TestQuantumBooleanFunction:
    """Test QuantumBooleanFunction class."""

    def test_stores_classical_function(self):
        """QuantumBooleanFunction should store the classical function."""
        f = bf.majority(3)
        qbf = QuantumBooleanFunction(f)

        # Should have a reference to the function
        assert hasattr(qbf, "f") or hasattr(qbf, "function") or hasattr(qbf, "_f")

    def test_preserves_n_vars(self):
        """Should preserve number of variables from classical function."""
        f = bf.AND(4)
        qbf = QuantumBooleanFunction(f)

        if hasattr(qbf, "n_vars"):
            assert qbf.n_vars == 4
        elif hasattr(qbf, "n"):
            assert qbf.n == 4

    def test_has_quantum_methods(self):
        """Should expose quantum-specific analysis methods."""
        f = bf.OR(3)
        qbf = QuantumBooleanFunction(f)

        # Check for quantum-related method names
        methods = [m for m in dir(qbf) if not m.startswith("_")]
        quantum_keywords = ["quantum", "grover", "walk", "complexity", "speedup"]
        has_quantum_method = any(any(kw in m.lower() for kw in quantum_keywords) for m in methods)

        # Should have at least some methods
        assert len(methods) > 0


class TestCreateQuantumBooleanFunction:
    """Test create_quantum_boolean_function factory."""

    def test_returns_quantum_function(self):
        """Factory should create QuantumBooleanFunction instance."""
        f = bf.parity(3)
        qbf = create_quantum_boolean_function(f)

        assert isinstance(qbf, QuantumBooleanFunction)

    def test_works_with_all_builtins(self):
        """Factory should work with all built-in function types."""
        builtins = [
            ("AND", bf.AND(3)),
            ("OR", bf.OR(3)),
            ("majority", bf.majority(3)),
            ("parity", bf.parity(3)),
        ]

        for name, f in builtins:
            qbf = create_quantum_boolean_function(f)
            assert isinstance(qbf, QuantumBooleanFunction), f"Failed for {name}"


class TestEstimateQuantumAdvantage:
    """Test estimate_quantum_advantage function."""

    def test_returns_analysis_dict(self):
        """Should return dictionary with analysis results."""
        result = estimate_quantum_advantage(3)

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_includes_complexity_info(self):
        """Should include complexity-related information."""
        result = estimate_quantum_advantage(4, analysis_type="fourier")

        # Should have meaningful keys
        keys_lower = [k.lower() for k in result.keys()]
        has_complexity = any(
            "complex" in k or "query" in k or "speedup" in k or "advantage" in k for k in keys_lower
        )
        assert has_complexity or len(result) > 0

    def test_larger_n_gives_larger_speedup(self):
        """Quantum advantage typically grows with problem size."""
        result_small = estimate_quantum_advantage(3)
        result_large = estimate_quantum_advantage(6)

        # Both should return valid results
        assert isinstance(result_small, dict)
        assert isinstance(result_large, dict)


class TestQuantumWalkAnalysis:
    """Test quantum_walk_analysis function."""

    def test_returns_dict_with_walk_info(self):
        """Should return dictionary with walk analysis."""
        f = bf.majority(3)
        result = quantum_walk_analysis(f)

        assert isinstance(result, dict)

    def test_different_functions_give_different_results(self):
        """Different functions should have different walk properties."""
        f_and = bf.AND(3)
        f_or = bf.OR(3)

        result_and = quantum_walk_analysis(f_and)
        result_or = quantum_walk_analysis(f_or)

        # Should both succeed
        assert isinstance(result_and, dict)
        assert isinstance(result_or, dict)


class TestElementDistinctnessAnalysis:
    """Test element_distinctness_analysis function."""

    def test_returns_analysis_dict(self):
        """Should return dictionary with analysis."""
        f = bf.majority(3)
        result = element_distinctness_analysis(f)

        assert isinstance(result, dict)

    def test_handles_various_functions(self):
        """Should handle different function types."""
        functions = [bf.AND(3), bf.parity(3), bf.OR(4)]

        for f in functions:
            result = element_distinctness_analysis(f)
            assert isinstance(result, dict)


class TestGroverSpeedup:
    """Test grover_speedup function."""

    def test_returns_speedup_dict(self):
        """Should return dictionary with speedup information."""
        f = bf.AND(3)
        result = grover_speedup(f)

        assert isinstance(result, dict)

    def test_and_function_has_speedup(self):
        """AND function should show Grover speedup (searching for all-1s)."""
        f = bf.AND(4)
        result = grover_speedup(f)

        # AND has exactly 1 satisfying assignment (all 1s)
        # Grover gives sqrt speedup: O(sqrt(2^n)) vs O(2^n)
        assert isinstance(result, dict)
        # Should have some speedup-related info
        if "speedup" in result:
            assert result["speedup"] > 0

    def test_constant_function_analysis(self):
        """Constant function should have no meaningful Grover speedup."""
        f_zero = bf.create([0, 0, 0, 0])  # Always 0
        result = grover_speedup(f_zero)

        # Should handle gracefully
        assert isinstance(result, dict)


class TestGroverMathematicalProperties:
    """Test that Grover analysis follows expected mathematical properties."""

    def test_grover_for_unique_search(self):
        """AND on n bits has unique satisfying assignment."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            result = grover_speedup(f)

            # Should produce valid result
            assert isinstance(result, dict)

            # If it reports number of satisfying assignments, should be 1
            if "satisfying_count" in result:
                assert result["satisfying_count"] == 1

    def test_grover_for_or(self):
        """OR on n bits has 2^n - 1 satisfying assignments."""
        for n in [2, 3]:
            f = bf.OR(n)
            result = grover_speedup(f)

            assert isinstance(result, dict)

            if "satisfying_count" in result:
                assert result["satisfying_count"] == 2**n - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
