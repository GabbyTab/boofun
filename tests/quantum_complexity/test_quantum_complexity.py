"""
Tests for quantum_complexity module.

Tests for classical computation of quantum complexity bounds.
Verifies both API existence AND mathematical correctness.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.quantum_complexity import (
    QuantumComplexityAnalyzer,
    create_complexity_analyzer,
    element_distinctness_analysis,
    grover_speedup,
    quantum_walk_bounds,
)


class TestQuantumComplexityAnalyzer:
    """Test QuantumComplexityAnalyzer class."""

    def test_stores_classical_function(self):
        """QuantumComplexityAnalyzer should store the classical function."""
        f = bf.majority(3)
        qca = QuantumComplexityAnalyzer(f)

        assert hasattr(qca, "function")
        assert qca.function is f

    def test_preserves_n_vars(self):
        """Should preserve number of variables from classical function."""
        f = bf.AND(4)
        qca = QuantumComplexityAnalyzer(f)

        assert qca.n_vars == 4

    def test_has_analysis_methods(self):
        """Should expose complexity analysis methods."""
        f = bf.OR(3)
        qca = QuantumComplexityAnalyzer(f)

        assert hasattr(qca, "grover_analysis")
        assert hasattr(qca, "grover_amplitude_analysis")
        assert hasattr(qca, "create_quantum_oracle")


class TestCreateComplexityAnalyzer:
    """Test create_complexity_analyzer factory."""

    def test_returns_analyzer(self):
        """Factory should create QuantumComplexityAnalyzer instance."""
        f = bf.parity(3)
        qca = create_complexity_analyzer(f)

        assert isinstance(qca, QuantumComplexityAnalyzer)

    def test_works_with_all_builtins(self):
        """Factory should work with all built-in function types."""
        builtins = [
            ("AND", bf.AND(3)),
            ("OR", bf.OR(3)),
            ("majority", bf.majority(3)),
            ("parity", bf.parity(3)),
        ]

        for name, f in builtins:
            qca = create_complexity_analyzer(f)
            assert isinstance(qca, QuantumComplexityAnalyzer), f"Failed for {name}"


class TestQuantumWalkBounds:
    """Test quantum_walk_bounds function."""

    def test_returns_dict_with_walk_info(self):
        """Should return dictionary with walk analysis."""
        f = bf.majority(3)
        result = quantum_walk_bounds(f)

        assert isinstance(result, dict)
        assert "spectral_gap" in result
        assert "mixing_time" in result
        assert "quantum_walk_complexity" in result

    def test_different_functions_give_different_results(self):
        """Different functions should have different walk properties."""
        f_and = bf.AND(3)
        f_or = bf.OR(3)

        result_and = quantum_walk_bounds(f_and)
        result_or = quantum_walk_bounds(f_or)

        assert isinstance(result_and, dict)
        assert isinstance(result_or, dict)

    def test_speedup_over_classical(self):
        """Quantum walk should show speedup over classical."""
        f = bf.AND(4)
        result = quantum_walk_bounds(f)

        assert result["speedup_over_classical"] > 1


class TestElementDistinctnessAnalysis:
    """Test element_distinctness_analysis function."""

    def test_returns_analysis_dict(self):
        """Should return dictionary with analysis."""
        f = bf.majority(3)
        result = element_distinctness_analysis(f)

        assert isinstance(result, dict)
        assert "has_collision" in result
        assert "quantum_complexity" in result

    def test_handles_various_functions(self):
        """Should handle different function types."""
        functions = [bf.AND(3), bf.parity(3), bf.OR(4)]

        for f in functions:
            result = element_distinctness_analysis(f)
            assert isinstance(result, dict)

    def test_quantum_speedup_positive(self):
        """Element distinctness should always show quantum speedup."""
        f = bf.parity(4)
        result = element_distinctness_analysis(f)

        assert result["speedup"] > 1


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
        assert result["num_solutions"] == 1
        assert result["speedup"] > 0

    def test_constant_function_analysis(self):
        """Constant-zero function should handle gracefully."""
        f_zero = bf.create([0, 0, 0, 0])
        result = grover_speedup(f_zero)

        assert isinstance(result, dict)
        assert result["has_solutions"] is False


class TestGroverMathematicalProperties:
    """Test that Grover analysis follows expected mathematical properties."""

    def test_grover_for_unique_search(self):
        """AND on n bits has unique satisfying assignment."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            result = grover_speedup(f)

            assert result["num_solutions"] == 1

    def test_grover_for_or(self):
        """OR on n bits has 2^n - 1 satisfying assignments."""
        for n in [2, 3]:
            f = bf.OR(n)
            result = grover_speedup(f)

            assert result["num_solutions"] == 2**n - 1

    def test_grover_speedup_grows_with_n(self):
        """Grover speedup should grow with problem size for AND (1 solution)."""
        speedups = []
        for n in [3, 4, 5, 6]:
            f = bf.AND(n)
            result = grover_speedup(f)
            speedups.append(result["speedup"])

        # Speedup should be monotonically increasing
        for i in range(len(speedups) - 1):
            assert speedups[i + 1] > speedups[i]


class TestBackwardsCompatibility:
    """Test that deprecated aliases still work."""

    def test_old_class_name(self):
        """QuantumBooleanFunction should still work as an alias."""
        from boofun.quantum_complexity import QuantumBooleanFunction

        f = bf.AND(3)
        qbf = QuantumBooleanFunction(f)
        assert isinstance(qbf, QuantumComplexityAnalyzer)

    def test_old_factory_name(self):
        """create_quantum_boolean_function should still work as an alias."""
        from boofun.quantum_complexity import create_quantum_boolean_function

        f = bf.AND(3)
        qbf = create_quantum_boolean_function(f)
        assert isinstance(qbf, QuantumComplexityAnalyzer)

    def test_old_walk_function_name(self):
        """quantum_walk_analysis should still work as an alias."""
        from boofun.quantum_complexity import quantum_walk_analysis

        f = bf.AND(4)
        result = quantum_walk_analysis(f)
        assert "spectral_gap" in result

    def test_old_walk_search_name(self):
        """quantum_walk_search should still work as an alias."""
        from boofun.quantum_complexity import quantum_walk_search

        f = bf.OR(4)
        result = quantum_walk_search(f)
        assert "marked_vertices" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
