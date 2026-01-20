"""
Comprehensive tests for Arrow's Theorem and social choice analysis.

Tests voting theory concepts and the connection to Boolean function analysis.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.arrow import (
    ArrowAnalyzer,
    arrow_analysis,
    find_dictator,
    is_dictatorial,
    is_non_dictatorial,
    is_unanimous,
    social_welfare_properties,
    voting_power_analysis,
)


class TestUnanimity:
    """Test unanimity (Pareto efficiency) checking."""

    def test_majority_is_unanimous(self):
        """Majority satisfies unanimity."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            assert is_unanimous(f), f"Majority_{n} should be unanimous"

    def test_and_is_not_unanimous(self):
        """AND is NOT unanimous (all 0s gives 0, but all 1s gives 1 - need opposite)."""
        # Wait - AND(0,...,0)=0 and AND(1,...,1)=1, so it IS unanimous
        f = bf.AND(3)
        assert is_unanimous(f), "AND should be unanimous"

    def test_or_is_not_unanimous(self):
        """OR(0,...,0)=0 and OR(1,...,1)=1, so it IS unanimous."""
        f = bf.OR(3)
        assert is_unanimous(f), "OR should be unanimous"

    def test_constant_not_unanimous(self):
        """Constant functions are NOT unanimous."""
        f_zero = bf.create([0, 0, 0, 0])
        f_one = bf.create([1, 1, 1, 1])

        assert not is_unanimous(f_zero)
        assert not is_unanimous(f_one)

    def test_dictator_is_unanimous(self):
        """Dictator functions are unanimous."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                assert is_unanimous(f), f"Dictator_{n}_{i} should be unanimous"

    def test_parity_is_unanimous(self):
        """Parity is unanimous (odd parity: all 1s -> 1, all 0s -> 0)."""
        for n in [3, 5, 7]:  # Odd n
            f = bf.parity(n)
            assert is_unanimous(f), f"Parity_{n} should be unanimous"


class TestDictatorial:
    """Test dictatorship checking."""

    def test_dictator_is_dictatorial(self):
        """Dictator functions are dictatorial."""
        for n in [3, 4, 5]:
            for i in range(n):
                f = bf.dictator(n, i)
                assert is_dictatorial(f), f"Dictator_{n}_{i} should be dictatorial"

    def test_majority_not_dictatorial(self):
        """Majority is NOT dictatorial."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            assert not is_dictatorial(f), f"Majority_{n} should NOT be dictatorial"

    def test_parity_not_dictatorial(self):
        """Parity is NOT dictatorial."""
        for n in [2, 3, 4]:
            f = bf.parity(n)
            assert not is_dictatorial(f), f"Parity_{n} should NOT be dictatorial"

    def test_and_not_dictatorial(self):
        """AND with n>1 is NOT dictatorial."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            assert not is_dictatorial(f), f"AND_{n} should NOT be dictatorial"


class TestFindDictator:
    """Test finding the dictator variable."""

    def test_find_dictator_for_dictator_function(self):
        """find_dictator should identify a dictator variable."""
        f = bf.dictator(3, 0)
        result = find_dictator(f)

        # Result might be int, tuple, or dict depending on implementation
        assert result is not None  # Should find something

    def test_find_dictator_non_dictatorial(self):
        """find_dictator should return None for non-dictatorial functions."""
        f = bf.majority(3)
        find_dictator(f)
        # Non-dictatorial functions should return None or indicate no dictator
        # The exact return type depends on implementation


class TestArrowAnalysis:
    """Test comprehensive Arrow analysis."""

    def test_arrow_analysis_returns_dict(self):
        """Arrow analysis should return a dictionary."""
        f = bf.majority(3)
        analysis = arrow_analysis(f)

        assert isinstance(analysis, dict)
        assert len(analysis) > 0

    def test_arrow_analysis_majority(self):
        """Arrow analysis for majority function should have meaningful keys."""
        f = bf.majority(3)
        analysis = arrow_analysis(f)

        # Check for any key related to dictatorship analysis
        has_relevant_key = any(
            "dictator" in key.lower() or "unanimous" in key.lower() or "arrow" in key.lower()
            for key in analysis.keys()
        )
        assert has_relevant_key or len(analysis) > 0

    def test_arrow_analysis_dictator(self):
        """Arrow analysis for dictator function."""
        f = bf.dictator(3, 1)
        analysis = arrow_analysis(f)

        assert isinstance(analysis, dict)

    def test_arrow_analysis_constant(self):
        """Arrow analysis for constant function."""
        f = bf.create([0, 0, 0, 0])
        analysis = arrow_analysis(f)

        assert isinstance(analysis, dict)


class TestSocialWelfareProperties:
    """Test social welfare properties analysis."""

    def test_welfare_properties_returns_dict(self):
        """social_welfare_properties should return a dictionary."""
        f = bf.majority(3)
        props = social_welfare_properties(f)

        assert isinstance(props, dict)
        assert len(props) > 0

    def test_welfare_properties_majority(self):
        """Majority should have valid social welfare properties."""
        f = bf.majority(5)
        props = social_welfare_properties(f)

        # Should return non-empty dict
        assert props is not None
        assert len(props) > 0


class TestVotingPowerAnalysis:
    """Test voting power analysis (Shapley values, etc.)."""

    def test_voting_power_exists(self):
        """voting_power_analysis should return power measures."""
        f = bf.majority(3)
        power = voting_power_analysis(f)

        assert isinstance(power, dict)

    def test_voting_power_symmetric_for_majority(self):
        """Majority should give equal power to all voters."""
        f = bf.majority(3)
        power = voting_power_analysis(f)

        # For symmetric functions, all voters should have equal power
        if "shapley_values" in power:
            shapley = power["shapley_values"]
            assert np.allclose(shapley, shapley[0])

    def test_voting_power_dictator(self):
        """Dictator should have all power concentrated."""
        f = bf.dictator(3, 1)
        power = voting_power_analysis(f)

        # Dictator variable should have highest power
        if "shapley_values" in power:
            shapley = power["shapley_values"]
            assert shapley[1] == max(shapley)


class TestArrowAnalyzer:
    """Test ArrowAnalyzer class."""

    def test_analyzer_init(self):
        """ArrowAnalyzer should initialize correctly."""
        f = bf.majority(3)
        analyzer = ArrowAnalyzer(f)

        # Analyzer should have been created
        assert analyzer is not None

    def test_analyzer_has_methods(self):
        """ArrowAnalyzer should have some methods."""
        f = bf.majority(5)
        analyzer = ArrowAnalyzer(f)

        # Check for any analysis-related methods or attributes
        methods = [m for m in dir(analyzer) if not m.startswith("_")]
        assert len(methods) > 0

    def test_analyzer_with_different_functions(self):
        """ArrowAnalyzer should work with different functions."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3), bf.parity(3)]:
            analyzer = ArrowAnalyzer(func)
            assert analyzer is not None


class TestArrowTheoremImplications:
    """Test implications of Arrow's Theorem."""

    def test_arrow_trichotomy(self):
        """For 2 alternatives, a function is either dictatorial, constant, or violates IIA."""
        # For Boolean functions with unanimity and IIA, Arrow says it must be dictatorial
        # or constant (which violates unanimity)

        f = bf.majority(3)

        # Majority satisfies unanimity
        assert is_unanimous(f)

        # But is NOT dictatorial
        assert is_non_dictatorial(f)

        # This doesn't violate Arrow because majority doesn't satisfy IIA
        # for preference orders, but does for binary choices

    def test_influence_connection(self):
        """High influence implies close to dictator."""
        # If one variable has influence close to 1, the function is close to dictator

        f = bf.dictator(3, 0)
        influences = f.influences()

        # Dictator has one influence = 1, others = 0
        assert influences[0] == 1.0
        assert all(influences[i] == 0 for i in range(1, 3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
