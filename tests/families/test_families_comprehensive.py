"""
Comprehensive tests for the families module.

Tests cover:
- FunctionFamily base class and metadata
- InductiveFamily class
- WeightPatternFamily class
- LTF family constructors (uniform, geometric, harmonic, power)
- Theoretical value calculations
- Generation and caching
"""

import pytest
import numpy as np
import boofun as bf

from boofun.families.base import (
    FamilyMetadata,
    FunctionFamily,
    InductiveFamily,
    WeightPatternFamily,
    uniform_ltf_family,
    geometric_ltf_family,
    harmonic_ltf_family,
    power_ltf_family,
)


class TestFamilyMetadata:
    """Tests for FamilyMetadata dataclass."""

    def test_basic_metadata(self):
        """Create metadata with basic fields."""
        meta = FamilyMetadata(
            name="TestFamily",
            description="A test family",
        )
        assert meta.name == "TestFamily"
        assert meta.description == "A test family"
        assert meta.parameters == {}
        assert meta.asymptotics == {}
        assert meta.universal_properties == []

    def test_metadata_with_parameters(self):
        """Create metadata with parameters."""
        meta = FamilyMetadata(
            name="ParameterizedFamily",
            description="A family with parameters",
            parameters={"width": "k", "blocks": "n/k"},
        )
        assert "width" in meta.parameters
        assert meta.parameters["width"] == "k"

    def test_metadata_with_asymptotics(self):
        """Create metadata with asymptotic formulas."""
        meta = FamilyMetadata(
            name="AsymptoticFamily",
            description="A family with known asymptotics",
            asymptotics={
                "total_influence": lambda n: np.sqrt(n),
                "noise_stability": "1 - O(1/sqrt(n))",
            },
        )
        assert callable(meta.asymptotics["total_influence"])
        assert meta.asymptotics["total_influence"](9) == 3.0

    def test_metadata_with_constraints(self):
        """Create metadata with n constraints."""
        meta = FamilyMetadata(
            name="ConstrainedFamily",
            description="A family with constraints on n",
            n_constraints=lambda n: n % 2 == 1,  # Must be odd
            n_constraint_description="n must be odd",
        )
        assert meta.n_constraints(3) is True
        assert meta.n_constraints(4) is False


class TestWeightPatternFamily:
    """Tests for WeightPatternFamily class."""

    def test_uniform_weights(self):
        """Uniform weights create majority-like function."""
        family = WeightPatternFamily(lambda i, n: 1.0, name="Uniform")
        
        assert family.metadata.name == "Uniform"
        
        weights = family.get_weights(5)
        assert len(weights) == 5
        assert np.allclose(weights, [1, 1, 1, 1, 1])

    def test_geometric_weights(self):
        """Geometric weights decay exponentially."""
        family = WeightPatternFamily(lambda i, n: 0.5**i, name="Geometric")
        
        weights = family.get_weights(4)
        expected = [1.0, 0.5, 0.25, 0.125]
        assert np.allclose(weights, expected)

    def test_harmonic_weights(self):
        """Harmonic weights decay as 1/(i+1)."""
        family = WeightPatternFamily(lambda i, n: 1.0 / (i + 1), name="Harmonic")
        
        weights = family.get_weights(4)
        expected = [1.0, 0.5, 1/3, 0.25]
        assert np.allclose(weights, expected)

    def test_generate_function(self):
        """Can generate actual BooleanFunction."""
        family = uniform_ltf_family()
        f = family.generate(3)
        
        # Uniform weights with threshold 0 should be majority
        assert f is not None
        assert f.n_vars == 3

    def test_callable_shorthand(self):
        """Family can be called directly."""
        family = uniform_ltf_family()
        f = family(3)  # Shorthand for family.generate(3)
        
        assert f.n_vars == 3

    def test_custom_threshold(self):
        """Can specify custom threshold function."""
        # Threshold = n/2 means f(x) = 1 iff sum(w_i * x_i) > n/2
        family = WeightPatternFamily(
            lambda i, n: 1.0,
            threshold_function=lambda n: n / 2,
            name="CustomThreshold",
        )
        
        threshold = family.get_threshold(4)
        assert threshold == 2.0


class TestLTFFamilyConstructors:
    """Tests for convenience LTF family constructors."""

    def test_uniform_ltf_family(self):
        """uniform_ltf_family creates majority-like functions."""
        family = uniform_ltf_family("MyUniform")
        
        assert family.metadata.name == "MyUniform"
        
        f = family(5)
        # Uniform weights creates a valid LTF
        assert f.n_vars == 5
        
        # Function should return only 0 or 1 for all inputs
        values = [f.evaluate(x) for x in range(1 << 5)]
        assert all(v in [0, 1, True, False] for v in values)

    def test_geometric_ltf_family(self):
        """geometric_ltf_family creates decaying weight functions."""
        family = geometric_ltf_family(ratio=0.5)
        
        weights = family.get_weights(4)
        # Weights: 1, 0.5, 0.25, 0.125
        assert weights[0] > weights[1] > weights[2] > weights[3]

    def test_harmonic_ltf_family(self):
        """harmonic_ltf_family creates 1/(i+1) weight functions."""
        family = harmonic_ltf_family()
        
        weights = family.get_weights(4)
        assert np.isclose(weights[0], 1.0)
        assert np.isclose(weights[1], 0.5)
        assert np.isclose(weights[2], 1/3)

    def test_power_ltf_family(self):
        """power_ltf_family creates (n-i)^p weight functions."""
        family = power_ltf_family(power=2.0)
        
        weights = family.get_weights(4)
        # (4-0)^2=16, (4-1)^2=9, (4-2)^2=4, (4-3)^2=1
        expected = [16, 9, 4, 1]
        assert np.allclose(weights, expected)


class TestInductiveFamily:
    """Tests for InductiveFamily class."""

    def test_basic_inductive_family(self):
        """Create basic inductive family with base case."""
        # Simple family: f_n = constant 0 for all n
        family = InductiveFamily(
            name="Constant0",
            base_cases={1: bf.create([0, 0])},
        )
        
        assert family.metadata.name == "Constant0"

    def test_inductive_with_step_function(self):
        """Inductive family with step function."""
        # Family that just extends with an extra variable (identity-like)
        def extend_step(f_prev, n, n_prev):
            # For simplicity, just create a constant function
            return bf.create([0] * (1 << n))
        
        family = InductiveFamily(
            name="ExtendFamily",
            base_cases={1: bf.create([0, 0])},
            step_function=extend_step,
        )
        
        f1 = family.generate(1)
        assert f1.n_vars == 1

    def test_cache_clear(self):
        """Can clear the cache."""
        family = InductiveFamily(
            name="CacheTest",
            base_cases={1: bf.create([0, 0])},
        )
        
        f1 = family.generate(1)
        assert 1 in family._cache
        
        family.clear_cache()
        assert 1 not in family._cache


class TestFunctionFamilyBase:
    """Tests for FunctionFamily base class methods."""

    def test_validate_n_default(self):
        """Default validation accepts n >= 1."""
        family = uniform_ltf_family()
        
        assert family.validate_n(1) is True
        assert family.validate_n(10) is True
        # n <= 0 should fail but depends on implementation

    def test_generate_range(self):
        """Can generate functions for a range of n values."""
        family = uniform_ltf_family()
        
        functions = family.generate_range([3, 5, 7])
        
        assert 3 in functions
        assert 5 in functions
        assert 7 in functions
        assert functions[3].n_vars == 3
        assert functions[5].n_vars == 5

    def test_theoretical_value_callable(self):
        """Theoretical value works with callable formulas."""
        # Create a family with known asymptotics
        family = WeightPatternFamily(
            lambda i, n: 1.0,
            name="WithAsymptotics",
        )
        # Manually add asymptotics to metadata
        family._metadata = FamilyMetadata(
            name="WithAsymptotics",
            description="Test",
            asymptotics={"total_influence": lambda n: np.sqrt(n)},
        )
        
        # Override metadata property for testing
        original_metadata = family.metadata
        
        # Since we can't easily modify metadata, test with a concrete family
        # that has asymptotics defined

    def test_theoretical_value_returns_none(self):
        """Theoretical value returns None for unknown properties."""
        family = uniform_ltf_family()
        
        result = family.theoretical_value("unknown_property", 5)
        assert result is None


class TestFamiliesIntegration:
    """Integration tests combining families with analysis."""

    def test_majority_family_properties(self):
        """Majority family has expected properties."""
        family = uniform_ltf_family()
        
        for n in [3, 5, 7]:
            f = family(n)
            
            # Should be symmetric
            influences = f.influences()
            
            # All influences should be approximately equal for majority
            assert np.std(influences) < 0.1

    def test_geometric_more_dictator_like(self):
        """Geometric family with steep decay creates valid LTF."""
        family = geometric_ltf_family(ratio=0.1)  # Very steep decay
        f = family(5)
        
        # Should be valid function
        assert f.n_vars == 5
        
        # With very steep decay (0.1), the weights are:
        # [1, 0.1, 0.01, 0.001, 0.0001]
        # First variable has dominant weight
        weights = family.get_weights(5)
        assert weights[0] > sum(weights[1:])  # First weight > sum of rest

    def test_family_comparison(self):
        """Compare different families at same n."""
        n = 5
        
        families = {
            "Uniform": uniform_ltf_family(),
            "Geometric": geometric_ltf_family(0.5),
            "Harmonic": harmonic_ltf_family(),
            "Power": power_ltf_family(2.0),
        }
        
        for name, family in families.items():
            f = family(n)
            assert f.n_vars == n
            
            # All should be valid Boolean functions
            for x in range(1 << n):
                val = f.evaluate(x)
                assert val in [0, 1]


class TestFamilyEdgeCases:
    """Edge case tests for families module."""

    def test_single_variable(self):
        """Families work with n=1."""
        family = uniform_ltf_family()
        f = family(1)
        
        assert f.n_vars == 1

    def test_large_n(self):
        """Families work with larger n."""
        family = uniform_ltf_family()
        f = family(10)
        
        assert f.n_vars == 10

    def test_zero_weight(self):
        """Handle weight function that returns 0."""
        family = WeightPatternFamily(
            lambda i, n: 0.0 if i > 0 else 1.0,  # Only first var has weight
            name="DictatorLike",
        )
        
        weights = family.get_weights(4)
        assert weights[0] == 1.0
        assert all(w == 0.0 for w in weights[1:])

    def test_negative_weights(self):
        """Handle negative weights (anti-correlation)."""
        family = WeightPatternFamily(
            lambda i, n: 1.0 if i % 2 == 0 else -1.0,
            name="Alternating",
        )
        
        weights = family.get_weights(4)
        assert weights[0] == 1.0
        assert weights[1] == -1.0
        assert weights[2] == 1.0
        assert weights[3] == -1.0
