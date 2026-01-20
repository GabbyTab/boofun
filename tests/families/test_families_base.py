import sys
sys.path.insert(0, "src")
"""
Tests for families/base module.

Tests for:
- FamilyMetadata dataclass
- FunctionFamily ABC
- InductiveFamily class
- WeightPatternFamily class
- Convenience constructors
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
    
    def test_creation_minimal(self):
        """Create metadata with minimal args."""
        meta = FamilyMetadata(
            name="Test",
            description="A test family"
        )
        
        assert meta.name == "Test"
        assert meta.description == "A test family"
        assert meta.parameters == {}
    
    def test_creation_with_parameters(self):
        """Create metadata with parameters."""
        meta = FamilyMetadata(
            name="Tribes",
            description="Tribes function",
            parameters={"k": "number of tribes", "w": "tribe width"}
        )
        
        assert "k" in meta.parameters
        assert "w" in meta.parameters
    
    def test_asymptotics(self):
        """Metadata can have asymptotics."""
        meta = FamilyMetadata(
            name="Parity",
            description="XOR",
            asymptotics={"total_influence": lambda n: n}
        )
        
        assert "total_influence" in meta.asymptotics
        assert meta.asymptotics["total_influence"](5) == 5
    
    def test_n_constraints(self):
        """Metadata can have n constraints."""
        meta = FamilyMetadata(
            name="Majority",
            description="MAJ",
            n_constraints=lambda n: n % 2 == 1,
            n_constraint_description="n must be odd"
        )
        
        assert meta.n_constraints(3) == True
        assert meta.n_constraints(4) == False


class TestInductiveFamily:
    """Tests for InductiveFamily."""
    
    def test_initialization(self):
        """InductiveFamily initializes."""
        family = InductiveFamily(name="Test")
        
        assert family._name == "Test"
        assert family._step_size == 1
    
    def test_metadata(self):
        """InductiveFamily has metadata."""
        family = InductiveFamily(name="MyFamily")
        
        meta = family.metadata
        assert meta.name == "MyFamily"
        assert "Inductively defined" in meta.description
    
    def test_with_base_cases(self):
        """InductiveFamily with provided base cases."""
        f1 = bf.parity(1)
        
        family = InductiveFamily(
            name="Test",
            base_cases={1: f1}
        )
        
        result = family.base_case(1)
        assert result is f1
    
    def test_base_case_not_found(self):
        """Base case returns None if not found."""
        family = InductiveFamily(name="Test")
        
        result = family.base_case(5)
        assert result is None
    
    def test_generate_no_base_raises(self):
        """Generate without base case raises."""
        family = InductiveFamily(name="Test")
        
        with pytest.raises(ValueError, match="no base case"):
            family.generate(5)
    
    def test_clear_cache(self):
        """Can clear cache."""
        family = InductiveFamily(
            name="Test",
            base_cases={1: bf.parity(1)}
        )
        
        family._cache[5] = bf.parity(5)
        assert 5 in family._cache
        
        family.clear_cache()
        assert family._cache == {}


class TestWeightPatternFamily:
    """Tests for WeightPatternFamily."""
    
    def test_initialization(self):
        """WeightPatternFamily initializes."""
        family = WeightPatternFamily(
            weight_function=lambda i, n: 1.0,
            name="Uniform"
        )
        
        assert family._name == "Uniform"
    
    def test_metadata(self):
        """WeightPatternFamily has metadata."""
        family = WeightPatternFamily(
            weight_function=lambda i, n: 1.0,
            name="Uniform"
        )
        
        meta = family.metadata
        assert meta.name == "Uniform"
        assert "LTF" in meta.description
    
    def test_get_weights(self):
        """get_weights returns weight array."""
        family = WeightPatternFamily(
            weight_function=lambda i, n: float(i + 1),
            name="Linear"
        )
        
        weights = family.get_weights(3)
        
        assert np.array_equal(weights, [1.0, 2.0, 3.0])
    
    def test_get_threshold_default(self):
        """Default threshold is 0."""
        family = WeightPatternFamily(
            weight_function=lambda i, n: 1.0
        )
        
        threshold = family.get_threshold(3)
        assert threshold == 0
    
    def test_get_threshold_custom(self):
        """Custom threshold function."""
        family = WeightPatternFamily(
            weight_function=lambda i, n: 1.0,
            threshold_function=lambda n: n / 2
        )
        
        threshold = family.get_threshold(10)
        assert threshold == 5
    
    def test_generate(self):
        """Generate creates a BooleanFunction."""
        family = WeightPatternFamily(
            weight_function=lambda i, n: 1.0,
            name="Uniform"
        )
        
        f = family.generate(3)
        
        assert f is not None
        assert f.n_vars == 3


class TestConvenienceConstructors:
    """Tests for convenience constructors."""
    
    def test_uniform_ltf_family(self):
        """uniform_ltf_family creates uniform weights."""
        family = uniform_ltf_family()
        
        weights = family.get_weights(5)
        assert all(w == 1.0 for w in weights)
    
    def test_geometric_ltf_family(self):
        """geometric_ltf_family creates geometric weights."""
        family = geometric_ltf_family(ratio=0.5)
        
        weights = family.get_weights(4)
        expected = [1.0, 0.5, 0.25, 0.125]
        assert np.allclose(weights, expected)
    
    def test_harmonic_ltf_family(self):
        """harmonic_ltf_family creates harmonic weights."""
        family = harmonic_ltf_family()
        
        weights = family.get_weights(4)
        expected = [1.0, 0.5, 1/3, 0.25]
        assert np.allclose(weights, expected)
    
    def test_power_ltf_family(self):
        """power_ltf_family creates power-law weights."""
        family = power_ltf_family(power=2.0)
        
        weights = family.get_weights(3)
        # (3-0)^2, (3-1)^2, (3-2)^2 = 9, 4, 1
        expected = [9.0, 4.0, 1.0]
        assert np.allclose(weights, expected)
    
    def test_constructors_generate(self):
        """All constructors can generate functions."""
        families = [
            uniform_ltf_family(),
            geometric_ltf_family(),
            harmonic_ltf_family(),
            power_ltf_family()
        ]
        
        for family in families:
            f = family.generate(3)
            assert f is not None
            assert f.n_vars == 3


class TestFunctionFamilyInterface:
    """Test FunctionFamily interface via concrete implementations."""
    
    def test_call_shorthand(self):
        """__call__ is shorthand for generate."""
        family = uniform_ltf_family()
        
        f1 = family.generate(3)
        f2 = family(3)
        
        # Should produce equivalent functions
        assert f1.n_vars == f2.n_vars
    
    def test_validate_n_default(self):
        """Default validate_n accepts n >= 1."""
        family = uniform_ltf_family()
        
        assert family.validate_n(1) == True
        assert family.validate_n(10) == True
        # n=0 might be edge case
    
    def test_theoretical_value_none(self):
        """theoretical_value returns None if not defined."""
        family = uniform_ltf_family()
        
        result = family.theoretical_value("total_influence", 5)
        assert result is None
    
    def test_generate_range(self):
        """generate_range creates dict of functions."""
        family = uniform_ltf_family()
        
        funcs = family.generate_range([2, 3, 4])
        
        assert 2 in funcs
        assert 3 in funcs
        assert 4 in funcs
        assert funcs[3].n_vars == 3
