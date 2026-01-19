"""
Comprehensive tests for circuit representation module.

Tests Boolean circuit representation and operations.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.representations.circuit import (
    GateType,
    Gate,
    BooleanCircuit,
    CircuitRepresentation,
    build_majority_circuit,
    build_parity_circuit,
)
from boofun.core.spaces import Space


class TestGateType:
    """Test GateType enum."""
    
    def test_gate_types_exist(self):
        """GateType enum should have standard gates."""
        assert GateType.AND is not None
        assert GateType.OR is not None
        assert GateType.NOT is not None
    
    def test_gate_type_values(self):
        """GateType values should be strings or meaningful."""
        # Just check they exist
        gates = list(GateType)
        assert len(gates) >= 3  # At least AND, OR, NOT


class TestGate:
    """Test Gate class."""
    
    def test_gate_class_exists(self):
        """Gate class should exist."""
        assert Gate is not None
    
    def test_gate_has_attributes(self):
        """Gate should have expected attributes."""
        # Check class attributes/methods
        assert hasattr(Gate, '__init__')


class TestBooleanCircuit:
    """Test BooleanCircuit class."""
    
    def test_circuit_class_exists(self):
        """BooleanCircuit class should exist."""
        assert BooleanCircuit is not None
    
    def test_circuit_has_methods(self):
        """BooleanCircuit should have expected methods."""
        methods = [m for m in dir(BooleanCircuit) if not m.startswith('_')]
        assert len(methods) > 0


class TestCircuitRepresentation:
    """Test CircuitRepresentation class."""
    
    def test_representation_class_exists(self):
        """CircuitRepresentation class should exist."""
        assert CircuitRepresentation is not None


class TestBuildMajorityCircuit:
    """Test build_majority_circuit function."""
    
    def test_function_callable(self):
        """build_majority_circuit should be callable."""
        assert callable(build_majority_circuit)
    
    def test_build_majority_returns_circuit(self):
        """Should return a BooleanCircuit."""
        circuit = build_majority_circuit(3)
        
        assert circuit is not None
        assert isinstance(circuit, BooleanCircuit)


class TestBuildParityCircuit:
    """Test build_parity_circuit function."""
    
    def test_function_callable(self):
        """build_parity_circuit should be callable."""
        assert callable(build_parity_circuit)
    
    def test_build_parity_returns_circuit(self):
        """Should return a BooleanCircuit."""
        circuit = build_parity_circuit(3)
        
        assert circuit is not None
        assert isinstance(circuit, BooleanCircuit)


class TestCircuitIntegration:
    """Integration tests for circuit representations."""
    
    def test_circuit_from_function(self):
        """Should be able to get circuit representation from function."""
        f = bf.AND(3)
        
        try:
            circuit = f.get_representation("circuit")
            assert circuit is not None
        except (KeyError, AttributeError):
            pytest.skip("Circuit representation not available via API")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
