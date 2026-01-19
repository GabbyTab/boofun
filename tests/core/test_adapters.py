"""
Tests for core/adapters module.

Tests for:
- LegacyAdapter
- CallableAdapter
- NumPyAdapter
- create_adapter factory
- Convenience functions
"""

import pytest
import numpy as np

from boolfunc.core.adapters import (
    LegacyAdapter,
    CallableAdapter,
    NumPyAdapter,
    create_adapter,
    adapt_callable,
    adapt_numpy_function,
)


class MockLegacyFunction:
    """Mock legacy function for testing."""
    
    def evaluate(self, inputs):
        """XOR-like evaluation."""
        if isinstance(inputs, list):
            return sum(inputs) % 2
        return inputs % 2


class TestLegacyAdapter:
    """Tests for LegacyAdapter."""
    
    def test_initialization(self):
        """LegacyAdapter initializes with defaults."""
        adapter = LegacyAdapter()
        
        assert adapter.evaluation_method == "evaluate"
        assert adapter.input_format == "auto"
    
    def test_adapt_basic(self):
        """Adapt a legacy function."""
        adapter = LegacyAdapter(n_vars=3)
        legacy = MockLegacyFunction()
        
        f = adapter.adapt(legacy)
        
        assert f is not None
    
    def test_adapt_missing_method_raises(self):
        """Raises if evaluation method not found."""
        adapter = LegacyAdapter(evaluation_method="nonexistent")
        
        class NoEvalMethod:
            pass
        
        with pytest.raises(AttributeError, match="nonexistent"):
            adapter.adapt(NoEvalMethod())
    
    def test_adapt_inputs_auto(self):
        """_adapt_inputs with auto format."""
        adapter = LegacyAdapter(n_vars=3)
        
        # Scalar
        result = adapter._adapt_inputs(np.array(5))
        assert result == 5
        
        # Vector
        result = adapter._adapt_inputs(np.array([0, 1, 1]))
        assert result == [0, 1, 1]
    
    def test_adapt_inputs_integer(self):
        """_adapt_inputs with integer format."""
        adapter = LegacyAdapter(input_format="integer", n_vars=3)
        
        # Convert binary vector to integer
        result = adapter._adapt_inputs(np.array([1, 0, 1]))
        assert isinstance(result, int)
    
    def test_adapt_inputs_binary(self):
        """_adapt_inputs with binary format."""
        adapter = LegacyAdapter(input_format="binary", n_vars=3)
        
        # Convert integer to binary
        result = adapter._adapt_inputs(np.array(5))
        assert isinstance(result, list)
    
    def test_adapt_output_auto(self):
        """_adapt_output with auto format."""
        adapter = LegacyAdapter()
        
        assert adapter._adapt_output(True) == True
        assert adapter._adapt_output(1) == True
        assert adapter._adapt_output(0) == False
        assert adapter._adapt_output(0.7) == True
        assert adapter._adapt_output(0.3) == False


class TestCallableAdapter:
    """Tests for CallableAdapter."""
    
    def test_initialization(self):
        """CallableAdapter initializes."""
        adapter = CallableAdapter(n_vars=3)
        
        assert adapter.n_vars == 3
        assert adapter.input_type == "binary_vector"
    
    def test_adapt_simple_callable(self):
        """Adapt a simple callable."""
        adapter = CallableAdapter(n_vars=2, input_type="binary_vector")
        
        # XOR function
        xor_func = lambda x: x[0] ^ x[1]
        
        f = adapter.adapt(xor_func)
        assert f is not None
    
    def test_call_with_binary_vector(self):
        """Call with binary vector input type."""
        adapter = CallableAdapter(n_vars=2, input_type="binary_vector")
        
        xor_func = lambda x: x[0] ^ x[1]
        
        result = adapter._call_with_adapted_inputs(xor_func, np.array([0, 1]))
        assert result == True
    
    def test_call_with_individual_args(self):
        """Call with individual arguments."""
        adapter = CallableAdapter(n_vars=2, input_type="individual_args")
        
        xor_func = lambda a, b: a ^ b
        
        result = adapter._call_with_adapted_inputs(xor_func, np.array([0, 1]))
        assert result == True
    
    def test_call_with_integer(self):
        """Call with integer input type."""
        adapter = CallableAdapter(n_vars=2, input_type="integer")
        
        parity = lambda x: bin(x).count('1') % 2
        
        result = adapter._call_with_adapted_inputs(parity, np.array(3))
        assert result == False  # 3 = 11 in binary, 2 ones -> even


class TestNumPyAdapter:
    """Tests for NumPyAdapter."""
    
    def test_initialization(self):
        """NumPyAdapter initializes."""
        adapter = NumPyAdapter(vectorized=True)
        
        assert adapter.vectorized == True
    
    def test_adapt_vectorized(self):
        """Adapt vectorized NumPy function."""
        adapter = NumPyAdapter(vectorized=True)
        
        # Simple parity
        np_func = lambda x: np.sum(x, axis=-1) % 2
        
        f = adapter.adapt(np_func, n_vars=3)
        assert f is not None
    
    def test_adapt_non_vectorized(self):
        """Adapt non-vectorized function."""
        adapter = NumPyAdapter(vectorized=False)
        
        # Simple function
        np_func = lambda x: np.sum(x) % 2
        
        f = adapter.adapt(np_func, n_vars=3)
        assert f is not None


class TestCreateAdapterFactory:
    """Tests for create_adapter factory."""
    
    def test_create_legacy(self):
        """Create legacy adapter."""
        adapter = create_adapter("legacy", n_vars=3)
        
        assert isinstance(adapter, LegacyAdapter)
    
    def test_create_callable(self):
        """Create callable adapter."""
        adapter = create_adapter("callable", n_vars=3)
        
        assert isinstance(adapter, CallableAdapter)
    
    def test_create_numpy(self):
        """Create NumPy adapter."""
        adapter = create_adapter("numpy", vectorized=True)
        
        assert isinstance(adapter, NumPyAdapter)
    
    def test_create_unknown_raises(self):
        """Unknown adapter type raises."""
        with pytest.raises(ValueError, match="Unknown adapter type"):
            create_adapter("unknown")


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_adapt_callable_function(self):
        """adapt_callable convenience function."""
        xor = lambda x: x[0] ^ x[1]
        
        f = adapt_callable(xor, n_vars=2)
        
        assert f is not None
    
    def test_adapt_numpy_function(self):
        """adapt_numpy_function convenience function."""
        np_parity = lambda x: np.sum(x) % 2
        
        f = adapt_numpy_function(np_parity, n_vars=3, vectorized=False)
        
        assert f is not None
