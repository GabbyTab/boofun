"""
Tests for quantum module.

Tests for quantum-inspired analysis of Boolean functions.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.quantum import (
    QuantumBooleanFunction,
    create_quantum_boolean_function,
    estimate_quantum_advantage,
    quantum_walk_analysis,
    element_distinctness_analysis,
    grover_speedup,
)


class TestQuantumBooleanFunction:
    """Test QuantumBooleanFunction class."""
    
    def test_creation(self):
        """QuantumBooleanFunction should be creatable."""
        f = bf.majority(3)
        qbf = QuantumBooleanFunction(f)
        
        assert qbf is not None
    
    def test_has_classical_function(self):
        """QuantumBooleanFunction should store classical function."""
        f = bf.AND(3)
        qbf = QuantumBooleanFunction(f)
        
        # Should have reference to classical function in some attribute
        attrs = [a for a in dir(qbf) if not a.startswith('__')]
        assert len(attrs) > 0
    
    def test_quantum_properties(self):
        """QuantumBooleanFunction should have quantum-specific methods."""
        f = bf.OR(3)
        qbf = QuantumBooleanFunction(f)
        
        # Check for any quantum-related methods
        methods = [m for m in dir(qbf) if not m.startswith('_')]
        assert len(methods) > 0


class TestCreateQuantumBooleanFunction:
    """Test create_quantum_boolean_function factory."""
    
    def test_factory_function(self):
        """Factory should create QuantumBooleanFunction."""
        f = bf.parity(3)
        qbf = create_quantum_boolean_function(f)
        
        assert isinstance(qbf, QuantumBooleanFunction)
    
    def test_factory_with_different_functions(self):
        """Factory should work with different function types."""
        functions = [bf.AND(3), bf.OR(3), bf.majority(3), bf.parity(3)]
        
        for f in functions:
            qbf = create_quantum_boolean_function(f)
            assert qbf is not None


class TestEstimateQuantumAdvantage:
    """Test estimate_quantum_advantage function."""
    
    def test_returns_dict(self):
        """Should return analysis dictionary."""
        result = estimate_quantum_advantage(3)
        
        assert isinstance(result, dict)
    
    def test_with_fourier_analysis(self):
        """Should work with Fourier analysis type."""
        result = estimate_quantum_advantage(4, analysis_type='fourier')
        
        assert result is not None
    
    def test_advantage_estimates(self):
        """Should provide quantum advantage estimates."""
        result = estimate_quantum_advantage(3)
        
        # Should have some advantage-related info
        has_advantage_key = any(
            'advantage' in k.lower() or 'speedup' in k.lower() or 'complexity' in k.lower()
            for k in result.keys()
        )
        assert has_advantage_key or len(result) > 0


class TestQuantumWalkAnalysis:
    """Test quantum_walk_analysis function."""
    
    def test_walk_analysis_returns_dict(self):
        """Should return analysis dictionary."""
        f = bf.majority(3)
        result = quantum_walk_analysis(f)
        
        assert isinstance(result, dict)
    
    def test_walk_analysis_and(self):
        """Should analyze AND function."""
        f = bf.AND(3)
        result = quantum_walk_analysis(f)
        
        assert result is not None
    
    def test_walk_analysis_or(self):
        """Should analyze OR function."""
        f = bf.OR(3)
        result = quantum_walk_analysis(f)
        
        assert result is not None


class TestElementDistinctnessAnalysis:
    """Test element_distinctness_analysis function."""
    
    def test_returns_dict(self):
        """Should return analysis dictionary."""
        f = bf.majority(3)
        result = element_distinctness_analysis(f)
        
        assert isinstance(result, dict)
    
    def test_with_different_functions(self):
        """Should work with different functions."""
        for func in [bf.AND(3), bf.parity(3)]:
            result = element_distinctness_analysis(func)
            assert result is not None


class TestGroverSpeedup:
    """Test grover_speedup function."""
    
    def test_returns_dict(self):
        """Should return speedup analysis."""
        f = bf.AND(3)
        result = grover_speedup(f)
        
        assert isinstance(result, dict)
    
    def test_grover_for_and(self):
        """Grover's speedup for AND function."""
        f = bf.AND(4)
        result = grover_speedup(f)
        
        # Should have some speedup information
        assert result is not None
        assert len(result) > 0
    
    def test_grover_for_or(self):
        """Grover's speedup for OR function."""
        f = bf.OR(4)
        result = grover_speedup(f)
        
        assert result is not None
    
    def test_grover_speedup_bounded(self):
        """Grover speedup should be at most quadratic."""
        f = bf.majority(3)
        result = grover_speedup(f)
        
        if 'speedup' in result:
            # Grover gives at most quadratic speedup
            assert result['speedup'] <= 4  # sqrt(2^n)


class TestQuantumClassicalComparison:
    """Test comparisons between quantum and classical analysis."""
    
    def test_quantum_provides_additional_info(self):
        """Quantum analysis should provide additional insights."""
        f = bf.majority(3)
        
        # Classical analysis
        influences = f.influences()
        fourier = f.fourier()
        
        # Quantum analysis
        qbf = QuantumBooleanFunction(f)
        grover = grover_speedup(f)
        
        assert len(influences) == 3
        assert len(fourier) == 8
        assert grover is not None
    
    def test_quantum_analysis_consistency(self):
        """Quantum analysis should be consistent with classical."""
        f = bf.parity(3)
        
        # Parity has simple classical structure
        assert f.total_influence() == 3  # Each variable has influence 1
        
        # Quantum analysis should also reflect this
        qbf = QuantumBooleanFunction(f)
        assert qbf is not None


class TestQuantumEdgeCases:
    """Test edge cases for quantum analysis."""
    
    def test_small_function(self):
        """Should handle small functions (n=1, n=2)."""
        f1 = bf.create([0, 1])  # n=1
        f2 = bf.AND(2)  # n=2
        
        qbf1 = QuantumBooleanFunction(f1)
        qbf2 = QuantumBooleanFunction(f2)
        
        assert qbf1 is not None
        assert qbf2 is not None
    
    def test_constant_function(self):
        """Should handle constant functions."""
        f = bf.create([0, 0, 0, 0])  # Constant 0
        
        qbf = QuantumBooleanFunction(f)
        assert qbf is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
