"""Tests for restriction operations on Boolean functions."""

import pytest
import numpy as np
import boolfunc as bf


class TestFixOperations:
    """Tests for the fix/restrict operations."""
    
    def test_fix_single_xor(self):
        """Test fixing a single variable in XOR function."""
        # XOR: f(x0, x1) = x0 XOR x1
        # Truth table: [0, 1, 1, 0] for inputs (0,0), (0,1), (1,0), (1,1)
        xor = bf.create([0, 1, 1, 0])
        
        # Fix x0 = 0: should give f(x1) = x1 (identity)
        f_x0_0 = xor.fix(0, 0)
        assert f_x0_0.n_vars == 1
        assert f_x0_0.evaluate(0) == False  # f(0) = 0 XOR 0 = 0
        assert f_x0_0.evaluate(1) == True   # f(1) = 0 XOR 1 = 1
        
        # Fix x0 = 1: should give f(x1) = NOT x1
        f_x0_1 = xor.fix(0, 1)
        assert f_x0_1.n_vars == 1
        assert f_x0_1.evaluate(0) == True   # f(0) = 1 XOR 0 = 1
        assert f_x0_1.evaluate(1) == False  # f(1) = 1 XOR 1 = 0
    
    def test_fix_single_and(self):
        """Test fixing a single variable in AND function."""
        # AND: f(x0, x1) = x0 AND x1
        # Truth table: [0, 0, 0, 1]
        and_func = bf.create([0, 0, 0, 1])
        
        # Fix x0 = 0: should give constant 0
        f_x0_0 = and_func.fix(0, 0)
        assert f_x0_0.n_vars == 1
        assert f_x0_0.evaluate(0) == False
        assert f_x0_0.evaluate(1) == False
        
        # Fix x0 = 1: should give f(x1) = x1 (identity)
        f_x0_1 = and_func.fix(0, 1)
        assert f_x0_1.n_vars == 1
        assert f_x0_1.evaluate(0) == False
        assert f_x0_1.evaluate(1) == True
    
    def test_fix_multi(self):
        """Test fixing multiple variables."""
        # 3-variable function
        majority = bf.BooleanFunctionBuiltins.majority(3)
        
        # Fix x0 = 1 and x1 = 1: should give constant 1
        f_fixed = majority.fix([0, 1], [1, 1])
        assert f_fixed.n_vars == 1
        # With x0=x1=1, majority(1,1,x2) = 1 for any x2
        assert f_fixed.evaluate(0) == True
        assert f_fixed.evaluate(1) == True
        
        # Fix x0 = 0 and x1 = 0: should give constant 0
        f_fixed2 = majority.fix([0, 1], [0, 0])
        assert f_fixed2.n_vars == 1
        # With x0=x1=0, majority(0,0,x2) = 0 for any x2
        assert f_fixed2.evaluate(0) == False
        assert f_fixed2.evaluate(1) == False
    
    def test_restrict_alias(self):
        """Test that restrict() is an alias for fix()."""
        xor = bf.create([0, 1, 1, 0])
        
        f_fix = xor.fix(0, 1)
        f_restrict = xor.restrict(0, 1)
        
        # Should produce same results
        assert f_fix.evaluate(0) == f_restrict.evaluate(0)
        assert f_fix.evaluate(1) == f_restrict.evaluate(1)
    
    def test_fix_invalid_value(self):
        """Test that invalid values raise errors."""
        xor = bf.create([0, 1, 1, 0])
        
        with pytest.raises(ValueError):
            xor.fix(0, 2)  # Value must be 0 or 1
        
        with pytest.raises(ValueError):
            xor.fix(0, -1)
    
    def test_fix_invalid_variable(self):
        """Test that invalid variable indices raise errors."""
        xor = bf.create([0, 1, 1, 0])
        
        with pytest.raises(ValueError):
            xor.fix(5, 0)  # Variable index out of range
        
        with pytest.raises(ValueError):
            xor.fix(-1, 0)


class TestDerivative:
    """Tests for the derivative operation."""
    
    def test_derivative_xor(self):
        """Derivative of XOR should be constant 1 (always influential)."""
        xor = bf.create([0, 1, 1, 0])
        
        # D_0 f for XOR
        d0 = xor.derivative(0)
        assert d0.n_vars == 2
        # XOR's derivative wrt any variable is always 1
        for i in range(4):
            assert d0.evaluate(i) == True
    
    def test_derivative_and(self):
        """Derivative of AND wrt x0 should be x1."""
        and_func = bf.create([0, 0, 0, 1])
        
        # D_0(x0 AND x1) = (0 AND x1) XOR (1 AND x1) = 0 XOR x1 = x1
        d0 = and_func.derivative(0)
        assert d0.evaluate(0) == False  # x1=0
        assert d0.evaluate(1) == True   # x1=1
        assert d0.evaluate(2) == False  # x1=0
        assert d0.evaluate(3) == True   # x1=1
    
    def test_derivative_constant(self):
        """Derivative of constant function should be 0."""
        const_0 = bf.BooleanFunctionBuiltins.constant(False, 2)
        
        d0 = const_0.derivative(0)
        for i in range(4):
            assert d0.evaluate(i) == False


class TestShift:
    """Tests for the shift operation."""
    
    def test_shift_xor(self):
        """Shifting XOR should produce expected results."""
        xor = bf.create([0, 1, 1, 0])
        
        # Shift by 0 should be identity
        shifted_0 = xor.shift(0)
        for i in range(4):
            assert shifted_0.evaluate(i) == xor.evaluate(i)
        
        # Shift by 1: f_1(x) = f(x XOR 1)
        shifted_1 = xor.shift(1)
        for x in range(4):
            assert shifted_1.evaluate(x) == xor.evaluate(x ^ 1)
    
    def test_shift_inverse(self):
        """Shifting twice by same amount should give original."""
        and_func = bf.create([0, 0, 0, 1])
        
        shifted = and_func.shift(3).shift(3)
        for i in range(4):
            assert shifted.evaluate(i) == and_func.evaluate(i)


class TestBias:
    """Tests for bias and balance operations."""
    
    def test_bias_balanced(self):
        """Balanced function should have bias 0."""
        xor = bf.create([0, 1, 1, 0])
        assert abs(xor.bias()) < 1e-10
    
    def test_bias_constant_0(self):
        """Constant 0 should have bias 1."""
        const_0 = bf.BooleanFunctionBuiltins.constant(False, 2)
        assert abs(const_0.bias() - 1.0) < 1e-10
    
    def test_bias_constant_1(self):
        """Constant 1 should have bias -1."""
        const_1 = bf.BooleanFunctionBuiltins.constant(True, 2)
        assert abs(const_1.bias() - (-1.0)) < 1e-10
    
    def test_is_balanced(self):
        """Test is_balanced check."""
        xor = bf.create([0, 1, 1, 0])
        assert xor.is_balanced() == True
        
        and_func = bf.create([0, 0, 0, 1])
        assert and_func.is_balanced() == False


class TestNegation:
    """Tests for the negation operation."""
    
    def test_negation(self):
        """Negation should flip all outputs."""
        and_func = bf.create([0, 0, 0, 1])
        not_and = and_func.negation()
        
        for i in range(4):
            assert not_and.evaluate(i) == (not and_func.evaluate(i))
    
    def test_double_negation(self):
        """Double negation should give original."""
        xor = bf.create([0, 1, 1, 0])
        double_neg = xor.negation().negation()
        
        for i in range(4):
            assert double_neg.evaluate(i) == xor.evaluate(i)
