import pytest
import numpy as np
from boofun.core.representations.truth_table import TruthTableRepresentation
from boofun.core.representations.fourier_expansion import FourierExpansionRepresentation
from boofun.core.representations.symbolic import SymbolicRepresentation
from boofun.core.spaces import Space

# Dummy BooleanFunction wrapper for testing
class DummyBooleanFunction:
    def __init__(self, repr_type, data, space, n_vars):
        self.repr_type = repr_type
        self.data = data
        self.n_vars = n_vars
        self.space = space

    def get_n_vars(self):
        return self.n_vars

    def evaluate(self, inputs, rep_type=None, **kwargs):
        return self.repr_type.evaluate(inputs, self.data, self.space, self.n_vars)


AND_TRUTH_TABLE = np.array([0, 0, 0, 1])  # [00, 01, 10, 11]
AND_FOURIER_COEFFS = np.array([0.5, 0.5, 0.5, -0.5])  # Correct AND coefficients

XOR_TRUTH_TABLE = np.array([0, 1, 1, 0])
XOR_FOURIER_COEFFS = np.array([0, 0, 0, 1])  # Only {0,1} term

# Fixtures for common test objects

@pytest.fixture
def sym_rep():
    return SymbolicRepresentation()

@pytest.fixture
def tt_rep():
    return TruthTableRepresentation()

@pytest.fixture
def fourier_rep():
    return FourierExpansionRepresentation()

@pytest.fixture
def boo_cube():
    return Space.BOOLEAN_CUBE


## Truth Table Representation Tests ##

def test_truth_table_evaluate_single(tt_rep):
    """Test evaluation of single inputs"""
    # AND function
    assert tt_rep.evaluate(np.array(0), AND_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    assert tt_rep.evaluate(np.array(1), AND_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    assert tt_rep.evaluate(np.array(2), AND_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    
    # XOR function
    assert tt_rep.evaluate(np.array(0), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0
    assert tt_rep.evaluate(np.array(1), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 1
    assert tt_rep.evaluate(np.array(2), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 1
    assert tt_rep.evaluate(np.array(3), XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2) == 0

def test_truth_table_evaluate_batch(tt_rep):
    """Test batch evaluation"""
    inputs = np.array([0, 1, 2, 3])
    
    # AND function
    results = tt_rep.evaluate(inputs, AND_TRUTH_TABLE, space = boo_cube, n_vars = 2)
    assert np.array_equal(results, np.array([0, 0, 0, 1]))
    
    # XOR function
    results = tt_rep.evaluate(inputs, XOR_TRUTH_TABLE, space = boo_cube, n_vars = 2)
    assert np.array_equal(results, np.array([0, 1, 1, 0]))

def test_truth_table_dump(tt_rep):
    """Test serialization of truth table"""
    dumped = tt_rep.dump(AND_TRUTH_TABLE)
    assert dumped['type'] == 'truth_table'
    assert dumped['values'] == [0, 0, 0, 1]
    assert dumped['n'] == 2
    assert dumped['size'] == 4

def test_truth_table_create_empty(tt_rep):
    """Test empty truth table creation"""
    empty_tt = tt_rep.create_empty(3)
    assert len(empty_tt) == 8
    assert np.all(empty_tt == 0)

def test_truth_table_storage_requirements(tt_rep):
    """Test storage requirements calculation"""
    requirements = tt_rep.get_storage_requirements(3)
    assert requirements['entries'] == 8
    assert requirements['bytes'] == 1  # packed bits


## Fourier Expansion Representation Tests ##
def test_fourier_evaluate_single(fourier_rep):
    """Test evaluation of single inputs"""
    # AND function in ±1 domain: [1, 1, 1, -1] (from Fourier evaluation)
    assert fourier_rep.evaluate(0, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(1, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(2, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(3, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == -1.0
    
    # XOR function in ±1 domain
    assert fourier_rep.evaluate(0, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0
    assert fourier_rep.evaluate(1, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == -1.0
    assert fourier_rep.evaluate(2, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == -1.0
    assert fourier_rep.evaluate(3, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) == 1.0

def test_fourier_evaluate_batch(fourier_rep):
    """Test batch evaluation"""
    inputs = np.array([0, 1, 2, 3])
    
    # AND function: [1, 1, 1, -1] in ±1 domain (from Fourier evaluation)
    results = fourier_rep.evaluate(inputs, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2) 
    expected = np.array([1.0, 1.0, 1.0, -1.0])
    assert np.allclose(results, expected)
    
    # XOR function
    results = fourier_rep.evaluate(inputs, XOR_FOURIER_COEFFS, space = boo_cube, n_vars = 2) 
    expected = np.array([1.0, -1.0, -1.0, 1.0])
    assert np.allclose(results, expected)

def test_fourier_dump(fourier_rep):
    """Test serialization of Fourier coefficients"""
    dumped = fourier_rep.dump(AND_FOURIER_COEFFS)
    assert dumped['type'] == 'fourier_expansion'
    assert dumped['coefficients'] == [0.5, 0.5, 0.5, -0.5]
    assert dumped['metadata']['num_vars'] == 2
    assert dumped['metadata']['norm'] == pytest.approx(1.0)

def test_fourier_create_empty(fourier_rep):
    """Test empty Fourier coefficients creation"""
    empty_coeffs = fourier_rep.create_empty(2)
    assert len(empty_coeffs) == 4
    assert np.all(empty_coeffs == 0.0)

def test_fourier_storage_requirements(fourier_rep):
    """Test storage requirements calculation"""
    requirements = fourier_rep.get_storage_requirements(4)
    assert requirements['elements'] == 16
    assert requirements['bytes'] == 128  # 16 floats * 8 bytes each


## Symbolic representation tests
def test_symbolic_evaluate_single(sym_rep):
    """Test evaluating a symbolic expression with one wrapped function"""
    expr = "x0"
    vars = [DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2)]
    
    # should evaluate AND(x0, x1)
    result = sym_rep.evaluate(np.array(3), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=2)
    assert result == 1

    result = sym_rep.evaluate(np.array(1), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=2)
    assert result == 0

def test_symbolic_evaluate_sum_two_functions(sym_rep):
    """Test evaluating a symbolic expression that sums two subfunctions"""
    expr = "x0 + x1"  # sum of outputs from two subfunctions

    # Two identical AND functions
    vars = [
        DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2),
        DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2)
    ]

    # Evaluate input [0, 0, 0, 0] (2 functions * 2 bits each)
    result = sym_rep.evaluate(np.array(0), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=4)
    assert result == 0

    # Evaluate input [1, 1, 1, 1] → each AND([1, 1]) = 1, so 1 + 1 = 2
    result = sym_rep.evaluate(np.array(15), (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=4)
    assert result == 2


def test_symbolic_evaluate_batch(sym_rep):
    """Test batch evaluation of symbolic expressions"""
    expr = "x0"
    vars = [DummyBooleanFunction(TruthTableRepresentation(), AND_TRUTH_TABLE, Space.BOOLEAN_CUBE, 2)]

    inputs = np.array([0, 1, 2, 3])
    expected = np.array([0, 0, 0, 1])
    result = sym_rep.evaluate(inputs, (expr, vars), space=Space.BOOLEAN_CUBE, n_vars=2)
    assert np.array_equal(result, expected)


def test_symbolic_create_empty(sym_rep):
    """Test empty symbolic creation"""
    empty_expr, var_list = sym_rep.create_empty(3)
    assert empty_expr == ""
    assert var_list == ["x0", "x1", "x2"]

def test_symbolic_dump(sym_rep):
    """Test dumping symbolic data"""
    expr = "x0 and x1"
    vars = ["x0", "x1"]
    dumped = sym_rep.dump((expr, vars))
    assert dumped["expression"] == "x0 and x1"
    assert dumped["variables"] == ["x0", "x1"]


## Conversion Tests ##
def test_truth_table_to_fourier_conversion(tt_rep, fourier_rep):
    """Test conversion from truth table to Fourier coefficients"""
    # Convert AND truth table to Fourier coefficients
    fourier_coeffs = fourier_rep.convert_from(tt_rep, AND_TRUTH_TABLE, space = boo_cube, n_vars = 2)
    
    # Should match known Fourier coefficients
    assert fourier_coeffs.shape == (4,)
    assert np.allclose(fourier_coeffs, AND_FOURIER_COEFFS, atol=1e-5)

def test_fourier_to_truth_table_conversion(tt_rep, fourier_rep):
    """Test conversion from Fourier coefficients to truth table"""
    # Convert Fourier coefficients to truth table
    truth_table = tt_rep.convert_from(fourier_rep, AND_FOURIER_COEFFS, space = boo_cube, n_vars = 2)
    
    # Should match known truth table
    assert truth_table.shape == (4,)
    assert np.array_equal(truth_table, AND_TRUTH_TABLE)



## Edge Case Tests ##

class TestCircuitRepresentation:
    """Test circuit representation."""
    
    def test_circuit_evaluate_simple(self):
        """Test evaluation of simple circuit."""
        from boofun.core.representations.circuit import CircuitRepresentation, BooleanCircuit, GateType
        
        repr_obj = CircuitRepresentation()
        
        # Create simple AND circuit
        circuit = BooleanCircuit(2)
        and_gate = circuit.add_gate(GateType.AND, [circuit.input_gates[0], circuit.input_gates[1]])
        circuit.set_output(and_gate)
        
        # Test evaluation
        result = repr_obj.evaluate(np.array([0, 0]), circuit, Space.BOOLEAN_CUBE, 2)
        assert result == False
        
        result = repr_obj.evaluate(np.array([1, 1]), circuit, Space.BOOLEAN_CUBE, 2)
        assert result == True
    
    def test_circuit_convert_from_truth_table(self):
        """Test conversion from truth table to circuit."""
        from boofun.core.representations.circuit import CircuitRepresentation
        
        repr_obj = CircuitRepresentation()
        truth_repr = TruthTableRepresentation()
        
        # Simple AND function
        truth_table = np.array([False, False, False, True])
        
        circuit = repr_obj.convert_from(truth_repr, truth_table, Space.BOOLEAN_CUBE, 2)
        
        # Verify conversion
        for i in range(4):
            circuit_val = repr_obj.evaluate(i, circuit, Space.BOOLEAN_CUBE, 2)
            truth_val = truth_repr.evaluate(i, truth_table, Space.BOOLEAN_CUBE, 2)
            assert circuit_val == truth_val


class TestBDDRepresentation:
    """Test BDD representation."""
    
    def test_bdd_evaluate_simple(self):
        """Test evaluation of simple BDD."""
        from boofun.core.representations.bdd import BDDRepresentation, BDD
        
        repr_obj = BDDRepresentation()
        
        # Create simple BDD for x0
        bdd = BDD(2)
        bdd.root = bdd.create_node(0, bdd.terminal_false, bdd.terminal_true)
        
        # Test evaluation
        result = repr_obj.evaluate(np.array([0, 0]), bdd, Space.BOOLEAN_CUBE, 2)
        assert result == False
        
        result = repr_obj.evaluate(np.array([1, 0]), bdd, Space.BOOLEAN_CUBE, 2)
        assert result == True
    
    def test_bdd_convert_from_truth_table(self):
        """Test conversion from truth table to BDD."""
        from boofun.core.representations.bdd import BDDRepresentation
        
        repr_obj = BDDRepresentation()
        truth_repr = TruthTableRepresentation()
        
        # XOR function
        truth_table = np.array([False, True, True, False])
        
        bdd = repr_obj.convert_from(truth_repr, truth_table, Space.BOOLEAN_CUBE, 2)
        
        # Verify conversion
        for i in range(4):
            bdd_val = repr_obj.evaluate(i, bdd, Space.BOOLEAN_CUBE, 2)
            truth_val = truth_repr.evaluate(i, truth_table, Space.BOOLEAN_CUBE, 2)
            assert bdd_val == truth_val


class TestPolynomialRepresentation:
    """Test polynomial (ANF) representation."""
    
    def test_polynomial_evaluate_simple(self):
        """Test evaluation of polynomial representation."""
        from boofun.core.representations.polynomial import PolynomialRepresentation
        
        repr_obj = PolynomialRepresentation()
        
        # XOR function: x0 ⊕ x1 as polynomial dictionary
        # Polynomial form: {frozenset(): 0, frozenset({0}): 1, frozenset({1}): 1, frozenset({0,1}): 0}
        coeffs = {
            frozenset(): 0,        # constant term
            frozenset({0}): 1,     # x0 coefficient  
            frozenset({1}): 1,     # x1 coefficient
            frozenset({0, 1}): 0   # x0*x1 coefficient
        }
        
        result = repr_obj.evaluate(0, coeffs, Space.BOOLEAN_CUBE, 2)
        assert result == 0  # 0 ⊕ 0 = 0
        
        result = repr_obj.evaluate(1, coeffs, Space.BOOLEAN_CUBE, 2)
        assert result == 1  # 0 ⊕ 1 = 1
        
        result = repr_obj.evaluate(3, coeffs, Space.BOOLEAN_CUBE, 2)
        assert result == 0  # 1 ⊕ 1 = 0 (XOR of 1,1 is 0)
    
    def test_polynomial_convert_from_truth_table(self):
        """Test conversion from truth table to polynomial."""
        from boofun.core.representations.polynomial import PolynomialRepresentation
        
        repr_obj = PolynomialRepresentation()
        truth_repr = TruthTableRepresentation()
        
        # XOR function
        truth_table = np.array([0, 1, 1, 0])
        
        polynomial = repr_obj.convert_from(truth_repr, truth_table, Space.BOOLEAN_CUBE, 2)
        
        # Verify conversion by evaluating both
        for i in range(4):
            truth_val = truth_repr.evaluate(i, truth_table, Space.BOOLEAN_CUBE, 2)
            poly_val = repr_obj.evaluate(i, polynomial, Space.BOOLEAN_CUBE, 2)
            assert truth_val == poly_val


class TestSparseTruthTableRepresentation:
    """Test sparse truth table representation."""
    
    def test_sparse_evaluate(self):
        """Test evaluation of sparse representation."""
        from boofun.core.representations.sparse_truth_table import SparseTruthTableRepresentation
        
        repr_obj = SparseTruthTableRepresentation()
        
        # Function that's True only at indices 1 and 3 (proper sparse format)
        sparse_data = {
            'default_value': False,
            'exceptions': {1: True, 3: True},
            'size': 4
        }
        
        result = repr_obj.evaluate(0, sparse_data, Space.BOOLEAN_CUBE, 2)
        assert result == False
        
        result = repr_obj.evaluate(1, sparse_data, Space.BOOLEAN_CUBE, 2)
        assert result == True
        
        result = repr_obj.evaluate(3, sparse_data, Space.BOOLEAN_CUBE, 2)
        assert result == True
    
    def test_sparse_convert_from_truth_table(self):
        """Test conversion from truth table to sparse representation."""
        from boofun.core.representations.sparse_truth_table import SparseTruthTableRepresentation
        
        repr_obj = SparseTruthTableRepresentation()
        truth_repr = TruthTableRepresentation()
        
        truth_table = np.array([False, True, False, True])
        
        sparse_data = repr_obj.convert_from(truth_repr, truth_table, Space.BOOLEAN_CUBE, 2)
        
        # Should be proper sparse format with only indices 1 and 3 True
        expected = {
            'default_value': False,
            'exceptions': {1: True, 3: True},
            'size': 4,
            'n_vars': 2  # Added by the representation
        }
        assert sparse_data == expected


class TestRepresentationConversions:
    """Test conversions between different representations."""
    
    def test_round_trip_conversions(self):
        """Test round-trip conversions preserve function behavior."""
        from boofun.core.representations.polynomial import PolynomialRepresentation
        from boofun.core.representations.sparse_truth_table import SparseTruthTableRepresentation
        
        truth_repr = TruthTableRepresentation()
        poly_repr = PolynomialRepresentation()
        sparse_repr = SparseTruthTableRepresentation()
        
        # Original XOR function
        original_truth_table = np.array([0, 1, 1, 0])
        
        # Truth table -> Polynomial -> Truth table
        polynomial = poly_repr.convert_from(truth_repr, original_truth_table, Space.BOOLEAN_CUBE, 2)
        recovered_truth_table = poly_repr.convert_to(truth_repr, polynomial, Space.BOOLEAN_CUBE, 2)
        np.testing.assert_array_equal(original_truth_table, recovered_truth_table)
        
        # Truth table -> Sparse -> Truth table
        sparse_data = sparse_repr.convert_from(truth_repr, original_truth_table, Space.BOOLEAN_CUBE, 2)
        recovered_truth_table2 = sparse_repr.convert_to(truth_repr, sparse_data, Space.BOOLEAN_CUBE, 2)
        np.testing.assert_array_equal(original_truth_table, recovered_truth_table2)


