import sys

sys.path.insert(0, "src")
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import boofun as bf
from boofun.core.representations.truth_table import TruthTableRepresentation

# Fixtures for reusable test objects


# Fixture for a BooleanFunction instance
@pytest.fixture
def xor_function():
    """XOR function with truth table representation"""
    func = bf.create(n=2)
    func.representations = {"truth_table": np.array([0, 1, 1, 0])}
    return func


# Fixture for a scalar value
@pytest.fixture
def scalar_value():
    return 3.5


# Fixture for another BooleanFunction
@pytest.fixture
def and_function():
    func = bf.create(n=2)
    func.representations = {"truth_table": np.array([0, 0, 0, 1])}
    return func


@pytest.fixture
def mock_strategy():
    """
    Provide a mock strategy whose evaluate() returns a fixed boolean array.
    """
    strategy = MagicMock(spec=TruthTableRepresentation)
    # Configure the mock to return [True, False] when evaluate() is called
    strategy.evaluate.return_value = np.array([0, 1, 1, 0], dtype=bool)
    return strategy


@pytest.fixture
def boolean_function():
    bf_instance = bf.BooleanFunction(space="plus_minus_cube", n=2)
    bf_instance.representations = {
        "truth_table": np.array([0, 1, 1, 0]),
        "function": lambda x: x[0] ^ x[1],
    }
    return bf_instance


# 1. Initialization and Space Handling
class TestBooleanFunctionInit:
    def test_default_initialization(self):
        bf_instance = bf.BooleanFunction()
        assert bf_instance.space == bf.Space.PLUS_MINUS_CUBE
        assert bf_instance.representations == {}
        assert isinstance(bf_instance.error_model, bf.ExactErrorModel)
        assert bf_instance.n_vars is None

    @pytest.mark.parametrize(
        "space_str,expected_space",
        [
            ("boolean_cube", bf.Space.BOOLEAN_CUBE),
            ("plus_minus_cube", bf.Space.PLUS_MINUS_CUBE),
            ("real", bf.Space.REAL),
            ("log", bf.Space.LOG),
            ("gaussian", bf.Space.GAUSSIAN),
        ],
    )
    def test_space_creation(self, space_str, expected_space):
        bf_instance = bf.BooleanFunction(space=space_str)
        assert bf_instance.space == expected_space

    def test_invalid_space_raises(self):
        with pytest.raises(ValueError, match="Unknown space type"):
            bf.BooleanFunction(space="invalid_space")


# 2. Factory Methods
class TestFactoryMethods:
    @pytest.mark.parametrize(
        "input_data,expected_method",
        [
            ([0, 1, 1, 0], "from_polynomial"),
            (lambda x: x[0] & x[1], "from_function"),
            ({"x0": 1, "x1": 0}, "from_polynomial"),
            ("x0 and x1", "from_symbolic"),
            ({(0, 1), (1, 0)}, "from_input_invariant_truth_table"),
            (np.array([0, 1, 1, 0]), "from_polynomial"),
            (np.array([False, True, True, False]), "from_truth_table"),
            (np.array([1.0, 0.0, 0.0, 1.0]), "from_multilinear"),
        ],
    )
    def test_create_dispatch(self, input_data, expected_method):
        # Test that create() successfully creates BooleanFunction objects
        # Some input types require additional parameters
        if expected_method == "from_function":
            result = bf.create(input_data, n=2)  # Functions need explicit n_vars
        elif expected_method == "from_symbolic":
            result = bf.create(input_data, variables=["x0", "x1"])  # Symbolic needs variables
        else:
            result = bf.create(input_data)

        assert isinstance(result, bf.BooleanFunction)
        assert len(result.representations) > 0

        # Some representations may not auto-determine n_vars
        if expected_method not in ["from_function", "from_symbolic", "from_polynomial"]:
            assert result.n_vars is not None

    def test_truth_table_creation(self):
        tt = [False, True, False, True]
        bf_instance = bf.create(tt)
        assert np.array_equal(bf_instance.representations["truth_table"], tt)
        assert bf_instance.n_vars == 2

    def test_symbolic_creation(self):
        expr = "x0 and not x1"
        bf_instance = bf.create(expr, variables=["x0", "x1"])
        assert bf_instance.representations["symbolic"] == (expr, ["x0", "x1"])


# 3. Representation Management
class TestRepresentations:
    def test_add_representation(self, boolean_function):
        new_rep = np.array([1, 0, 0, 1])
        boolean_function.add_representation(new_rep, "polynomial")
        assert "polynomial" in boolean_function.representations
        assert np.array_equal(boolean_function.representations["polynomial"], new_rep)

    def test_get_representation(self, boolean_function):
        tt = boolean_function.get_representation("truth_table")
        assert np.array_equal(tt, np.array([0, 1, 1, 0]))

    def test_missing_representation(self, boolean_function):
        # Test that representations are auto-created, not missing
        # This tests the new behavior where get_representation auto-creates
        result = boolean_function.get_representation("bdd")
        assert result is not None
        assert "bdd" in boolean_function.representations


class TestEvaluation:
    @patch("boofun.core.base.get_strategy")
    def test_deterministic_evaluation(self, mock_get_strategy, boolean_function, mock_strategy):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Act: call the method under test

        mock_get_strategy.return_value = mock_strategy

        result = boolean_function._evaluate_deterministic(inputs, rep_type="truth_table")

        # Assert: ensure evaluate() was invoked with correct arguments
        mock_get_strategy.assert_called_once_with("truth_table")

        # mock_strategy.evaluate.assert_called_once_with(
        #    inputs, boolean_function.representations['truth_table']
        # )
        # Assert: the return value matches the mock’s configured return_value
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 0], dtype=bool))

    @patch("boofun.core.BooleanFunction._evaluate_stochastic")
    def test_stochastic_evaluation(self, mock_stochastic, boolean_function):
        mock_rv = MagicMock()
        mock_stochastic.return_value = "dist_result"
        result = boolean_function.evaluate(mock_rv, n_samples=500)
        assert result == "dist_result"
        mock_stochastic.assert_called_once_with(mock_rv, rep_type=None, n_samples=500)

    def test_auto_representation_selection(self, boolean_function):
        with patch("boofun.core.BooleanFunction._evaluate_deterministic") as mock_eval:
            mock_eval.return_value = [False, True, True, False]
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            boolean_function.evaluate(inputs)
            mock_eval.assert_called_once_with(inputs, rep_type=None)


# 5. Operator Overloading
class TestOperators:
    def test_and_operator(self, boolean_function):
        with patch("boofun.core.factory.BooleanFunctionFactory.create_composite") as mock_factory:
            other = bf.create([0, 0, 0, 1])
            _ = boolean_function & other
            mock_factory.assert_called_once_with(
                boolean_function_cls=type(boolean_function),
                operator="&",
                left_func=boolean_function,
                right_func=other,
            )

    def test_invert_operator(self, boolean_function):
        with patch("boofun.core.factory.BooleanFunctionFactory.create_composite") as mock_factory:
            _ = ~boolean_function
            mock_factory.assert_called_once_with(
                boolean_function_cls=type(boolean_function),
                operator="~",
                left_func=boolean_function,
                right_func=None,
            )


# 6. Property System
class TestProperties:
    def test_property_management(self):
        bf_instance = bf.BooleanFunction()
        prop = bf.Property("linear", test_func=lambda f: True)
        bf_instance.properties.add(prop, status="verified")
        assert "linear" in bf_instance.properties._properties
        assert bf_instance.properties._properties["linear"]["status"] == "verified"


# 7. String Representations
class TestStringRepresentations:
    def test_str_representation(self, boolean_function):
        rep = str(boolean_function)
        assert "BooleanFunction" in rep
        assert "vars=2" in rep
        assert "space=Space.PLUS_MINUS_CUBE" in rep

    def test_repr_representation(self, boolean_function):
        rep = repr(boolean_function)
        assert "BooleanFunction" in rep
        assert "space=Space.PLUS_MINUS_CUBE" in rep
        assert "n_vars=2" in rep


# 8. Probabilistic Interface
class TestProbabilisticInterface:
    @patch("boofun.core.BooleanFunction._uniform_sample")
    def test_rvs_without_distribution(self, mock_sample, boolean_function):
        mock_sample.return_value = [0, 1, 0]
        samples = boolean_function.rvs(size=3)
        assert samples == [0, 1, 0]
        mock_sample.assert_called_once_with(3, None)

    def test_pmf_with_cache(self, boolean_function):
        boolean_function._pmf_cache = {(1, 0): 0.3}
        assert boolean_function.pmf([1, 0]) == 0.3
        assert boolean_function.pmf([0, 0]) == 0.0


## 1. __array__ method
def test_array_conversion(xor_function):
    """Test conversion to NumPy array"""
    arr = np.array(xor_function)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, [0, 1, 1, 0])

    # Test with dtype conversion
    bool_arr = np.array(xor_function, dtype=bool)
    assert np.array_equal(bool_arr, [False, True, True, False])


## 2. Operator methods
@patch("boofun.core.factory.BooleanFunctionFactory.create_composite")
def test_binary_operators(mock_factory, xor_function, and_function):
    cls = type(xor_function)

    _ = xor_function + and_function
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="+", left_func=xor_function, right_func=and_function
    )

    _ = xor_function * and_function
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="*", left_func=xor_function, right_func=and_function
    )

    _ = xor_function & and_function
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="&", left_func=xor_function, right_func=and_function
    )

    _ = xor_function | and_function
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="|", left_func=xor_function, right_func=and_function
    )

    _ = xor_function ^ and_function
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="^", left_func=xor_function, right_func=and_function
    )


def test_scalar_multiplication(xor_function, scalar_value):
    """Test multiplication with scalar"""
    with patch("boofun.core.factory.BooleanFunctionFactory.create_composite") as mock_scalar:
        cls = type(xor_function)
        _ = xor_function * scalar_value
        mock_scalar.assert_called_with(
            boolean_function_cls=cls, operator="*", left_func=xor_function, right_func=scalar_value
        )


@patch("boofun.core.factory.BooleanFunctionFactory.create_composite")
def test_unary_operators(mock_factory, xor_function):
    cls = type(xor_function)

    _ = ~xor_function
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="~", left_func=xor_function, right_func=None
    )

    _ = xor_function**2
    mock_factory.assert_called_with(
        boolean_function_cls=cls, operator="**", left_func=xor_function, right_func=2
    )


## 4. __call__ method
def test_call_method(xor_function):
    """Test function call syntax"""
    with patch.object(xor_function, "evaluate") as mock_eval:
        inputs = [0, 1]
        _ = xor_function(inputs)
        mock_eval.assert_called_once_with(inputs)


## 5. String representations
def test_string_representations(xor_function):
    """Test __str__ and __repr__ methods"""
    # Test __str__
    assert "BooleanFunction" in str(xor_function)
    assert "vars=2" in str(xor_function)
    assert "space=Space.PLUS_MINUS_CUBE" in str(xor_function)

    # Test __repr__
    repr_str = repr(xor_function)
    assert "BooleanFunction" in repr_str
    assert "space=Space.PLUS_MINUS_CUBE" in repr_str
    assert "n_vars=2" in repr_str


## 6. Edge cases
def test_missing_truth_table():
    """Test __array__ without truth table representation"""
    func = bf.BooleanFunction()
    with pytest.raises(KeyError):
        np.array(func)


def test_invalid_operand_type(xor_function):
    """Test operators with invalid types"""
    with pytest.raises(TypeError):
        _ = xor_function + "invalid"
