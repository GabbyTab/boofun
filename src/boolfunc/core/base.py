import warnings
import operator
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional
import numpy as np
from .errormodels import ExactErrorModel
from collections.abc import Iterable
from .spaces import Space
from .representations.registry import get_strategy
from .factory import BooleanFunctionFactory
from .conversion_graph import find_conversion_path

# The BooleanFunctionRepresentations and Spaces and ErrorModels are in separate files in the same directory, should I import them?

try:
    from numba import jit

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    warnings.warn("Numba not installed - using pure Python mode")


class Property:
    def __init__(self, name, test_func=None, doc=None, closed_under=None):
        self.name = name
        self.test_func = test_func
        self.doc = doc
        self.closed_under = closed_under or set()


class PropertyStore:
    def __init__(self):
        self._properties = {}

    def add(self, prop: Property, status="user"):
        self._properties[prop.name] = {"property": prop, "status": status}

    def has(self, name):
        return name in self._properties


class Evaluable(Protocol):
    def evaluate(self, inputs): ...


class Representable(Protocol):
    def to_representation(self, rep_type: str): ...


class BooleanFunction(Evaluable, Representable):
    def __new__(cls, *args, **kwargs):
        # Allocate without calling __init__
        self = super().__new__(cls)
        # Delegate actual setup to a private initializer
        self._init(*args, **kwargs)
        return self

    def _init(
        self,
        space: str = "plus_minus_cube",
        error_model: Optional[Any] = None,
        storage_manager=None,
        **kwargs,
    ):
        # Original __init__ logic moved here
        self.space = self._create_space(space)
        self.representations: Dict[str, Any] = {}
        self.properties = PropertyStore()
        self.error_model = error_model or ExactErrorModel()
        self.tracking = kwargs.get("tracking")
        self.restrictions = kwargs.get("restrictions")
        self.n_vars = kwargs.get("n") or kwargs.get("n_vars")
        self._metadata = kwargs.get("metadata", {})
        self.nickname = kwargs.get("nickname") or "x_0"

    def __array__(self, dtype=None) -> np.ndarray:
        """Return the truth table as a NumPy array for NumPy compatibility."""
        truth_table = self.get_representation("truth_table")
        return np.asarray(truth_table, dtype=dtype)

    def __add__(self, other):
        """Addition operator - creates composite function with + operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="+",
            left_func=self,
            right_func=other,
        )

    def __sub__(self, other):
        """Subtraction operator - creates composite function with - operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="-",
            left_func=self,
            right_func=other,
        )

    def __mul__(self, other):
        """Multiplication operator - creates composite function with * operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="*",
            left_func=self,
            right_func=other,
        )

    def __and__(self, other):
        """Bitwise AND operator - creates composite function with & operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="&",
            left_func=self,
            right_func=other,
        )

    def __or__(self, other):
        """Bitwise OR operator - creates composite function with | operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="|",
            left_func=self,
            right_func=other,
        )

    def __xor__(self, other):
        """Bitwise XOR operator - creates composite function with ^ operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="^",
            left_func=self,
            right_func=other,
        )

    def __invert__(self):
        """Bitwise NOT operator - creates composite function with ~ operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="~",
            left_func=self,
            right_func=None,
        )

    def __pow__(self, exponent):
        """Power operator - creates composite function with ** operation"""
        return BooleanFunctionFactory.create_composite(
            boolean_function_cls=type(self),
            operator="**",
            left_func=self,
            right_func=exponent,  # Pass exponent as right_func for consistency
        )

    def __call__(self, inputs):
        return self.evaluate(inputs)

    def __str__(self):
        return f"BooleanFunction(vars={self.n_vars}, space={self.space})"  # TODO figure out what should be outputed here

    def __repr__(self):
        return f"BooleanFunction(space={self.space}, n_vars={self.n_vars})"  # TODO figure out what should be outputed here

    def _create_space(self, space_type):
        # Handle both string and Space enum inputs
        if isinstance(space_type, Space):
            return space_type
        elif space_type == "boolean_cube":
            return Space.BOOLEAN_CUBE
        elif space_type == "plus_minus_cube":
            return Space.PLUS_MINUS_CUBE
        elif space_type == "real":
            return Space.REAL
        elif space_type == "log":
            return Space.LOG
        elif space_type == "gaussian":
            return Space.GAUSSIAN
        else:
            raise ValueError(f"Unknown space type: {space_type}")

    def _compute_representation(self, rep_type: str):
        """
        Compute representation using intelligent conversion graph.
        
        Uses Dijkstra's algorithm to find optimal conversion path from
        available representations to target representation.
        """
        if rep_type in self.representations:
            return None

        if not self.representations:
            raise KeyError("Boolean Function is Empty (no representations)")

        # Find the best source representation using conversion graph
        best_path = None
        best_source_data = None
        
        for source_rep_type, source_data in self.representations.items():
            path = find_conversion_path(source_rep_type, rep_type, self.n_vars)
            if path and (best_path is None or path.total_cost < best_path.total_cost):
                best_path = path
                best_source_data = source_data

        if best_path is None:
            # Fallback to direct conversion from first available representation
            source_rep_type = next(iter(self.representations))
            data = self.representations[source_rep_type]
            source_strategy = get_strategy(source_rep_type)
            target_strategy = get_strategy(rep_type)
            
            try:
                result = source_strategy.convert_to(
                    target_strategy, data, self.space, self.n_vars
                )
            except NotImplementedError:
                raise ValueError(
                    f"No conversion path available from {source_rep_type} to {rep_type}"
                )
        else:
            # Use optimal path from conversion graph
            result = best_path.execute(best_source_data, self.space, self.n_vars)

        self.add_representation(result, rep_type)
        return None

    def get_representation(self, rep_type: str):
        """Retrieve or compute representation"""
        self._compute_representation(rep_type)
        rep_data = self.representations[rep_type]

        return rep_data

    def add_representation(self, data, rep_type=None):
        """Add a representation to this boolean function"""
        if rep_type == None:
            factory = BooleanFunctionFactory()
            rep_type = factory._determine_rep_type(data)

        self.representations[rep_type] = data
        return self

    def evaluate(self, inputs, rep_type=None, **kwargs):
        """
        Evaluate function with automatic input type detection and representation selection.

        Args:
            inputs: Input data (array, list, or scipy random variable)
            rep_type: Optional specific representation to use
            **kwargs: Additional evaluation parameters

        Returns:
            Boolean result(s) or distribution (with error model applied)
        """
        bit_strings = False or kwargs.get("bit_strings")
        if bit_strings:
            inputs = self._compute_index(inputs)

        # Get base result
        if hasattr(inputs, "rvs"):  # scipy.stats random variable
            result = self._evaluate_stochastic(inputs, rep_type=rep_type, **kwargs)
        elif isinstance(inputs, (list, np.ndarray, int, float)):
            # Check for empty inputs (only for lists and multi-dimensional arrays)
            if isinstance(inputs, list) and len(inputs) == 0:
                raise ValueError("Cannot evaluate empty input list")
            elif isinstance(inputs, np.ndarray) and inputs.ndim > 0 and inputs.size == 0:
                raise ValueError("Cannot evaluate empty input array")
                
            # Convert single values to array for consistent processing
            is_scalar_input = isinstance(inputs, (int, float))
            if is_scalar_input:
                inputs = np.array([inputs])
            result = self._evaluate_deterministic(inputs, rep_type=rep_type)
            # Return scalar if input was scalar
            if is_scalar_input and len(result) == 1:
                result = result[0]
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)}")
        
        # Apply error model if not exact
        if hasattr(self.error_model, 'apply_error'):
            try:
                result = self.error_model.apply_error(result)
            except Exception:
                # If error model fails, use original result
                pass
        
        return result

    def _compute_index(self, bits: np.ndarray) -> int:
        """Convert boolean vector to integer index using bit packing"""
        return np.array(int(np.packbits(bits.astype(np.uint8), bitorder="little")[0]))

    def _evaluate_deterministic(self, inputs, rep_type=None):
        """
        Evaluate using the specified or first available representation.
        
        Automatically uses batch processing for large input arrays.
        """
        inputs = np.asarray(inputs)
        if rep_type == None:
            rep_type = next(iter(self.representations))

        data = self.representations[rep_type]
        
        # Use batch processing for large arrays
        if inputs.size > 100:  # Threshold for batch processing
            from .batch_processing import process_batch
            try:
                return process_batch(inputs, data, rep_type, self.space, self.n_vars)
            except Exception:
                # Fallback to standard evaluation
                pass
        
        # Standard evaluation for small inputs or fallback
        strategy = get_strategy(rep_type)
        result = strategy.evaluate(inputs, data, self.space, self.n_vars)
        return result

    def _setup_probabilistic_interface(self):
        """Configure as scipy.stats-like random variable"""
        # Add methods that make this behave like rv_discrete/rv_continuous
        # self._configure_sampling_methods()
        pass

    def _evaluate_stochastic(self, rv_inputs, n_samples=1000):
        """Handle random variable inputs using Monte Carlo"""
        pass
        samples = rv_inputs.rvs(size=n_samples)
        results = [self._evaluate_deterministic(sample) for sample in samples]
        return self._create_result_distribution(results)

    def evaluate_range(self, inputs):
        pass

    def rvs(self, size=1, rng=None):
        """Generate random samples (like scipy.stats)"""
        if "distribution" in self.representations:
            return self.representations["distribution"].rvs(size=size, random_state=rng)
        # Fallback: uniform sampling from truth table
        return self._uniform_sample(size, rng)
    
    def _uniform_sample(self, size, rng=None):
        """Generate uniform random samples from the function's domain."""
        if rng is None:
            rng = np.random.default_rng()
        
        # Generate random inputs and evaluate
        domain_size = 2 ** self.n_vars
        random_indices = rng.integers(0, domain_size, size=size)
        
        # Evaluate function at random points
        results = []
        for idx in random_indices:
            result = self.evaluate(idx)
            results.append(int(result) if isinstance(result, (bool, np.bool_)) else result)
        
        return results

    def pmf(self, x):
        """Probability mass function"""
        if hasattr(self, "_pmf_cache"):
            return self._pmf_cache.get(tuple(x), 0.0)
        return self._compute_pmf(x)
    
    def _compute_pmf(self, x):
        """Compute probability mass function for input x."""
        # For Boolean functions, PMF is just the function value
        return float(self.evaluate(x))

    def cdf(self, x):
        """Cumulative distribution function"""
        # return self._compute_cdf(x)
        pass

    # get methods
    def get_n_vars(self):
        return self.n_vars

    # get methods
    def has_rep(self, rep_type):
        if rep_type in self.representations:
            return True
        return False
    
    def get_conversion_options(self, max_cost: Optional[float] = None) -> Dict[str, Any]:
        """
        Get available conversion options from current representations.
        
        Args:
            max_cost: Maximum acceptable conversion cost
            
        Returns:
            Dictionary with conversion options and costs
        """
        from .conversion_graph import get_conversion_options
        
        if not self.representations:
            return {}
        
        all_options = {}
        for source_rep in self.representations.keys():
            options = get_conversion_options(source_rep, max_cost)
            for target, path in options.items():
                if target not in all_options or path.total_cost < all_options[target]['cost']:
                    all_options[target] = {
                        'cost': path.total_cost,
                        'path': path,
                        'source': source_rep,
                        'exact': path.total_cost.is_exact
                    }
        
        return all_options
    
    def estimate_conversion_cost(self, target_rep: str) -> Optional[Any]:
        """
        Estimate cost to convert to target representation.
        
        Args:
            target_rep: Target representation name
            
        Returns:
            Conversion cost estimate or None if impossible
        """
        from .conversion_graph import estimate_conversion_cost
        
        if target_rep in self.representations:
            return None  # Already available
        
        best_cost = None
        for source_rep in self.representations.keys():
            cost = estimate_conversion_cost(source_rep, target_rep, self.n_vars)
            if cost and (best_cost is None or cost < best_cost):
                best_cost = cost
        
        return best_cost
    
    def to(self, representation_type: str):
        """
        Convert to specified representation (convenience method).
        
        Args:
            representation_type: Target representation type
            
        Returns:
            Self (for method chaining)
        """
        self.get_representation(representation_type)
        return self
