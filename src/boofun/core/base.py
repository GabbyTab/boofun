import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

import numpy as np

from ..utils.exceptions import ConversionError, EvaluationError, InvalidInputError
from .conversion_graph import find_conversion_path
from .errormodels import ExactErrorModel
from .factory import BooleanFunctionFactory
from .representations.registry import get_strategy
from .spaces import Space

if TYPE_CHECKING:
    from .query_model import AccessType, QueryModel

# Check Numba availability (optional optimization)
try:
    import numba  # noqa: F401

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    # Only warn once at import time, not on every operation
    # Using UserWarning instead of ImportWarning to avoid pytest treating it as error
    warnings.warn(
        "Numba not installed - using pure Python mode. "
        "Install numba for 10-100x faster computations: pip install numba",
        UserWarning,
    )


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
    def __init__(
        self,
        space: str = "plus_minus_cube",
        error_model: Optional[Any] = None,
        storage_manager=None,
        **kwargs,
    ):
        self.space = self._create_space(space)
        self.representations: Dict[str, Any] = {}
        self.properties = PropertyStore()
        self.error_model = error_model or ExactErrorModel()
        self.tracking = kwargs.get("tracking")
        self.restrictions = kwargs.get("restrictions")
        # Use explicit None check to handle n=0 (constant functions) correctly
        n = kwargs.get("n")
        self.n_vars: Optional[int] = n if n is not None else kwargs.get("n_vars")
        self._metadata: Dict[str, Any] = kwargs.get("metadata", {})
        self.nickname: str = kwargs.get("nickname") or "x_0"

    @property
    def _n(self) -> int:
        """n_vars as int. Returns 0 if not set (for adapted/partial functions)."""
        return self.n_vars if self.n_vars is not None else 0

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the truth table as a NumPy array for NumPy compatibility.

        Note: The 'copy' parameter is for NumPy 2.0 compatibility.
        """
        truth_table = self.get_representation("truth_table")
        arr = np.asarray(truth_table, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr

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

    def compose(self, other: "BooleanFunction") -> "BooleanFunction":
        """Compose this function with another BooleanFunction.

        Semantics mirror the legacy ``BooleanFunc.compose``: if ``self`` depends
        on ``n`` variables and ``other`` depends on ``m`` variables, the result is
        a function on ``n * m`` variables obtained by substituting ``other`` into
        each input of ``self`` on disjoint variable blocks.
        """

        return BooleanFunctionFactory.compose_truth_tables(
            boolean_function_cls=type(self),
            outer_func=self,
            inner_func=other,
        )

    def __call__(self, inputs):
        return self.evaluate(inputs)

    def __eq__(self, other):
        """Two BooleanFunctions are equal if they have the same truth table."""
        if not isinstance(other, BooleanFunction):
            return NotImplemented
        if self.n_vars != other.n_vars:
            return False
        try:
            tt_self = list(self.get_representation("truth_table"))
            tt_other = list(other.get_representation("truth_table"))
            return tt_self == tt_other
        except Exception:
            return NotImplemented

    def __hash__(self):
        """Hash based on truth table content."""
        try:
            tt = tuple(self.get_representation("truth_table"))
            return hash((self.n_vars, tt))
        except Exception:
            return id(self)

    def __str__(self):
        n = self.n_vars or 0
        name = getattr(self, "nickname", None)
        if name and name != "x_0":
            return f"{name} (n={n})"
        return f"BooleanFunction(n={n})"

    def __repr__(self):
        n = self.n_vars or 0
        name = getattr(self, "nickname", None)
        try:
            tt = list(self.get_representation("truth_table"))
            if len(tt) <= 16:
                tt_str = str(tt)
            else:
                tt_str = f"[{tt[0]}, {tt[1]}, ..., {tt[-1]}] ({len(tt)} entries)"
        except Exception:
            tt_str = "oracle"
        label = f'"{name}" ' if name and name != "x_0" else ""
        return f"BooleanFunction({label}n={n}, tt={tt_str})"

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

        Raises:
            ConversionError: If no conversion path exists or conversion fails
        """
        if rep_type in self.representations:
            return None

        if not self.representations:
            raise ConversionError(
                "Cannot compute representation: Boolean function has no representations",
                target_repr=rep_type,
                suggestion="Add a representation first using add_representation() or create()",
            )

        # Find the best source representation using conversion graph
        best_path = None
        best_source_data = None
        available_reps = list(self.representations.keys())

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
                result = source_strategy.convert_to(target_strategy, data, self.space, self._n)
            except NotImplementedError:
                raise ConversionError(
                    f"No conversion path available from '{source_rep_type}' to '{rep_type}'",
                    source_repr=source_rep_type,
                    target_repr=rep_type,
                    context={"available_representations": available_reps},
                    suggestion=f"Available representations: {', '.join(available_reps)}",
                )
            except Exception as e:
                raise ConversionError(
                    f"Conversion from '{source_rep_type}' to '{rep_type}' failed: {e}",
                    source_repr=source_rep_type,
                    target_repr=rep_type,
                ) from e
        else:
            # Use optimal path from conversion graph
            try:
                result = best_path.execute(best_source_data, self.space, self._n)
            except Exception as e:
                raise ConversionError(
                    f"Conversion to '{rep_type}' failed during path execution: {e}",
                    target_repr=rep_type,
                ) from e

        self.add_representation(result, rep_type)
        return None

    def get_representation(self, rep_type: str):
        """Retrieve or compute representation"""
        self._compute_representation(rep_type)
        rep_data = self.representations[rep_type]

        return rep_data

    def add_representation(self, data, rep_type=None):
        """Add a representation to this boolean function"""
        if rep_type is None:
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

        Raises:
            InvalidInputError: If inputs are empty or have unsupported type
            EvaluationError: If evaluation fails
        """
        bit_strings = False or kwargs.get("bit_strings")
        if bit_strings:
            inputs = self._compute_index(inputs)

        # Get base result
        if hasattr(inputs, "rvs"):  # scipy.stats random variable
            result = self._evaluate_stochastic(inputs, rep_type=rep_type, **kwargs)
        elif isinstance(inputs, (list, tuple, np.ndarray, int, float)):
            # Convert tuple to list for consistent processing
            if isinstance(inputs, tuple):
                inputs = list(inputs)
            # Check for empty inputs (only for lists and multi-dimensional arrays)
            if isinstance(inputs, list) and len(inputs) == 0:
                raise InvalidInputError(
                    "Cannot evaluate empty input list",
                    parameter="inputs",
                    received=[],
                    expected="non-empty list of inputs",
                )
            elif isinstance(inputs, np.ndarray) and inputs.ndim > 0 and inputs.size == 0:
                raise InvalidInputError(
                    "Cannot evaluate empty input array",
                    parameter="inputs",
                    received=f"empty array with shape {inputs.shape}",
                    expected="non-empty array",
                )

            # Convert single values to array for consistent processing
            is_scalar_input = isinstance(inputs, (int, float))
            if is_scalar_input:
                inputs = np.array([inputs])
            result = self._evaluate_deterministic(inputs, rep_type=rep_type)
            # Return scalar if input was scalar
            if is_scalar_input:
                # Handle case where result is already a scalar
                if isinstance(result, (bool, np.bool_)):
                    pass  # Already scalar
                elif hasattr(result, "__len__") and len(result) == 1:
                    result = result[0]
        else:
            raise InvalidInputError(
                f"Unsupported input type for evaluation",
                parameter="inputs",
                received=type(inputs).__name__,
                expected="list, tuple, np.ndarray, int, float, or scipy random variable",
            )

        # Apply error model if not exact
        if hasattr(self.error_model, "apply_error"):
            try:
                result = self.error_model.apply_error(result)
            except Exception as e:
                # Only warn if error model was explicitly configured (not default ExactErrorModel)
                if not isinstance(self.error_model, ExactErrorModel):
                    warnings.warn(
                        f"Error model {type(self.error_model).__name__} failed: {e}. "
                        f"Using unadjusted result.",
                        UserWarning,
                    )

        return result

    def _compute_index(self, bits) -> int:
        """Convert boolean vector to integer index.

        Uses positional summation instead of np.packbits, which only
        handles 8 bits. This works correctly for any n.
        """
        bits = np.asarray(bits, dtype=int)
        return int(sum(int(b) << i for i, b in enumerate(bits)))

    def _evaluate_deterministic(self, inputs, rep_type=None):
        """
        Evaluate using the specified or first available representation.

        Automatically uses batch processing for large input arrays.
        """
        inputs = np.asarray(inputs)
        if rep_type is None:
            rep_type = next(iter(self.representations))

        data = self.representations[rep_type]

        # Use batch processing for large arrays
        if inputs.size > 100:  # Threshold for batch processing
            from .batch_processing import process_batch

            try:
                return process_batch(inputs, data, rep_type, self.space, self._n)
            except ImportError:
                # Batch processing not available, use standard path
                pass
            except Exception as e:
                # Log the fallback so users know about potential performance impact
                warnings.warn(
                    f"Batch processing failed ({type(e).__name__}: {e}), "
                    f"falling back to sequential evaluation. "
                    f"This may be slower for large inputs.",
                    UserWarning,
                )

        # Standard evaluation for small inputs or fallback
        strategy = get_strategy(rep_type)
        result = strategy.evaluate(inputs, data, self.space, self._n)
        return result

    def _setup_probabilistic_interface(self):
        """Configure as scipy.stats-like random variable"""
        # Add methods that make this behave like rv_discrete/rv_continuous
        # self._configure_sampling_methods()

    def _evaluate_stochastic(self, rv_inputs, n_samples=1000, **kwargs):
        """Handle random variable inputs using Monte Carlo."""
        samples = rv_inputs.rvs(size=n_samples)
        results = [self._evaluate_deterministic(sample) for sample in samples]
        return np.array(results)

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
        domain_size = 2**self._n
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

    # get methods
    def get_n_vars(self):
        return self.n_vars

    # Backwards-compatible property aliases
    @property
    def num_variables(self) -> int:
        """Alias for n_vars for backwards compatibility."""
        return self._n

    @property
    def num_vars(self) -> int:
        """Alias for n_vars for backwards compatibility."""
        return self._n

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

        all_options: Dict[str, Any] = {}
        for source_rep in self.representations.keys():
            options = get_conversion_options(source_rep, max_cost)
            for target, path in options.items():
                if target not in all_options or path.total_cost < all_options[target]["cost"]:
                    all_options[target] = {
                        "cost": path.total_cost,
                        "path": path,
                        "source": source_rep,
                        "exact": path.total_cost.is_exact,
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

    # =========================================================================
    # Restriction Operations (ported from legacy BooleanFunc)
    # =========================================================================

    def fix(self, var, val):
        """
        Fix variable(s) to specific value(s), returning a new function on fewer variables.

        This is a fundamental operation for Boolean function analysis, used in:
        - Decision tree computation
        - Certificate analysis
        - Influence computation via derivatives

        Args:
            var: Variable index (int) or list of variable indices
            val: Value (0 or 1) or list of values to fix variables to

        Returns:
            New BooleanFunction with fixed variables removed

        Example:
            >>> f = bf.create([0, 1, 1, 0])  # XOR on 2 vars
            >>> g = f.fix(0, 1)  # Fix x_0 = 1, get function on x_1 only
        """
        if isinstance(var, (list, tuple)):
            return self._fix_multi(list(var), val)
        else:
            return self._fix_single(var, val)

    def _fix_single(self, var: int, val: int) -> "BooleanFunction":
        """
        Fix a single variable to a specific value.

        Args:
            var: Variable index (0-indexed from MSB)
            val: Value to fix (0 or 1)

        Returns:
            New BooleanFunction with one fewer variable
        """
        if val not in (0, 1):
            raise ValueError(f"Value must be 0 or 1, got {val}")
        if var < 0 or var >= self._n:
            raise ValueError(f"Variable index {var} out of range [0, {self._n-1}]")

        n = self._n
        new_n = n - 1

        if new_n == 0:
            # Special case: fixing last variable gives constant function
            truth_table = self.get_representation("truth_table")
            result_val = bool(truth_table[val])
            new_tt = np.array([result_val], dtype=bool)
        else:
            # Build new truth table by selecting entries where var has value val
            old_tt = self.get_representation("truth_table")
            new_size = 1 << new_n
            new_tt = np.zeros(new_size, dtype=bool)

            for i in range(new_size):
                # Insert the fixed bit at position var
                # Bits to the left of var stay in place
                # Bits at and to the right of var shift right by 1
                left_mask = ((1 << (n - 1 - var)) - 1) << (var + 1)
                right_mask = (1 << var) - 1

                left_bits = (i << 1) & left_mask
                right_bits = i & right_mask
                fixed_bit = val << var

                old_idx = left_bits | fixed_bit | right_bits
                new_tt[i] = old_tt[old_idx]

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=new_n)

    def _fix_multi(self, vars: list, vals: list) -> "BooleanFunction":
        """
        Fix multiple variables to specific values.

        Args:
            vars: List of variable indices
            vals: List of values (0 or 1) to fix each variable to

        Returns:
            New BooleanFunction with len(vars) fewer variables
        """
        if len(vars) != len(vals):
            raise ValueError("vars and vals must have same length")

        # Sort by variable index descending to fix from right to left
        # This ensures indices stay valid as we remove variables
        pairs = sorted(zip(vars, vals), reverse=True)

        result = self
        for var, val in pairs:
            result = result._fix_single(var, val)

        return result

    def restrict(self, var: int, val: int) -> "BooleanFunction":
        """
        Alias for fix() - restrict a variable to a specific value.

        This terminology is often used in the literature (e.g., random restrictions).
        """
        return self.fix(var, val)

    def derivative(self, var: int) -> "BooleanFunction":
        """
        Compute the discrete derivative with respect to variable var.

        The derivative D_i f is defined as:
            D_i f(x) = f(x) XOR f(x ⊕ e_i)

        where e_i is the i-th unit vector. The derivative is 1 exactly when
        variable i is influential at input x.

        Args:
            var: Variable index to differentiate with respect to

        Returns:
            New BooleanFunction representing the derivative

        Note:
            The influence of variable i equals E[D_i f] = Pr[D_i f(x) = 1]
        """
        if var < 0 or var >= self._n:
            raise ValueError(f"Variable index {var} out of range [0, {self._n-1}]")

        n = self._n
        size = 1 << n
        old_tt = self.get_representation("truth_table")
        new_tt = np.zeros(size, dtype=bool)

        for x in range(size):
            # Flip bit at position var (LSB=x₀ convention)
            x_flipped = x ^ (1 << var)
            new_tt[x] = old_tt[x] ^ old_tt[x_flipped]

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=n)

    def shift(self, s: int) -> "BooleanFunction":
        """
        Shift the function: f_s(x) = f(x ⊕ s).

        This applies an XOR mask to all inputs, effectively translating
        the function in the Boolean cube.

        Args:
            s: Shift mask (integer representing the XOR offset)

        Returns:
            New BooleanFunction with shifted inputs
        """
        if s < 0 or s >= (1 << self._n):
            raise ValueError(f"Shift {s} out of range [0, {(1 << self._n) - 1}]")

        n = self._n
        size = 1 << n
        old_tt = self.get_representation("truth_table")
        new_tt = np.zeros(size, dtype=bool)

        for x in range(size):
            new_tt[x] = old_tt[x ^ s]

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=n)

    def negation(self) -> "BooleanFunction":
        """
        Return the negation of this function: NOT f.

        Returns:
            New BooleanFunction where all outputs are flipped
        """
        old_tt = np.asarray(self.get_representation("truth_table"), dtype=bool)
        new_tt = ~old_tt

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=self.n_vars)

    def bias(self) -> float:
        """
        Compute the bias of the function: E[(-1)^f(x)] = 1 - 2*Pr[f(x)=1].

        The bias is in [-1, 1]:
        - bias = 1 means f is constantly 0
        - bias = -1 means f is constantly 1
        - bias = 0 means f is balanced

        Returns:
            Bias value in [-1, 1]
        """
        tt = np.asarray(self.get_representation("truth_table"), dtype=bool)
        ones_count = np.sum(tt)
        total = len(tt)
        return 1.0 - 2.0 * (ones_count / total)

    def is_balanced(self) -> bool:
        """
        Check if the function is balanced (equal 0s and 1s in truth table).

        Returns:
            True if the function outputs 0 and 1 equally often
        """
        tt = np.asarray(self.get_representation("truth_table"), dtype=bool)
        ones_count = np.sum(tt)
        return ones_count == len(tt) // 2

    # =========================================================================
    # Query Model (for production safety)
    # =========================================================================

    def query_model(self, max_queries: int = 10_000_000) -> "QueryModel":
        """
        Get the query model for this function.

        The query model helps understand computational costs and prevents
        accidentally running exponential-time operations on huge functions.

        Args:
            max_queries: Maximum acceptable number of function evaluations

        Returns:
            QueryModel instance with cost estimation methods

        Example:
            >>> f = bf.create(huge_neural_net, n=100)
            >>> qm = f.query_model()
            >>> qm.can_compute("is_linear")  # True - O(k) queries
            >>> qm.can_compute("fourier")     # False - would need 2^100 queries
        """
        from .query_model import QueryModel

        return QueryModel(self, max_queries)

    def access_type(self) -> "AccessType":
        """
        Get the access type for this function.

        Returns:
            AccessType.EXPLICIT if we have the full truth table
            AccessType.QUERY if we can only evaluate on demand
            AccessType.SYMBOLIC if we have a formula
        """
        from .query_model import get_access_type

        return get_access_type(self)

    def is_explicit(self) -> bool:
        """Check if we have explicit access (full truth table)."""
        from .query_model import AccessType, get_access_type

        return get_access_type(self) == AccessType.EXPLICIT

    def is_query_access(self) -> bool:
        """Check if this is a query-access function (no truth table)."""
        from .query_model import AccessType, get_access_type

        return get_access_type(self) == AccessType.QUERY

    # =========================================================================
    # Spectral Analysis Methods (mathematician-friendly API)
    # =========================================================================

    def fourier(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute Fourier coefficients f̂(S) for all subsets S ⊆ [n].

        The Fourier expansion is: f(x) = Σ_S f̂(S) χ_S(x)
        where χ_S(x) = ∏_{i∈S} x_i are the Walsh characters.

        Returns:
            Array of Fourier coefficients indexed by subset bitmask

        Example:
            >>> xor = bf.create([0, 1, 1, 0])
            >>> coeffs = xor.fourier()
            >>> coeffs[3]  # f̂({0,1}) = -1 for XOR
        """
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self)
        return analyzer.fourier_expansion(force_recompute=force_recompute)

    def spectrum(self, force_recompute: bool = False) -> np.ndarray:
        """Alias for fourier() - returns Fourier spectrum."""
        return self.fourier(force_recompute=force_recompute)

    def degree(self, gf2: bool = False) -> int:
        """
        Compute the degree of the function.

        Args:
            gf2: If True, compute GF(2) (algebraic) degree
                 If False, compute Fourier (real) degree

        Returns:
            Maximum degree of non-zero coefficient

        Example:
            >>> xor = bf.create([0, 1, 1, 0])
            >>> xor.degree()        # Fourier degree = 2
            >>> xor.degree(gf2=True)  # GF(2) degree = 1
        """
        if gf2:
            from ..analysis.gf2 import gf2_degree

            return gf2_degree(self)
        else:
            from ..analysis.fourier import fourier_degree

            return fourier_degree(self)

    def influences(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute influences of all variables: Inf_i[f] = Pr[f(x) ≠ f(x ⊕ e_i)].

        Returns:
            Array of influences, one per variable

        Example:
            >>> maj = bf.majority(3)
            >>> maj.influences()  # All equal for symmetric function
        """
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self)
        return analyzer.influences(force_recompute=force_recompute)

    def influence(self, var: int) -> float:
        """
        Compute influence of a single variable.

        Args:
            var: Variable index (0-indexed)

        Returns:
            Influence value in [0, 1]
        """
        return self.influences()[var]

    def total_influence(self) -> float:
        """
        Compute total influence: I[f] = Σ_i Inf_i[f].

        Also called average sensitivity.

        Returns:
            Sum of all variable influences
        """
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self)
        return analyzer.total_influence()

    def noise_stability(self, rho: float) -> float:
        """
        Compute noise stability at correlation ρ.

        Stab_ρ[f] = E[f(x)f(y)] where y is ρ-correlated with x.
        In Fourier: Stab_ρ[f] = Σ_S f̂(S)² ρ^|S|

        Args:
            rho: Correlation parameter in [-1, 1]

        Returns:
            Noise stability value
        """
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self)
        return analyzer.noise_stability(rho)

    def W(self, k: int) -> float:
        """
        Compute Fourier weight at exactly degree k: W^{=k}[f] = Σ_{|S|=k} f̂(S)².

        Args:
            k: Degree level

        Returns:
            Sum of squared Fourier coefficients at degree k
        """
        coeffs = self.fourier()
        total = 0.0
        for s, c in enumerate(coeffs):
            if bin(s).count("1") == k:
                total += c * c
        return total

    def W_leq(self, k: int) -> float:
        """
        Compute Fourier weight up to degree k: W^{≤k}[f] = Σ_{|S|≤k} f̂(S)².

        This measures spectral concentration on low-degree coefficients.

        Args:
            k: Maximum degree

        Returns:
            Sum of squared Fourier coefficients up to degree k
        """
        from ..analysis import SpectralAnalyzer

        analyzer = SpectralAnalyzer(self)
        return analyzer.spectral_concentration(k)

    def sparsity(self, threshold: float = 1e-10) -> int:
        """
        Count non-zero Fourier coefficients.

        From O'Donnell: degree-k functions have at most 4^k non-zero coefficients.

        Args:
            threshold: Minimum magnitude to count as non-zero

        Returns:
            Number of significant Fourier coefficients
        """
        from ..analysis.fourier import fourier_sparsity

        return fourier_sparsity(self, threshold)

    def spectral_weight_by_degree(self) -> dict:
        """
        Compute spectral weight at each degree level.

        Returns:
            Dict mapping degree k -> W^{=k}[f] = Σ_{|S|=k} f̂(S)²

        Example:
            >>> maj = bf.majority(5)
            >>> maj.spectral_weight_by_degree()
            {0: 0.0, 1: 0.625, 3: 0.3125, 5: 0.0625}
        """
        coeffs = self.fourier()
        weights: Dict[int, float] = {}
        for s, c in enumerate(coeffs):
            deg = bin(s).count("1")
            weights[deg] = weights.get(deg, 0) + c * c
        return dict(sorted(weights.items()))

    def heavy_coefficients(self, tau: float = 0.1) -> list:
        """
        Find Fourier coefficients with |f̂(S)| ≥ τ.

        Args:
            tau: Threshold for "heavy" coefficient

        Returns:
            List of (subset_tuple, coefficient) pairs sorted by magnitude

        Example:
            >>> maj = bf.majority(3)
            >>> maj.heavy_coefficients(0.3)
            [((0,), 0.5), ((1,), 0.5), ((2,), 0.5)]
        """
        coeffs = self.fourier()
        heavy = []
        for s, c in enumerate(coeffs):
            if abs(c) >= tau:
                # Convert bitmask to tuple of variable indices
                subset = tuple(i for i in range(self._n) if (s >> i) & 1)
                heavy.append((subset, float(c)))
        return sorted(heavy, key=lambda x: -abs(x[1]))

    def variance(self) -> float:
        """
        Compute variance: Var[f] = E[f²] - E[f]² = Σ_{S≠∅} f̂(S)².

        Returns:
            Variance of the function (0 for constant, 1 for balanced ±1 function)
        """
        coeffs = self.fourier()
        return sum(c * c for c in coeffs[1:])  # Skip S=∅

    def max_influence(self) -> float:
        """
        Compute maximum influence: max_i Inf_i[f].

        Important for KKL theorem: max_i Inf_i[f] ≥ Ω(log n / n) for balanced f.

        Returns:
            Maximum influence value
        """
        return max(self.influences())

    def analyze(self) -> dict:
        """
        Quick analysis returning common metrics.

        Returns:
            Dict with: n_vars, is_balanced, variance, degree,
                       total_influence, max_influence, noise_stability_0.5

        Example:
            >>> bf.majority(5).analyze()
            {'n_vars': 5, 'is_balanced': True, 'variance': 1.0, ...}
        """
        return {
            "n_vars": self.n_vars,
            "is_balanced": self.is_balanced(),
            "variance": self.variance(),
            "degree": self.degree(),
            "total_influence": self.total_influence(),
            "max_influence": self.max_influence(),
            "noise_stability_0.5": self.noise_stability(0.5),
        }

    def negate_inputs(self) -> "BooleanFunction":
        """
        Compute g(x) = f(-x) where -x flips all bits.

        In Fourier: ĝ(S) = (-1)^|S| f̂(S) (odd-degree coefficients flip sign)

        Returns:
            New function with negated inputs
        """
        from ..analysis.fourier import negate_inputs

        return negate_inputs(self)

    def __neg__(self) -> "BooleanFunction":
        """
        Unary minus: -f returns f(-x) (input negation).

        This is the natural mathematical notation for input negation.
        For output negation (NOT), use ~f.
        """
        return self.negate_inputs()

    # =========================================================================
    # Property Testing (convenient methods)
    # =========================================================================

    def is_linear(self, num_tests: int = 100) -> bool:
        """
        Test if function is linear (affine over GF(2)).

        Uses BLR linearity test.
        """
        from ..analysis import PropertyTester

        tester = PropertyTester(self)
        return tester.blr_linearity_test(num_queries=num_tests)

    def is_monotone(self, num_tests: int = 100) -> bool:
        """
        Test if function is monotone: x ≤ y implies f(x) ≤ f(y).
        """
        from ..analysis import PropertyTester

        tester = PropertyTester(self)
        return tester.monotonicity_test(num_queries=num_tests)

    def is_junta(self, k: int) -> bool:
        """
        Test if function depends on at most k variables.
        """
        from ..analysis import PropertyTester

        tester = PropertyTester(self)
        return tester.junta_test(k)

    def is_global(self, alpha: float, max_set_size: int = 3) -> bool:
        """
        Check if f has alpha-small generalized influences (is alpha-global).

        A function is alpha-global if I_S(f) <= alpha * E[f^2] for all small
        sets S. Global functions satisfy hypercontractivity under p-biased
        measures (Keevash-Lifshitz-Long-Minzer).

        Args:
            alpha: Threshold for generalized influences
            max_set_size: Maximum |S| to check (default 3)

        Returns:
            True if all generalized influences are <= alpha * E[f^2]

        References:
            - Keevash, Lifshitz, Long & Minzer (arXiv:1906.05039)
        """
        from ..analysis.global_hypercontractivity import is_alpha_global

        result, _ = is_alpha_global(self, alpha, max_set_size)
        return result

    def is_symmetric(self, num_tests: int = 100) -> bool:
        """
        Test if function is symmetric (invariant under variable permutations).
        """
        from ..analysis import PropertyTester

        tester = PropertyTester(self)
        return tester.symmetry_test(num_queries=num_tests)

    # =========================================================================
    # Additional Analysis Methods
    # =========================================================================

    def hamming_weight(self) -> int:
        """
        Count number of 1s in truth table (outputs where f(x) = 1).

        Also called the "weight" or "on-set size" of the function.

        Returns:
            Number of inputs x where f(x) = 1

        Example:
            >>> bf.majority(3).hamming_weight()  # 4 (inputs with ≥2 ones)
            >>> bf.AND(3).hamming_weight()       # 1 (only 111)
        """
        try:
            tt = np.asarray(self.get_representation("truth_table"))
            return int(np.sum(tt))
        except Exception:
            return int(sum(self.evaluate(x) for x in range(2**self._n)))

    def support(self) -> list:
        """
        Return all inputs where f(x) = 1.

        Also called the "on-set" or "satisfying assignments".

        Returns:
            List of input indices (as integers) where f(x) = 1

        Example:
            >>> bf.AND(2).support()  # [3] (binary 11)
            >>> bf.OR(2).support()   # [1, 2, 3] (01, 10, 11)
        """
        try:
            tt = np.asarray(self.get_representation("truth_table"))
            return [int(x) for x in np.nonzero(tt)[0]]
        except Exception:
            return [x for x in range(2**self._n) if self.evaluate(x) == 1]

    def restriction(self, fixed_vars: dict) -> "BooleanFunction":
        """
        Create restriction of f by fixing some variables.

        Alias for fix() with more standard mathematical terminology.

        Args:
            fixed_vars: Dict mapping variable index -> fixed value (0 or 1)

        Returns:
            Restricted function on remaining variables

        Example:
            >>> f = bf.majority(3)
            >>> g = f.restriction({0: 1})  # Fix x₀=1, get 2-variable function
        """
        vars_list = list(fixed_vars.keys())
        vals_list = list(fixed_vars.values())
        if len(vars_list) == 1:
            return self.fix(vars_list[0], vals_list[0])
        return self.fix(vars_list, vals_list)

    def cofactor(self, var: int, val: int) -> "BooleanFunction":
        """
        Compute Shannon cofactor f|_{x_i=b}.

        The cofactor is the restriction of f with variable i fixed to val.
        Shannon expansion: f = x_i · f|_{x_i=1} + (1-x_i) · f|_{x_i=0}

        Args:
            var: Variable index to fix
            val: Value to fix it to (0 or 1)

        Returns:
            Cofactor function (one fewer variable)

        Example:
            >>> f = bf.majority(3)
            >>> f0 = f.cofactor(0, 0)  # f with x₀=0
            >>> f1 = f.cofactor(0, 1)  # f with x₀=1
        """
        return self.fix(var, val)

    def sensitivity_at(self, x: int) -> int:
        """
        Compute sensitivity of f at input x.

        s(f, x) = |{i : f(x) ≠ f(x ⊕ eᵢ)}|

        Args:
            x: Input (as integer)

        Returns:
            Number of sensitive coordinates at x
        """
        f_x = self.evaluate(x)
        count = 0
        for i in range(self._n):
            neighbor = x ^ (1 << i)  # Flip bit i
            if self.evaluate(neighbor) != f_x:
                count += 1
        return count

    def sensitivity(self) -> int:
        """
        Compute sensitivity: s(f) = max_x s(f, x).

        Maximum number of sensitive bits over all inputs.
        Huang's theorem: s(f) ≥ √deg(f)

        Returns:
            Maximum sensitivity
        """
        return max(self.sensitivity_at(x) for x in range(2**self._n))

    # =========================================================================
    # Fluent/Chainable API Methods
    # =========================================================================

    def xor(self, other: "BooleanFunction") -> "BooleanFunction":
        """
        XOR with another function (chainable).

        Equivalent to f ^ g but more readable in chains.

        Example:
            >>> f.xor(g).fourier()
            >>> f.restrict(0, 1).xor(g).influences()
        """
        return self ^ other

    def and_(self, other: "BooleanFunction") -> "BooleanFunction":
        """
        AND with another function (chainable).

        Named with underscore to avoid Python keyword conflict.
        Equivalent to f & g.
        """
        return self & other

    def or_(self, other: "BooleanFunction") -> "BooleanFunction":
        """
        OR with another function (chainable).

        Named with underscore to avoid Python keyword conflict.
        Equivalent to f | g.
        """
        return self | other

    def not_(self) -> "BooleanFunction":
        """
        Negate output (chainable).

        Equivalent to ~f. Returns function g where g(x) = NOT f(x).
        """
        return ~self

    def apply_noise(self, rho: float, samples: int = 100) -> "BooleanFunction":
        """
        Apply noise to get a new Boolean function via sampling.

        For each input x, outputs the majority vote of f(y) over multiple y,
        where each y is independently ρ-correlated with x.

        This gives a Boolean approximation to the noise operator T_ρ.

        Args:
            rho: Correlation parameter in [-1, 1]
            samples: Number of samples for majority vote (default: 100)

        Returns:
            New Boolean function representing noisy version of f

        Example:
            >>> f = bf.parity(5)
            >>> noisy_f = f.apply_noise(0.9)  # High noise correlation
            >>> # Noisy version has lower degree (high-degree parts attenuated)
        """
        if not -1 <= rho <= 1:
            raise ValueError("rho must be in [-1, 1]")

        n = self._n
        new_tt = []

        for x in range(2**n):
            # Sample multiple y's correlated with x, take majority of f(y)
            votes = 0
            for _ in range(samples):
                # Generate y: each bit equals x_i with prob (1+rho)/2
                y = 0
                for i in range(n):
                    x_bit = (x >> i) & 1
                    if np.random.random() < (1 + rho) / 2:
                        y_bit = x_bit  # Keep same
                    else:
                        y_bit = 1 - x_bit  # Flip
                    y |= y_bit << i

                if self.evaluate(y):
                    votes += 1

            # Majority vote
            new_tt.append(1 if votes > samples // 2 else 0)

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=n)

    def noise_expectation(self, rho: float) -> np.ndarray:
        """
        Compute (T_ρ f)(x) = E_y[f(y)] for all inputs x.

        This returns the real-valued expectations, not a Boolean function.
        Useful for analysis of noise stability.

        In Fourier: (T_ρ f)^(S) = ρ^|S| · f̂(S)

        Args:
            rho: Correlation parameter in [-1, 1]

        Returns:
            Array of expectations E[f(y)|x] in {-1,+1} representation

        Example:
            >>> f = bf.parity(5)
            >>> expectations = f.noise_expectation(0.9)
            >>> # All values close to 0 (noise destroys parity signal)
        """
        if not -1 <= rho <= 1:
            raise ValueError("rho must be in [-1, 1]")

        fourier = self.fourier()

        # Apply T_ρ: multiply each coefficient by ρ^|S|
        new_fourier = np.zeros_like(fourier)
        for s in range(len(fourier)):
            k = bin(s).count("1")
            new_fourier[s] = fourier[s] * (rho**k)

        # Convert back to values via inverse WHT
        from .optimizations import fast_walsh_hadamard

        return fast_walsh_hadamard(new_fourier.copy())

    def permute(self, perm: list) -> "BooleanFunction":
        """
        Permute variables according to given permutation.

        Creates g where g(x_{perm[0]}, ..., x_{perm[n-1]}) = f(x_0, ..., x_{n-1}).

        Args:
            perm: List defining the permutation, where perm[i] = j means
                  variable i in the new function corresponds to variable j in self.

        Returns:
            New function with permuted variables

        Example:
            >>> f = bf.dictator(3, 0)  # f(x) = x_0
            >>> g = f.permute([2, 0, 1])  # g(x) = x_2 (old position 0 → new position 2)
        """
        n = self._n
        if len(perm) != n:
            raise ValueError(f"Permutation must have {n} elements")
        if set(perm) != set(range(n)):
            raise ValueError("Permutation must be a valid permutation of 0..n-1")

        # Build new truth table
        old_tt = self.get_representation("truth_table")
        new_tt = [0] * (2**n)

        for x in range(2**n):
            # Apply permutation: bit i of new input → bit perm[i] of old input
            old_x = 0
            for i in range(n):
                if (x >> i) & 1:
                    old_x |= 1 << perm[i]
            new_tt[x] = old_tt[old_x]

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=n)

    def dual(self) -> "BooleanFunction":
        """
        Compute the dual function f*(x) = 1 - f(1-x) = NOT f(NOT x).

        The dual swaps the roles of AND and OR.
        For monotone functions, f* is the De Morgan dual.

        Returns:
            Dual function

        Example:
            >>> bf.AND(3).dual()  # Returns OR(3)
            >>> bf.OR(3).dual()   # Returns AND(3)
        """
        # NOT input, then f, then NOT output
        return ~(-self)  # -f = f(-x), then ~

    def extend(self, new_n: int, method: str = "dummy") -> "BooleanFunction":
        """
        Extend function to more variables.

        Args:
            new_n: New number of variables (must be >= current n)
            method: How to extend:
                    - "dummy": New variables don't affect output (default)
                    - "xor": XOR new variables with output

        Returns:
            Extended function

        Example:
            >>> f = bf.AND(2)  # f(x0, x1) = x0 AND x1
            >>> g = f.extend(4)  # g(x0,x1,x2,x3) = x0 AND x1 (x2,x3 ignored)
        """
        n = self._n
        if new_n < n:
            raise ValueError(f"new_n ({new_n}) must be >= current n ({n})")
        if new_n == n:
            return self

        old_tt = self.get_representation("truth_table")
        new_size = 2**new_n
        new_tt = [0] * new_size

        for x in range(new_size):
            # Extract lower n bits for the original function
            orig_x = x & ((1 << n) - 1)
            extra_bits = x >> n

            if method == "dummy":
                new_tt[x] = old_tt[orig_x]
            elif method == "xor":
                # XOR with parity of extra bits
                extra_parity = bin(extra_bits).count("1") % 2
                new_tt[x] = old_tt[orig_x] ^ extra_parity
            else:
                raise ValueError(f"Unknown extension method: {method}")

        return BooleanFunctionFactory.from_truth_table(type(self), new_tt, n=new_n)

    def named(self, name: str) -> "BooleanFunction":
        """
        Return same function with a descriptive name (for display/debugging).

        This is a fluent method that returns self with updated nickname.

        Example:
            >>> f = bf.majority(5).named("MAJ_5")
            >>> f.nickname
            'MAJ_5'
        """
        self.nickname = name
        return self

    def pipe(self, func, *args, **kwargs):
        """
        Apply an arbitrary function to self (for maximum fluency).

        Allows inserting custom transformations into a chain.

        Args:
            func: Function to apply, receives self as first argument
            *args, **kwargs: Additional arguments to func

        Returns:
            Result of func(self, *args, **kwargs)

        Example:
            >>> def custom_transform(f, scale):
            ...     return f.apply_noise(scale)
            >>> f.pipe(custom_transform, 0.9).fourier()
        """
        return func(self, *args, **kwargs)
