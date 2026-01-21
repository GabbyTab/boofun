import numbers
from collections.abc import Iterable
from typing import Any, Type

import numpy as np

from ..utils.exceptions import InvalidRepresentationError, InvalidTruthTableError

# Import DNF/CNF types for type checking
try:
    from .representations.cnf_form import CNFFormula
    from .representations.dnf_form import DNFFormula

    _DNF_CNF_AVAILABLE = True
except ImportError:
    _DNF_CNF_AVAILABLE = False
    DNFFormula = None
    CNFFormula = None


# Supported representation types for better error messages
_SUPPORTED_REP_TYPES = [
    "truth_table",
    "packed_truth_table",
    "sparse_truth_table",
    "adaptive_truth_table",
    "function",
    "distribution",
    "polynomial",
    "fourier_expansion",
    "fourier",
    "symbolic",
    "dnf",
    "cnf",
    "file",
]


class BooleanFunctionFactory:
    """Factory for creating BooleanFunction instances from various representations."""

    @classmethod
    def _determine_rep_type(cls, data: Any) -> str:
        """
        Determine the representation type based on data type.

        Args:
            data: Input data to analyze

        Returns:
            String representation type name

        Raises:
            InvalidRepresentationError: If data type cannot be mapped to a representation
        """
        # Check for file path (string ending with known extensions or Path object)
        from pathlib import Path

        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.suffix.lower() in (".json", ".bf", ".cnf", ".dimacs"):
                return "file"
            # If it looks like a file path but doesn't have known extension, check if it exists
            if path.exists() and path.is_file():
                return "file"

        if callable(data):
            return "function"
        if hasattr(data, "rvs"):
            return "distribution"
        # Check for DNF/CNF formulas
        if _DNF_CNF_AVAILABLE:
            if isinstance(data, DNFFormula):
                return "dnf"
            if isinstance(data, CNFFormula):
                return "cnf"
        if isinstance(data, np.ndarray):
            if data.dtype == bool or np.issubdtype(data.dtype, np.bool_):
                return "truth_table"
            if np.issubdtype(data.dtype, np.floating):
                return "fourier_expansion"
            # For integer arrays, assume truth table (most common case)
            # Polynomial coefficients should be passed as dict
            return "truth_table"
        if isinstance(data, list):
            if len(data) == 0:
                raise InvalidTruthTableError(
                    "Cannot create Boolean function from empty list",
                    size=0,
                    suggestion="Provide a non-empty truth table or other representation",
                )
            return cls._determine_rep_type(np.array(data))
        if isinstance(data, dict):
            return "polynomial"
        if isinstance(data, str):
            return "symbolic"
        if isinstance(data, set):
            return "invariant_truth_table"
        if isinstance(data, Iterable):
            return "iterable_rep"

        raise InvalidRepresentationError(
            f"Cannot determine representation type for data of type '{type(data).__name__}'",
            representation=type(data).__name__,
            available=_SUPPORTED_REP_TYPES,
            suggestion=(
                "Expected: list, np.ndarray (truth table), callable (function), "
                "dict (polynomial), str (symbolic expression), or DNF/CNF formula"
            ),
        )

    @classmethod
    def create(cls, boolean_function_cls: Type, data: Any = None, **kwargs):
        """
        Main factory method that dispatches to specialized creators based on input data type.

        Args:
            boolean_function_cls: The BooleanFunction class to instantiate
            data: Input data in one of the supported formats
            **kwargs: Additional arguments (n_vars, space, rep_type, etc.)

        Returns:
            BooleanFunction instance

        Raises:
            InvalidRepresentationError: If data type is not supported
            InvalidTruthTableError: If truth table has invalid structure

        Example:
            >>> bf.create([0, 1, 1, 0])  # From truth table
            >>> bf.create(lambda x: x[0] ^ x[1], n=2)  # From function
        """
        if data is None:
            return boolean_function_cls(**kwargs)

        # Determine representation type and dispatch accordingly
        rep_type = kwargs.get("rep_type")
        if rep_type is None:
            rep_type = cls._determine_rep_type(data)

        if rep_type == "function":
            return cls.from_function(boolean_function_cls, data, **kwargs)
        elif rep_type == "distribution":
            return cls.from_scipy_distribution(boolean_function_cls, data, **kwargs)
        elif rep_type == "truth_table":
            return cls.from_truth_table(boolean_function_cls, data, **kwargs)
        elif rep_type in ("packed_truth_table", "sparse_truth_table", "adaptive_truth_table"):
            # These are specialized truth table representations
            # Remove rep_type from kwargs if present to avoid duplicate
            kwargs_clean = {k: v for k, v in kwargs.items() if k != "rep_type"}
            return cls.from_truth_table(boolean_function_cls, data, rep_type=rep_type, **kwargs_clean)
        elif rep_type == "invariant_truth_table":
            return cls.from_input_invariant_truth_table(boolean_function_cls, data, **kwargs)
        elif rep_type == "polynomial":
            return cls.from_polynomial(boolean_function_cls, data, **kwargs)
        elif rep_type == "fourier_expansion" or rep_type == "fourier":
            return cls.from_multilinear(boolean_function_cls, data, **kwargs)
        elif rep_type == "symbolic":
            return cls.from_symbolic(boolean_function_cls, data, **kwargs)
        elif rep_type == "iterable_rep":
            return cls.from_iterable(boolean_function_cls, data, **kwargs)
        elif rep_type == "dnf":
            return cls.from_dnf(boolean_function_cls, data, **kwargs)
        elif rep_type == "cnf":
            return cls.from_cnf(boolean_function_cls, data, **kwargs)
        elif rep_type == "file":
            return cls.from_file(boolean_function_cls, data, **kwargs)

        raise InvalidRepresentationError(
            f"Unknown representation type '{rep_type}'",
            representation=rep_type,
            available=_SUPPORTED_REP_TYPES,
        )

    @classmethod
    def from_truth_table(cls, boolean_function_cls, truth_table, rep_type="truth_table", **kwargs):
        """
        Create from truth table data.

        Args:
            boolean_function_cls: The BooleanFunction class to instantiate
            truth_table: List or array of boolean values (length must be power of 2)
            rep_type: Representation type name (default "truth_table")
            **kwargs: Additional arguments

        Returns:
            BooleanFunction instance

        Raises:
            InvalidTruthTableError: If truth table size is not a power of 2
        """
        # Convert to array for validation
        tt_array = np.asarray(truth_table)
        size = len(tt_array)

        # Validate size is power of 2
        if size == 0:
            raise InvalidTruthTableError(
                "Truth table cannot be empty",
                size=0,
                suggestion="Provide at least one value for a constant function",
            )

        if size & (size - 1) != 0:
            # Not a power of 2
            # Find the closest powers of 2
            import math

            lower_power = 2 ** int(math.floor(math.log2(size)))
            upper_power = 2 ** int(math.ceil(math.log2(size + 1)))
            raise InvalidTruthTableError(
                f"Truth table size must be a power of 2, got {size}",
                size=size,
                expected_size=f"{lower_power} or {upper_power}",
                suggestion=(
                    f"For {int(math.log2(lower_power))}-variable function, "
                    f"use {lower_power} entries. "
                    f"For {int(math.log2(upper_power))}-variable function, "
                    f"use {upper_power} entries."
                ),
            )

        n_vars = kwargs.get("n")
        if n_vars is None:
            n_vars = int(np.log2(size))
            kwargs["n"] = n_vars

        instance = boolean_function_cls(**kwargs)
        instance.add_representation(truth_table, rep_type)
        return instance

    @classmethod
    def from_function(
        cls, boolean_function_cls, func, rep_type="function", domain_size=None, **kwargs
    ):
        """Create from callable function"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(func, rep_type)
        if domain_size:
            instance.n_vars = int(np.log2(domain_size))
        return instance

    @classmethod
    def from_scipy_distribution(
        cls, boolean_function_cls, distribution, rep_type="distribution", **kwargs
    ):
        """Create from scipy.stats distribution"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(distribution, rep_type)
        instance._setup_probabilistic_interface()
        return instance

    @classmethod
    def from_polynomial(cls, boolean_function_cls, coeffs, rep_type="polynomial", **kwargs):
        """Create from polynomial coefficients"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(coeffs, rep_type)
        return instance

    @classmethod
    def from_multilinear(cls, boolean_function_cls, coeffs, rep_type="fourier_expansion", **kwargs):
        """Create from multilinear polynomial coefficients"""
        n_vars = kwargs.get("n")
        if n_vars is None:
            n_vars = int(np.log2(len(coeffs)))
            kwargs["n"] = n_vars

        instance = boolean_function_cls(**kwargs)
        instance.add_representation(coeffs, rep_type)
        return instance

    @classmethod
    def from_iterable(cls, boolean_function_cls, data, rep_type="iterable_rep", **kwargs):
        """Create from streaming truth table"""
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(list(data), rep_type)
        return instance

    @classmethod
    def from_symbolic(cls, boolean_function_cls, expression, rep_type="symbolic", **kwargs):
        """Create from symbolic expression string"""
        instance = boolean_function_cls(**kwargs)
        variables = kwargs.get("variables")
        # if variables is None:
        #    variables = [f'x{i}' for i in range(instance.n_vars)]
        # kwargs.get('variables', [f'x{i}' for i in range(instance.n_vars)])
        instance.add_representation((expression, variables), rep_type)
        return instance

    @classmethod
    def from_input_invariant_truth_table(
        cls, boolean_function_cls, true_inputs, rep_type="truth_table", **kwargs
    ):
        """Create from set of true input vectors"""
        n_vars = len(next(iter(true_inputs))) if true_inputs else kwargs.get("n_vars", 0)
        size = 1 << n_vars
        truth_table = np.zeros(size, dtype=bool)

        for i in range(size):
            vec = tuple(int(b) for b in np.binary_repr(i, width=n_vars))
            truth_table[i] = vec in true_inputs

        kwargs["n_vars"] = n_vars
        instance = boolean_function_cls(**kwargs)
        instance.add_representation(truth_table, rep_type)
        return instance

    @classmethod
    def create_composite(
        cls,
        boolean_function_cls,
        operator,
        left_func,
        right_func,
        rep_type="symbolic",
        **kwargs,
    ):
        """Create composite function from BooleanFunctions or scalars.

        Prefers truth-table composition when both operands are BooleanFunction
        instances on the same domain/space, while still recording the symbolic
        expression for readability. Falls back to symbolic-only composition when
        truth tables are unavailable (e.g., mixing scalars or mismatched domains).
        """

        kwargs = kwargs.copy()

        def _is_boolean_function(obj):
            return hasattr(obj, "get_representation") and hasattr(obj, "n_vars")

        left_is_func = _is_boolean_function(left_func)
        right_is_func = _is_boolean_function(right_func) if right_func is not None else False

        if left_is_func and "space" not in kwargs and getattr(left_func, "space", None) is not None:
            kwargs["space"] = left_func.space
        if (
            right_is_func
            and "space" not in kwargs
            and getattr(right_func, "space", None) is not None
        ):
            kwargs["space"] = right_func.space

        variables = []
        if isinstance(left_func, numbers.Number):
            left_sym = str(left_func)
            left_n_vars = 0
        else:
            left_sym = "x0"
            if left_is_func:
                variables.append(left_func)
                left_n_vars = left_func.get_n_vars() or 0
            else:
                left_n_vars = 0

        if right_func is None:
            right_sym = ""
            right_n_vars = 0
        elif isinstance(right_func, numbers.Number):
            right_sym = str(right_func)
            right_n_vars = 0
        elif right_is_func:
            right_sym = f"x{len(variables)}"
            variables.append(right_func)
            right_n_vars = right_func.get_n_vars() or 0
        else:
            raise TypeError(
                f"Invalid operand type: {type(right_func)}. Expected BooleanFunction or number."
            )

        same_domain = (
            left_is_func
            and right_is_func
            and left_func.n_vars is not None
            and right_func.n_vars is not None
            and left_func.n_vars == right_func.n_vars
        )

        same_space = same_domain and getattr(left_func, "space", None) == getattr(
            right_func, "space", None
        )

        if right_func is None:
            result_n_vars = left_n_vars
        elif same_domain:
            result_n_vars = left_func.n_vars
        else:
            result_n_vars = left_n_vars + right_n_vars

        def _truth_table_unary(op, func):
            if not left_is_func:
                return None
            try:
                table = np.asarray(func.get_representation("truth_table"), dtype=bool)
            except Exception:
                return None
            if op == "~":
                return np.logical_not(table)
            return None

        def _truth_table_binary(op, left, right):
            if not (same_domain and same_space):
                return None
            try:
                left_tt = np.asarray(left.get_representation("truth_table"), dtype=bool)
                right_tt = np.asarray(right.get_representation("truth_table"), dtype=bool)
            except Exception:
                return None
            if left_tt.shape != right_tt.shape:
                return None
            if op in {"+", "-", "^"}:
                return np.logical_xor(left_tt, right_tt)
            if op in {"*", "&"}:
                return np.logical_and(left_tt, right_tt)
            if op == "|":
                return np.logical_or(left_tt, right_tt)
            return None

        if operator == "~":
            truth_table = _truth_table_unary(operator, left_func)
        else:
            truth_table = _truth_table_binary(operator, left_func, right_func)

        if truth_table is not None:
            if result_n_vars is None:
                raise ValueError("Cannot infer number of variables for composite function")
            kwargs.setdefault("n_vars", result_n_vars)
            if (
                left_is_func
                and "space" not in kwargs
                and getattr(left_func, "space", None) is not None
            ):
                kwargs["space"] = left_func.space

            instance = boolean_function_cls(**kwargs)
            instance.add_representation(truth_table.astype(bool), "truth_table")

            if right_func is None:
                expression = f"not {left_sym}" if operator == "~" else f"{operator}{left_sym}"
            else:
                expression = f"({left_sym} {operator} {right_sym})"

            if variables:
                instance.add_representation((expression, variables), rep_type)
            return instance

        # Fallback to symbolic composition
        if result_n_vars is not None:
            kwargs["n_vars"] = result_n_vars

        if right_func is None:
            if operator == "~":
                expression = f"not {left_sym}"
            else:
                expression = f"{operator}{left_sym}"
        else:
            expression = f"({left_sym} {operator} {right_sym})"

        instance = boolean_function_cls(**kwargs)
        instance.add_representation((expression, variables), rep_type)
        return instance

    @classmethod
    def compose_truth_tables(
        cls,
        boolean_function_cls,
        outer_func,
        inner_func,
        rep_type: str = "truth_table",
        **kwargs,
    ):
        """Compose two BooleanFunction instances via truth tables.

        Mirrors the legacy ``BooleanFunc.compose`` semantics: if ``outer`` has
        ``n`` variables and ``inner`` has ``m`` variables, the result is a
        function on ``n * m`` variables obtained by feeding disjoint copies of
        ``inner`` into each input of ``outer``.
        """

        if not (
            hasattr(outer_func, "get_representation") and hasattr(inner_func, "get_representation")
        ):
            raise TypeError("Both operands must be BooleanFunction instances")

        if outer_func.n_vars is None or inner_func.n_vars is None:
            raise ValueError("Both functions must have defined n_vars for composition")

        if getattr(outer_func, "space", None) != getattr(inner_func, "space", None):
            raise ValueError("Composition requires both functions to share the same space")

        outer_tt = np.asarray(outer_func.get_representation("truth_table"), dtype=bool)
        inner_tt = np.asarray(inner_func.get_representation("truth_table"), dtype=bool)

        outer_n = outer_func.n_vars
        inner_n = inner_func.n_vars
        total_n = outer_n * inner_n
        size = 1 << total_n

        result = np.zeros(size, dtype=bool)
        mask = (1 << inner_n) - 1

        for idx in range(size):
            outer_index = 0
            for j in range(outer_n):
                block = (idx >> (inner_n * j)) & mask
                bit = int(inner_tt[block])
                outer_index |= bit << (outer_n - 1 - j)
            result[idx] = bool(outer_tt[outer_index])

        kwargs.setdefault("n_vars", total_n)
        kwargs.setdefault("space", outer_func.space)

        instance = boolean_function_cls(**kwargs)
        instance.add_representation(result, rep_type)
        return instance

    @classmethod
    def from_dnf(cls, boolean_function_cls, dnf_formula, rep_type="dnf", **kwargs):
        """Create from DNF (Disjunctive Normal Form) formula.

        Args:
            boolean_function_cls: The BooleanFunction class to instantiate
            dnf_formula: A DNFFormula object
            rep_type: Representation type (default "dnf")
            **kwargs: Additional arguments (n_vars, space, etc.)

        Returns:
            BooleanFunction instance
        """
        if not _DNF_CNF_AVAILABLE:
            raise ImportError("DNF representation not available")

        n_vars = kwargs.get("n_vars") or kwargs.get("n") or dnf_formula.n_vars
        kwargs["n_vars"] = n_vars
        kwargs.pop("n", None)  # Remove 'n' if present to avoid conflict

        instance = boolean_function_cls(**kwargs)
        instance.add_representation(dnf_formula, rep_type)
        return instance

    @classmethod
    def from_cnf(cls, boolean_function_cls, cnf_formula, rep_type="cnf", **kwargs):
        """Create from CNF (Conjunctive Normal Form) formula.

        Args:
            boolean_function_cls: The BooleanFunction class to instantiate
            cnf_formula: A CNFFormula object
            rep_type: Representation type (default "cnf")
            **kwargs: Additional arguments (n_vars, space, etc.)

        Returns:
            BooleanFunction instance
        """
        if not _DNF_CNF_AVAILABLE:
            raise ImportError("CNF representation not available")

        n_vars = kwargs.get("n_vars") or kwargs.get("n") or cnf_formula.n_vars
        kwargs["n_vars"] = n_vars
        kwargs.pop("n", None)  # Remove 'n' if present to avoid conflict

        instance = boolean_function_cls(**kwargs)
        instance.add_representation(cnf_formula, rep_type)
        return instance

    @classmethod
    def from_file(cls, boolean_function_cls, path, **kwargs):
        """Create from file (JSON, .bf, or DIMACS CNF).

        Args:
            boolean_function_cls: The BooleanFunction class to instantiate
            path: Path to file (str or Path object)
            **kwargs: Additional arguments passed to the loader

        Returns:
            BooleanFunction instance

        Example:
            >>> bf.create("function.json")
            >>> bf.create("function.bf")
            >>> bf.create("function.cnf")
        """
        from .io import load

        return load(path, **kwargs)
