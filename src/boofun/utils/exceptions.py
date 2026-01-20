"""
Exception hierarchy for the BooFun library.

This module provides a structured exception taxonomy that enables:
- Clear differentiation between user errors and internal errors
- Machine-readable error categorization via error codes
- Consistent error handling across the library
- Actionable error messages with context

Exception Hierarchy:
    BooleanFunctionError (base)
    ├── ValidationError          - Invalid user input (E1xxx)
    │   ├── InvalidInputError    - Bad function arguments (E11xx)
    │   ├── InvalidRepresentationError - Unsupported representation (E12xx)
    │   └── InvalidTruthTableError - Malformed truth table (E13xx)
    ├── EvaluationError          - Function evaluation failures (E2xxx)
    ├── ConversionError          - Representation conversion failures (E3xxx)
    ├── ConfigurationError       - Setup/configuration errors (E4xxx)
    ├── ResourceUnavailableError - Optional deps unavailable (E5xxx)
    └── InvariantViolationError  - Internal library bugs (E9xxx)

Error Code Ranges:
    E1000-E1999: Validation errors (user input problems)
    E2000-E2999: Evaluation errors (function execution problems)
    E3000-E3999: Conversion errors (representation problems)
    E4000-E4999: Configuration errors (setup problems)
    E5000-E5999: Resource errors (dependency problems)
    E9000-E9999: Internal errors (library bugs)

Usage:
    import boofun as bf

    try:
        f = bf.create([0, 1, 1])  # Invalid size
    except bf.BooleanFunctionError as e:
        print(f"Error {e.code}: {e.message}")
        if e.suggestion:
            print(f"Fix: {e.suggestion}")
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCode(Enum):
    """
    Machine-readable error codes for BooFun exceptions.

    Error codes enable programmatic error handling and logging aggregation.
    Each code maps to a specific error condition.

    Ranges:
        E1000-E1999: Validation errors
        E2000-E2999: Evaluation errors
        E3000-E3999: Conversion errors
        E4000-E4999: Configuration errors
        E5000-E5999: Resource errors
        E9000-E9999: Internal errors
    """

    # Validation errors (E1xxx)
    VALIDATION_ERROR = "E1000"
    INVALID_INPUT = "E1100"
    INVALID_PARAMETER_VALUE = "E1101"
    INVALID_PARAMETER_TYPE = "E1102"
    PARAMETER_OUT_OF_RANGE = "E1103"
    EMPTY_INPUT = "E1104"
    INVALID_REPRESENTATION = "E1200"
    UNKNOWN_REPRESENTATION = "E1201"
    REPRESENTATION_NOT_AVAILABLE = "E1202"
    INVALID_TRUTH_TABLE = "E1300"
    TRUTH_TABLE_WRONG_SIZE = "E1301"
    TRUTH_TABLE_EMPTY = "E1302"
    TRUTH_TABLE_INVALID_VALUES = "E1303"

    # Evaluation errors (E2xxx)
    EVALUATION_ERROR = "E2000"
    EVALUATION_FAILED = "E2001"
    CALLABLE_RAISED = "E2002"
    INDEX_OUT_OF_BOUNDS = "E2003"
    CORRUPTED_DATA = "E2004"

    # Conversion errors (E3xxx)
    CONVERSION_ERROR = "E3000"
    NO_CONVERSION_PATH = "E3001"
    CONVERSION_FAILED = "E3002"
    INCOMPATIBLE_REPRESENTATIONS = "E3003"
    NO_REPRESENTATIONS = "E3004"

    # Configuration errors (E4xxx)
    CONFIGURATION_ERROR = "E4000"
    INVALID_ERROR_MODEL = "E4001"
    INCOMPATIBLE_SPACE = "E4002"
    INVALID_OPTIMIZATION = "E4003"

    # Resource errors (E5xxx)
    RESOURCE_UNAVAILABLE = "E5000"
    NUMBA_UNAVAILABLE = "E5001"
    CUPY_UNAVAILABLE = "E5002"
    MATPLOTLIB_UNAVAILABLE = "E5003"
    SCIPY_UNAVAILABLE = "E5004"
    SYMPY_UNAVAILABLE = "E5005"

    # Internal errors (E9xxx)
    INTERNAL_ERROR = "E9000"
    INVARIANT_VIOLATION = "E9001"
    STATE_CORRUPTION = "E9002"
    ALGORITHM_ERROR = "E9003"


class BooleanFunctionError(Exception):
    """
    Base exception for all BooFun library errors.

    All library-specific exceptions inherit from this class, allowing
    users to catch all library errors with a single except clause.

    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (ErrorCode enum)
        context: Dictionary with additional error context
        suggestion: Optional suggestion for how to fix the error

    Raised By:
        This base class is not raised directly. Use specific subclasses.

    Example:
        >>> try:
        ...     result = bf.create(data).fourier()
        ... except bf.BooleanFunctionError as e:
        ...     logger.error(f"[{e.code.value}] {e.message}")
        ...     if e.suggestion:
        ...         logger.info(f"Suggestion: {e.suggestion}")
    """

    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.code = code or self.default_code
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with code, context and suggestion."""
        parts = [f"[{self.code.value}] {self.message}"]

        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging/serialization.

        Returns:
            Dictionary with error details suitable for JSON logging.
        """
        return {
            "error_code": self.code.value,
            "error_type": type(self).__name__,
            "message": self.message,
            "context": self.context,
            "suggestion": self.suggestion,
        }


# =============================================================================
# Validation Errors - User input problems (E1xxx)
# =============================================================================


class ValidationError(BooleanFunctionError):
    """
    Raised when user input fails validation.

    This is the parent class for all input validation errors.
    Use specific subclasses when possible for more precise error handling.

    Raised By:
        - bf.create() when data format is unrecognized
        - Analysis functions when parameters are invalid
        - Any function receiving malformed input

    Error Codes:
        E1000: Generic validation error
        E1100-E1199: Input parameter errors
        E1200-E1299: Representation errors
        E1300-E1399: Truth table errors

    Example:
        >>> try:
        ...     bf.create("invalid")
        ... except bf.ValidationError as e:
        ...     print(f"Invalid input: {e.message}")
    """

    default_code = ErrorCode.VALIDATION_ERROR


class InvalidInputError(ValidationError):
    """
    Raised when function arguments are invalid.

    Raised By:
        - bf.BooleanFunction.evaluate() with empty or wrong-type inputs
        - bf.BooleanFunction.fix() with invalid variable index or value
        - bf.BooleanFunction.noise_stability() with rho outside [-1, 1]
        - Any method receiving out-of-range parameters

    Error Codes:
        E1100: Generic invalid input
        E1101: Invalid parameter value
        E1102: Invalid parameter type
        E1103: Parameter out of range
        E1104: Empty input

    Example:
        >>> f = bf.create([0, 1, 1, 0])
        >>> f.fix(0, 5)  # Raises InvalidInputError (value must be 0 or 1)
    """

    default_code = ErrorCode.INVALID_INPUT

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        parameter: Optional[str] = None,
        received: Any = None,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        ctx = context or {}
        if parameter:
            ctx["parameter"] = parameter
        if received is not None:
            ctx["received"] = received
        if expected:
            ctx["expected"] = expected
        super().__init__(message, code, ctx, suggestion)


class InvalidRepresentationError(ValidationError):
    """
    Raised when requesting an unsupported or unknown representation.

    Raised By:
        - bf.create() with rep_type parameter set to unknown value
        - bf.BooleanFunction.get_representation() for unsupported type
        - Factory methods when representation cannot be determined

    Error Codes:
        E1200: Generic representation error
        E1201: Unknown representation type
        E1202: Representation not available

    Example:
        >>> f = bf.create([0, 1, 1, 0])
        >>> f.get_representation("unknown_type")  # Raises InvalidRepresentationError
    """

    default_code = ErrorCode.INVALID_REPRESENTATION

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        representation: Optional[str] = None,
        available: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        ctx = context or {}
        if representation:
            ctx["representation"] = representation
        if available:
            ctx["available"] = available
            if not suggestion:
                suggestion = f"Available representations: {', '.join(available)}"
        super().__init__(message, code, ctx, suggestion)


class InvalidTruthTableError(ValidationError):
    """
    Raised when a truth table has invalid structure.

    Raised By:
        - bf.create() with list/array that is not power of 2
        - bf.create() with empty list
        - Factory.from_truth_table() with malformed data

    Error Codes:
        E1300: Generic truth table error
        E1301: Wrong size (not power of 2)
        E1302: Empty truth table
        E1303: Invalid values (not boolean-convertible)

    Example:
        >>> bf.create([0, 1, 1])  # Raises InvalidTruthTableError (size=3, not power of 2)
    """

    default_code = ErrorCode.INVALID_TRUTH_TABLE

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        size: Optional[int] = None,
        expected_size: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        ctx = context or {}
        if size is not None:
            ctx["size"] = size
        if expected_size is not None:
            ctx["expected_size"] = expected_size
        super().__init__(message, code, ctx, suggestion)


# =============================================================================
# Evaluation Errors - Function evaluation problems (E2xxx)
# =============================================================================


class EvaluationError(BooleanFunctionError):
    """
    Raised when function evaluation fails.

    Raised By:
        - bf.BooleanFunction.evaluate() when underlying callable fails
        - TruthTableRepresentation.convert_from() during truth table generation
        - Any operation that evaluates the function on inputs

    Error Codes:
        E2000: Generic evaluation error
        E2001: Evaluation failed
        E2002: Underlying callable raised exception
        E2003: Index out of bounds
        E2004: Corrupted representation data

    Example:
        >>> def bad_func(x):
        ...     raise ValueError("oops")
        >>> f = bf.create(bad_func, n=2)
        >>> f.get_representation("truth_table")  # Raises EvaluationError
    """

    default_code = ErrorCode.EVALUATION_ERROR

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        input_value: Any = None,
        representation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        ctx = context or {}
        if input_value is not None:
            ctx["input"] = input_value
        if representation:
            ctx["representation"] = representation
        super().__init__(message, code, ctx, suggestion)


# =============================================================================
# Conversion Errors - Representation conversion problems (E3xxx)
# =============================================================================


class ConversionError(BooleanFunctionError):
    """
    Raised when representation conversion fails.

    Raised By:
        - bf.BooleanFunction.get_representation() when no path exists
        - bf.BooleanFunction._compute_representation() on conversion failure
        - Representation strategies during convert_to/convert_from

    Error Codes:
        E3000: Generic conversion error
        E3001: No conversion path exists
        E3002: Conversion algorithm failed
        E3003: Incompatible representations
        E3004: No representations available (empty function)

    Example:
        >>> f = bf.BooleanFunction(n=2)  # No representations
        >>> f.get_representation("fourier")  # Raises ConversionError
    """

    default_code = ErrorCode.CONVERSION_ERROR

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        source_repr: Optional[str] = None,
        target_repr: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        ctx = context or {}
        if source_repr:
            ctx["source"] = source_repr
        if target_repr:
            ctx["target"] = target_repr
        super().__init__(message, code, ctx, suggestion)


# =============================================================================
# Configuration Errors - Setup and configuration problems (E4xxx)
# =============================================================================


class ConfigurationError(BooleanFunctionError):
    """
    Raised when library configuration is invalid.

    Raised By:
        - Error model initialization with invalid parameters
        - Space configuration conflicts
        - Optimization settings that are incompatible

    Error Codes:
        E4000: Generic configuration error
        E4001: Invalid error model
        E4002: Incompatible space settings
        E4003: Invalid optimization settings

    Example:
        >>> from boofun import PACErrorModel
        >>> PACErrorModel(epsilon=2.0)  # Raises ConfigurationError (epsilon must be in (0,1))
    """

    default_code = ErrorCode.CONFIGURATION_ERROR


# =============================================================================
# Resource Errors - External dependency problems (E5xxx)
# =============================================================================


class ResourceUnavailableError(BooleanFunctionError):
    """
    Raised when an optional resource is unavailable.

    Raised By:
        - GPU acceleration code when CuPy is not installed
        - JIT compilation when Numba is not installed
        - Visualization when Matplotlib is not installed
        - Any feature requiring optional dependencies

    Error Codes:
        E5000: Generic resource unavailable
        E5001: Numba unavailable
        E5002: CuPy unavailable
        E5003: Matplotlib unavailable
        E5004: SciPy unavailable
        E5005: SymPy unavailable

    Example:
        >>> # When CuPy is not installed:
        >>> f.to_gpu()  # Raises ResourceUnavailableError
    """

    default_code = ErrorCode.RESOURCE_UNAVAILABLE

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        resource: Optional[str] = None,
        install_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        ctx = context or {}
        if resource:
            ctx["resource"] = resource
        suggestion = install_hint or (
            f"Install {resource} to enable this feature" if resource else None
        )
        super().__init__(message, code, ctx, suggestion)


# =============================================================================
# Internal Errors - Library bugs (E9xxx)
# =============================================================================


class InvariantViolationError(BooleanFunctionError):
    """
    Raised when an internal invariant is violated.

    This indicates a bug in the library itself, not a user error.
    If you encounter this exception, please report it as a bug.

    Raised By:
        - Internal consistency checks that fail
        - Algorithm outputs that violate postconditions
        - State machine transitions to invalid states

    Error Codes:
        E9000: Generic internal error
        E9001: Invariant violation
        E9002: State corruption
        E9003: Algorithm error

    Example:
        If you see this error, please report it at:
        https://github.com/boofun/boofun/issues
    """

    default_code = ErrorCode.INVARIANT_VIOLATION

    def __init__(
        self,
        message: str,
        code: Optional[ErrorCode] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        suggestion = "This is likely a bug in BooFun. Please report it at https://github.com/boofun/boofun/issues"
        super().__init__(message, code, context, suggestion)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Error codes
    "ErrorCode",
    # Base
    "BooleanFunctionError",
    # Validation
    "ValidationError",
    "InvalidInputError",
    "InvalidRepresentationError",
    "InvalidTruthTableError",
    # Evaluation
    "EvaluationError",
    # Conversion
    "ConversionError",
    # Configuration
    "ConfigurationError",
    # Resources
    "ResourceUnavailableError",
    # Internal
    "InvariantViolationError",
]
