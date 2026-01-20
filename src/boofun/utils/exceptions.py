"""
Exception hierarchy for the BooFun library.

This module provides a structured exception taxonomy that enables:
- Clear differentiation between user errors and internal errors
- Machine-readable error categorization
- Consistent error handling across the library
- Actionable error messages with context

Exception Hierarchy:
    BooleanFunctionError (base)
    ├── ValidationError          - Invalid user input
    │   ├── InvalidInputError    - Bad function arguments
    │   └── InvalidRepresentationError - Unsupported representation
    ├── EvaluationError          - Function evaluation failures
    ├── ConversionError          - Representation conversion failures
    ├── ConfigurationError       - Setup/configuration errors
    ├── ResourceUnavailableError - Optional deps unavailable
    └── InvariantViolationError  - Internal library bugs
"""

from typing import Any, Dict, List, Optional, Union


class BooleanFunctionError(Exception):
    """
    Base exception for all BooFun library errors.

    All library-specific exceptions inherit from this class, allowing
    users to catch all library errors with a single except clause:

        try:
            result = bf.create(data).fourier()
        except BooleanFunctionError as e:
            logger.error(f"BooFun error: {e}")

    Attributes:
        message: Human-readable error description
        context: Dictionary with additional error context
        suggestion: Optional suggestion for how to fix the error
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.context = context or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with context and suggestion."""
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        return " | ".join(parts)


# =============================================================================
# Validation Errors - User input problems
# =============================================================================


class ValidationError(BooleanFunctionError):
    """
    Raised when user input fails validation.

    This is the parent class for all input validation errors.
    Use specific subclasses when possible for more precise error handling.

    Examples:
        - Invalid parameter values (negative n_vars, rho outside [-1,1])
        - Malformed data structures (non-power-of-2 truth tables)
        - Type mismatches
    """

    pass


class InvalidInputError(ValidationError):
    """
    Raised when function arguments are invalid.

    Examples:
        - n_vars must be positive
        - rho must be in [-1, 1]
        - Variable index out of range
    """

    def __init__(
        self,
        message: str,
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
        super().__init__(message, ctx, suggestion)


class InvalidRepresentationError(ValidationError):
    """
    Raised when requesting an unsupported or unknown representation.

    Examples:
        - Unknown representation type name
        - Representation not available for this function
        - Cannot convert between incompatible representations
    """

    def __init__(
        self,
        message: str,
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
        super().__init__(message, ctx, suggestion)


class InvalidTruthTableError(ValidationError):
    """
    Raised when a truth table has invalid structure.

    Examples:
        - Size not a power of 2
        - Contains non-boolean values
        - Empty truth table
    """

    def __init__(
        self,
        message: str,
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
        super().__init__(message, ctx, suggestion)


# =============================================================================
# Evaluation Errors - Function evaluation problems
# =============================================================================


class EvaluationError(BooleanFunctionError):
    """
    Raised when function evaluation fails.

    This occurs when the library cannot compute the value of a Boolean
    function at a given input. Common causes include:
        - Underlying callable raises an exception
        - Input index out of bounds
        - Representation data is corrupted or incomplete
    """

    def __init__(
        self,
        message: str,
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
        super().__init__(message, ctx, suggestion)


# =============================================================================
# Conversion Errors - Representation conversion problems
# =============================================================================


class ConversionError(BooleanFunctionError):
    """
    Raised when representation conversion fails.

    This occurs during get_representation() or explicit conversion
    operations when the target representation cannot be computed.

    Examples:
        - No conversion path exists
        - Conversion algorithm failed
        - Data is not convertible (e.g., non-LTF function to LTF representation)
    """

    def __init__(
        self,
        message: str,
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
        super().__init__(message, ctx, suggestion)


# =============================================================================
# Configuration Errors - Setup and configuration problems
# =============================================================================


class ConfigurationError(BooleanFunctionError):
    """
    Raised when library configuration is invalid.

    Examples:
        - Invalid error model configuration
        - Incompatible space settings
        - Invalid optimization settings
    """

    pass


# =============================================================================
# Resource Errors - External dependency problems
# =============================================================================


class ResourceUnavailableError(BooleanFunctionError):
    """
    Raised when an optional resource is unavailable.

    This exception is raised when code attempts to use a feature
    that requires an optional dependency that is not installed.

    Examples:
        - Numba not available for JIT compilation
        - CuPy not available for GPU acceleration
        - Matplotlib not available for visualization
    """

    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        install_hint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        ctx = context or {}
        if resource:
            ctx["resource"] = resource
        suggestion = install_hint or (f"Install {resource} to enable this feature" if resource else None)
        super().__init__(message, ctx, suggestion)


# =============================================================================
# Internal Errors - Library bugs (should never happen in correct code)
# =============================================================================


class InvariantViolationError(BooleanFunctionError):
    """
    Raised when an internal invariant is violated.

    This indicates a bug in the library itself, not a user error.
    If you encounter this exception, please report it as a bug.

    Examples:
        - Data structure corruption
        - Algorithm produced invalid output
        - State machine in invalid state
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        suggestion = "This is likely a bug in BooFun. Please report it at https://github.com/boofun/boofun/issues"
        super().__init__(message, context, suggestion)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
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
