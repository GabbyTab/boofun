"""
Tests for the improved error handling and exception hierarchy.

This module tests:
1. Exception taxonomy is correct and exceptions can be caught hierarchically
2. Error messages contain appropriate context
3. Error codes are correct and machine-readable
4. Fail-fast behavior works correctly
5. Silent failures are properly eliminated
"""

import numpy as np
import pytest

import boofun as bf
from boofun.utils.exceptions import (
    BooleanFunctionError,
    ConfigurationError,
    ConversionError,
    ErrorCode,
    EvaluationError,
    InvalidInputError,
    InvalidRepresentationError,
    InvalidTruthTableError,
    InvariantViolationError,
    ResourceUnavailableError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions should inherit from BooleanFunctionError."""
        exceptions = [
            ValidationError,
            InvalidInputError,
            InvalidRepresentationError,
            InvalidTruthTableError,
            EvaluationError,
            ConversionError,
            ConfigurationError,
            ResourceUnavailableError,
            InvariantViolationError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, BooleanFunctionError)

    def test_validation_errors_inherit_from_validation_error(self):
        """Validation-related exceptions should inherit from ValidationError."""
        validation_exceptions = [
            InvalidInputError,
            InvalidRepresentationError,
            InvalidTruthTableError,
        ]
        for exc_class in validation_exceptions:
            assert issubclass(exc_class, ValidationError)

    def test_catch_all_library_errors(self):
        """Should be able to catch all library errors with base exception."""
        with pytest.raises(BooleanFunctionError):
            raise InvalidInputError("test")

        with pytest.raises(BooleanFunctionError):
            raise ConversionError("test")

        with pytest.raises(BooleanFunctionError):
            raise EvaluationError("test")


class TestExceptionContext:
    """Test that exceptions contain appropriate context."""

    def test_invalid_input_error_has_context(self):
        """InvalidInputError should include parameter details."""
        exc = InvalidInputError(
            "Value out of range",
            parameter="rho",
            received=2.5,
            expected="value in [-1, 1]",
        )
        assert "rho" in str(exc)
        assert "2.5" in str(exc)

    def test_invalid_truth_table_error_has_size_info(self):
        """InvalidTruthTableError should include size information."""
        exc = InvalidTruthTableError(
            "Invalid size",
            size=5,
            expected_size=4,
        )
        assert "5" in str(exc)
        assert "4" in str(exc)

    def test_conversion_error_has_repr_info(self):
        """ConversionError should include source and target info."""
        exc = ConversionError(
            "Cannot convert",
            source_repr="truth_table",
            target_repr="bdd",
        )
        assert "truth_table" in str(exc)
        assert "bdd" in str(exc)

    def test_resource_unavailable_has_install_hint(self):
        """ResourceUnavailableError should include install hint."""
        exc = ResourceUnavailableError(
            "GPU not available",
            resource="cupy",
            install_hint="pip install cupy",
        )
        assert "cupy" in str(exc)
        assert "pip install" in str(exc)

    def test_exception_suggestion_is_included(self):
        """Exceptions with suggestions should include them in message."""
        exc = BooleanFunctionError(
            "Something went wrong",
            suggestion="Try doing X instead",
        )
        assert "Try doing X instead" in str(exc)


class TestTruthTableValidation:
    """Test truth table validation catches errors early."""

    def test_empty_truth_table_raises(self):
        """Empty truth table should raise InvalidTruthTableError."""
        with pytest.raises(InvalidTruthTableError) as exc_info:
            bf.create([])
        assert "empty" in str(exc_info.value).lower()

    def test_non_power_of_two_raises(self):
        """Non-power-of-2 truth table should raise InvalidTruthTableError."""
        with pytest.raises(InvalidTruthTableError) as exc_info:
            bf.create([0, 1, 1])  # 3 elements, not power of 2
        assert "power of 2" in str(exc_info.value).lower()
        # Should suggest correct sizes
        assert "2" in str(exc_info.value) or "4" in str(exc_info.value)

    def test_valid_truth_table_works(self):
        """Valid truth tables should work without errors."""
        f = bf.create([0, 1, 1, 0])  # XOR - 4 elements = 2^2
        assert f.n_vars == 2

        g = bf.create([0] * 8)  # Constant 0 - 8 elements = 2^3
        assert g.n_vars == 3


class TestEvaluationErrors:
    """Test evaluation error handling."""

    def test_empty_input_list_raises(self):
        """Empty input list should raise InvalidInputError."""
        f = bf.create([0, 1, 1, 0])
        with pytest.raises(InvalidInputError) as exc_info:
            f.evaluate([])
        assert "empty" in str(exc_info.value).lower()

    def test_empty_input_array_raises(self):
        """Empty numpy array should raise InvalidInputError."""
        f = bf.create([0, 1, 1, 0])
        with pytest.raises(InvalidInputError) as exc_info:
            f.evaluate(np.array([]))
        assert "empty" in str(exc_info.value).lower()

    def test_unsupported_input_type_raises(self):
        """Unsupported input type should raise InvalidInputError."""
        f = bf.create([0, 1, 1, 0])
        with pytest.raises(InvalidInputError) as exc_info:
            f.evaluate("not a valid input")  # type: ignore
        assert "Unsupported" in str(exc_info.value)


class TestConversionErrors:
    """Test representation conversion error handling."""

    def test_conversion_error_on_empty_function(self):
        """Converting from empty function should raise ConversionError."""
        f = bf.BooleanFunction(n=2)  # No representations added
        with pytest.raises(ConversionError) as exc_info:
            f.get_representation("fourier_expansion")
        assert "no representations" in str(exc_info.value).lower()


class TestTruthTableConversionStrictMode:
    """Test truth table conversion strict vs lenient modes."""

    def test_strict_mode_raises_on_failure(self):
        """Strict mode (default) should raise on evaluation failure."""
        # Create a function that will fail during evaluation
        def failing_func(x):
            if x == 2:
                raise ValueError("Intentional failure")
            return x % 2

        # This should fail during truth table conversion
        f = bf.BooleanFunction(n=2)
        f.add_representation(failing_func, "function")

        # Getting truth table should fail in strict mode
        from boofun.core.representations.truth_table import TruthTableRepresentation
        from boofun.core.representations.registry import get_strategy

        tt_strategy = get_strategy("truth_table")
        func_strategy = get_strategy("function")

        with pytest.raises(EvaluationError):
            tt_strategy.convert_from(
                func_strategy,
                failing_func,
                bf.Space.BOOLEAN_CUBE,
                2,
                lenient=False,
            )

    def test_lenient_mode_substitutes_false(self):
        """Lenient mode should substitute False and warn on failure."""
        def failing_func(x):
            if x == 2:
                raise ValueError("Intentional failure")
            return x % 2

        from boofun.core.representations.truth_table import TruthTableRepresentation
        from boofun.core.representations.registry import get_strategy

        tt_strategy = get_strategy("truth_table")
        func_strategy = get_strategy("function")

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = tt_strategy.convert_from(
                func_strategy,
                failing_func,
                bf.Space.BOOLEAN_CUBE,
                2,
                lenient=True,
            )
            # Should have emitted a warning
            assert len(w) >= 1
            assert "failed" in str(w[-1].message).lower()

        # Result should have False at index 2
        assert result[2] == False


class TestInvariantViolation:
    """Test internal invariant violation errors."""

    def test_invariant_violation_suggests_bug_report(self):
        """InvariantViolationError should suggest reporting a bug."""
        exc = InvariantViolationError("Internal state corrupted")
        assert "bug" in str(exc).lower() or "report" in str(exc).lower()


class TestExceptionExports:
    """Test that exceptions are properly exported from package."""

    def test_exceptions_available_from_boofun(self):
        """All exceptions should be importable from boofun."""
        assert hasattr(bf, "BooleanFunctionError")
        assert hasattr(bf, "ValidationError")
        assert hasattr(bf, "InvalidInputError")
        assert hasattr(bf, "InvalidRepresentationError")
        assert hasattr(bf, "InvalidTruthTableError")
        assert hasattr(bf, "EvaluationError")
        assert hasattr(bf, "ConversionError")
        assert hasattr(bf, "ConfigurationError")
        assert hasattr(bf, "ResourceUnavailableError")
        assert hasattr(bf, "InvariantViolationError")


class TestBackwardCompatibility:
    """Test that existing error handling still works."""

    def test_value_error_for_invalid_range(self):
        """ValueError should still be raised for simple range errors."""
        f = bf.create([0, 1, 1, 0])
        with pytest.raises((ValueError, InvalidInputError)):
            f.fix(0, 2)  # Value must be 0 or 1

    def test_index_error_for_out_of_bounds(self):
        """IndexError should still be raised for out of bounds access."""
        f = bf.create([0, 1, 1, 0])
        with pytest.raises(IndexError):
            f.evaluate(10)  # Out of range for 2-variable function


class TestErrorCodes:
    """Test machine-readable error codes."""

    def test_error_code_enum_exists(self):
        """ErrorCode enum should be importable."""
        assert hasattr(bf, "ErrorCode")
        assert ErrorCode.INVALID_TRUTH_TABLE is not None

    def test_exception_has_error_code(self):
        """Exceptions should have error code attribute."""
        exc = InvalidTruthTableError("test", size=5)
        assert hasattr(exc, "code")
        assert isinstance(exc.code, ErrorCode)

    def test_truth_table_error_code(self):
        """Truth table errors should have correct code."""
        try:
            bf.create([0, 1, 1])  # Invalid size
        except InvalidTruthTableError as e:
            assert e.code == ErrorCode.INVALID_TRUTH_TABLE
            assert "E13" in e.code.value  # E13xx range

    def test_conversion_error_code(self):
        """Conversion errors should have correct code."""
        f = bf.BooleanFunction(n=2)
        try:
            f.get_representation("fourier")
        except ConversionError as e:
            assert e.code == ErrorCode.CONVERSION_ERROR
            assert "E3" in e.code.value  # E3xxx range

    def test_error_code_in_message(self):
        """Error code should appear in formatted message."""
        exc = InvalidInputError("test error", parameter="x")
        assert "[E1" in str(exc)  # E1xxx range in message

    def test_to_dict_method(self):
        """Exception should serialize to dict correctly."""
        exc = InvalidTruthTableError(
            "Bad size",
            size=5,
            suggestion="Use power of 2"
        )
        d = exc.to_dict()
        assert "error_code" in d
        assert "error_type" in d
        assert "message" in d
        assert "context" in d
        assert d["error_type"] == "InvalidTruthTableError"
        assert d["context"]["size"] == 5

    def test_error_codes_are_unique(self):
        """All error codes should have unique values."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values))


class TestLogging:
    """Test that logging utilities work correctly."""

    def test_logging_module_exists(self):
        """Logging module should be importable."""
        from boofun.utils.logging import get_logger, enable_debug_logging
        assert callable(get_logger)
        assert callable(enable_debug_logging)

    def test_get_logger(self):
        """get_logger should return a logger."""
        from boofun.utils.logging import get_logger
        logger = get_logger("test")
        assert logger.name == "boofun.test"
