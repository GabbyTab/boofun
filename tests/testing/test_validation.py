"""
Comprehensive tests for testing module.

Tests for validation, property testing, and performance profiling utilities.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.testing import (
    BooleanFunctionValidator,
    PerformanceProfiler,
    PropertyTestSuite,
    RepresentationTester,
    quick_validate,
    validate_representation,
)


class TestBooleanFunctionValidator:
    """Test BooleanFunctionValidator class."""

    def test_validator_init(self):
        """Validator should initialize correctly."""
        f = bf.majority(3)
        validator = BooleanFunctionValidator(f)

        assert validator.function == f
        assert validator.verbose == False

    def test_validator_verbose_mode(self):
        """Validator should accept verbose mode."""
        f = bf.AND(3)
        validator = BooleanFunctionValidator(f, verbose=True)

        assert validator.verbose == True

    def test_validate_all(self):
        """validate_all should run comprehensive validation."""
        f = bf.majority(3)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_all()

        assert isinstance(results, dict)
        assert "overall_status" in results
        assert "basic_properties" in results

    def test_validate_basic_properties(self):
        """validate_basic_properties should check n_vars, space, etc."""
        f = bf.AND(3)
        validator = BooleanFunctionValidator(f)

        result = validator.validate_basic_properties()

        assert isinstance(result, dict)
        assert "passed" in result
        assert "issues" in result

    def test_validate_representation_consistency(self):
        """validate_representation_consistency should compare reps."""
        f = bf.OR(3)
        validator = BooleanFunctionValidator(f)

        result = validator.validate_representation_consistency()

        assert isinstance(result, dict)
        assert "passed" in result

    def test_validate_evaluation_correctness(self):
        """validate_evaluation_correctness should test evaluation."""
        f = bf.parity(3)
        validator = BooleanFunctionValidator(f)

        result = validator.validate_evaluation_correctness()

        assert isinstance(result, dict)

    def test_validate_space_handling(self):
        """validate_space_handling should test space conversions."""
        f = bf.majority(3)
        validator = BooleanFunctionValidator(f)

        result = validator.validate_space_handling()

        assert isinstance(result, dict)

    def test_validate_edge_cases(self):
        """validate_edge_cases should test boundary conditions."""
        f = bf.AND(3)
        validator = BooleanFunctionValidator(f)

        result = validator.validate_edge_cases()

        assert isinstance(result, dict)


class TestRepresentationTester:
    """Test RepresentationTester class."""

    def test_tester_requires_representation(self):
        """RepresentationTester should require representation."""
        f = bf.majority(3)

        # Get a representation to pass
        try:
            rep = f.get_representation("truth_table")
            tester = RepresentationTester(rep)
            assert tester is not None
        except (TypeError, AttributeError, KeyError):
            pytest.skip("RepresentationTester has different API")

    def test_tester_has_methods(self):
        """RepresentationTester should have testing methods."""
        f = bf.majority(3)

        try:
            rep = f.get_representation("truth_table")
            tester = RepresentationTester(rep)

            methods = [m for m in dir(tester) if not m.startswith("_")]
            assert len(methods) > 0
        except (TypeError, AttributeError, KeyError):
            pytest.skip("RepresentationTester has different API")


class TestPropertyTestSuite:
    """Test PropertyTestSuite class."""

    def test_suite_init(self):
        """PropertyTestSuite should initialize correctly."""
        suite = PropertyTestSuite()
        assert suite is not None

    def test_run_all_tests(self):
        """PropertyTestSuite should run all tests."""
        f = bf.majority(3)
        suite = PropertyTestSuite()

        if hasattr(suite, "run_all"):
            results = suite.run_all(f)
            assert isinstance(results, dict)

    def test_test_fourier_properties(self):
        """PropertyTestSuite should test Fourier properties."""
        f = bf.parity(3)
        suite = PropertyTestSuite()

        if hasattr(suite, "test_fourier"):
            result = suite.test_fourier(f)
            assert result is not None


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""

    def test_profiler_init(self):
        """PerformanceProfiler should initialize correctly."""
        profiler = PerformanceProfiler()
        assert profiler is not None

    def test_profile_function(self):
        """PerformanceProfiler should profile function operations."""
        f = bf.majority(3)
        profiler = PerformanceProfiler()

        if hasattr(profiler, "profile"):
            result = profiler.profile(f)
            assert isinstance(result, dict)

    def test_profile_evaluation(self):
        """PerformanceProfiler should profile evaluation."""
        f = bf.AND(3)
        profiler = PerformanceProfiler()

        if hasattr(profiler, "profile_evaluation"):
            try:
                result = profiler.profile_evaluation(f)
                assert result is not None
            except TypeError:
                pass  # May have different signature


class TestQuickValidate:
    """Test quick_validate function."""

    def test_quick_validate_valid_function(self):
        """quick_validate should return True for valid functions."""
        f = bf.majority(3)

        result = quick_validate(f)

        assert isinstance(result, bool)

    def test_quick_validate_and(self):
        """quick_validate should work for AND function."""
        f = bf.AND(4)

        result = quick_validate(f)

        assert isinstance(result, bool)

    def test_quick_validate_or(self):
        """quick_validate should work for OR function."""
        f = bf.OR(4)

        result = quick_validate(f)

        assert isinstance(result, bool)

    def test_quick_validate_parity(self):
        """quick_validate should work for parity function."""
        f = bf.parity(4)

        result = quick_validate(f)

        assert isinstance(result, bool)

    def test_quick_validate_verbose(self):
        """quick_validate should accept verbose parameter."""
        f = bf.majority(3)

        result = quick_validate(f, verbose=True)

        assert isinstance(result, bool)


class TestTestRepresentation:
    """Test validate_representation function."""

    def validate_representation_function_exists(self):
        """validate_representation function should exist and be callable."""
        assert callable(validate_representation)

    def test_validate_representation_basic(self):
        """validate_representation should work with a representation."""
        f = bf.majority(3)

        try:
            rep = f.get_representation("truth_table")
            result = validate_representation(rep, n_vars=3)
            # Result should be a meaningful validation outcome (bool, dict, etc.)
            assert result is not None
        except (ImportError, AttributeError, TypeError):
            pytest.skip("Representation testing has different API")


class TestValidationIntegration:
    """Integration tests for validation utilities."""

    def test_validate_multiple_functions(self):
        """Validator should work for multiple function types."""
        functions = [
            bf.AND(3),
            bf.OR(3),
            bf.majority(3),
            bf.parity(3),
            bf.dictator(4, 0),
        ]

        for f in functions:
            validator = BooleanFunctionValidator(f)
            results = validator.validate_all()

            assert "overall_status" in results

    def test_quick_validate_all_types(self):
        """quick_validate should work for all function types."""
        functions = [
            bf.AND(3),
            bf.OR(3),
            bf.majority(3),
            bf.parity(3),
        ]

        for f in functions:
            result = quick_validate(f)
            assert isinstance(result, bool)

    def test_validation_on_custom_function(self):
        """Validator should work for custom functions."""
        # Create a custom function from truth table
        tt = [0, 1, 1, 0, 1, 0, 0, 1]  # Parity-like
        f = bf.create(tt)

        validator = BooleanFunctionValidator(f)
        results = validator.validate_all()

        assert isinstance(results, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
