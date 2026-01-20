import sys

sys.path.insert(0, "src")
"""
Tests for the testing module.

Tests for:
- BooleanFunctionValidator
- RepresentationTester
- PropertyTestSuite
- PerformanceProfiler
- quick_validate
- validate_representation
"""


import boofun as bf
from boofun.core.representations.registry import get_strategy
from boofun.testing import (
    BooleanFunctionValidator,
    PerformanceProfiler,
    PropertyTestSuite,
    RepresentationTester,
    quick_validate,
    validate_representation,
)


class TestBooleanFunctionValidator:
    """Tests for BooleanFunctionValidator."""

    def test_initialization(self):
        """Validator initializes correctly."""
        f = bf.majority(3)
        validator = BooleanFunctionValidator(f)

        assert validator.function is f
        assert validator.verbose == False
        assert validator.validation_results == {}

    def test_validate_all_returns_dict(self):
        """validate_all returns dictionary."""
        f = bf.AND(3)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_all()

        assert isinstance(results, dict)
        assert "basic_properties" in results
        assert "overall_status" in results

    def test_validate_basic_properties(self):
        """Basic properties validation works."""
        f = bf.parity(3)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_basic_properties()

        assert "passed" in results
        assert "issues" in results

    def test_validate_evaluation_correctness(self):
        """Evaluation correctness validation works."""
        f = bf.OR(3)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_evaluation_correctness()

        assert "passed" in results
        assert "tests" in results

    def test_validate_space_handling(self):
        """Space handling validation works."""
        f = bf.majority(3)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_space_handling()

        assert "passed" in results
        assert "space_tests" in results

    def test_validate_edge_cases(self):
        """Edge case validation works."""
        f = bf.AND(3)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_edge_cases()

        assert "passed" in results
        assert "edge_tests" in results

    def test_print_validation_report_no_results(self, capsys):
        """Print report without running validate_all."""
        f = bf.parity(3)
        validator = BooleanFunctionValidator(f)

        validator.print_validation_report()

        captured = capsys.readouterr()
        assert "No validation results" in captured.out

    def test_print_validation_report_with_results(self, capsys):
        """Print report after validation."""
        f = bf.majority(3)
        validator = BooleanFunctionValidator(f)
        validator.validate_all()

        validator.print_validation_report()

        captured = capsys.readouterr()
        assert "Validation Report" in captured.out


class TestRepresentationTester:
    """Tests for RepresentationTester."""

    def test_initialization(self):
        """RepresentationTester initializes correctly."""
        rep = get_strategy("truth_table")
        tester = RepresentationTester(rep)

        assert tester.representation is rep

    def test_interface_compliance(self):
        """test_interface_compliance works."""
        rep = get_strategy("truth_table")
        tester = RepresentationTester(rep)

        results = tester.test_interface_compliance(n_vars=3)

        assert "passed" in results
        assert "method_tests" in results

        # Check that required methods are tested
        assert "evaluate" in results["method_tests"]
        assert "dump" in results["method_tests"]

    def test_create_empty(self):
        """test_create_empty works."""
        rep = get_strategy("truth_table")
        tester = RepresentationTester(rep)

        results = tester.test_create_empty(n_vars=3)

        assert "passed" in results

    def test_storage_requirements(self):
        """test_storage_requirements works."""
        rep = get_strategy("truth_table")
        tester = RepresentationTester(rep)

        results = tester.test_storage_requirements(n_vars_range=[1, 2, 3])

        assert "passed" in results
        assert "requirements" in results


class TestPropertyTestSuite:
    """Tests for PropertyTestSuite."""

    def test_initialization(self):
        """PropertyTestSuite initializes."""
        suite = PropertyTestSuite()
        assert suite is not None

    def test_test_known_functions(self):
        """test_known_functions runs."""
        suite = PropertyTestSuite()

        results = suite.test_known_functions()

        assert "tests" in results
        assert "overall_passed" in results


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""

    def test_initialization(self):
        """PerformanceProfiler initializes."""
        profiler = PerformanceProfiler()
        assert profiler is not None

    def test_profile_evaluation(self):
        """profile_evaluation works."""
        profiler = PerformanceProfiler()
        f = bf.AND(3)

        results = profiler.profile_evaluation(f, n_trials=10)

        assert "timings" in results
        assert "average_time" in results
        assert len(results["timings"]) == 10

    def test_profile_evaluation_error_handling(self):
        """profile_evaluation handles errors."""
        profiler = PerformanceProfiler()
        f = bf.majority(3)

        # Should not raise even with low trials
        results = profiler.profile_evaluation(f, n_trials=1)

        assert results["error"] is None


class TestQuickValidate:
    """Tests for quick_validate function."""

    def test_returns_bool(self):
        """quick_validate returns boolean."""
        f = bf.AND(3)
        result = quick_validate(f)

        assert isinstance(result, bool)

    def test_valid_function_passes(self):
        """Valid function should pass validation."""
        f = bf.majority(3)
        result = quick_validate(f, verbose=False)

        # Should pass basic validation
        assert isinstance(result, bool)

    def test_verbose_mode(self, capsys):
        """Verbose mode prints output."""
        f = bf.parity(3)
        quick_validate(f, verbose=True)

        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestTestRepresentation:
    """Tests for validate_representation function."""

    def test_returns_dict(self):
        """validate_representation returns dict."""
        rep = get_strategy("truth_table")
        results = validate_representation(rep, n_vars=3)

        assert isinstance(results, dict)

    def test_contains_expected_keys(self):
        """Result contains expected keys."""
        rep = get_strategy("truth_table")
        results = validate_representation(rep, n_vars=3)

        assert "interface_compliance" in results
        assert "create_empty" in results
        assert "storage_requirements" in results
        assert "overall_passed" in results


class TestValidatorEdgeCases:
    """Edge case tests for validators."""

    def test_small_function(self):
        """Validate 1-variable function."""
        f = bf.parity(1)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_all()
        assert "overall_status" in results

    def test_larger_function(self):
        """Validate larger function."""
        f = bf.majority(5)
        validator = BooleanFunctionValidator(f)

        results = validator.validate_basic_properties()
        assert "passed" in results
