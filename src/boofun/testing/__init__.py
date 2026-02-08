"""
Testing utilities and validation tools for BooFun library.

This module provides comprehensive testing tools for Boolean function analysis,
including property validation, representation consistency checks, and performance testing.
"""

from typing import Any, Dict, List

import numpy as np

from ..analysis import PropertyTester, SpectralAnalyzer
from ..core.base import BooleanFunction
from ..core.representations.base import BooleanFunctionRepresentation


class BooleanFunctionValidator:
    """
    Comprehensive validator for Boolean function implementations.

    Validates correctness, consistency, and performance of Boolean functions
    across different representations and operations.
    """

    def __init__(self, function: BooleanFunction, verbose: bool = False):
        """
        Initialize validator.

        Args:
            function: Boolean function to validate
            verbose: Whether to print detailed validation messages
        """
        self.function = function
        self.verbose = verbose
        self.validation_results: Dict[str, Any] = {}

    def validate_all(self) -> Dict[str, Any]:
        """
        Run comprehensive validation suite.

        Returns:
            Dictionary with validation results
        """
        results: Dict[str, Any] = {
            "basic_properties": self.validate_basic_properties(),
            "representation_consistency": self.validate_representation_consistency(),
            "evaluation_correctness": self.validate_evaluation_correctness(),
            "space_handling": self.validate_space_handling(),
            "edge_cases": self.validate_edge_cases(),
        }

        # Overall validation status
        results["overall_status"] = all(result.get("passed", False) for result in results.values())

        self.validation_results = results
        return results

    def validate_basic_properties(self) -> Dict[str, Any]:
        """Validate basic Boolean function properties."""
        results: Dict[str, Any] = {"passed": True, "issues": []}

        try:
            # Check n_vars is properly set
            if self.function.n_vars is None:
                results["issues"].append("n_vars not set")
                results["passed"] = False
            elif self.function.n_vars < 0:
                results["issues"].append(f"Invalid n_vars: {self.function.n_vars}")
                results["passed"] = False

            # Check space is valid
            if self.function.space is None:
                results["issues"].append("Space not set")
                results["passed"] = False

            # Check representations exist
            if not self.function.representations:
                results["issues"].append("No representations available")
                results["passed"] = False

            # Check error model exists
            if self.function.error_model is None:
                results["issues"].append("Error model not set")
                results["passed"] = False

        except Exception as e:
            results["issues"].append(f"Exception during basic validation: {e}")
            results["passed"] = False

        return results

    def validate_representation_consistency(self) -> Dict[str, Any]:
        """Validate consistency across different representations."""
        results: Dict[str, Any] = {"passed": True, "issues": [], "comparisons": {}}

        if len(self.function.representations) < 2:
            results["issues"].append("Need at least 2 representations for consistency check")
            results["passed"] = False
            return results

        try:
            # Test evaluation consistency across representations
            test_inputs = self._generate_test_inputs()

            rep_names = list(self.function.representations.keys())
            for i, rep1 in enumerate(rep_names):
                for rep2 in rep_names[i + 1 :]:
                    consistency_result = self._compare_representations(rep1, rep2, test_inputs)
                    results["comparisons"][f"{rep1}_vs_{rep2}"] = consistency_result

                    if not consistency_result["consistent"]:
                        results["passed"] = False
                        results["issues"].append(f"Inconsistency between {rep1} and {rep2}")

        except Exception as e:
            results["issues"].append(f"Exception during consistency validation: {e}")
            results["passed"] = False

        return results

    def validate_evaluation_correctness(self) -> Dict[str, Any]:
        """Validate evaluation correctness."""
        results: Dict[str, Any] = {"passed": True, "issues": [], "tests": {}}

        try:
            # Test different input formats
            test_cases = [
                ("integer_index", 0),
                (
                    "binary_vector",
                    np.array([0, 1]) if (self.function.n_vars or 0) >= 2 else np.array([0]),
                ),
                (
                    "batch_integers",
                    (
                        np.array([0, 1, 2, 3])
                        if (self.function.n_vars or 0) >= 2
                        else np.array([0, 1])
                    ),
                ),
            ]

            for test_name, test_input in test_cases:
                try:
                    result = self.function.evaluate(test_input)
                    results["tests"][test_name] = {
                        "passed": True,
                        "result": str(result),
                        "input": str(test_input),
                    }
                except Exception as e:
                    results["tests"][test_name] = {
                        "passed": False,
                        "error": str(e),
                        "input": str(test_input),
                    }
                    results["passed"] = False
                    results["issues"].append(f"Evaluation failed for {test_name}: {e}")

        except Exception as e:
            results["issues"].append(f"Exception during evaluation validation: {e}")
            results["passed"] = False

        return results

    def validate_space_handling(self) -> Dict[str, Any]:
        """Validate space conversion and handling."""
        results: Dict[str, Any] = {"passed": True, "issues": [], "space_tests": {}}

        # This is a placeholder - full space validation would require more representations
        try:
            current_space = self.function.space
            results["space_tests"]["current_space"] = {
                "space": current_space.name if current_space else None,
                "valid": current_space is not None,
            }

            if current_space is None:
                results["passed"] = False
                results["issues"].append("No space defined")

        except Exception as e:
            results["issues"].append(f"Exception during space validation: {e}")
            results["passed"] = False

        return results

    def validate_edge_cases(self) -> Dict[str, Any]:
        """Validate handling of edge cases."""
        results: Dict[str, Any] = {"passed": True, "issues": [], "edge_tests": {}}

        edge_cases = [
            ("empty_input", np.array([])),
            ("out_of_range_index", 2 ** (self.function.n_vars or 4)),
            ("negative_index", -1),
        ]

        for case_name, test_input in edge_cases:
            try:
                result = self.function.evaluate(test_input)
                results["edge_tests"][case_name] = {
                    "passed": False,  # Should have raised exception
                    "unexpected_result": str(result),
                }
                results["issues"].append(f"Edge case {case_name} should have failed but didn't")
                results["passed"] = False

            except Exception as e:
                results["edge_tests"][case_name] = {
                    "passed": True,  # Expected exception
                    "expected_error": str(e),
                }

        return results

    def _generate_test_inputs(self) -> List[Any]:
        """Generate test inputs for validation."""
        if self.function.n_vars is None or self.function.n_vars == 0:
            return [0]

        n_vars = min(self.function.n_vars, 4)  # Limit for performance
        max_inputs = min(2**n_vars, 16)  # Limit number of test inputs

        return list(range(max_inputs))

    def _compare_representations(
        self, rep1: str, rep2: str, test_inputs: List[Any]
    ) -> Dict[str, Any]:
        """Compare two representations for consistency."""
        mismatches = 0
        total_tests = len(test_inputs)

        for test_input in test_inputs:
            try:
                result1 = self.function.evaluate(test_input, rep_type=rep1)
                result2 = self.function.evaluate(test_input, rep_type=rep2)

                if result1 != result2:
                    mismatches += 1

            except Exception as e:
                mismatches += 1
                if self.verbose:
                    print(f"Error comparing {rep1} vs {rep2} on input {test_input}: {e}")

        return {
            "consistent": mismatches == 0,
            "mismatches": mismatches,
            "total_tests": total_tests,
            "accuracy": (total_tests - mismatches) / total_tests if total_tests > 0 else 0,
        }

    def print_validation_report(self) -> None:
        """Print detailed validation report."""
        if not self.validation_results:
            print("No validation results available. Run validate_all() first.")
            return

        print("🔍 Boolean Function Validation Report")
        print("=" * 50)

        overall_status = self.validation_results.get("overall_status", False)
        status_icon = "✅" if overall_status else "❌"
        print(f"{status_icon} Overall Status: {'PASSED' if overall_status else 'FAILED'}")
        print()

        for category, results in self.validation_results.items():
            if category == "overall_status":
                continue

            print(f"📋 {category.replace('_', ' ').title()}")
            if isinstance(results, dict):
                passed = results.get("passed", False)
                icon = "✅" if passed else "❌"
                print(f"  {icon} Status: {'PASSED' if passed else 'FAILED'}")

                if "issues" in results and results["issues"]:
                    print("  Issues:")
                    for issue in results["issues"]:
                        print(f"    - {issue}")
                print()


class RepresentationTester:
    """
    Specialized tester for Boolean function representations.

    Tests individual representations for correctness, performance, and compliance
    with the representation interface.
    """

    def __init__(self, representation: BooleanFunctionRepresentation):
        """
        Initialize representation tester.

        Args:
            representation: Representation to test
        """
        self.representation = representation

    def test_interface_compliance(self, n_vars: int = 3) -> Dict[str, Any]:
        """Test that representation implements required interface methods."""
        results: Dict[str, Any] = {"passed": True, "method_tests": {}}

        required_methods = [
            "evaluate",
            "dump",
            "convert_from",
            "convert_to",
            "create_empty",
            "is_complete",
            "time_complexity_rank",
            "get_storage_requirements",
        ]

        for method_name in required_methods:
            if hasattr(self.representation, method_name):
                method = getattr(self.representation, method_name)
                results["method_tests"][method_name] = {
                    "exists": True,
                    "callable": callable(method),
                }
                if not callable(method):
                    results["passed"] = False
            else:
                results["method_tests"][method_name] = {"exists": False, "callable": False}
                results["passed"] = False

        return results

    def test_create_empty(self, n_vars: int = 3) -> Dict[str, Any]:
        """Test create_empty method."""
        results: Dict[str, Any] = {"passed": True, "error": None}

        try:
            empty_data = self.representation.create_empty(n_vars)

            # Test that empty data is valid
            is_complete = self.representation.is_complete(empty_data)
            results["empty_is_complete"] = is_complete
            results["empty_data_type"] = type(empty_data).__name__

        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)

        return results

    def test_storage_requirements(self, n_vars_range: List[int] = [1, 2, 3, 4]) -> Dict[str, Any]:
        """Test storage requirements computation."""
        results: Dict[str, Any] = {"passed": True, "requirements": {}, "error": None}

        try:
            for n_vars in n_vars_range:
                reqs: Any = self.representation.get_storage_requirements(n_vars)
                results["requirements"][n_vars] = reqs

                # Basic validation of requirements format
                if not isinstance(reqs, dict):
                    results["passed"] = False
                    results["error"] = f"Storage requirements should be dict, got {type(reqs)}"
                    break

        except Exception as e:
            results["passed"] = False
            results["error"] = str(e)

        return results


class PropertyTestSuite:
    """
    Comprehensive test suite for Boolean function properties.

    Tests various mathematical properties and their detection algorithms.
    """

    def __init__(self):
        """Initialize property test suite."""

    def test_known_functions(self) -> Dict[str, Any]:
        """Test property detection on functions with known properties."""
        results: Dict[str, Any] = {"tests": {}, "overall_passed": True}

        # Test cases: (function_name, truth_table, expected_properties)
        test_cases = [
            ("constant_zero", [False] * 4, {"constant": True, "balanced": False}),
            ("constant_one", [True] * 4, {"constant": True, "balanced": False}),
            ("xor", [False, True, True, False], {"balanced": True, "linear": True}),
            ("and", [False, False, False, True], {"monotone": True}),
        ]

        for func_name, truth_table, expected_props in test_cases:
            try:
                # Create function and test properties
                from ..api import create

                func = create(truth_table)
                tester = PropertyTester(func)

                detected_props = tester.run_all_tests()

                # Compare with expected properties
                test_result: Dict[str, Any] = {"passed": True, "mismatches": []}
                for prop, expected_value in expected_props.items():
                    if prop in detected_props:
                        detected_value = detected_props[prop]
                        if detected_value != expected_value:
                            test_result["mismatches"].append(
                                {
                                    "property": prop,
                                    "expected": expected_value,
                                    "detected": detected_value,
                                }
                            )
                            test_result["passed"] = False

                results["tests"][func_name] = test_result
                if not test_result["passed"]:
                    results["overall_passed"] = False

            except Exception as e:
                results["tests"][func_name] = {"passed": False, "error": str(e)}
                results["overall_passed"] = False

        return results


class PerformanceProfiler:
    """
    Performance profiler for Boolean function operations.

    Measures timing and memory usage of various operations.
    """

    def __init__(self):
        """Initialize performance profiler."""

    def profile_evaluation(self, function: BooleanFunction, n_trials: int = 1000) -> Dict[str, Any]:
        """Profile evaluation performance."""
        import time

        results: Dict[str, Any] = {"timings": [], "average_time": 0, "error": None}

        try:
            # Generate test inputs
            if function.n_vars and function.n_vars <= 10:
                test_inputs = list(range(min(2**function.n_vars, 100)))
            else:
                test_inputs = list(range(100))

            # Time evaluations
            for _ in range(n_trials):
                start_time = time.perf_counter()
                for test_input in test_inputs:
                    function.evaluate(test_input)
                end_time = time.perf_counter()
                results["timings"].append(end_time - start_time)

            results["average_time"] = sum(results["timings"]) / len(results["timings"])
            results["min_time"] = min(results["timings"])
            results["max_time"] = max(results["timings"])

        except Exception as e:
            results["error"] = str(e)

        return results


# Convenience functions for quick testing
def quick_validate(function: BooleanFunction, verbose: bool = False) -> bool:
    """
    Quick validation of a Boolean function.

    Args:
        function: Function to validate
        verbose: Whether to print detailed results

    Returns:
        True if validation passed, False otherwise
    """
    validator = BooleanFunctionValidator(function, verbose=verbose)
    results = validator.validate_all()

    if verbose:
        validator.print_validation_report()

    return results["overall_status"]


def validate_representation(
    representation: BooleanFunctionRepresentation, n_vars: int = 3
) -> Dict[str, Any]:
    """
    Quick validation of a representation implementation.

    This is a utility function for validating representation implementations,
    not a pytest test.

    Args:
        representation: Representation to validate
        n_vars: Number of variables for testing

    Returns:
        Validation results dictionary
    """
    tester = RepresentationTester(representation)

    results: Dict[str, Any] = {
        "interface_compliance": tester.test_interface_compliance(n_vars),
        "create_empty": tester.test_create_empty(n_vars),
        "storage_requirements": tester.test_storage_requirements(),
    }

    results["overall_passed"] = all(result.get("passed", False) for result in results.values())

    return results


# Export main classes and functions
__all__ = [
    "BooleanFunctionValidator",
    "RepresentationTester",
    "PropertyTestSuite",
    "PerformanceProfiler",
    "quick_validate",
    "validate_representation",
]
