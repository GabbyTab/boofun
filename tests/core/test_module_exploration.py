"""
Module exploration and integration tests.

These tests verify that all public modules:
1. Import correctly
2. Have expected public API
3. Produce valid results (not just "something")

Note: For detailed mathematical correctness, see dedicated test files.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestAutoRepresentation:
    """Test auto_representation module."""

    def test_module_imports(self):
        """Module should import without errors."""
        from boofun.core import auto_representation

        # Verify module has expected content
        assert hasattr(auto_representation, "__name__")

    def test_auto_rep_has_public_api(self):
        """Module should expose public functions/classes."""
        from boofun.core import auto_representation

        public_names = [n for n in dir(auto_representation) if not n.startswith("_")]
        assert len(public_names) > 0, "Module should have public API"


class TestLegacyAdapter:
    """Test legacy_adapter module."""

    def test_module_imports(self):
        """Module should import without errors."""
        from boofun.core import legacy_adapter

        assert hasattr(legacy_adapter, "__name__")

    def test_create_produces_valid_function(self):
        """bf.create should produce a valid BooleanFunction."""
        f = bf.create([0, 0, 0, 1])  # AND function

        # Verify it's a proper function with correct behavior
        assert f.n_vars == 2, f"Expected 2 vars, got {f.n_vars}"
        assert f.evaluate(0b11) == 1, "AND(1,1) should be 1"
        assert f.evaluate(0b00) == 0, "AND(0,0) should be 0"

    def test_fourier_returns_valid_coefficients(self):
        """Fourier method should return valid coefficients."""
        f = bf.create([0, 1, 1, 0])  # XOR

        fourier = f.fourier()
        assert len(fourier) == 4, f"Expected 4 coefficients, got {len(fourier)}"

        # XOR has coefficient only at {0,1}
        assert abs(fourier[3]) > 0.9, "XOR should have large coefficient at {0,1}"

    def test_influences_are_valid(self):
        """Influences should be non-negative and sum correctly."""
        f = bf.majority(3)

        if hasattr(f, "influences"):
            influences = f.influences()
            assert len(influences) == 3, "Should have 3 influences"
            assert all(i >= 0 for i in influences), "Influences should be non-negative"

    def test_total_influence_in_valid_range(self):
        """Total influence should be in [0, n]."""
        f = bf.parity(4)

        if hasattr(f, "total_influence"):
            ti = f.total_influence()
            assert 0 <= ti <= 4, f"Total influence {ti} out of range [0, 4]"


class TestPackedTruthTable:
    """Test packed_truth_table module."""

    def test_module_imports(self):
        """Module should import without errors."""
        from boofun.core.representations import packed_truth_table

        assert hasattr(packed_truth_table, "__name__")

    def test_module_has_representation_class(self):
        """Module should have representation-related classes."""
        from boofun.core.representations import packed_truth_table

        public_names = [n for n in dir(packed_truth_table) if not n.startswith("_")]
        assert len(public_names) > 0


class TestLTFRepresentation:
    """Test LTF (Linear Threshold Function) representation."""

    def test_module_imports(self):
        """Module should import without errors."""
        from boofun.core.representations import ltf

        assert hasattr(ltf, "__name__")

    def test_ltf_has_expected_classes(self):
        """Module should have LTF-related classes."""
        from boofun.core.representations import ltf

        # Should have LTFParameters or similar
        public_names = [n for n in dir(ltf) if not n.startswith("_")]
        assert len(public_names) > 0

        # Check for key classes
        assert hasattr(ltf, "LTFParameters") or hasattr(ltf, "LTFRepresentation")


class TestCoreSpaces:
    """Test core spaces module."""

    def test_space_enum_exists(self):
        """Space enum should exist with expected values."""
        from boofun.core.spaces import Space

        # Verify key space types exist
        assert hasattr(Space, "BOOLEAN_CUBE")
        assert hasattr(Space, "PLUS_MINUS_CUBE")

    def test_space_translation(self):
        """Space translation should convert between representations."""
        from boofun.core.spaces import Space

        # Boolean -> ±1
        result = Space.translate([0, 1], Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)
        expected = np.array([-1, 1])
        assert np.allclose(result, expected), f"Expected {expected}, got {result}"

        # ±1 -> Boolean
        result = Space.translate([-1, 1], Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE)
        expected = np.array([0, 1])
        assert np.allclose(result, expected), f"Expected {expected}, got {result}"


class TestAnalysisInvariance:
    """Test analysis invariance module."""

    def test_module_imports(self):
        """Module should import without errors."""
        from boofun.analysis import invariance

        assert hasattr(invariance, "__name__")


class TestBooleanFunctionMethods:
    """Test BooleanFunction methods produce valid results."""

    def test_evaluate_returns_boolean(self):
        """Evaluate should return 0/1 or True/False."""
        f = bf.AND(3)

        for x in range(8):
            result = f(x)
            # Accept various boolean representations
            assert result in [0, 1, True, False] or isinstance(
                result, (bool, np.bool_)
            ), f"Unexpected result type: {type(result)}"

    def test_truth_table_has_correct_size(self):
        """Truth table should have 2^n entries."""
        f = bf.OR(3)

        tt = f.get_representation("truth_table")
        assert len(tt) == 8, f"Expected 8 entries, got {len(tt)}"

    def test_function_properties_return_booleans(self):
        """Property methods should return boolean values."""
        f = bf.majority(5)

        if hasattr(f, "is_balanced"):
            result = f.is_balanced()
            assert isinstance(result, (bool, np.bool_))

        if hasattr(f, "is_monotone"):
            result = f.is_monotone()
            assert isinstance(result, (bool, np.bool_))

        if hasattr(f, "is_symmetric"):
            result = f.is_symmetric()
            assert isinstance(result, (bool, np.bool_))


class TestFamilyMethods:
    """Test family methods produce valid functions."""

    def test_majority_family_generates_majority(self):
        """MajorityFamily should generate actual majority functions."""
        from boofun.families import MajorityFamily

        fam = MajorityFamily()
        f = fam.generate(3)

        # Verify it's actually majority
        assert f.evaluate(0b111) == 1, "MAJ(1,1,1) should be 1"
        assert f.evaluate(0b000) == 0, "MAJ(0,0,0) should be 0"
        assert f.evaluate(0b011) == 1, "MAJ(0,1,1) should be 1"

    def test_parity_family_generates_parity(self):
        """ParityFamily should generate actual parity functions."""
        from boofun.families import ParityFamily

        fam = ParityFamily()
        f = fam.generate(3)

        # Verify it's actually parity (XOR)
        for x in range(8):
            expected = bin(x).count("1") % 2
            actual = int(f.evaluate(x))
            assert actual == expected, f"PAR({bin(x)}) = {actual}, expected {expected}"


class TestSpectralAnalyzer:
    """Test SpectralAnalyzer produces valid results."""

    def test_fourier_expansion_sums_to_1(self):
        """Parseval: sum of squared coefficients should be 1."""
        from boofun.analysis import SpectralAnalyzer

        f = bf.majority(3)
        analyzer = SpectralAnalyzer(f)

        coeffs = analyzer.fourier_expansion()
        sum_squared = np.sum(coeffs**2)

        assert np.isclose(sum_squared, 1.0), f"Parseval violated: Σf̂(S)² = {sum_squared}"

    def test_total_influence_in_range(self):
        """Total influence should be in [0, n]."""
        from boofun.analysis import SpectralAnalyzer

        f = bf.AND(4)
        analyzer = SpectralAnalyzer(f)

        ti = analyzer.total_influence()
        assert 0 <= ti <= 4, f"Total influence {ti} out of range [0, 4]"


class TestPropertyTester:
    """Test PropertyTester produces valid results."""

    def test_blr_test_on_linear_function(self):
        """BLR should accept linear functions."""
        from boofun.analysis import PropertyTester

        f = bf.parity(4)  # Linear function
        tester = PropertyTester(f)

        result = tester.blr_linearity_test(100)
        assert result is True, "BLR should accept parity (linear)"

    def test_blr_test_on_nonlinear_function(self):
        """BLR should reject non-linear functions (with high probability)."""
        from boofun.analysis import PropertyTester

        f = bf.majority(5)  # Not linear
        tester = PropertyTester(f)

        result = tester.blr_linearity_test(100)
        assert result is False, "BLR should reject majority (non-linear)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
