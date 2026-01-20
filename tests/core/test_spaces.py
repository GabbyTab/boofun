import sys

sys.path.insert(0, "src")
"""
Tests for spaces module.

Tests for:
- Space enum values
- Space.translate method
- Space utility methods
"""

import numpy as np
import pytest

from boofun.core.spaces import Space


class TestSpaceEnum:
    """Tests for Space enum."""

    def test_enum_values(self):
        """Enum has expected values."""
        assert Space.BOOLEAN_CUBE is not None
        assert Space.PLUS_MINUS_CUBE is not None
        assert Space.REAL is not None
        assert Space.LOG is not None
        assert Space.GAUSSIAN is not None


class TestSpaceTranslate:
    """Tests for Space.translate method."""

    def test_same_space_identity(self):
        """Translation to same space is identity."""
        arr = np.array([0, 1, 0, 1])

        result = Space.translate(arr, Space.BOOLEAN_CUBE, Space.BOOLEAN_CUBE)

        assert np.array_equal(result, arr)

    def test_boolean_to_pm(self):
        """Boolean to plus-minus conversion."""
        arr = np.array([0, 1, 0, 1])

        result = Space.translate(arr, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)

        expected = np.array([-1, 1, -1, 1])
        assert np.array_equal(result, expected)

    def test_pm_to_boolean(self):
        """Plus-minus to boolean conversion."""
        arr = np.array([-1, 1, -1, 1])

        result = Space.translate(arr, Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE)

        expected = np.array([0, 1, 0, 1])
        assert np.array_equal(result, expected)

    def test_boolean_to_real(self):
        """Boolean to real conversion."""
        arr = np.array([0, 1, 0, 1])

        result = Space.translate(arr, Space.BOOLEAN_CUBE, Space.REAL)

        expected = np.array([0.0, 1.0, 0.0, 1.0])
        assert np.allclose(result, expected)

    def test_real_to_boolean(self):
        """Real to boolean conversion (mod 2)."""
        arr = np.array([0.0, 1.0, 2.0, 3.0])

        result = Space.translate(arr, Space.REAL, Space.BOOLEAN_CUBE)

        expected = np.array([0, 1, 0, 1])  # mod 2
        assert np.array_equal(result, expected)

    def test_pm_to_real(self):
        """Plus-minus to real conversion."""
        arr = np.array([-1, 1, -1, 1])

        result = Space.translate(arr, Space.PLUS_MINUS_CUBE, Space.REAL)

        expected = np.array([-1.0, 1.0, -1.0, 1.0])
        assert np.allclose(result, expected)

    def test_real_to_pm(self):
        """Real to plus-minus conversion (sign)."""
        arr = np.array([-0.5, 0.5, -2.0, 3.0])

        result = Space.translate(arr, Space.REAL, Space.PLUS_MINUS_CUBE)

        expected = np.array([-1, 1, -1, 1])
        assert np.array_equal(result, expected)

    def test_real_to_pm_zero(self):
        """Zero maps to +1 in plus-minus."""
        arr = np.array([0.0])

        result = Space.translate(arr, Space.REAL, Space.PLUS_MINUS_CUBE)

        assert result[0] == 1  # >= 0 -> 1

    def test_log_to_boolean(self):
        """Log to boolean conversion."""
        # log(0.3) ≈ -1.2 (prob < 0.5)
        # log(0.7) ≈ -0.36 (prob > 0.5)
        arr = np.array([np.log(0.3), np.log(0.7)])

        result = Space.translate(arr, Space.LOG, Space.BOOLEAN_CUBE)

        expected = np.array([0, 1])
        assert np.array_equal(result, expected)

    def test_boolean_to_log(self):
        """Boolean to log conversion."""
        arr = np.array([0, 1])

        result = Space.translate(arr, Space.BOOLEAN_CUBE, Space.LOG)

        # 0 -> log(~0) = large negative
        # 1 -> log(~1) = small negative
        assert result[0] < result[1]

    def test_log_to_pm(self):
        """Log to plus-minus conversion (through boolean)."""
        arr = np.array([np.log(0.3), np.log(0.7)])

        result = Space.translate(arr, Space.LOG, Space.PLUS_MINUS_CUBE)

        expected = np.array([-1, 1])
        assert np.array_equal(result, expected)

    def test_pm_to_log(self):
        """Plus-minus to log conversion (through boolean)."""
        arr = np.array([-1, 1])

        result = Space.translate(arr, Space.PLUS_MINUS_CUBE, Space.LOG)

        # Should go through boolean
        assert result[0] < result[1]

    def test_scalar_input(self):
        """Scalar input works."""
        result = Space.translate(1, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)

        assert result == 1

    def test_unsupported_raises(self):
        """Unsupported translation raises."""
        arr = np.array([0, 1])

        # GAUSSIAN conversions require scipy, but let's test the error path
        # for a truly unsupported combination
        with pytest.raises(NotImplementedError):
            # This should be unsupported
            Space.translate(arr, Space.LOG, Space.GAUSSIAN)


class TestSpaceUtilities:
    """Tests for Space utility methods."""

    def test_get_canonical_space(self):
        """Canonical space is boolean cube."""
        canonical = Space.get_canonical_space()

        assert canonical == Space.BOOLEAN_CUBE

    def test_is_discrete_boolean(self):
        """Boolean cube is discrete."""
        assert Space.is_discrete(Space.BOOLEAN_CUBE) == True

    def test_is_discrete_pm(self):
        """Plus-minus cube is discrete."""
        assert Space.is_discrete(Space.PLUS_MINUS_CUBE) == True

    def test_is_discrete_real(self):
        """Real is not discrete."""
        assert Space.is_discrete(Space.REAL) == False

    def test_is_continuous_real(self):
        """Real is continuous."""
        assert Space.is_continuous(Space.REAL) == True

    def test_is_continuous_log(self):
        """Log is continuous."""
        assert Space.is_continuous(Space.LOG) == True

    def test_is_continuous_gaussian(self):
        """Gaussian is continuous."""
        assert Space.is_continuous(Space.GAUSSIAN) == True

    def test_is_continuous_boolean(self):
        """Boolean is not continuous."""
        assert Space.is_continuous(Space.BOOLEAN_CUBE) == False

    def test_get_default_threshold_real(self):
        """Default threshold for real is 0.5."""
        threshold = Space.get_default_threshold(Space.REAL)

        assert threshold == 0.5

    def test_get_default_threshold_log(self):
        """Default threshold for log is 0."""
        threshold = Space.get_default_threshold(Space.LOG)

        assert threshold == 0.0

    def test_get_default_threshold_gaussian(self):
        """Default threshold for gaussian is 0."""
        threshold = Space.get_default_threshold(Space.GAUSSIAN)

        assert threshold == 0.0


class TestRoundTrips:
    """Test round-trip conversions."""

    def test_boolean_pm_roundtrip(self):
        """Boolean -> PM -> Boolean is identity."""
        original = np.array([0, 1, 0, 1, 1, 0])

        pm = Space.translate(original, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)
        back = Space.translate(pm, Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE)

        assert np.array_equal(back, original)

    def test_pm_boolean_roundtrip(self):
        """PM -> Boolean -> PM is identity."""
        original = np.array([-1, 1, -1, 1, 1, -1])

        boolean = Space.translate(original, Space.PLUS_MINUS_CUBE, Space.BOOLEAN_CUBE)
        back = Space.translate(boolean, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)

        assert np.array_equal(back, original)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_array(self):
        """Empty array translates to empty."""
        arr = np.array([])

        result = Space.translate(arr, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)

        assert len(result) == 0

    def test_single_element(self):
        """Single element array works."""
        arr = np.array([1])

        result = Space.translate(arr, Space.BOOLEAN_CUBE, Space.PLUS_MINUS_CUBE)

        assert result[0] == 1
