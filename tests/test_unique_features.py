"""
Smoke tests for BooFun-specific feature surfaces.

Moved out of the old tests/test_cross_validation.py: these are not
cross-validation (there is no external reference), just checks that the
distinguishing modules exist and produce sane output.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestUniqueFeatures:
    """Test features that only BooFun has."""

    def test_query_complexity_exists(self):
        """Verify query complexity module works."""
        from boofun.analysis.query_complexity import QueryComplexityProfile

        f = bf.AND(3)
        profile = QueryComplexityProfile(f)
        measures = profile.compute()

        # Should have key measures (D = deterministic complexity)
        assert "D" in measures
        assert "Q2" in measures  # Quantum complexity
        assert "bs" in measures  # Block sensitivity
        assert measures["D"] == 3  # D(AND_3) = 3

    def test_property_testing_exists(self):
        """Verify property testing works."""
        from boofun.analysis import PropertyTester

        f = bf.parity(4)
        tester = PropertyTester(f, random_seed=42)

        # Should have key tests
        assert hasattr(tester, "blr_linearity_test")
        assert hasattr(tester, "junta_test")
        assert hasattr(tester, "monotonicity_test")

    def test_quantum_complexity_module_exists(self):
        """Verify quantum complexity module works."""
        from boofun.quantum_complexity import QuantumComplexityAnalyzer

        f = bf.AND(3)
        qca = QuantumComplexityAnalyzer(f)

        # Should have complexity analysis methods
        assert hasattr(qca, "create_quantum_oracle")
        assert hasattr(qca, "grover_analysis")
        assert hasattr(qca, "grover_amplitude_analysis")

    def test_noise_stability_exists(self):
        """Verify noise stability (unique to our Fourier focus)."""
        f = bf.majority(5)

        # Should have noise_stability method
        stab = f.noise_stability(0.5)

        # Should be in valid range
        assert -1 <= stab <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
