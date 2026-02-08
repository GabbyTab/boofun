"""
Integration test: weighted majority -> Chow parameters -> verify match.

This is the test described in external review: "create a weighted majority
from weights, compute its Chow parameters, verify they match."

The Chow parameters of an LTF f with weights w and threshold θ are:
    chow[0] = E[f]  (= f̂(∅))
    chow[i] = f̂({i})  (the degree-1 Fourier coefficient on variable i)

For a weighted majority, these should satisfy:
    sign(chow[i]) = sign(w[i])  for non-degenerate cases
    |chow[i]| correlated with |w[i]|

This test catches bit-ordering bugs because if chow_parameters extracts
coefficients at the wrong index (e.g. 1 << (n-1-i) instead of 1 << i),
the Chow parameters will be scrambled relative to the weights.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.ltf_analysis import chow_parameters


class TestWeightedMajorityChowIntegration:
    """The single highest-value test for bit-ordering correctness."""

    def test_asymmetric_weights_chow_ordering(self):
        """Chow parameter magnitudes should track weight magnitudes.

        Uses 4 variables so that no single weight dominates (w_0 < sum of
        remaining), ensuring all variables have non-zero influence.
        If bit ordering is wrong, the Chow parameters will be permuted.
        """
        weights = [4.0, 3.0, 2.0, 1.0]
        f = bf.weighted_majority(weights)
        chow = chow_parameters(f)

        n = len(weights)
        assert len(chow) == n + 1

        # All degree-1 Chow parameters should be non-zero
        for i in range(n):
            assert chow[i + 1] != 0, f"chow[{i + 1}] is zero -- likely bit-ordering bug"

        # The variable with the largest weight should have the largest |Chow param|
        chow_magnitudes = [abs(chow[i + 1]) for i in range(n)]
        assert (
            chow_magnitudes[0] >= chow_magnitudes[1] >= chow_magnitudes[2] >= chow_magnitudes[3]
        ), (
            f"Chow magnitudes {chow_magnitudes} don't match weight ordering "
            f"{weights} -- likely bit-ordering bug"
        )

    def test_nassau_county_chow_ordering(self):
        """Nassau County board of supervisors: 6 districts, very asymmetric.

        The famous case study in weighted voting. District 0 has weight 31,
        districts 4-5 have weight 2. If Chow parameters are scrambled by
        bit ordering, the heaviest-weight district won't have the largest
        Chow parameter.
        """
        weights = [31, 31, 28, 21, 2, 2]
        f = bf.weighted_majority(weights)
        chow = chow_parameters(f)

        n = len(weights)
        assert len(chow) == n + 1

        # Districts 0,1 (weight 31) should have larger |Chow| than
        # districts 4,5 (weight 2)
        heavy = min(abs(chow[1]), abs(chow[2]))  # weights 31, 31
        light = max(abs(chow[5]), abs(chow[6]))  # weights 2, 2

        assert heavy > light, (
            f"|Chow| for heavy districts ({heavy}) should exceed " f"light districts ({light})"
        )

    def test_single_weight_dominant_is_almost_dictator(self):
        """When one weight dominates, the function is close to a dictator.

        If w_0 = 100 and w_1 = w_2 = 1, the function is nearly x_0.
        The Chow parameter for variable 0 should dominate.
        """
        weights = [100.0, 1.0, 1.0]
        f = bf.weighted_majority(weights)
        chow = chow_parameters(f)

        # Variable 0 should dominate
        assert abs(chow[1]) > 5 * abs(chow[2]), (
            f"|chow[1]|={abs(chow[1]):.4f} should dominate " f"|chow[2]|={abs(chow[2]):.4f}"
        )
        assert abs(chow[1]) > 5 * abs(chow[3]), (
            f"|chow[1]|={abs(chow[1]):.4f} should dominate " f"|chow[3]|={abs(chow[3]):.4f}"
        )

    def test_equal_weights_equal_chow(self):
        """Equal weights => equal Chow parameters (by symmetry)."""
        f = bf.weighted_majority([1, 1, 1, 1, 1])
        chow = chow_parameters(f)

        # All degree-1 Chow parameters should be equal (by symmetry)
        degree1 = [chow[i + 1] for i in range(5)]
        for i in range(1, 5):
            assert abs(degree1[i] - degree1[0]) < 1e-10, (
                f"chow[{i + 1}]={degree1[i]} != chow[1]={degree1[0]} " f"for equal-weight majority"
            )

    def test_chow_roundtrip_with_analyze_ltf(self):
        """The full analyze_ltf pipeline should produce consistent Chow params."""
        from boofun.analysis.ltf_analysis import analyze_ltf

        weights = [4.0, 3.0, 2.0, 1.0]
        f = bf.weighted_majority(weights)

        result = analyze_ltf(f)
        assert result.is_ltf
        assert result.chow_parameters is not None

        # Cross-check with standalone chow_parameters
        standalone_chow = chow_parameters(f)
        np.testing.assert_allclose(result.chow_parameters, standalone_chow, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
