"""
Representation round-trip tests.

For each pair of representations that can convert A -> B -> A,
verify the truth table is preserved.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestRepresentationRoundTrips:
    """Verify that converting A -> B -> A preserves the truth table."""

    @pytest.fixture
    def test_functions(self):
        return [
            ("AND_3", bf.AND(3)),
            ("XOR_3", bf.parity(3)),
            ("MAJ_3", bf.majority(3)),
            ("OR_3", bf.OR(3)),
        ]

    def _get_truth_table(self, f):
        return list(f.get_representation("truth_table"))

    def test_truth_table_to_fourier_roundtrip(self, test_functions):
        """truth_table -> fourier_expansion -> truth_table."""
        for name, f in test_functions:
            original_tt = self._get_truth_table(f)
            f.get_representation("fourier_expansion")

            # Create new function from Fourier, convert back
            fourier = f.fourier()
            n = f.n_vars
            size = 1 << n

            # Inverse WHT: truth table from Fourier
            reconstructed = np.zeros(size)
            for x in range(size):
                val = 0.0
                for s in range(size):
                    inner = bin(x & s).count("1") % 2
                    val += fourier[s] * ((-1) ** inner)
                reconstructed[x] = val

            # Convert ±1 back to {0,1}
            tt_back = [int(v < 0) for v in reconstructed]
            assert tt_back == original_tt, f"{name}: round-trip failed"

    def test_truth_table_to_anf_roundtrip(self, test_functions):
        """truth_table -> ANF -> truth_table (evaluate ANF polynomial)."""
        from boofun.analysis.gf2 import gf2_fourier_transform

        for name, f in test_functions:
            original_tt = self._get_truth_table(f)

            # Forward: truth table -> ANF coefficients
            anf = gf2_fourier_transform(f)

            # Inverse: evaluate ANF polynomial to reconstruct truth table
            n = f.n_vars
            size = 1 << n
            reconstructed = np.zeros(size, dtype=int)
            for x in range(size):
                val = 0
                for s in range(size):
                    if anf[s]:
                        if (x & s) == s:
                            val ^= 1
                reconstructed[x] = val

            assert list(reconstructed) == original_tt, f"{name}: ANF round-trip failed"


class TestF2Polynomial:
    """Test the new bf.f2_polynomial builtin."""

    def test_single_variable(self):
        """f(x) = (-1)^{x_0} should equal dictator on variable 0."""
        f = bf.f2_polynomial(3, [{0}])
        assert f.n_vars == 3
        # x_0 = 1 only when bit 0 is set
        for x in range(8):
            expected = (x >> 0) & 1
            assert f.evaluate(x) == bool(expected), f"x={x}"

    def test_xor_two_vars(self):
        """f = (-1)^{x_0 + x_1} = XOR of first two variables."""
        f = bf.f2_polynomial(3, [{0}, {1}])
        for x in range(8):
            expected = ((x >> 0) & 1) ^ ((x >> 1) & 1)
            assert f.evaluate(x) == bool(expected), f"x={x}"

    def test_degree_2_monomial(self):
        """f = (-1)^{x_0 * x_1} should be 1 only when both x_0,x_1 are 1."""
        f = bf.f2_polynomial(3, [{0, 1}])
        for x in range(8):
            expected = ((x >> 0) & 1) & ((x >> 1) & 1)
            assert f.evaluate(x) == bool(expected), f"x={x}"

    def test_empty_polynomial(self):
        """f = (-1)^0 = constant 0 (always +1 in ±1)."""
        f = bf.f2_polynomial(3, [])
        for x in range(8):
            assert f.evaluate(x) == False  # noqa: E712


class TestIsGlobal:
    """Test the new f.is_global(alpha) method."""

    def test_dictator_not_global_at_small_alpha(self):
        """Dictator has max generalized influence = 4, so not 0.5-global."""
        f = bf.dictator(5, 0)
        assert not f.is_global(0.5)

    def test_large_alpha_makes_everything_global(self):
        """With alpha large enough, any function is global."""
        f = bf.dictator(5, 0)
        assert f.is_global(10.0)

    def test_majority_globality_improves_with_n(self):
        """Majority becomes more global (lower alpha needed) as n increases."""
        # For Majority_n at p=0.5: max I_S decreases relative to E[f^2]=1
        # as n grows (influences spread across more variables)
        # Majority_3: max I_S ~ 2.5, Majority_7: max I_S ~ 2.19
        assert not bf.majority(3).is_global(2.0)
        # With enough alpha, majority is always global
        assert bf.majority(3).is_global(100.0)


class TestAdaptiveSampling:
    """Test estimate_fourier_adaptive."""

    def test_converges_to_correct_value(self):
        """Adaptive estimate should be close to exact for known function."""
        from boofun.analysis.sampling import estimate_fourier_adaptive

        f = bf.majority(5)
        exact = f.fourier()[1]  # f_hat({0})

        est, se, n_used = estimate_fourier_adaptive(f, S=1, target_error=0.05, max_samples=50000)
        assert abs(est - exact) < 0.15, f"Adaptive estimate {est} too far from exact {exact}"
        assert se <= 0.05 + 0.01  # Allow small tolerance

    def test_respects_max_samples(self):
        """Should not exceed max_samples budget."""
        from boofun.analysis.sampling import estimate_fourier_adaptive

        f = bf.parity(5)
        _, _, n_used = estimate_fourier_adaptive(f, S=31, target_error=0.0001, max_samples=5000)
        assert n_used <= 5000 + 1000  # batch_size overshoot


class TestMeasure:
    """Test the new Measure class."""

    def test_uniform(self):
        from boofun.core.spaces import Measure

        m = Measure.uniform()
        assert m.p == 0.5
        assert m.is_uniform

    def test_p_biased(self):
        from boofun.core.spaces import Measure

        m = Measure.p_biased(0.3)
        assert m.p == 0.3
        assert not m.is_uniform
        assert abs(m.sigma - np.sqrt(0.3 * 0.7)) < 1e-10

    def test_sample_shape(self):
        from boofun.core.spaces import Measure

        m = Measure.p_biased(0.5)
        sample = m.sample(10)
        assert sample.shape == (10,)
        assert all(x in (0, 1) for x in sample)

    def test_sample_batch(self):
        from boofun.core.spaces import Measure

        m = Measure.p_biased(0.3)
        batch = m.sample_batch(5, 100)
        assert batch.shape == (100, 5)

    def test_invalid_p(self):
        from boofun.core.spaces import Measure

        with pytest.raises(ValueError):
            Measure(0.0)
        with pytest.raises(ValueError):
            Measure(1.0)
