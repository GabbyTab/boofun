import sys

sys.path.insert(0, "src")
"""
Internal cross-validation tests for the BooFun library.

These tests verify that different computation paths for the same mathematical
quantity produce consistent results. This is the most powerful class of tests
because any inconsistency reveals either a bug or a convention mismatch.

Categories:
1. Influence consistency (6+ computation paths)
2. Total influence / average sensitivity consistency (9+ paths)
3. Sensitivity consistency (5+ paths)
4. Block sensitivity consistency
5. Fourier coefficient consistency
6. Degree consistency
7. Noise stability consistency
8. Expectation / bias consistency
9. Decision tree depth consistency
10. Certificate complexity consistency
11. Fourier sparsity consistency
12. Variance consistency
13. Sensitive coordinates consistency
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis import SpectralAnalyzer

# ---------------------------------------------------------------------------
# Test fixtures: standard functions with known properties
# ---------------------------------------------------------------------------

FUNCTIONS_3 = {
    "parity3": bf.parity(3),
    "majority3": bf.majority(3),
    "and3": bf.AND(3),
    "or3": bf.OR(3),
    "dictator3_0": bf.dictator(3, i=0),
}

FUNCTIONS_4 = {
    "parity4": bf.parity(4),
    "and4": bf.AND(4),
}

ALL_FUNCTIONS = {**FUNCTIONS_3, **FUNCTIONS_4}

# Tolerance for floating-point comparisons
TOL = 1e-10
SAMPLING_TOL = 0.15  # Looser tolerance for Monte Carlo estimates


# ===========================================================================
# 1. INFLUENCE CONSISTENCY
# ===========================================================================


class TestInfluenceConsistency:
    """
    Verify that per-variable influence Inf_i[f] is consistent across paths:
    - SpectralAnalyzer.influences()
    - BooleanFunction.influences()
    - Fourier identity: Inf_i = sum_{S containing i} f_hat(S)^2
    - p_biased_influence at p=0.5
    - annealed_influence at rho=1.0
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_fourier_identity(self, name):
        """SpectralAnalyzer influences match Fourier coefficient identity."""
        f = FUNCTIONS_3[name]
        n = f.n_vars
        sa = SpectralAnalyzer(f)
        influences = sa.influences()
        fourier = np.asarray(f.fourier())

        for i in range(n):
            # Fourier identity: Inf_i = sum_{S: i in S} f_hat(S)^2
            inf_fourier = sum(fourier[S] ** 2 for S in range(1 << n) if (S >> i) & 1)
            assert (
                abs(influences[i] - inf_fourier) < TOL
            ), f"{name}: Inf_{i} mismatch: SA={influences[i]}, Fourier={inf_fourier}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_bf_influences(self, name):
        """BooleanFunction.influences() matches SpectralAnalyzer.influences()."""
        f = FUNCTIONS_3[name]
        sa = SpectralAnalyzer(f)
        sa_infs = sa.influences()
        bf_infs = f.influences()

        for i in range(f.n_vars):
            assert (
                abs(sa_infs[i] - bf_infs[i]) < TOL
            ), f"{name}: Inf_{i} mismatch: SA={sa_infs[i]}, BF={bf_infs[i]}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_p_biased_at_half(self, name):
        """p-biased influence at p=0.5 matches standard influence."""
        from boofun.analysis.p_biased import p_biased_influence

        f = FUNCTIONS_3[name]
        sa = SpectralAnalyzer(f)
        influences = sa.influences()

        for i in range(f.n_vars):
            inf_pb = p_biased_influence(f, i, p=0.5)
            assert (
                abs(influences[i] - inf_pb) < TOL
            ), f"{name}: Inf_{i} mismatch: SA={influences[i]}, p-biased={inf_pb}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_annealed_influence_at_rho_one(self, name):
        """Annealed influence at rho=1.0 equals standard influence."""
        from boofun.analysis.fourier import annealed_influence

        f = FUNCTIONS_3[name]
        sa = SpectralAnalyzer(f)
        influences = sa.influences()

        for i in range(f.n_vars):
            inf_ann = annealed_influence(f, i, rho=1.0)
            assert (
                abs(influences[i] - inf_ann) < TOL
            ), f"{name}: Inf_{i} mismatch: SA={influences[i]}, annealed={inf_ann}"


# ===========================================================================
# 2. TOTAL INFLUENCE / AVERAGE SENSITIVITY CONSISTENCY
# ===========================================================================


class TestTotalInfluenceConsistency:
    """
    Verify that total influence I[f] = sum_i Inf_i[f] = E[s(f,x)] is
    consistent across all computation paths.
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_sensitivity_average(self, name):
        """SpectralAnalyzer total influence matches sensitivity.average_sensitivity."""
        from boofun.analysis.sensitivity import average_sensitivity

        f = FUNCTIONS_3[name]
        ti_spectral = SpectralAnalyzer(f).total_influence()
        ti_sens = average_sensitivity(f)
        assert (
            abs(ti_spectral - ti_sens) < TOL
        ), f"{name}: TI mismatch: spectral={ti_spectral}, sensitivity={ti_sens}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_complexity_average(self, name):
        """SpectralAnalyzer total influence matches complexity.average_sensitivity."""
        from boofun.analysis.complexity import average_sensitivity

        f = FUNCTIONS_3[name]
        ti_spectral = SpectralAnalyzer(f).total_influence()
        ti_comp = average_sensitivity(f)
        assert (
            abs(ti_spectral - ti_comp) < TOL
        ), f"{name}: TI mismatch: spectral={ti_spectral}, complexity={ti_comp}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_huang_average(self, name):
        """SpectralAnalyzer total influence matches huang.average_sensitivity."""
        from boofun.analysis.huang import average_sensitivity

        f = FUNCTIONS_3[name]
        ti_spectral = SpectralAnalyzer(f).total_influence()
        ti_huang = average_sensitivity(f)
        assert (
            abs(ti_spectral - ti_huang) < TOL
        ), f"{name}: TI mismatch: spectral={ti_spectral}, huang={ti_huang}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_fourier_identity(self, name):
        """Total influence from Fourier: sum_S |S| * f_hat(S)^2."""
        f = FUNCTIONS_3[name]
        n = f.n_vars
        ti_spectral = SpectralAnalyzer(f).total_influence()
        fourier = np.asarray(f.fourier())
        ti_fourier = sum(bin(S).count("1") * fourier[S] ** 2 for S in range(1 << n))
        assert (
            abs(ti_spectral - ti_fourier) < TOL
        ), f"{name}: TI mismatch: spectral={ti_spectral}, Fourier={ti_fourier}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_spectral_vs_sum_of_influences(self, name):
        """Total influence equals sum of individual influences."""
        f = FUNCTIONS_3[name]
        sa = SpectralAnalyzer(f)
        ti = sa.total_influence()
        inf_sum = sum(sa.influences())
        assert abs(ti - inf_sum) < TOL, f"{name}: TI={ti} != sum(Inf_i)={inf_sum}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_p_biased_total_at_half(self, name):
        """p-biased total influence at p=0.5 matches standard."""
        from boofun.analysis.p_biased import p_biased_total_influence

        f = FUNCTIONS_3[name]
        ti_spectral = SpectralAnalyzer(f).total_influence()
        ti_pb = p_biased_total_influence(f, p=0.5)
        assert (
            abs(ti_spectral - ti_pb) < TOL
        ), f"{name}: TI mismatch: spectral={ti_spectral}, p-biased={ti_pb}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_p_biased_fourier_total_at_half(self, name):
        """p-biased total influence (Fourier formula) at p=0.5 matches standard."""
        from boofun.analysis.p_biased import p_biased_total_influence_fourier

        f = FUNCTIONS_3[name]
        ti_spectral = SpectralAnalyzer(f).total_influence()
        ti_pb_fourier = p_biased_total_influence_fourier(f, p=0.5)
        assert (
            abs(ti_spectral - ti_pb_fourier) < TOL
        ), f"{name}: TI mismatch: spectral={ti_spectral}, p-biased-fourier={ti_pb_fourier}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bf_method_matches_spectral(self, name):
        """BooleanFunction.total_influence() matches SpectralAnalyzer."""
        f = FUNCTIONS_3[name]
        ti_bf = f.total_influence()
        ti_sa = SpectralAnalyzer(f).total_influence()
        assert abs(ti_bf - ti_sa) < TOL


# ===========================================================================
# 3. SENSITIVITY CONSISTENCY
# ===========================================================================


class TestSensitivityConsistency:
    """
    Verify sensitivity is consistent across:
    - sensitivity.sensitivity_at
    - complexity.sensitivity
    - huang.sensitivity_at
    - block_sensitivity.sensitive_coordinates (via len)
    - BooleanFunction.sensitivity_at (if present)
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_sensitivity_at_all_modules(self, name):
        """All sensitivity_at implementations agree for every input."""
        from boofun.analysis.complexity import sensitivity as sens_complexity
        from boofun.analysis.huang import sensitivity_at as sens_huang
        from boofun.analysis.sensitivity import sensitivity_at as sens_sensitivity

        f = FUNCTIONS_3[name]
        for x in range(1 << f.n_vars):
            s_sens = sens_sensitivity(f, x)
            s_comp = sens_complexity(f, x)
            s_huang = sens_huang(f, x)

            assert s_sens == s_comp, f"{name} at x={x}: sensitivity={s_sens}, complexity={s_comp}"
            assert s_sens == s_huang, f"{name} at x={x}: sensitivity={s_sens}, huang={s_huang}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_max_sensitivity_all_modules(self, name):
        """All max_sensitivity implementations agree."""
        from boofun.analysis.complexity import max_sensitivity as ms_complexity
        from boofun.analysis.huang import max_sensitivity as ms_huang
        from boofun.analysis.sensitivity import max_sensitivity as ms_sensitivity

        f = FUNCTIONS_3[name]
        s1 = ms_sensitivity(f)
        s2 = ms_complexity(f)
        s3 = ms_huang(f)

        assert s1 == s2 == s3, f"{name}: max_sensitivity disagrees: {s1}, {s2}, {s3}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_sensitive_coordinates_consistency(self, name):
        """sensitivity.sensitive_coordinates matches block_sensitivity.sensitive_coordinates."""
        from boofun.analysis.block_sensitivity import sensitive_coordinates as sc_bs
        from boofun.analysis.sensitivity import sensitive_coordinates as sc_sens

        f = FUNCTIONS_3[name]
        for x in range(1 << f.n_vars):
            coords_sens = sorted(sc_sens(f, x))
            coords_bs = sorted(sc_bs(f, x))
            assert coords_sens == coords_bs, f"{name} at x={x}: sens={coords_sens}, bs={coords_bs}"


# ===========================================================================
# 4. BLOCK SENSITIVITY CONSISTENCY
# ===========================================================================


class TestBlockSensitivityConsistency:
    """
    Verify block sensitivity is consistent between:
    - block_sensitivity.max_block_sensitivity
    - huang.block_sensitivity
    And that bs(f) >= s(f) always holds.
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bs_geq_sensitivity(self, name):
        """Block sensitivity >= max sensitivity (mathematical fact)."""
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.sensitivity import max_sensitivity

        f = FUNCTIONS_3[name]
        bs = max_block_sensitivity(f)
        s = max_sensitivity(f)
        assert bs >= s, f"{name}: bs={bs} < s={s}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_block_sensitivity_vs_huang(self, name):
        """block_sensitivity.max_block_sensitivity == huang.block_sensitivity."""
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        from boofun.analysis.huang import block_sensitivity

        f = FUNCTIONS_3[name]
        bs_exact = max_block_sensitivity(f)
        bs_huang = block_sensitivity(f)

        assert bs_exact == bs_huang, f"{name}: exact bs={bs_exact} != huang bs={bs_huang}"


# ===========================================================================
# 5. FOURIER COEFFICIENT CONSISTENCY
# ===========================================================================


class TestFourierCoefficientConsistency:
    """
    Verify Fourier coefficients are consistent across:
    - BooleanFunction.fourier()
    - BooleanFunction.spectrum() (alias)
    - SpectralAnalyzer.fourier_expansion()
    - Walsh transform (cryptographic module) / 2^n
    - p_biased_fourier_coefficient at p=0.5
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_fourier_vs_spectrum_alias(self, name):
        """f.fourier() and f.spectrum() return the same array."""
        f = FUNCTIONS_3[name]
        fourier = np.asarray(f.fourier())
        spectrum = np.asarray(f.spectrum())
        assert np.allclose(fourier, spectrum), f"{name}: fourier() != spectrum()"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_fourier_vs_walsh(self, name):
        """Walsh transform / 2^n matches Fourier coefficients (with sign convention)."""
        from boofun.analysis.cryptographic import walsh_transform

        f = FUNCTIONS_3[name]
        n = f.n_vars
        fourier = np.asarray(f.fourier())
        walsh = np.asarray(walsh_transform(f))

        # Walsh transform W_f(a) = sum_x (-1)^{f(x) + a.x}
        # vs Fourier: f_hat(S) = (1/2^n) sum_x f(x)*chi_S(x)
        # Relationship depends on convention. Typically walsh / 2^n = f_hat
        walsh_normalized = walsh / (1 << n)
        assert np.allclose(fourier, walsh_normalized, atol=1e-10), f"{name}: Fourier != Walsh/2^n"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_parseval_identity(self, name):
        """Parseval: sum f_hat(S)^2 = 1 for +-1 valued functions."""
        f = FUNCTIONS_3[name]
        fourier = np.asarray(f.fourier())
        total = np.sum(fourier**2)
        assert abs(total - 1.0) < TOL, f"{name}: Parseval violation: sum f_hat^2 = {total}"


# ===========================================================================
# 6. DEGREE CONSISTENCY
# ===========================================================================


class TestDegreeConsistency:
    """
    Verify degree consistency across computation paths.

    Important: Fourier degree and GF(2)/algebraic degree are DIFFERENT quantities!
    - Fourier degree: max |S| with f_hat(S) != 0 (real multilinear polynomial)
    - GF(2) degree: max |S| with ANF coefficient c_S != 0 (polynomial over GF(2))
    Example: Parity(3) has Fourier degree 3 but GF(2) degree 1 (it's linear over GF(2)).

    What SHOULD be consistent:
    - f.degree() == fourier_degree(f) (both use Fourier)
    - algebraic_degree(f) == gf2_degree(f) (both use ANF)
    - GF(2) degree <= Fourier degree (always)
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bf_degree_vs_fourier_degree(self, name):
        """BooleanFunction.degree() matches fourier_degree."""
        from boofun.analysis.fourier import fourier_degree

        f = FUNCTIONS_3[name]
        assert f.degree() == fourier_degree(f)

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_algebraic_degree_vs_gf2(self, name):
        """cryptographic.algebraic_degree matches gf2_degree."""
        from boofun.analysis.cryptographic import algebraic_degree
        from boofun.analysis.gf2 import gf2_degree

        f = FUNCTIONS_3[name]
        assert algebraic_degree(f) == gf2_degree(f)

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_gf2_degree_leq_fourier_degree(self, name):
        """GF(2) degree <= Fourier degree (always true)."""
        from boofun.analysis.fourier import fourier_degree
        from boofun.analysis.gf2 import gf2_degree

        f = FUNCTIONS_3[name]
        d_gf2 = gf2_degree(f)
        d_fourier = fourier_degree(f)
        assert d_gf2 <= d_fourier, f"{name}: GF2 degree={d_gf2} > Fourier degree={d_fourier}"

    def test_fourier_vs_gf2_known_difference(self):
        """Parity(n) has Fourier degree n but GF(2) degree 1 (it's linear over GF(2))."""
        from boofun.analysis.fourier import fourier_degree
        from boofun.analysis.gf2 import gf2_degree

        f = bf.parity(3)
        assert fourier_degree(f) == 3
        assert gf2_degree(f) == 1


# ===========================================================================
# 7. NOISE STABILITY CONSISTENCY
# ===========================================================================


class TestNoiseStabilityConsistency:
    """
    Verify noise stability is consistent across:
    - BooleanFunction.noise_stability(rho)
    - SpectralAnalyzer.noise_stability(rho)
    - p_biased_noise_stability at p=0.5
    - Direct Fourier computation: sum_S f_hat(S)^2 * rho^|S|
    """

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.9, 1.0])
    def test_spectral_vs_fourier_identity(self, rho):
        """Noise stability matches direct Fourier formula."""
        f = bf.majority(3)
        n = f.n_vars
        ns_sa = SpectralAnalyzer(f).noise_stability(rho)
        fourier = np.asarray(f.fourier())

        ns_fourier = sum(fourier[S] ** 2 * rho ** bin(S).count("1") for S in range(1 << n))
        assert abs(ns_sa - ns_fourier) < TOL, f"rho={rho}: SA={ns_sa}, Fourier={ns_fourier}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bf_method_vs_spectral(self, name):
        """BooleanFunction.noise_stability matches SpectralAnalyzer."""
        f = FUNCTIONS_3[name]
        rho = 0.5
        ns_bf = f.noise_stability(rho)
        ns_sa = SpectralAnalyzer(f).noise_stability(rho)
        assert abs(ns_bf - ns_sa) < TOL

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_p_biased_at_half(self, name):
        """p-biased noise stability at p=0.5 matches standard."""
        from boofun.analysis.p_biased import p_biased_noise_stability

        f = FUNCTIONS_3[name]
        rho = 0.5
        ns_sa = SpectralAnalyzer(f).noise_stability(rho)
        ns_pb = p_biased_noise_stability(f, rho=rho, p=0.5)
        assert abs(ns_sa - ns_pb) < TOL, f"{name}: SA={ns_sa}, p-biased={ns_pb}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_rho_one_is_one(self, name):
        """At rho=1, noise stability = sum f_hat(S)^2 = 1 (by Parseval)."""
        f = FUNCTIONS_3[name]
        ns = SpectralAnalyzer(f).noise_stability(1.0)
        assert abs(ns - 1.0) < TOL

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_rho_zero_is_mean_squared(self, name):
        """At rho=0, noise stability = f_hat(emptyset)^2 = E[f]^2."""
        f = FUNCTIONS_3[name]
        ns = SpectralAnalyzer(f).noise_stability(0.0)
        expected = float(f.fourier()[0]) ** 2
        assert abs(ns - expected) < TOL


# ===========================================================================
# 8. EXPECTATION / BIAS CONSISTENCY
# ===========================================================================


class TestExpectationBiasConsistency:
    """
    Verify expectation/bias consistency:
    - f.bias() (returns E[f] in +-1, i.e., 1 - 2*Pr[f=1])
    - f.fourier()[0] = E[f] in +-1 representation
    - basic_properties.bias(f) = Pr[f=1]
    - basic_properties.weight(f) / 2^n = Pr[f=1]
    - p_biased_expectation(f, 0.5) = E[f] in +-1
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bias_vs_fourier_constant_term(self, name):
        """f.bias() equals f.fourier()[0]."""
        f = FUNCTIONS_3[name]
        bias = f.bias()
        f_hat_0 = float(f.fourier()[0])
        assert abs(bias - f_hat_0) < TOL, f"{name}: bias={bias}, f_hat(0)={f_hat_0}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bias_vs_basic_properties(self, name):
        """f.bias() = 1 - 2 * basic_properties.bias(f)."""
        from boofun.analysis.basic_properties import bias as bp_bias

        f = FUNCTIONS_3[name]
        bf_bias = f.bias()
        bp_val = bp_bias(f)
        # BooleanFunction.bias() = E[f] in +-1 = 1 - 2*Pr[f=1]
        # basic_properties.bias(f) = Pr[f=1]
        expected = 1 - 2 * bp_val
        assert (
            abs(bf_bias - expected) < TOL
        ), f"{name}: bf.bias={bf_bias}, expected 1-2*{bp_val}={expected}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_weight_vs_bias(self, name):
        """basic_properties.weight(f) / 2^n = basic_properties.bias(f)."""
        from boofun.analysis.basic_properties import bias as bp_bias
        from boofun.analysis.basic_properties import weight

        f = FUNCTIONS_3[name]
        n = f.n_vars
        w = weight(f)
        b = bp_bias(f)
        assert abs(w / (1 << n) - b) < TOL

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_p_biased_expectation_at_half(self, name):
        """p_biased_expectation at p=0.5 matches f.bias()."""
        from boofun.analysis.p_biased import p_biased_expectation

        f = FUNCTIONS_3[name]
        exp_pb = p_biased_expectation(f, p=0.5)
        exp_bias = f.bias()
        assert abs(exp_pb - exp_bias) < TOL, f"{name}: p-biased={exp_pb}, bias={exp_bias}"


# ===========================================================================
# 9. DECISION TREE DEPTH CONSISTENCY
# ===========================================================================


class TestDecisionTreeDepthConsistency:
    """
    Verify decision tree depth is consistent across:
    - complexity.decision_tree_depth
    - complexity.D
    - decision_trees.decision_tree_depth_dp
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_complexity_vs_decision_trees(self, name):
        """complexity.decision_tree_depth matches decision_trees.decision_tree_depth_dp."""
        from boofun.analysis.complexity import decision_tree_depth
        from boofun.analysis.decision_trees import decision_tree_depth_dp

        f = FUNCTIONS_3[name]
        d1 = decision_tree_depth(f)
        d2 = decision_tree_depth_dp(f)
        assert d1 == d2, f"{name}: complexity.D={d1}, decision_trees.dp={d2}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_d_alias(self, name):
        """complexity.D is an alias for decision_tree_depth."""
        from boofun.analysis.complexity import D, decision_tree_depth

        f = FUNCTIONS_3[name]
        assert D(f) == decision_tree_depth(f)


# ===========================================================================
# 10. CERTIFICATE COMPLEXITY CONSISTENCY
# ===========================================================================


class TestCertificateComplexityConsistency:
    """
    Verify certificate complexity across:
    - complexity.certificate_complexity
    - certificates.certificate
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_complexity_vs_certificates(self, name):
        """Both modules agree on certificate size for every input."""
        from boofun.analysis.certificates import certificate
        from boofun.analysis.complexity import certificate_complexity

        f = FUNCTIONS_3[name]
        for x in range(1 << f.n_vars):
            size_comp, _ = certificate_complexity(f, x)
            size_cert, _ = certificate(f, x)
            assert (
                size_comp == size_cert
            ), f"{name} at x={x}: complexity={size_comp}, certificates={size_cert}"


# ===========================================================================
# 11. FOURIER SPARSITY CONSISTENCY
# ===========================================================================


class TestFourierSparsityConsistency:
    """
    Verify Fourier sparsity consistency across:
    - fourier.fourier_sparsity
    - sparsity.fourier_sparsity
    - BooleanFunction.sparsity() (if present)
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_fourier_vs_sparsity_module(self, name):
        """fourier.fourier_sparsity matches sparsity.fourier_sparsity."""
        from boofun.analysis.fourier import fourier_sparsity as fs_fourier
        from boofun.analysis.sparsity import fourier_sparsity as fs_sparsity

        f = FUNCTIONS_3[name]
        s1 = fs_fourier(f)
        s2 = fs_sparsity(f)
        assert s1 == s2, f"{name}: fourier.sparsity={s1}, sparsity.sparsity={s2}"


# ===========================================================================
# 12. VARIANCE CONSISTENCY
# ===========================================================================


class TestVarianceConsistency:
    """
    Verify variance computation across:
    - BooleanFunction.variance()
    - Fourier identity: Var[f] = sum_{S != 0} f_hat(S)^2
    - 1 - E[f]^2 (for Boolean-valued +-1 functions)
    - p_biased_variance at p=0.5
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bf_variance_vs_fourier(self, name):
        """Variance matches sum of non-constant Fourier coefficients squared."""
        f = FUNCTIONS_3[name]
        var_bf = f.variance()
        fourier = np.asarray(f.fourier())
        var_fourier = np.sum(fourier[1:] ** 2)
        assert abs(var_bf - var_fourier) < TOL

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_variance_plus_mean_squared_is_one(self, name):
        """For +-1 functions: Var[f] + E[f]^2 = 1."""
        f = FUNCTIONS_3[name]
        var = f.variance()
        mean_sq = float(f.fourier()[0]) ** 2
        assert abs(var + mean_sq - 1.0) < TOL, f"{name}: Var + E^2 = {var + mean_sq}, expected 1"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_p_biased_variance_at_half(self, name):
        """p_biased_variance at p=0.5 matches standard variance."""
        from boofun.analysis.p_biased import p_biased_variance

        f = FUNCTIONS_3[name]
        var_bf = f.variance()
        var_pb = p_biased_variance(f, p=0.5)
        assert abs(var_bf - var_pb) < TOL, f"{name}: BF.variance={var_bf}, p-biased={var_pb}"


# ===========================================================================
# 13. IS_BALANCED CONSISTENCY
# ===========================================================================


class TestIsBalancedConsistency:
    """
    Verify is_balanced across:
    - BooleanFunction.is_balanced()
    - basic_properties.is_balanced()
    - PropertyTester.balanced_test()
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_bf_vs_basic_properties(self, name):
        """BooleanFunction.is_balanced matches basic_properties.is_balanced."""
        from boofun.analysis.basic_properties import is_balanced

        f = FUNCTIONS_3[name]
        assert f.is_balanced() == is_balanced(
            f
        ), f"{name}: BF={f.is_balanced()}, basic_props={is_balanced(f)}"

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_balanced_iff_zero_bias(self, name):
        """Balanced iff E[f] = 0 (f_hat(0) = 0)."""
        f = FUNCTIONS_3[name]
        fourier_0 = abs(float(f.fourier()[0]))
        is_bal = f.is_balanced()
        assert is_bal == (fourier_0 < TOL), f"{name}: is_balanced={is_bal}, f_hat(0)={fourier_0}"


# ===========================================================================
# 14. CROSS-MODULE KNOWN VALUES
# ===========================================================================


class TestKnownValues:
    """
    Verify known mathematical facts for specific functions.
    These serve as ground truth across all computation paths.
    """

    def test_parity3_total_influence(self):
        """Parity(3): I[f] = 3."""
        f = bf.parity(3)
        assert abs(SpectralAnalyzer(f).total_influence() - 3.0) < TOL

    def test_parity3_degree(self):
        """Parity(3): deg = 3."""
        f = bf.parity(3)
        assert f.degree() == 3

    def test_parity3_all_influences_one(self):
        """Parity(3): Inf_i = 1 for all i."""
        f = bf.parity(3)
        infs = SpectralAnalyzer(f).influences()
        for inf in infs:
            assert abs(inf - 1.0) < TOL

    def test_and3_total_influence(self):
        """AND(3): I[f] = 3 * (1/4) = 3/4."""
        f = bf.AND(3)
        ti = SpectralAnalyzer(f).total_influence()
        assert abs(ti - 0.75) < TOL

    def test_dictator_influence_one(self):
        """Dictator: exactly one variable has influence 1, rest 0."""
        f = bf.dictator(3, i=1)
        infs = SpectralAnalyzer(f).influences()
        assert abs(infs[1] - 1.0) < TOL
        assert abs(infs[0]) < TOL
        assert abs(infs[2]) < TOL

    def test_majority3_balanced(self):
        """MAJ(3) is balanced: Pr[f=1] = 1/2."""
        f = bf.majority(3)
        assert f.is_balanced()

    def test_and3_not_balanced(self):
        """AND(3) is not balanced: Pr[f=1] = 1/8."""
        f = bf.AND(3)
        assert not f.is_balanced()

    def test_parity_noise_stability_at_half(self):
        """Parity(3) noise stability at rho=0.5: rho^n = 0.125."""
        f = bf.parity(3)
        ns = SpectralAnalyzer(f).noise_stability(0.5)
        # Parity has all weight at level n, so NS = rho^n
        assert abs(ns - 0.5**3) < TOL

    @pytest.mark.parametrize("n", [3, 5])
    def test_majority_symmetric_influences(self, n):
        """MAJ(n): all variables have equal influence (by symmetry)."""
        f = bf.majority(n)
        infs = SpectralAnalyzer(f).influences()
        for i in range(1, n):
            assert abs(infs[i] - infs[0]) < TOL


# ===========================================================================
# 15. ANF / GF2 CONSISTENCY
# ===========================================================================


class TestANFConsistency:
    """
    Verify ANF / GF(2) representations are consistent:
    - gf2.gf2_fourier_transform matches cryptographic.algebraic_normal_form
    """

    @pytest.mark.parametrize("name", FUNCTIONS_3.keys())
    def test_gf2_vs_crypto_anf(self, name):
        """GF(2) transform matches cryptographic ANF."""
        from boofun.analysis.cryptographic import algebraic_normal_form
        from boofun.analysis.gf2 import gf2_fourier_transform

        f = FUNCTIONS_3[name]
        gf2_coeffs = gf2_fourier_transform(f)
        anf_coeffs = algebraic_normal_form(f)

        assert np.array_equal(
            np.asarray(gf2_coeffs), np.asarray(anf_coeffs)
        ), f"{name}: GF2 transform != cryptographic ANF"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
