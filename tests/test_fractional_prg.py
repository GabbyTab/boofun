"""
Tests for Fourier tail computations and multilinear extension properties
used in the fractional PRG notebook (CHLT ITCS 2019).

These verify mathematical invariants from the paper and cross-validate
the boofun library's Fourier and Gaussian analysis modules.
"""

import sys
from math import comb

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.analysis.fourier import fourier_level_lp_norm, fourier_tail_profile
from boofun.analysis.gaussian import multilinear_extension


class TestMultilinearExtension:
    """Verify multilinear extension properties from the PRG framework."""

    def test_origin_equals_expectation(self):
        """f_tilde(0) = f_hat(emptyset) = E[f] for +-1 valued f.

        This is the key identity: the origin gives the expectation.
        """
        for name, f in [
            ("majority(5)", bf.majority(5)),
            ("AND(4)", bf.AND(4)),
            ("parity(3)", bf.parity(3)),
        ]:
            p = multilinear_extension(f)
            origin_val = p(np.zeros(f.n_vars))
            fourier_empty = f.fourier()[0]
            assert (
                abs(origin_val - fourier_empty) < 1e-10
            ), f"{name}: f_tilde(0)={origin_val}, f_hat(empty)={fourier_empty}"

    def test_agrees_on_vertices(self):
        """f_tilde agrees with f on {-1,+1}^n vertices."""
        f = bf.majority(3)
        p = multilinear_extension(f)
        n = f.n_vars
        tt = list(f.get_representation("truth_table"))

        for idx in range(2**n):
            # Build +-1 vertex from index
            vertex = np.array([1.0 - 2.0 * ((idx >> j) & 1) for j in range(n)])
            pm_val = 1.0 - 2.0 * tt[idx]  # Convert {0,1} to {+1,-1}
            ext_val = p(vertex)
            assert (
                abs(ext_val - pm_val) < 1e-10
            ), f"Vertex {idx}: extension={ext_val}, boolean={pm_val}"

    def test_bounded_on_cube(self):
        """f_tilde maps [-1,1]^n to [-1,1] for +-1 valued f.

        Lemma 12 in the paper: |f(x) - f(y)| <= n * ||x-y||_inf.
        Combined with |f(0)| <= 1, this gives |f(x)| <= 1 + n on [-1,1]^n.
        But for +-1 valued f, the tighter bound f_tilde: [-1,1]^n -> [-1,1] holds.
        """
        f = bf.majority(5)
        p = multilinear_extension(f)
        n = f.n_vars

        np.random.seed(42)
        for _ in range(100):
            x = np.random.uniform(-1, 1, n)
            val = p(x)
            assert -1 - 1e-10 <= val <= 1 + 1e-10, f"Extension out of [-1,1]: f_tilde({x}) = {val}"


class TestFourierTails:
    """Verify Fourier tail computations and theoretical bounds."""

    def test_L1k_parseval_relationship(self):
        """L_{1,k}^2 >= W^k (by Cauchy-Schwarz: (sum |a_i|)^2 >= sum a_i^2 when only 1 term).

        More precisely: L_{1,k}^2 <= C(n,k) * W^k (by Cauchy-Schwarz).
        """
        f = bf.majority(5)
        fourier = f.fourier()
        n = f.n_vars

        for k in range(n + 1):
            l1k = fourier_level_lp_norm(f, k, p=1)
            wk = sum(fourier[s] ** 2 for s in range(len(fourier)) if bin(s).count("1") == k)
            nk = comb(n, k)  # Number of subsets of size k
            # Cauchy-Schwarz: L_{1,k}^2 <= C(n,k) * W^k
            assert l1k**2 <= nk * wk + 1e-10, (
                f"Cauchy-Schwarz violated at k={k}: " f"L1k^2={l1k**2}, C(n,k)*Wk={nk * wk}"
            )

    def test_parity_L1_profile(self):
        """Parity_n: all Fourier weight at degree n, so L_{1,n}=1, L_{1,k}=0 for k<n."""
        for n in [3, 5]:
            f = bf.parity(n)
            for k in range(n):
                assert fourier_level_lp_norm(f, k) < 1e-10, f"Parity({n}): L_{{1,{k}}} should be 0"
            assert (
                abs(fourier_level_lp_norm(f, n) - 1.0) < 1e-10
            ), f"Parity({n}): L_{{1,{n}}} should be 1"

    def test_dictator_L1_profile(self):
        """Dictator: f_hat({i})=1, all others 0. So L_{1,1}=1, L_{1,k}=0 for k!=1."""
        f = bf.dictator(5, 0)
        assert abs(fourier_level_lp_norm(f, 0)) < 1e-10
        assert abs(fourier_level_lp_norm(f, 1) - 1.0) < 1e-10
        for k in range(2, 6):
            assert fourier_level_lp_norm(f, k) < 1e-10


class TestTheoremFive:
    """Verify Theorem 5: L_{1,1}(f) <= 4^d for degree-d F2-polynomials.

    Reference: CHLT ITCS 2019, Theorem 13.
    """

    @staticmethod
    def create_f2_polynomial(n, monomials):
        """Create f(x) = (-1)^{p(x)} where p = sum of monomials over F2."""

        def truth_table_entry(idx):
            bits = [(idx >> j) & 1 for j in range(n)]
            p_val = 0
            for mon in monomials:
                prod = 1
                for i in mon:
                    prod *= bits[i]
                p_val ^= prod
            return 0 if p_val == 0 else 1

        tt = [truth_table_entry(idx) for idx in range(2**n)]
        return bf.create(tt)

    @pytest.mark.parametrize(
        "name,n,monomials,degree",
        [
            ("x_0", 5, [{0}], 1),
            ("x_0+x_1+x_2", 5, [{0}, {1}, {2}], 1),
            ("x_0*x_1", 5, [{0, 1}], 2),
            ("x_0*x_1+x_2*x_3", 5, [{0, 1}, {2, 3}], 2),
            ("x_0*x_1*x_2", 5, [{0, 1, 2}], 3),
            ("x_0*x_1*x_2+x_3*x_4*x_5", 6, [{0, 1, 2}, {3, 4, 5}], 3),
        ],
    )
    def test_theorem_5_bound(self, name, n, monomials, degree):
        """L_{1,1}(f) <= 4^d for degree-d F2-polynomial f."""
        f = self.create_f2_polynomial(n, monomials)
        l11 = fourier_level_lp_norm(f, 1)
        bound = 4**degree
        assert l11 <= bound + 1e-10, f"{name}: L_{{1,1}}={l11} > 4^{degree}={bound}"


class TestFractionalPRGProperty:
    """Verify that fractional samples (truncated Gaussians) fool low-L1,2 functions."""

    def test_fractional_expectation_close_to_true(self):
        """For functions with small L_{1,2}, fractional PRG error should be small."""
        np.random.seed(42)
        f = bf.majority(5)
        p_ext = multilinear_extension(f)
        n = f.n_vars
        exact = p_ext(np.zeros(n))

        # Fractional samples: truncated Gaussian
        sigma = 0.2
        total = sum(p_ext(np.clip(np.random.randn(n) * sigma, -1, 1)) for _ in range(5000))
        frac_mean = total / 5000
        error = abs(frac_mean - exact)

        # Error should be small (< 0.1 for sigma=0.2 with 5000 samples)
        assert error < 0.1, f"Fractional PRG error too large: {error:.4f}"

    def test_vertex_and_fractional_agree_on_mean(self):
        """Both vertex and fractional samples should estimate E[f] correctly."""
        np.random.seed(42)
        f = bf.majority(5)
        p_ext = multilinear_extension(f)
        n = f.n_vars
        exact = p_ext(np.zeros(n))

        # Vertex samples
        v_total = sum(
            p_ext((2.0 * np.random.randint(0, 2, n) - 1).astype(float)) for _ in range(5000)
        )
        v_mean = v_total / 5000

        # Fractional samples
        f_total = sum(p_ext(np.clip(np.random.randn(n) * 0.3, -1, 1)) for _ in range(5000))
        f_mean = f_total / 5000

        assert abs(v_mean - exact) < 0.1
        assert abs(f_mean - exact) < 0.1
