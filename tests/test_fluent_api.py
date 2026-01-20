"""
Tests for the fluent/chainable API.

These tests verify that transformation methods can be chained
and that terminal methods (analysis) work correctly at the end of chains.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf


class TestMethodAliases:
    """Test readable method aliases for operators."""

    def test_xor_method(self):
        """f.xor(g) should equal f ^ g."""
        f = bf.parity(3)
        g = bf.majority(3)

        result1 = f.xor(g)
        result2 = f ^ g

        assert np.array_equal(result1.fourier(), result2.fourier())

    def test_and_method(self):
        """f.and_(g) should equal f & g."""
        f = bf.parity(3)
        g = bf.majority(3)

        result1 = f.and_(g)
        result2 = f & g

        assert np.array_equal(result1.fourier(), result2.fourier())

    def test_or_method(self):
        """f.or_(g) should equal f | g."""
        f = bf.parity(3)
        g = bf.majority(3)

        result1 = f.or_(g)
        result2 = f | g

        assert np.array_equal(result1.fourier(), result2.fourier())

    def test_not_method(self):
        """f.not_() should equal ~f."""
        f = bf.majority(5)

        result1 = f.not_()
        result2 = ~f

        assert np.array_equal(result1.fourier(), result2.fourier())


class TestPermute:
    """Test variable permutation."""

    def test_permute_changes_influences(self):
        """Permuting should change which variable has influence."""
        f = bf.majority(3)

        # Get original influences
        orig_inf = f.influences()

        # Permute variables
        g = f.permute([1, 2, 0])
        perm_inf = g.influences()

        # Influences should be permuted (symmetric function, all equal)
        assert np.allclose(orig_inf, perm_inf)  # Majority is symmetric

    def test_permute_identity(self):
        """Identity permutation should not change function."""
        f = bf.majority(4)
        g = f.permute([0, 1, 2, 3])

        for x in range(16):
            assert f.evaluate(x) == g.evaluate(x)

    def test_permute_invalid(self):
        """Invalid permutation should raise error."""
        f = bf.AND(3)

        with pytest.raises(ValueError):
            f.permute([0, 1])  # Wrong length

        with pytest.raises(ValueError):
            f.permute([0, 0, 1])  # Not a valid permutation

    def test_permute_and_function(self):
        """Permuting AND should preserve its structure."""
        f = bf.AND(3)
        g = f.permute([2, 1, 0])  # Reverse order

        # AND is symmetric, so output should be same
        for x in range(8):
            assert f.evaluate(x) == g.evaluate(x)


class TestExtend:
    """Test extending to more variables."""

    def test_extend_dummy(self):
        """Extension with dummy variables should preserve original behavior."""
        f = bf.AND(2)
        g = f.extend(4)

        assert g.n_vars == 4

        # Original behavior preserved for lower bits
        assert g.evaluate(0b0011) == f.evaluate(0b11)  # 1 AND 1 = 1
        assert g.evaluate(0b0010) == f.evaluate(0b10)  # 0 AND 1 = 0

        # Extra bits don't matter
        assert g.evaluate(0b1111) == g.evaluate(0b0011)
        assert g.evaluate(0b1100) == g.evaluate(0b0000)

    def test_extend_xor(self):
        """Extension with XOR should flip based on extra bit parity."""
        f = bf.AND(2)
        g = f.extend(3, method="xor")

        # Extra bit 0: same as original
        assert g.evaluate(0b011) == f.evaluate(0b11)

        # Extra bit 1: XOR flips the output
        assert g.evaluate(0b111) != f.evaluate(0b11)

    def test_extend_same_n(self):
        """Extending to same n should return equivalent function."""
        f = bf.majority(5)
        g = f.extend(5)

        assert np.array_equal(f.fourier(), g.fourier())


class TestDual:
    """Test dual function computation."""

    def test_dual_and_or(self):
        """AND and OR should be duals."""
        and3 = bf.AND(3)
        or3 = bf.OR(3)

        # dual(AND) = OR
        assert and3.dual().hamming_weight() == or3.hamming_weight()

        # dual(OR) = AND
        assert or3.dual().hamming_weight() == and3.hamming_weight()

    def test_dual_involution(self):
        """Dual of dual should be original."""
        f = bf.majority(5)

        # f** = f
        ff = f.dual().dual()

        for x in range(32):
            assert f.evaluate(x) == ff.evaluate(x)


class TestNoise:
    """Test noise-related methods."""

    def test_noise_expectation_parity(self):
        """Parity noise expectation should be small for high-degree function."""
        f = bf.parity(5)

        expectations = f.noise_expectation(0.9)

        # For parity, T_ρ f ≈ 0 because rho^5 is small
        assert np.abs(expectations).max() < 0.1

    def test_noise_expectation_degree_1(self):
        """Degree-1 function noise expectation should scale with rho."""
        # Use parity on 1 variable = dictator
        f = bf.parity(1)

        # For degree-1, T_ρ f = rho * f
        exp_09 = f.noise_expectation(0.9)
        exp_05 = f.noise_expectation(0.5)

        # Higher rho should give larger expectations
        assert np.abs(exp_09).max() > np.abs(exp_05).max()

    def test_apply_noise_high_correlation(self):
        """High correlation noise should preserve function structure."""
        np.random.seed(42)
        f = bf.majority(5)

        noisy = f.apply_noise(0.99, samples=100)

        # Should be very similar to original
        original_hw = f.hamming_weight()
        noisy_hw = noisy.hamming_weight()

        assert abs(original_hw - noisy_hw) <= 2  # Allow small variation

    def test_apply_noise_chainable(self):
        """apply_noise should be chainable."""
        np.random.seed(42)

        # Should not raise
        result = bf.majority(3).apply_noise(0.9, samples=20).degree()
        assert isinstance(result, int)


class TestNamed:
    """Test named method for fluent naming."""

    def test_named_returns_self(self):
        """named() should return the same function."""
        f = bf.majority(5)
        g = f.named("MAJ_5")

        assert f is g  # Same object

    def test_named_sets_nickname(self):
        """named() should set the nickname."""
        f = bf.majority(5).named("MAJ_5")

        assert f.nickname == "MAJ_5"


class TestPipe:
    """Test pipe method for custom transformations."""

    def test_pipe_basic(self):
        """pipe() should apply function to self."""

        def double_restrict(f, var, val):
            return f.restrict(var, val)

        f = bf.majority(5)
        g = f.pipe(double_restrict, 0, 1)

        assert g.n_vars == 4

    def test_pipe_in_chain(self):
        """pipe() should work in a chain."""

        def negate_and_restrict(f, var):
            return (~f).restrict(var, 1)

        result = bf.majority(5).pipe(negate_and_restrict, 0).degree()

        assert isinstance(result, int)


class TestChaining:
    """Test complex chaining scenarios."""

    def test_multiple_restrictions(self):
        """Multiple restrictions should chain correctly."""
        f = bf.majority(5)

        g = f.restrict(0, 1).restrict(0, 0).restrict(0, 1)  # On 2-var function

        assert g.n_vars == 2

    def test_restrict_then_analyze(self):
        """Restriction followed by analysis should work."""
        f = bf.majority(5)

        degree = f.restrict(0, 1).restrict(1, 0).degree()
        influences = f.restrict(0, 1).influences()

        assert degree == 3
        assert len(influences) == 4

    def test_derivative_chain(self):
        """Derivatives should chain."""
        f = bf.majority(5)

        # Second derivative
        g = f.derivative(0).derivative(1)

        # Should be a valid function
        assert g.n_vars == 5

    def test_complex_chain(self):
        """Complex chain with multiple operations."""
        f = bf.majority(7)

        result = (
            f.restrict(0, 1)  # 6 vars
            .derivative(0)  # Still 6 vars
            .xor(bf.parity(6))  # XOR with parity
            .degree()
        )

        assert isinstance(result, int)
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
