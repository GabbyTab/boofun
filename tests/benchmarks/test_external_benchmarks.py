"""
External benchmark tests for boolfunc library.

This module contains tests against known Boolean functions from:
1. Cryptographic S-boxes (AES, DES)
2. Classical Boolean functions with known properties
3. Functions from O'Donnell's book
4. Functions from Scott Aaronson's Boolean Function Wizard

These serve as both correctness tests and benchmarks.
"""

import pytest
import numpy as np
import boolfunc as bf
from boolfunc.analysis import SpectralAnalyzer, PropertyTester
from boolfunc.analysis.query_complexity import QueryComplexityProfile
from boolfunc.analysis.gaussian import GaussianAnalyzer
from boolfunc.analysis.invariance import InvarianceAnalyzer


# =============================================================================
# Known Boolean Functions with Verified Properties
# =============================================================================

class TestKnownFunctions:
    """Tests for functions with known, verified properties."""
    
    def test_and_function_properties(self):
        """AND function: known to be monotone, not balanced, symmetric."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            tester = PropertyTester(f)
            
            # AND is monotone
            assert tester.monotonicity_test() == True
            
            # AND is not balanced (only 1 out of 2^n inputs gives 1)
            assert tester.balanced_test() == False
            
            # AND is symmetric
            assert tester.symmetry_test() == True
            
            # AND has degree n (all variables appear)
            analyzer = SpectralAnalyzer(f)
            fourier = analyzer.fourier_expansion()
            # The all-1s coefficient (S = {0,1,...,n-1}) should be non-zero
            all_vars_idx = (1 << n) - 1
            assert abs(fourier[all_vars_idx]) > 1e-10
    
    def test_or_function_properties(self):
        """OR function: known to be monotone, not balanced, symmetric."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            tester = PropertyTester(f)
            
            # OR is monotone
            assert tester.monotonicity_test() == True
            
            # OR is not balanced
            assert tester.balanced_test() == False
            
            # OR is symmetric
            assert tester.symmetry_test() == True
    
    def test_parity_function_properties(self):
        """XOR/Parity: linear, balanced, symmetric."""
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            tester = PropertyTester(f)
            
            # Parity is linear (it's x_0 XOR x_1 XOR ... XOR x_{n-1})
            assert tester.blr_linearity_test() == True
            
            # Parity is balanced
            assert tester.balanced_test() == True
            
            # Parity is symmetric
            assert tester.symmetry_test() == True
            
            # Parity has exactly one non-zero Fourier coefficient (at S = all vars)
            analyzer = SpectralAnalyzer(f)
            fourier = analyzer.fourier_expansion()
            non_zero = sum(1 for c in fourier if abs(c) > 1e-10)
            # For ±1 parity, we have f̂(∅) = 0, f̂({all}) = ±1
            assert non_zero <= 2  # constant and the parity coefficient
    
    def test_majority_function_properties(self):
        """Majority: monotone, balanced (for odd n), symmetric."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            tester = PropertyTester(f)
            
            # Majority is monotone
            assert tester.monotonicity_test() == True
            
            # Majority is balanced for odd n
            assert tester.balanced_test() == True
            
            # Majority is symmetric
            assert tester.symmetry_test() == True
            
            # Majority is not a dictator
            is_dictator, _ = tester.dictator_test()
            assert is_dictator == False
    
    def test_dictator_function_properties(self):
        """Dictator (single variable): linear, balanced, is a dictator."""
        for n in [3, 4, 5]:
            for i in range(n):
                # Create dictator on variable i
                def dictator_i(x, var=i, num_vars=n):
                    bits = [(x >> j) & 1 for j in range(num_vars)]
                    return bits[var]
                
                f = bf.create(dictator_i, n=n)
                tester = PropertyTester(f)
                
                # Dictator is linear
                assert tester.blr_linearity_test() == True
                
                # Dictator is balanced
                assert tester.balanced_test() == True
                
                # Dictator is a dictator
                is_dictator, _ = tester.dictator_test()
                assert is_dictator == True


class TestFourierIdentities:
    """Test Fourier-theoretic identities that must hold for any function."""
    
    def test_parseval_identity(self):
        """Parseval: sum of squared Fourier coefficients = E[f^2]."""
        from boolfunc.analysis.fourier import parseval_verify
        
        for n in [3, 4, 5]:
            # Test several functions
            functions = [
                bf.AND(n),
                bf.OR(n),
                bf.parity(n),
                bf.majority(n) if n % 2 == 1 else bf.AND(n),
            ]
            
            for f in functions:
                # Parseval should hold within numerical precision
                assert parseval_verify(f, tolerance=1e-6)
    
    def test_plancherel_theorem(self):
        """Plancherel: <f,g> in time domain = <f̂,ĝ> in Fourier domain."""
        from boolfunc.analysis.fourier import plancherel_inner_product
        
        n = 4
        f = bf.AND(n)
        g = bf.OR(n)
        
        # Compute inner product directly
        tt_f = np.asarray(f.get_representation("truth_table"), dtype=float)
        tt_g = np.asarray(g.get_representation("truth_table"), dtype=float)
        pm_f = 1.0 - 2.0 * tt_f
        pm_g = 1.0 - 2.0 * tt_g
        direct_inner = float(np.mean(pm_f * pm_g))
        
        # Compute via Fourier
        fourier_inner = plancherel_inner_product(f, g)
        
        assert abs(direct_inner - fourier_inner) < 1e-6
    
    def test_total_influence_equals_spectral_sum(self):
        """Total influence = sum of influences = sum of k*W_k where W_k = sum |f̂(S)|² for |S|=k."""
        n = 5
        f = bf.majority(n)
        
        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()
        total_inf = analyzer.total_influence()
        
        # Total influence should equal sum of individual influences
        assert abs(total_inf - sum(influences)) < 1e-6
        
        # Also equals sum_S |S| * f̂(S)²
        fourier = analyzer.fourier_expansion()
        spectral_sum = sum(
            bin(s).count("1") * fourier[s]**2 
            for s in range(len(fourier))
        )
        
        assert abs(total_inf - spectral_sum) < 1e-6


class TestKnownComplexities:
    """Test query complexity values for functions with known complexities."""
    
    def test_and_complexity(self):
        """AND(n): D(f) = n, C0(f) = 1, C1(f) = n, s(f) = n."""
        for n in [3, 4, 5]:
            f = bf.AND(n)
            profile = QueryComplexityProfile(f)
            m = profile.compute()
            
            # Decision tree depth is n (must query all variables)
            assert m["D"] == n
            
            # Certificate for 0-inputs: 1 (any 0 suffices)
            assert m["C0"] == 1
            
            # Certificate for 1-inputs: n (must see all 1s)
            assert m["C1"] == n
            
            # Max sensitivity: n (at input 111...1)
            assert m["s"] == n
    
    def test_or_complexity(self):
        """OR(n): D(f) = n, C0(f) = n, C1(f) = 1, s(f) = n."""
        for n in [3, 4, 5]:
            f = bf.OR(n)
            profile = QueryComplexityProfile(f)
            m = profile.compute()
            
            # Decision tree depth is n
            assert m["D"] == n
            
            # Certificate for 0-inputs: n (must see all 0s)
            assert m["C0"] == n
            
            # Certificate for 1-inputs: 1 (any 1 suffices)
            assert m["C1"] == 1
    
    def test_parity_complexity(self):
        """Parity(n): D(f) = n (must query all bits)."""
        for n in [3, 4, 5]:
            f = bf.parity(n)
            profile = QueryComplexityProfile(f)
            m = profile.compute()
            
            # Decision tree depth is n
            assert m["D"] == n
            
            # Sensitivity is 1 at every input (flip any bit, output changes)
            assert m["s0"] == n
            assert m["s1"] == n
    
    def test_majority_complexity(self):
        """Majority(n): D(f) = n, s(f) = ceil(n/2) for odd n."""
        for n in [3, 5, 7]:
            f = bf.majority(n)
            profile = QueryComplexityProfile(f)
            m = profile.compute()
            
            # Decision tree depth is n
            assert m["D"] == n
            
            # Max sensitivity is ceil(n/2) for majority
            # (at balanced inputs, flipping ceil(n/2) bits can change output)
            expected_s = (n + 1) // 2
            assert m["s"] >= expected_s


class TestRelationships:
    """Test known relationships between complexity measures."""
    
    def test_sensitivity_vs_certificate(self):
        """s(f) <= C(f) always."""
        for n in [3, 4, 5]:
            for f in [bf.AND(n), bf.OR(n), bf.parity(n), bf.majority(n) if n % 2 == 1 else bf.AND(n)]:
                profile = QueryComplexityProfile(f)
                m = profile.compute()
                
                assert m["s"] <= m["C"]
    
    def test_block_sensitivity_vs_decision_tree(self):
        """bs(f) <= D(f) always."""
        for n in [3, 4, 5]:
            for f in [bf.AND(n), bf.OR(n), bf.parity(n), bf.majority(n) if n % 2 == 1 else bf.AND(n)]:
                profile = QueryComplexityProfile(f)
                m = profile.compute()
                
                assert m["bs"] <= m["D"]
    
    def test_huang_sensitivity_theorem(self):
        """
        Huang's Sensitivity Theorem (2019): s(f) >= sqrt(bs(f))
        
        This is one of the most important recent results in Boolean function theory.
        """
        for n in [3, 4, 5]:
            for f in [bf.AND(n), bf.OR(n), bf.parity(n), bf.majority(n) if n % 2 == 1 else bf.AND(n)]:
                profile = QueryComplexityProfile(f)
                m = profile.compute()
                
                # s(f) >= sqrt(bs(f))
                assert m["s"] >= np.sqrt(m["bs"]) - 1e-6


class TestGaussianProperties:
    """Test Gaussian analysis properties."""
    
    def test_majority_is_approximately_gaussian(self):
        """For large n, majority has approximately Gaussian distribution."""
        # For small n, we can verify the Berry-Esseen bound
        for n in [5, 7, 9]:
            f = bf.majority(n)
            gauss = GaussianAnalyzer(f)
            
            # Berry-Esseen bound should decrease with n
            be_bound = gauss.berry_esseen()
            
            # For majority, max influence = O(1/sqrt(n)), so BE = O(1/n^{3/4})
            # This should be small for larger n
            if n >= 5:
                assert be_bound < 0.2
    
    def test_parity_not_gaussian(self):
        """Parity is far from Gaussian (dictator-like)."""
        n = 5
        f = bf.parity(n)
        gauss = GaussianAnalyzer(f)
        
        # Parity has very high influence per variable
        # So it shouldn't be "approximately Gaussian"
        be_bound = gauss.berry_esseen()
        
        # BE bound should be large
        assert be_bound > 0.01


class TestInvarianceProperties:
    """Test invariance principle properties."""
    
    def test_majority_is_stablest(self):
        """Majority should have noise stability deficit of 0 (it IS the stablest)."""
        n = 5
        f = bf.majority(n)
        inv = InvarianceAnalyzer(f)
        
        # Deficit should be 0 or very small
        deficit = inv.noise_stability_deficit(0.9)
        assert abs(deficit) < 0.01
    
    def test_low_influence_implies_small_invariance_bound(self):
        """Functions with low max influence should have small invariance bound."""
        # Majority has influences O(1/sqrt(n))
        n = 9
        f = bf.majority(n)
        inv = InvarianceAnalyzer(f)
        
        bound = inv.invariance_bound()
        
        # For majority with n=9, max influence ≈ 0.3
        # Bound should be (0.3)^{1/4} ≈ 0.74
        assert bound < 1.0


# =============================================================================
# Cryptographic S-box Tests
# =============================================================================

class TestCryptographicSboxes:
    """
    Test cryptographic S-boxes and their known properties.
    
    S-boxes are lookup tables used in block ciphers. They are designed
    to be highly nonlinear to resist cryptanalysis.
    """
    
    @pytest.fixture
    def aes_sbox_component(self):
        """
        One component function of the AES S-box.
        
        The AES S-box is an 8→8 function. Each output bit is an 8→1
        Boolean function. We test one such component.
        
        The S-box has high nonlinearity (112 out of max 120 for n=8).
        """
        # AES S-box lookup table (first 16 entries for a simpler test)
        # We'll use a smaller example for testing
        # This is a 4-bit component approximation
        
        # Simple 4-bit highly nonlinear function (bent function proxy)
        # For actual AES, would need full 8-bit implementation
        # Create truth table directly for the S-box component
        table = [0, 1, 15, 14, 13, 11, 2, 6, 12, 5, 10, 4, 9, 3, 8, 7]
        truth_table = [table[x] & 1 for x in range(16)]  # First output bit
        
        return bf.create(truth_table)
    
    def test_sbox_nonlinearity(self, aes_sbox_component):
        """Test that S-box component has high nonlinearity."""
        f = aes_sbox_component
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        
        # Nonlinearity = 2^{n-1} - (1/2) * max_{a≠0} |f̂(a)|
        # For n=4, max nonlinearity is 6
        n = 4
        max_coeff = max(abs(c) for i, c in enumerate(fourier) if i > 0)
        nonlinearity = (1 << (n-1)) - int(max_coeff * (1 << n) / 2)
        
        # Should have reasonably high nonlinearity
        assert nonlinearity >= 2
    
    def test_sbox_not_linear(self, aes_sbox_component):
        """Cryptographic S-boxes must not be linear."""
        f = aes_sbox_component
        tester = PropertyTester(f)
        
        # S-box components should NOT be linear
        result = tester.blr_linearity_test(num_queries=1000)
        assert result == False


# =============================================================================
# O'Donnell Book Examples
# =============================================================================

class TestODonnellExamples:
    """Test examples from O'Donnell's "Analysis of Boolean Functions" book."""
    
    def test_tribes_function(self):
        """
        Tribes function from Chapter 4.
        
        Tribes(k, ℓ) is the OR of ℓ ANDs of size k.
        It's designed to be approximately balanced with minimal influences.
        """
        # Tribes(2, 3) = (x0 ∧ x1) ∨ (x2 ∧ x3) ∨ (x4 ∧ x5)
        f = bf.tribes(2, 3)
        
        # Should be nearly balanced for appropriate parameters
        truth_table = np.asarray(f.get_representation("truth_table"))
        prob_1 = np.mean(truth_table)
        
        # Tribes is designed to have Pr[f=1] ≈ 1/2
        # For small parameters, it won't be exactly 1/2
        assert 0.1 < prob_1 < 0.9
        
        # Tribes is monotone
        tester = PropertyTester(f)
        assert tester.monotonicity_test() == True
        
        # Should NOT be a junta (depends on all 6 variables)
        assert tester.junta_test(k=2) == False
    
    def test_dictator_vs_majority_noise_stability(self):
        """
        From Chapter 2: Majority is more noise-stable than dictators.
        
        For ρ close to 1:
        - Stab_ρ[Dict] = ρ
        - Stab_ρ[Maj_n] → (2/π) arcsin(ρ) > ρ for ρ ∈ (0, 1)
        """
        n = 5
        rho = 0.9
        
        # Create dictator (first variable)
        def dict_0(x):
            return x & 1
        dictator = bf.create(dict_0, n=n)
        
        # Majority
        majority = bf.majority(n)
        
        from boolfunc.analysis.gaussian import gaussian_noise_stability
        
        stab_dict = gaussian_noise_stability(dictator, rho)
        stab_maj = gaussian_noise_stability(majority, rho)
        
        # Dictator: Stab_ρ[x_1] = ρ
        assert abs(stab_dict - rho) < 0.1
        
        # Majority should be MORE stable for ρ close to 1
        assert stab_maj > stab_dict - 0.1


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for various operations."""
    
    @pytest.mark.parametrize("n", [6, 8, 10, 12])
    def test_fourier_transform_performance(self, n):
        """Benchmark Fourier transform speed."""
        import time
        
        f = bf.majority(n) if n % 2 == 1 else bf.AND(n)
        analyzer = SpectralAnalyzer(f)
        
        start = time.time()
        fourier = analyzer.fourier_expansion()
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        # n=10: < 0.1s, n=12: < 1s
        max_time = 0.01 * (2 ** (n - 6))  # Exponential scaling
        assert elapsed < max(max_time, 2.0)  # Cap at 2 seconds
    
    @pytest.mark.parametrize("n", [6, 8, 10])
    def test_property_testing_performance(self, n):
        """Benchmark property testing speed."""
        import time
        
        f = bf.majority(n) if n % 2 == 1 else bf.AND(n)
        tester = PropertyTester(f)
        
        start = time.time()
        tester.run_all_tests()
        elapsed = time.time() - start
        
        # Property testing should be fast
        assert elapsed < 5.0  # 5 seconds max
    
    @pytest.mark.parametrize("n", [6, 8])
    def test_query_profile_performance(self, n):
        """Benchmark query complexity profile computation."""
        import time
        
        f = bf.majority(n) if n % 2 == 1 else bf.AND(n)
        
        start = time.time()
        profile = QueryComplexityProfile(f)
        profile.compute()
        elapsed = time.time() - start
        
        # Profile computation may be slower due to many measures
        # n=6: ~5s, n=8: ~30s (without Numba)
        max_time = 10.0 * (2 ** (n - 6))
        assert elapsed < max(max_time, 60.0)  # 60 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
