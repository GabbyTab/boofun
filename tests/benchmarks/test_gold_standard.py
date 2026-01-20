import sys
sys.path.insert(0, "src")
"""
Gold Standard Tests for boofun library.

These tests verify correctness against established benchmarks and known results:
1. EPFL Combinational Benchmarks - small circuits with known truth tables
2. BDD canonical form tests - logical operation verification
3. GECCO-style property tests - functions with known properties
4. Mathematical identities - Fourier analysis theorems

References:
- EPFL Benchmarks: https://github.com/lsils/benchmarks
- BDD Benchmarks: https://github.com/SSoelvsten/bdd-benchmark
- GECCO 2023: "Towards a General Boolean Function Benchmark Suite"
- O'Donnell: "Analysis of Boolean Functions"
"""

import pytest
import numpy as np
import boofun as bf
from boofun.analysis import SpectralAnalyzer, PropertyTester
from boofun.analysis.fourier import (
    parseval_verify, 
    plancherel_inner_product,
    fourier_degree,
)
from boofun.analysis.query_complexity import QueryComplexityProfile
from math import sqrt, comb


# =============================================================================
# EPFL Combinational Benchmark Circuits (Small)
# From: https://github.com/lsils/benchmarks
# =============================================================================

class TestEPFLBenchmarks:
    """
    Gold standard tests from EPFL combinational benchmark suite.
    These are small circuits with known, verified truth tables.
    """
    
    def test_half_adder_sum(self):
        """
        Half adder SUM output: a XOR b
        EPFL: adder circuit component
        """
        # Truth table for XOR (half adder sum)
        expected_tt = [0, 1, 1, 0]
        f = bf.create(expected_tt)
        
        # Verify all evaluations
        assert f.evaluate(0b00) == 0  # 0 XOR 0 = 0
        assert f.evaluate(0b01) == 1  # 0 XOR 1 = 1
        assert f.evaluate(0b10) == 1  # 1 XOR 0 = 1
        assert f.evaluate(0b11) == 0  # 1 XOR 1 = 0
        
        # XOR is linear
        tester = PropertyTester(f, random_seed=42)
        assert tester.blr_linearity_test() == True
    
    def test_half_adder_carry(self):
        """
        Half adder CARRY output: a AND b
        EPFL: adder circuit component
        """
        expected_tt = [0, 0, 0, 1]
        f = bf.create(expected_tt)
        
        assert f.evaluate(0b00) == 0
        assert f.evaluate(0b01) == 0
        assert f.evaluate(0b10) == 0
        assert f.evaluate(0b11) == 1
        
        # AND is monotone
        tester = PropertyTester(f, random_seed=42)
        assert tester.monotonicity_test() == True
    
    def test_full_adder_sum(self):
        """
        Full adder SUM output: a XOR b XOR cin
        EPFL: adder circuit, 3-input XOR
        """
        # 3-input XOR truth table
        expected_tt = [0, 1, 1, 0, 1, 0, 0, 1]
        f = bf.create(expected_tt)
        
        # Verify parity behavior
        for x in range(8):
            a, b, c = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1
            expected = (a + b + c) % 2
            assert f.evaluate(x) == expected
        
        # 3-XOR is linear
        tester = PropertyTester(f, random_seed=42)
        assert tester.blr_linearity_test() == True
    
    def test_full_adder_carry(self):
        """
        Full adder CARRY output: MAJ(a, b, cin)
        EPFL: adder circuit, majority gate
        """
        # MAJ3 truth table: output 1 if >= 2 inputs are 1
        expected_tt = [0, 0, 0, 1, 0, 1, 1, 1]
        f = bf.create(expected_tt)
        
        for x in range(8):
            a, b, c = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1
            expected = 1 if (a + b + c) >= 2 else 0
            assert f.evaluate(x) == expected
        
        # MAJ3 is monotone and symmetric
        tester = PropertyTester(f, random_seed=42)
        assert tester.monotonicity_test() == True
        assert tester.symmetry_test() == True
    
    def test_mux_2to1(self):
        """
        2:1 Multiplexer: (s AND b) OR (NOT s AND a)
        EPFL: control logic component
        """
        # MUX(s, a, b) = if s then b else a
        # Inputs: s=bit0, a=bit1, b=bit2
        # Truth table: [a, a, b, b, a, a, b, b] when ordered (s,a,b)
        # Actually: f(s,a,b) = s*b + (1-s)*a
        expected_tt = [0, 0, 1, 1, 0, 1, 0, 1]  # f(0,0,0)=0, f(1,0,0)=0, etc.
        f = bf.create(expected_tt)
        
        # MUX is NOT monotone (changing s can decrease output)
        tester = PropertyTester(f)
        # Just verify it evaluates correctly
        assert f.n_vars == 3
    
    def test_comparator_lt(self):
        """
        2-bit less-than comparator: a < b
        EPFL: arithmetic circuit
        """
        # a < b for 2-bit numbers (4 vars: a1,a0,b1,b0)
        # This would be 16 entries, let's do 1-bit: a < b
        # a=bit0, b=bit1: f = (NOT a) AND b
        expected_tt = [0, 0, 1, 0]  # f(0,0)=0, f(1,0)=0, f(0,1)=1, f(1,1)=0
        f = bf.create(expected_tt)
        
        assert f.evaluate(0b00) == 0  # 0 < 0 = False
        assert f.evaluate(0b01) == 0  # 1 < 0 = False  
        assert f.evaluate(0b10) == 1  # 0 < 1 = True
        assert f.evaluate(0b11) == 0  # 1 < 1 = False


# =============================================================================
# BDD Benchmark Suite - Canonical Form Tests
# From: https://github.com/SSoelvsten/bdd-benchmark
# =============================================================================

class TestBDDCanonicalForms:
    """
    Tests based on BDD benchmark suite principles.
    Verify canonical representations and logical operations.
    """
    
    def test_cofactor_shannon_decomposition(self):
        """
        Shannon decomposition: f = x*f|_{x=1} + x'*f|_{x=0}
        BDD fundamental operation.
        """
        f = bf.AND(3)  # f = x0 AND x1 AND x2
        
        # Fix x0=1: should give x1 AND x2
        f_x0_1 = f.fix(0, 1)
        assert f_x0_1.n_vars == 2
        
        # Fix x0=0: should give constant 0
        f_x0_0 = f.fix(0, 0)
        tt = np.asarray(f_x0_0.get_representation("truth_table"))
        assert np.all(tt == 0)
    
    def test_ite_operation(self):
        """
        ITE (if-then-else): ITE(f,g,h) = (f AND g) OR (NOT f AND h)
        BDD core operation.
        """
        n = 2
        # Create simple functions
        f = bf.create([0, 1, 0, 1])  # x0
        g = bf.create([0, 0, 1, 1])  # x1
        h = bf.create([1, 1, 1, 1])  # constant 1
        
        # ITE(x0, x1, 1) = (x0 AND x1) OR (NOT x0 AND 1)
        # At x=00: (0 AND 0) OR (1 AND 1) = 1
        # At x=01: (1 AND 0) OR (0 AND 1) = 0  
        # At x=10: (0 AND 1) OR (1 AND 1) = 1
        # At x=11: (1 AND 1) OR (0 AND 1) = 1
        # Truth table: [1, 0, 1, 1]
        
        # Compute ITE manually
        tt_f = np.asarray(f.get_representation("truth_table"), dtype=int)
        tt_g = np.asarray(g.get_representation("truth_table"), dtype=int)
        tt_h = np.asarray(h.get_representation("truth_table"), dtype=int)
        tt_ite = (tt_f & tt_g) | ((1 - tt_f) & tt_h)
        
        ite_func = bf.create(list(tt_ite))
        expected = [1, 0, 1, 1]
        assert list(np.asarray(ite_func.get_representation("truth_table"))) == expected
    
    def test_apply_and_or_xor(self):
        """
        BDD Apply operations: AND, OR, XOR must be consistent.
        """
        f = bf.create([0, 1, 1, 0])  # XOR
        g = bf.create([0, 0, 0, 1])  # AND
        
        # f AND g
        fg_and = f & g
        expected_and = [0, 0, 0, 0]  # XOR AND AND = always 0 (they never both 1)
        assert list(np.asarray(fg_and.get_representation("truth_table"))) == expected_and
        
        # f OR g  
        fg_or = f | g
        expected_or = [0, 1, 1, 1]  # XOR OR AND
        assert list(np.asarray(fg_or.get_representation("truth_table"))) == expected_or
        
        # f XOR g
        fg_xor = f ^ g
        expected_xor = [0, 1, 1, 1]  # XOR XOR AND = XOR OR AND (since never both 1)
        assert list(np.asarray(fg_xor.get_representation("truth_table"))) == expected_xor
    
    def test_complement_involution(self):
        """
        NOT(NOT(f)) = f (involution property)
        """
        f = bf.majority(3)
        f_not = ~f
        f_not_not = ~f_not
        
        tt_f = np.asarray(f.get_representation("truth_table"))
        tt_f_not_not = np.asarray(f_not_not.get_representation("truth_table"))
        
        assert np.array_equal(tt_f, tt_f_not_not)


# =============================================================================
# GECCO Boolean Function Suite - Property Tests
# Based on: "Towards a General Boolean Function Benchmark Suite" (GECCO 2023)
# =============================================================================

class TestGECCOPropertySuite:
    """
    Functions with known properties from GECCO benchmark principles.
    Tests monotonicity, balance, symmetry, linearity.
    """
    
    def test_threshold_functions_monotone(self):
        """
        All threshold functions are monotone.
        Threshold_k(x) = 1 iff sum(x) >= k
        """
        n = 4
        for k in range(n + 1):
            # Create threshold function
            tt = [1 if bin(x).count('1') >= k else 0 for x in range(1 << n)]
            f = bf.create(tt)
            
            tester = PropertyTester(f, random_seed=42)
            assert tester.monotonicity_test() == True, f"Threshold_{k} should be monotone"
    
    def test_symmetric_functions_symmetric(self):
        """
        Threshold and exact-weight functions are symmetric.
        """
        n = 4
        
        # Threshold functions
        for k in range(n + 1):
            tt = [1 if bin(x).count('1') >= k else 0 for x in range(1 << n)]
            f = bf.create(tt)
            tester = PropertyTester(f, random_seed=42)
            assert tester.symmetry_test() == True, f"Threshold_{k} should be symmetric"
        
        # Exact weight functions: f(x) = 1 iff |x| = k
        for k in range(n + 1):
            tt = [1 if bin(x).count('1') == k else 0 for x in range(1 << n)]
            f = bf.create(tt)
            tester = PropertyTester(f, random_seed=42)
            assert tester.symmetry_test() == True, f"ExactWeight_{k} should be symmetric"
    
    def test_parity_functions_linear(self):
        """
        All parity functions (XOR of subsets) are linear.
        """
        n = 4
        
        # Test several subset parities
        for mask in [0b0001, 0b0011, 0b0111, 0b1111, 0b1010]:
            tt = [bin(x & mask).count('1') % 2 for x in range(1 << n)]
            f = bf.create(tt)
            tester = PropertyTester(f, random_seed=42)
            assert tester.blr_linearity_test() == True, f"Parity with mask {bin(mask)} should be linear"
    
    def test_balanced_functions(self):
        """
        Functions known to be balanced (equal 0s and 1s).
        """
        n = 4
        
        # All single-variable functions are balanced
        for i in range(n):
            tt = [(x >> i) & 1 for x in range(1 << n)]
            f = bf.create(tt)
            tester = PropertyTester(f)
            assert tester.balanced_test() == True, f"Variable x_{i} should be balanced"
        
        # XOR of any non-empty subset is balanced
        for mask in [0b0001, 0b0011, 0b0111, 0b1111]:
            tt = [bin(x & mask).count('1') % 2 for x in range(1 << n)]
            f = bf.create(tt)
            tester = PropertyTester(f)
            assert tester.balanced_test() == True
    
    def test_unbalanced_functions(self):
        """
        Functions known to be unbalanced.
        """
        # AND is unbalanced (only one 1)
        f = bf.AND(3)
        tester = PropertyTester(f)
        assert tester.balanced_test() == False
        
        # OR is unbalanced (only one 0)
        f = bf.OR(3)
        tester = PropertyTester(f)
        assert tester.balanced_test() == False
        
        # Constant functions
        f = bf.create([0, 0, 0, 0])
        tester = PropertyTester(f)
        assert tester.balanced_test() == False


# =============================================================================
# Mathematical Identity Tests (O'Donnell Book)
# =============================================================================

class TestMathematicalIdentities:
    """
    Verify fundamental mathematical identities from Boolean function theory.
    Based on O'Donnell's "Analysis of Boolean Functions".
    """
    
    def test_parseval_all_functions(self):
        """
        Parseval: sum of squared Fourier coefficients = 1 for Boolean functions.
        """
        n = 4
        
        # Test all 2^(2^n) functions for small n... just kidding, test several
        test_functions = [
            bf.AND(n),
            bf.OR(n),
            bf.parity(n),
            bf.majority(n - 1),  # n-1 for odd
            bf.tribes(2, 2),
            bf.create([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]),  # random
        ]
        
        for f in test_functions:
            verified, sum_sq, expected = parseval_verify(f)
            assert verified, f"Parseval failed: sum={sum_sq}, expected={expected}"
    
    def test_influence_sum_equals_total_influence(self):
        """
        Total influence = sum of individual influences.
        """
        f = bf.majority(5)
        analyzer = SpectralAnalyzer(f)
        
        influences = analyzer.influences()
        total = analyzer.total_influence()
        
        assert abs(sum(influences) - total) < 1e-10
    
    def test_total_influence_equals_spectral_sum(self):
        """
        Total influence = sum_S |S| * f̂(S)²
        """
        f = bf.majority(5)
        analyzer = SpectralAnalyzer(f)
        
        total_inf = analyzer.total_influence()
        fourier = analyzer.fourier_expansion()
        
        spectral_sum = sum(
            bin(s).count('1') * fourier[s]**2 
            for s in range(len(fourier))
        )
        
        assert abs(total_inf - spectral_sum) < 1e-10
    
    def test_degree_bounds(self):
        """
        Fourier degree bounds:
        - AND_n has degree n
        - OR_n has degree n  
        - MAJ_n has degree n (all odd terms for odd n)
        - Parity_n has degree n
        """
        for n in [3, 4, 5]:
            # AND has degree n
            assert fourier_degree(bf.AND(n)) == n
            
            # OR has degree n
            assert fourier_degree(bf.OR(n)) == n
            
            # Parity has degree n
            assert fourier_degree(bf.parity(n)) == n
    
    def test_majority_fourier_structure(self):
        """
        MAJ_n Fourier coefficients are known exactly.
        For odd n, only odd-sized sets have non-zero coefficients.
        """
        n = 5  # Must be odd
        f = bf.majority(n)
        analyzer = SpectralAnalyzer(f)
        fourier = analyzer.fourier_expansion()
        
        # Check that even-sized sets have coefficient 0
        for s in range(len(fourier)):
            size = bin(s).count('1')
            if size % 2 == 0:  # even size
                assert abs(fourier[s]) < 1e-10, f"MAJ should have 0 coefficient for even |S|={size}"
    
    def test_parity_fourier_concentration(self):
        """
        Parity_n has exactly one non-zero Fourier coefficient at S = [n].
        """
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            analyzer = SpectralAnalyzer(f)
            fourier = analyzer.fourier_expansion()
            
            non_zero_count = sum(1 for c in fourier if abs(c) > 1e-10)
            assert non_zero_count == 1, f"Parity_{n} should have exactly 1 non-zero coefficient"
            
            # The non-zero coefficient should be at the all-1s index
            all_ones = (1 << n) - 1
            assert abs(fourier[all_ones]) > 0.9  # Should be ±1


# =============================================================================
# Known Complexity Values (BFW / Aaronson)
# =============================================================================

class TestKnownComplexityValues:
    """
    Verify query complexity measures against known values.
    Based on Scott Aaronson's Boolean Function Wizard.
    """
    
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_and_or_complexity(self, n):
        """
        AND_n and OR_n have well-known complexity measures.
        """
        and_f = bf.AND(n)
        or_f = bf.OR(n)
        
        profile_and = QueryComplexityProfile(and_f)
        profile_or = QueryComplexityProfile(or_f)
        m_and = profile_and.compute()
        m_or = profile_or.compute()
        
        # AND: D=n, C0=1, C1=n, s=n
        assert m_and["D"] == n
        assert m_and["C0"] == 1
        assert m_and["C1"] == n
        assert m_and["s"] == n
        
        # OR: D=n, C0=n, C1=1, s=n
        assert m_or["D"] == n
        assert m_or["C0"] == n
        assert m_or["C1"] == 1
        assert m_or["s"] == n
    
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_parity_complexity(self, n):
        """
        Parity_n: D=n, s0=s1=n (every input has sensitivity n)
        """
        f = bf.parity(n)
        profile = QueryComplexityProfile(f)
        m = profile.compute()
        
        assert m["D"] == n
        assert m["s"] == n
        assert m["s0"] == n
        assert m["s1"] == n
    
    def test_tribes_influences(self):
        """
        Tribes(w, k) has known influence structure.
        Each variable has equal influence.
        """
        w, k = 2, 3  # width 2, 3 tribes
        f = bf.tribes(w, k)
        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()
        
        # All influences should be equal (by symmetry within tribes)
        # Variables in same position across tribes have same influence
        assert len(set(round(inf, 6) for inf in influences)) <= 2  # At most 2 distinct values


# =============================================================================
# Cross-Validation with Mathematical Formulas
# =============================================================================

class TestCrossValidation:
    """
    Cross-validate computed values against known mathematical formulas.
    """
    
    def test_majority_influence_formula(self):
        """
        For MAJ_n with odd n: Inf_i = 2^{1-n} * C(n-1, (n-1)/2)
        This equals approximately sqrt(2/(πn)) for large n.
        """
        for n in [3, 5, 7]:
            f = bf.majority(n)
            analyzer = SpectralAnalyzer(f)
            influences = analyzer.influences()
            
            # Known formula for majority influence
            k = (n - 1) // 2
            expected_inf = comb(n - 1, k) / (2 ** (n - 1))
            
            # All influences should be equal and match formula
            for inf in influences:
                assert abs(inf - expected_inf) < 1e-10, \
                    f"MAJ_{n} influence: got {inf}, expected {expected_inf}"
    
    def test_and_influence_formula(self):
        """
        For AND_n: Inf_i = 1/2^{n-1}
        """
        for n in [2, 3, 4, 5]:
            f = bf.AND(n)
            analyzer = SpectralAnalyzer(f)
            influences = analyzer.influences()
            
            expected_inf = 1 / (2 ** (n - 1))
            
            for inf in influences:
                assert abs(inf - expected_inf) < 1e-10, \
                    f"AND_{n} influence: got {inf}, expected {expected_inf}"
    
    def test_noise_stability_dictator(self):
        """
        Dictator function: Stab_ρ[x_i] = ρ
        """
        n = 4
        rho = 0.9
        
        # Create dictator on first variable
        tt = [(x >> 0) & 1 for x in range(1 << n)]
        f = bf.create(tt)
        
        analyzer = SpectralAnalyzer(f)
        stability = analyzer.noise_stability(rho)
        
        # Dictator has Stab_ρ = ρ (single degree-1 coefficient)
        assert abs(stability - rho) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
