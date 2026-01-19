"""
Integration tests for S-box (cryptographic) analysis.

S-boxes are substitution boxes used in block ciphers (AES, DES).
Their cryptographic properties can be analyzed using Boolean function theory.

This validates:
1. Nonlinearity computation matches known values
2. Property testing works on S-box component functions
3. Fourier analysis reveals security properties
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis import SpectralAnalyzer, PropertyTester


# AES S-box (8-bit input/output)
# We analyze component functions (projections onto single output bits)
AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]


def get_sbox_component(sbox, output_bit):
    """
    Extract a component function from an S-box.
    
    The component function is f(x) = bit_i(S(x))
    where S is the S-box and bit_i extracts the i-th bit.
    
    Args:
        sbox: List of S-box values
        output_bit: Which output bit to extract (0-7 for 8-bit)
        
    Returns:
        BooleanFunction representing the component
    """
    n = int(np.log2(len(sbox)))  # 8 for AES
    truth_table = [(sbox[x] >> output_bit) & 1 for x in range(2**n)]
    return bf.create(truth_table)


def compute_nonlinearity(f):
    """
    Compute nonlinearity = 2^(n-1) - max|Walsh(a)|/2.
    
    Higher nonlinearity = more resistant to linear cryptanalysis.
    """
    fourier = f.fourier()
    n = f.n_vars
    
    # Maximum absolute Walsh coefficient (excluding constant)
    max_walsh = max(abs(fourier[s]) for s in range(1, 2**n))
    
    # Nonlinearity formula
    nonlinearity = 2**(n-1) - int(max_walsh * 2**(n-1))
    return nonlinearity


class TestSboxBasics:
    """Test basic S-box component function analysis."""
    
    def test_aes_sbox_component_sizes(self):
        """AES S-box components should be 8-variable functions."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            assert f.n_vars == 8
    
    def test_aes_sbox_components_balanced(self):
        """AES S-box components should be balanced."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            assert f.is_balanced(), f"Component {bit} not balanced"
    
    def test_aes_sbox_components_nonlinear(self):
        """AES S-box components should be highly nonlinear."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            
            # AES S-box has nonlinearity 112 (maximum for 8-bit is 120)
            nl = compute_nonlinearity(f)
            assert nl >= 100, f"Component {bit} nonlinearity {nl} too low"


class TestSboxPropertyTesting:
    """Test property testing on S-box components."""
    
    def test_sbox_not_linear(self):
        """S-box components should fail linearity test."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)
        
        # AES S-box is highly nonlinear
        assert not tester.blr_linearity_test(num_queries=50)
    
    def test_sbox_not_affine(self):
        """S-box components should fail affine test."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)
        
        assert not tester.affine_test(num_queries=50)
    
    def test_sbox_not_symmetric(self):
        """S-box components should not be symmetric."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)
        
        assert not tester.symmetry_test(num_queries=50)
    
    def test_sbox_not_monotone(self):
        """S-box components should not be monotone."""
        f = get_sbox_component(AES_SBOX, 0)
        tester = PropertyTester(f, random_seed=42)
        
        assert not tester.monotonicity_test(num_queries=50)


class TestSboxFourierAnalysis:
    """Test Fourier analysis on S-box components."""
    
    def test_sbox_high_degree(self):
        """AES S-box components have high Fourier degree."""
        f = get_sbox_component(AES_SBOX, 0)
        
        # AES S-box components should have high degree (near n=8)
        # Note: Fourier degree can differ from algebraic degree
        degree = f.degree()
        assert degree >= 6, f"Expected high degree, got {degree}"
    
    def test_sbox_parseval_identity(self):
        """Parseval's identity should hold."""
        f = get_sbox_component(AES_SBOX, 0)
        fourier = f.fourier()
        
        sum_sq = sum(c**2 for c in fourier)
        assert abs(sum_sq - 1.0) < 1e-10
    
    def test_sbox_spectral_weight_distribution(self):
        """S-box should have weight spread across degrees."""
        f = get_sbox_component(AES_SBOX, 0)
        weights = f.spectral_weight_by_degree()
        
        # Should have weight at multiple degrees (not concentrated)
        nonzero_degrees = sum(1 for w in weights.values() if w > 0.01)
        assert nonzero_degrees >= 3, "S-box spectral weight too concentrated"
    
    def test_sbox_variance(self):
        """S-box variance should be 1 (balanced function)."""
        f = get_sbox_component(AES_SBOX, 0)
        var = f.variance()
        
        # For balanced ±1 function, Var[f] = 1
        assert abs(var - 1.0) < 0.01


class TestSboxInfluences:
    """Test influence analysis on S-box components."""
    
    def test_sbox_total_influence(self):
        """S-box should have reasonably high total influence."""
        f = get_sbox_component(AES_SBOX, 0)
        
        # High degree functions typically have high total influence
        total_inf = f.total_influence()
        assert total_inf >= 2.0, f"Total influence {total_inf} too low"
    
    def test_sbox_influences_spread(self):
        """S-box influences should be spread across variables."""
        f = get_sbox_component(AES_SBOX, 0)
        influences = f.influences()
        
        # Check that no single variable dominates
        max_inf = max(influences)
        total_inf = sum(influences)
        
        # No variable should have more than 50% of total influence
        assert max_inf < 0.5 * total_inf


class TestSboxQueryComplexity:
    """Test query complexity measures on S-box components."""
    
    def test_sbox_decision_tree_depth(self):
        """S-box components should need many queries."""
        from boofun.analysis.query_complexity import deterministic_query_complexity
        
        f = get_sbox_component(AES_SBOX, 0)
        D = deterministic_query_complexity(f)
        
        # High degree function, should need close to n queries
        assert D >= 5, f"D(f)={D} unexpectedly low"
    
    def test_sbox_sensitivity(self):
        """S-box components should have high sensitivity."""
        f = get_sbox_component(AES_SBOX, 0)
        sens = f.sensitivity()
        
        # Good S-boxes have high sensitivity
        assert sens >= 4, f"Sensitivity {sens} too low"


class TestSimpleSbox:
    """Tests using smaller S-boxes for faster execution."""
    
    def test_4bit_sbox_analysis(self):
        """Test analysis on a 4-bit S-box (faster)."""
        # Simple 4-bit S-box 
        sbox_4bit = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        f = bf.create(sbox_4bit)
        
        assert f.n_vars == 4
        assert f.is_balanced()
        
        # Fourier analysis
        fourier = f.fourier()
        assert abs(sum(c**2 for c in fourier) - 1.0) < 1e-10
        
        # Property testing
        tester = PropertyTester(f, random_seed=42)
        results = tester.run_all_tests()
        
        assert 'linear' in results
        assert 'monotone' in results
    
    def test_balanced_4bit(self):
        """Test a balanced 4-bit function."""
        # A balanced function 
        balanced = [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]
        f = bf.create(balanced)
        
        # Should be balanced (8 zeros, 8 ones)
        assert f.is_balanced()
        
        # Test that nonlinearity is non-negative
        nl = compute_nonlinearity(f)
        assert nl >= 0


class TestCrossValidation:
    """Cross-validate S-box properties with known results."""
    
    def test_aes_known_properties(self):
        """AES S-box has well-documented properties."""
        # AES S-box component 0 properties from literature
        f = get_sbox_component(AES_SBOX, 0)
        
        # Known: high degree (Fourier degree can be 8, algebraic degree 7)
        assert f.degree() >= 6
        
        # Known: balanced
        assert f.is_balanced()
        
        # Known: nonlinearity = 112
        nl = compute_nonlinearity(f)
        assert nl == 112, f"Expected nonlinearity 112, got {nl}"
    
    def test_all_aes_components_nl_112(self):
        """All AES S-box components should have nonlinearity 112."""
        for bit in range(8):
            f = get_sbox_component(AES_SBOX, bit)
            nl = compute_nonlinearity(f)
            assert nl == 112, f"Component {bit}: expected NL=112, got {nl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
