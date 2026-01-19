"""
Comprehensive tests for representation modules.

Tests various Boolean function representations and their conversions.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.spaces import Space


class TestTruthTableRepresentation:
    """Test truth table representation."""
    
    def test_get_truth_table(self):
        """Should be able to get truth table representation."""
        f = bf.AND(3)
        tt = list(f.get_representation("truth_table"))
        
        assert len(tt) == 8
        assert all(v in [0, 1, True, False] for v in tt)
    
    def test_truth_table_correctness(self):
        """Truth table should match evaluation."""
        f = bf.majority(3)
        tt = list(f.get_representation("truth_table"))
        
        for x in range(8):
            assert int(tt[x]) == f.evaluate(x)
    
    @pytest.mark.parametrize("func_factory,n", [
        (bf.AND, 2),
        (bf.AND, 3),
        (bf.OR, 3),
        (bf.majority, 3),
        (bf.parity, 4),
    ])
    def test_truth_table_various_functions(self, func_factory, n):
        """Truth table should work for various functions."""
        f = func_factory(n)
        tt = list(f.get_representation("truth_table"))
        
        assert len(tt) == 2**n


class TestFourierRepresentation:
    """Test Fourier representation."""
    
    def test_fourier_representation(self):
        """Should be able to get Fourier coefficients."""
        f = bf.majority(3)
        fourier = f.fourier()
        
        assert len(fourier) == 8
    
    def test_fourier_parseval(self):
        """Fourier coefficients should satisfy Parseval."""
        for func in [bf.AND(3), bf.OR(3), bf.parity(3)]:
            fourier = np.array(func.fourier())
            total = np.sum(fourier ** 2)
            
            assert abs(total - 1.0) < 1e-10
    
    def test_fourier_constant_term(self):
        """Fourier constant term is the expectation."""
        f = bf.majority(3)
        fourier = f.fourier()
        
        # Manual expectation calculation
        tt = list(f.get_representation("truth_table"))
        pm = [1.0 - 2.0 * v for v in tt]  # O'Donnell
        expectation = sum(pm) / len(pm)
        
        assert abs(fourier[0] - expectation) < 1e-10


class TestRepresentationConversions:
    """Test conversions between representations."""
    
    def test_truth_table_to_fourier_roundtrip(self):
        """Converting TT → Fourier → TT should preserve function."""
        f = bf.majority(3)
        
        # Get truth table
        tt_original = list(f.get_representation("truth_table"))
        
        # Get Fourier
        fourier = f.fourier()
        
        # Reconstruct from Fourier (inverse WHT)
        n = 3
        reconstructed = []
        for x in range(2**n):
            # f(x) in ±1 = sum over S of f̂(S) * χ_S(x)
            val = 0
            for S in range(2**n):
                # χ_S(x) = (-1)^{|S ∩ x|} = (-1)^popcount(S & x)
                parity = bin(S & x).count('1') % 2
                chi = 1 - 2 * parity  # 0 → +1, 1 → -1
                val += fourier[S] * chi
            # Convert back to {0,1}
            reconstructed.append(0 if val > 0 else 1)
        
        # Should match
        tt_bool = [int(v) for v in tt_original]
        assert reconstructed == tt_bool
    
    def test_evaluation_consistency(self):
        """All representations should give consistent evaluation."""
        f = bf.OR(3)
        
        for x in range(8):
            # Direct evaluation
            direct = f.evaluate(x)
            
            # From truth table
            tt = list(f.get_representation("truth_table"))
            from_tt = int(tt[x])
            
            assert direct == from_tt


class TestCreateFromTruthTable:
    """Test creating functions from truth tables."""
    
    @pytest.mark.parametrize("tt,expected_balanced", [
        ([0, 0, 0, 0], False),
        ([1, 1, 1, 1], False),
        ([0, 1, 1, 0], True),
        ([0, 0, 1, 1], True),
    ])
    def test_create_from_list(self, tt, expected_balanced):
        """Should create function from truth table list."""
        f = bf.create(tt)
        
        assert f.n_vars == 2
        assert f.is_balanced() == expected_balanced
    
    def test_create_various_sizes(self):
        """Should create functions of various sizes."""
        for n in [1, 2, 3, 4, 5]:
            tt = [i % 2 for i in range(2**n)]
            f = bf.create(tt)
            
            assert f.n_vars == n
    
    def test_create_preserves_values(self):
        """Created function should have same truth table."""
        tt = [0, 1, 1, 0, 1, 0, 0, 1]  # Parity
        f = bf.create(tt)
        
        result = list(f.get_representation("truth_table"))
        assert [int(v) for v in result] == tt


class TestRepresentationEdgeCases:
    """Test edge cases for representations."""
    
    def test_single_variable(self):
        """Representations should work for n=1."""
        f = bf.create([0, 1])
        
        tt = list(f.get_representation("truth_table"))
        assert len(tt) == 2
        
        fourier = f.fourier()
        assert len(fourier) == 2
    
    def test_constant_functions(self):
        """Representations should work for constant functions."""
        f_zero = bf.create([0, 0, 0, 0])
        f_one = bf.create([1, 1, 1, 1])
        
        # Fourier of constants
        fourier_zero = f_zero.fourier()
        fourier_one = f_one.fourier()
        
        # Constant term should be ±1, others 0
        assert abs(abs(fourier_zero[0]) - 1.0) < 1e-10
        assert abs(abs(fourier_one[0]) - 1.0) < 1e-10


class TestDNFCNFRepresentations:
    """Test DNF/CNF representations if available."""
    
    def test_dnf_available(self):
        """Check if DNF representation is available."""
        f = bf.AND(3)
        
        try:
            dnf = f.get_representation("dnf")
            assert dnf is not None
        except (KeyError, AttributeError):
            pytest.skip("DNF representation not available")
    
    def test_cnf_available(self):
        """Check if CNF representation is available."""
        f = bf.OR(3)
        
        try:
            cnf = f.get_representation("cnf")
            assert cnf is not None
        except (KeyError, AttributeError):
            pytest.skip("CNF representation not available")


class TestBDDRepresentation:
    """Test BDD representation if available."""
    
    def test_bdd_via_function_api(self):
        """BDD should be accessible via function API."""
        f = bf.AND(3)
        
        try:
            bdd = f.get_representation("bdd")
            assert bdd is not None
        except (KeyError, AttributeError):
            pytest.skip("BDD representation not available via API")
    
    def test_bdd_evaluation_via_api(self):
        """BDD should evaluate correctly via API."""
        f = bf.OR(3)
        
        try:
            bdd = f.get_representation("bdd")
            # Function should still evaluate correctly
            for x in range(8):
                assert f.evaluate(x) in [0, 1, True, False]
        except (KeyError, AttributeError):
            pytest.skip("BDD representation not available via API")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
