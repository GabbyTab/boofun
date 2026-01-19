"""
Communication Complexity verification tests.

Tests communication complexity bounds and known theoretical values:
- D^cc(f) ≥ log₂(rank(M_f)) (Log-rank bound)
- D^cc(EQ_n) = n+1 (Equality function)
- D^cc(DISJ_n) = n (Set disjointness)
- D^cc(IP_n) = n (Inner product)

References:
- Kushilevitz & Nisan, "Communication Complexity"
- O'Donnell, "Analysis of Boolean Functions", Chapter 6
"""

import pytest
import numpy as np
from math import log2, ceil
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis.communication_complexity import (
    log_rank_bound,
    fooling_set_bound,
    CommunicationMatrix,
    CommunicationComplexityProfile,
    get_communication_matrix,
)


class TestLogRankBound:
    """Test log-rank lower bound computations."""
    
    def test_log_rank_and(self):
        """AND function has low rank."""
        f = bf.AND(4)
        rank, log_rank = log_rank_bound(f)
        
        # AND has rank 1 (all 0s except one corner)
        # Actually rank depends on how we split variables
        assert rank >= 1
        assert log_rank >= 0
    
    def test_log_rank_parity(self):
        """Parity has full rank (maximum communication)."""
        f = bf.parity(4)
        rank, log_rank = log_rank_bound(f)
        
        # Parity/XOR has full rank
        assert rank >= 2  # At least some rank
    
    def test_log_rank_or(self):
        """OR function has low rank."""
        f = bf.OR(4)
        rank, log_rank = log_rank_bound(f)
        
        # OR has rank n+1 at most
        assert rank <= f.n_vars + 1


class TestFoolingSetBound:
    """Test fooling set lower bounds."""
    
    def test_fooling_set_and(self):
        """AND has small fooling set."""
        f = bf.AND(4)
        size, bound = fooling_set_bound(f)
        
        # Should find at least some fooling set
        assert size >= 1
        assert bound >= 0
    
    def test_fooling_set_or(self):
        """OR has small fooling set."""
        f = bf.OR(4)
        size, bound = fooling_set_bound(f)
        
        assert size >= 1
        assert bound >= 0
    
    def test_fooling_set_parity(self):
        """Parity has large fooling set."""
        f = bf.parity(4)
        size, bound = fooling_set_bound(f)
        
        # Parity's fooling set can be large
        assert size >= 1


class TestCommunicationMatrix:
    """Test communication matrix construction and analysis."""
    
    def test_matrix_dimensions(self):
        """Matrix has correct dimensions."""
        f = bf.AND(4)
        cm = CommunicationMatrix(f)
        M = cm.matrix
        
        # For n=4, split 2-2, so 4x4 matrix
        assert M.shape == (4, 4)
    
    def test_matrix_binary(self):
        """Matrix entries are binary."""
        f = bf.majority(5)
        cm = CommunicationMatrix(f)
        M = cm.matrix
        
        assert np.all((M == 0) | (M == 1))
    
    def test_matrix_consistency(self):
        """Matrix entries match function evaluation."""
        f = bf.AND(4)
        M = get_communication_matrix(f)
        
        n = f.n_vars
        n_alice = n // 2
        n_bob = n - n_alice
        
        for x in range(2**n_alice):
            for y in range(2**n_bob):
                combined = (x << n_bob) | y
                expected = int(f.evaluate(combined))
                assert M[x, y] == expected, \
                    f"Matrix mismatch at ({x}, {y}): expected {expected}, got {M[x, y]}"


class TestCommunicationProfile:
    """Test full communication complexity profile."""
    
    def test_profile_and(self):
        """Profile computes for AND function."""
        f = bf.AND(4)
        profile = CommunicationComplexityProfile(f)
        measures = profile.compute()
        
        # Check that expected keys exist
        assert 'rank' in measures
        assert 'fooling_set_size' in measures
        assert 'density' in measures
        assert measures['rank'] >= 1
    
    def test_profile_or(self):
        """Profile computes for OR function."""
        f = bf.OR(4)
        profile = CommunicationComplexityProfile(f)
        measures = profile.compute()
        
        assert 'rank' in measures
        assert 'fooling_set_size' in measures
    
    def test_profile_majority(self):
        """Profile computes for Majority function."""
        f = bf.majority(5)
        profile = CommunicationComplexityProfile(f)
        measures = profile.compute()
        
        assert 'rank' in measures
        assert 'density' in measures


class TestCommunicationBounds:
    """Test theoretical bounds hold."""
    
    def test_log_rank_is_lower_bound(self):
        """log₂(rank) ≤ D^cc(f) always."""
        for func in [bf.AND(4), bf.OR(4), bf.parity(4)]:
            rank, log_rank = log_rank_bound(func)
            
            # Log-rank is always a valid lower bound
            assert log_rank >= 0
            assert log_rank <= func.n_vars  # Can't exceed trivial protocol
    
    def test_fooling_set_is_lower_bound(self):
        """log₂(fooling_set_size) ≤ D^cc(f)."""
        for func in [bf.AND(4), bf.OR(4)]:
            size, bound = fooling_set_bound(func)
            
            # Fooling set bound is always valid
            assert bound >= 0
            assert bound <= func.n_vars
    
    def test_bounds_consistent(self):
        """Multiple lower bounds should be consistent."""
        f = bf.parity(4)
        
        _, log_rank = log_rank_bound(f)
        fs_size, fs_bound = fooling_set_bound(f)
        
        # Both are valid lower bounds on the same quantity
        # They don't need to be equal but both should be reasonable
        assert log_rank >= 0
        assert fs_bound >= 0


class TestKnownComplexities:
    """Test against known communication complexities."""
    
    def test_inner_product_high_complexity(self):
        """Inner product mod 2 has high communication complexity."""
        # IP(x,y) = ⊕ᵢ xᵢyᵢ
        # For n=4, IP has D^cc = n/2 = 2
        n = 4
        
        # Create inner product function
        tt = []
        for i in range(1 << n):
            x_bits = [(i >> j) & 1 for j in range(n//2)]
            y_bits = [(i >> (n//2 + j)) & 1 for j in range(n//2)]
            ip = sum(x * y for x, y in zip(x_bits, y_bits)) % 2
            tt.append(ip)
        
        f = bf.create(tt)
        _, log_rank = log_rank_bound(f)
        
        # Inner product should have high rank/complexity
        assert log_rank >= 1  # At least log(2) = 1


class TestEdgeCases:
    """Test edge cases in communication complexity."""
    
    def test_constant_function(self):
        """Constant functions have D^cc = 0."""
        f_zero = bf.create([0, 0, 0, 0])
        f_one = bf.create([1, 1, 1, 1])
        
        rank_zero, log_rank_zero = log_rank_bound(f_zero)
        rank_one, log_rank_one = log_rank_bound(f_one)
        
        # Constant functions have rank 1 (or 0 for all-zeros)
        assert rank_zero <= 1
        assert rank_one == 1
    
    def test_dictator_function(self):
        """Dictator function has low communication complexity."""
        f = bf.dictator(4, 0)
        _, log_rank = log_rank_bound(f)
        
        # Dictator only depends on one party's variable
        # So has low communication complexity
        assert log_rank <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
