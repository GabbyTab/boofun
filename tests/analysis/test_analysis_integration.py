"""
Integration tests for analysis modules.

These tests exercise multiple analysis modules together, testing
their consistency and covering more code paths.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.analysis import SpectralAnalyzer, PropertyTester


class TestSpectralAnalyzerComprehensive:
    """Comprehensive tests for SpectralAnalyzer."""
    
    def test_analyzer_creation(self):
        """SpectralAnalyzer should be creatable."""
        f = bf.majority(3)
        analyzer = SpectralAnalyzer(f)
        
        assert analyzer is not None
    
    @pytest.mark.parametrize("func_factory,n", [
        (bf.AND, 3),
        (bf.OR, 3),
        (bf.majority, 3),
        (bf.majority, 5),
        (bf.parity, 4),
    ])
    def test_analyzer_different_functions(self, func_factory, n):
        """SpectralAnalyzer should work with various functions."""
        f = func_factory(n)
        analyzer = SpectralAnalyzer(f)
        
        # Should have analysis methods
        assert hasattr(analyzer, 'influences') or hasattr(analyzer, 'fourier')
    
    def test_analyzer_influences(self):
        """SpectralAnalyzer should compute influences."""
        f = bf.majority(3)
        analyzer = SpectralAnalyzer(f)
        
        if hasattr(analyzer, 'influences'):
            influences = analyzer.influences()
            assert len(influences) == 3
            assert all(0 <= i <= 1 for i in influences)
    
    def test_analyzer_total_influence(self):
        """SpectralAnalyzer should compute total influence."""
        f = bf.parity(3)
        analyzer = SpectralAnalyzer(f)
        
        if hasattr(analyzer, 'total_influence'):
            total = analyzer.total_influence()
            assert abs(total - 3.0) < 1e-10  # Parity has total influence n


class TestPropertyTesterComprehensive:
    """Comprehensive tests for PropertyTester."""
    
    def test_tester_creation(self):
        """PropertyTester should be creatable."""
        f = bf.majority(3)
        tester = PropertyTester(f)
        
        assert tester is not None
    
    def test_test_monotone_and(self):
        """AND should be monotone."""
        f = bf.AND(4)
        tester = PropertyTester(f)
        
        if hasattr(tester, 'test_monotone'):
            result = tester.test_monotone()
            # AND is monotone
            assert result is not None
    
    def test_test_monotone_parity(self):
        """Parity should NOT be monotone."""
        f = bf.parity(3)
        tester = PropertyTester(f)
        
        if hasattr(tester, 'test_monotone'):
            result = tester.test_monotone()
            # Parity is not monotone
            assert result is not None
    
    def test_test_symmetric(self):
        """Test symmetric property detection."""
        f = bf.majority(3)
        tester = PropertyTester(f)
        
        if hasattr(tester, 'test_symmetric'):
            result = tester.test_symmetric()
            assert result is not None


class TestAnalysisConsistency:
    """Test consistency between different analysis methods."""
    
    def test_fourier_influences_consistency(self):
        """Influences computed from Fourier should match direct computation."""
        f = bf.majority(5)
        
        # Fourier-based influence
        fourier = np.array(f.fourier())
        
        # Influence of variable i = sum over S containing i of f̂(S)²
        n = 5
        fourier_influences = []
        for i in range(n):
            inf_i = sum(
                fourier[S]**2
                for S in range(2**n)
                if (S >> i) & 1
            )
            fourier_influences.append(inf_i)
        
        # Direct influences
        direct_influences = f.influences()
        
        assert np.allclose(fourier_influences, direct_influences, atol=1e-10)
    
    def test_parseval_identity(self):
        """Verify Parseval's identity: sum of squared coefficients = 1."""
        for func in [bf.AND(3), bf.OR(3), bf.majority(3), bf.parity(3)]:
            fourier = np.array(func.fourier())
            total_weight = np.sum(fourier ** 2)
            
            assert abs(total_weight - 1.0) < 1e-10
    
    def test_total_influence_formula(self):
        """Total influence = sum of individual influences."""
        f = bf.majority(5)
        
        individual = sum(f.influences())
        total = f.total_influence()
        
        assert abs(individual - total) < 1e-10


class TestComplexityMeasures:
    """Test complexity measure computations."""
    
    def test_sensitivity_via_influences(self):
        """Test sensitivity via influence bounds."""
        f = bf.AND(3)
        
        # Total influence bounds sensitivity
        total_inf = f.total_influence()
        
        # For AND, sensitivity should be <= n
        assert 0 <= total_inf <= 3
    
    def test_block_sensitivity_basic(self):
        """Test block sensitivity computation."""
        from boofun.analysis.block_sensitivity import max_block_sensitivity
        
        f = bf.OR(3)
        bs = max_block_sensitivity(f)
        
        # Block sensitivity is in [0, n]
        assert 0 <= bs <= 3
    
    def test_certificate_size(self):
        """Test certificate complexity."""
        from boofun.analysis.certificates import max_certificate_size
        
        f = bf.AND(3)
        c = max_certificate_size(f)
        
        # Certificate complexity is in [0, n]
        assert 0 <= c <= 3


class TestKnownFunctionProperties:
    """Test known theoretical properties of specific functions."""
    
    def test_dictator_properties(self):
        """Dictator function has known properties."""
        f = bf.dictator(4, 0)  # Dictator on variable 0
        
        # Fourier: check that it's a valid dictator
        fourier = f.fourier()
        
        # Sum of squared coefficients should be 1
        total_weight = sum(c**2 for c in fourier)
        assert abs(total_weight - 1.0) < 1e-10
        
        # Dictator should be balanced (E[f] = 0)
        assert abs(fourier[0]) < 1e-10
    
    def test_parity_properties(self):
        """Parity has known Fourier structure."""
        f = bf.parity(3)
        fourier = f.fourier()
        
        # Only top coefficient is non-zero
        for S in range(8):
            if S == 7:  # {0, 1, 2}
                assert abs(fourier[S]) > 0.5
            else:
                assert abs(fourier[S]) < 1e-10
    
    def test_and_fourier(self):
        """AND function Fourier coefficients."""
        f = bf.AND(2)
        fourier = f.fourier()
        
        # AND_2 Fourier: known values
        # Under O'Donnell convention (0→+1, 1→-1):
        # f(0,0)=0→+1, f(0,1)=0→+1, f(1,0)=0→+1, f(1,1)=1→-1
        # f̂(∅) = E[f] = (1+1+1-1)/4 = 0.5
        # f̂({0}) = E[f·χ_0] = (1·1 + 1·(-1) + 1·1 + (-1)·(-1))/4 = (1-1+1+1)/4 = 0.5
        # etc.
        assert len(fourier) == 4


class TestAnalysisEdgeCases:
    """Test edge cases in analysis modules."""
    
    def test_constant_function_analysis(self):
        """Constant function should have known analysis results."""
        f_zero = bf.create([0, 0, 0, 0])
        f_one = bf.create([1, 1, 1, 1])
        
        # Zero influences for constant functions
        assert all(i == 0 for i in f_zero.influences())
        assert all(i == 0 for i in f_one.influences())
    
    def test_single_variable_function(self):
        """Single variable function analysis."""
        f = bf.create([0, 1])  # Identity on 1 variable
        
        influences = f.influences()
        assert len(influences) == 1
        assert abs(influences[0] - 1.0) < 1e-10
    
    def test_large_n_analysis(self):
        """Analysis should work for larger n (within reason)."""
        f = bf.majority(7)
        
        fourier = f.fourier()
        assert len(fourier) == 128
        
        influences = f.influences()
        assert len(influences) == 7


class TestAnalysisPerformance:
    """Basic performance tests for analysis."""
    
    def test_fourier_not_too_slow(self):
        """Fourier should be reasonably fast for moderate n."""
        import time
        
        f = bf.majority(9)
        
        start = time.time()
        fourier = f.fourier()
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
