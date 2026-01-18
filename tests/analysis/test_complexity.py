"""Tests for complexity measures module."""

import pytest
import boolfunc as bf
from boolfunc.analysis import complexity


class TestDecisionTreeDepth:
    """Tests for decision tree depth computation."""
    
    def test_constant_function(self):
        """Constant functions have depth 0."""
        const_0 = bf.BooleanFunctionBuiltins.constant(False, 3)
        const_1 = bf.BooleanFunctionBuiltins.constant(True, 3)
        
        assert complexity.decision_tree_depth(const_0) == 0
        assert complexity.decision_tree_depth(const_1) == 0
    
    def test_dictator_function(self):
        """Dictator functions have depth 1."""
        for n in [2, 3, 4]:
            for i in range(n):
                dictator = bf.BooleanFunctionBuiltins.dictator(n, i)
                assert complexity.decision_tree_depth(dictator) == 1
    
    def test_parity_function(self):
        """Parity requires all n queries."""
        for n in [2, 3, 4]:
            parity = bf.BooleanFunctionBuiltins.parity(n)
            assert complexity.decision_tree_depth(parity) == n
    
    def test_majority_3(self):
        """3-variable majority needs 3 queries in worst case."""
        majority_3 = bf.BooleanFunctionBuiltins.majority(3)
        depth = complexity.decision_tree_depth(majority_3)
        # Majority-3 requires 3 queries in the worst case
        # (when first two queried bits differ, need to query the third)
        assert depth == 3
    
    def test_and_function(self):
        """AND function on n vars has depth n."""
        and_2 = bf.create([0, 0, 0, 1])  # x0 AND x1
        assert complexity.decision_tree_depth(and_2) == 2
    
    def test_or_function(self):
        """OR function on n vars has depth n."""
        or_2 = bf.create([0, 1, 1, 1])  # x0 OR x1
        assert complexity.decision_tree_depth(or_2) == 2


class TestDecisionTreeSize:
    """Tests for decision tree size computation."""
    
    def test_constant_has_size_1(self):
        """Constant function has size 1 (single leaf)."""
        const = bf.BooleanFunctionBuiltins.constant(False, 2)
        size, depth = complexity.decision_tree_size(const)
        assert size == 1
        assert depth == 0
    
    def test_dictator_has_size_2(self):
        """Dictator has size 2 (two leaves)."""
        dictator = bf.BooleanFunctionBuiltins.dictator(2, 0)
        size, depth = complexity.decision_tree_size(dictator)
        assert size == 2
        assert depth == 1


class TestSensitivity:
    """Tests for sensitivity computation."""
    
    def test_sensitivity_xor(self):
        """XOR has sensitivity n at every point."""
        xor = bf.create([0, 1, 1, 0])
        
        for x in range(4):
            assert complexity.sensitivity(xor, x) == 2
    
    def test_sensitivity_and(self):
        """AND has varying sensitivity."""
        and_func = bf.create([0, 0, 0, 1])
        
        # At (0,0): flipping either bit doesn't change output -> sens = 0
        assert complexity.sensitivity(and_func, 0) == 0
        # At (1,1): flipping either bit changes output -> sens = 2
        assert complexity.sensitivity(and_func, 3) == 2
    
    def test_max_sensitivity_xor(self):
        """XOR has max sensitivity = n."""
        xor = bf.create([0, 1, 1, 0])
        assert complexity.max_sensitivity(xor) == 2
    
    def test_max_sensitivity_constant(self):
        """Constant function has max sensitivity = 0."""
        const = bf.BooleanFunctionBuiltins.constant(True, 3)
        assert complexity.max_sensitivity(const) == 0
    
    def test_average_sensitivity_xor(self):
        """XOR has average sensitivity = n."""
        xor = bf.create([0, 1, 1, 0])
        # Every point has sensitivity 2
        assert complexity.average_sensitivity(xor) == 2.0
    
    def test_average_sensitivity_equals_total_influence(self):
        """Average sensitivity should equal total influence."""
        majority = bf.BooleanFunctionBuiltins.majority(3)
        
        avg_sens = complexity.average_sensitivity(majority)
        
        analyzer = bf.SpectralAnalyzer(majority)
        total_inf = analyzer.total_influence()
        
        assert abs(avg_sens - total_inf) < 1e-10


class TestCertificateComplexity:
    """Tests for certificate complexity."""
    
    def test_certificate_constant(self):
        """Constant function has certificate size 0."""
        const = bf.BooleanFunctionBuiltins.constant(False, 2)
        
        for x in range(4):
            cert_size, _ = complexity.certificate_complexity(const, x)
            assert cert_size == 0
    
    def test_certificate_dictator(self):
        """Dictator has certificate size 1."""
        dictator = bf.BooleanFunctionBuiltins.dictator(3, 0)
        
        for x in range(8):
            cert_size, cert_vars = complexity.certificate_complexity(dictator, x)
            assert cert_size == 1
            # The algorithm finds the dictator variable (may vary by bit ordering)
            assert len(cert_vars) == 1
    
    def test_certificate_and(self):
        """AND function certificate analysis."""
        and_func = bf.create([0, 0, 0, 1])
        
        # At (1,1): both variables needed to certify 1
        cert_size, cert_vars = complexity.certificate_complexity(and_func, 3)
        assert cert_size == 2
        
        # At (0,0): just one 0 is enough to certify output is 0
        cert_size, cert_vars = complexity.certificate_complexity(and_func, 0)
        assert cert_size == 1


class TestComplexityProfile:
    """Tests for the ComplexityProfile class."""
    
    def test_profile_computation(self):
        """Test that profile computes all measures."""
        parity = bf.BooleanFunctionBuiltins.parity(3)
        profile = complexity.ComplexityProfile(parity)
        
        measures = profile.compute()
        
        assert "s" in measures
        assert "C" in measures
        assert "D" in measures
        assert "avg_sensitivity" in measures
    
    def test_profile_relations(self):
        """Test that complexity relations hold."""
        majority = bf.BooleanFunctionBuiltins.majority(3)
        profile = complexity.ComplexityProfile(majority)
        
        checks = profile.check_relations()
        
        # All known relations should hold
        for relation, holds in checks.items():
            assert holds, f"Relation '{relation}' failed"
    
    def test_profile_summary(self):
        """Test that summary is generated."""
        xor = bf.create([0, 1, 1, 0])
        profile = complexity.ComplexityProfile(xor)
        
        summary = profile.summary()
        
        assert "Sensitivity" in summary
        assert "Certificate" in summary
        assert "Decision Tree" in summary
