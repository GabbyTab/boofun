"""
Integration tests for newly added features.

Tests:
- Communication complexity
- Decision tree export
- LaTeX/TikZ export
- New function families (IteratedMajority, RandomDNF, Sbox)
- Quantum walks
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boolfunc as bf


class TestCommunicationComplexity:
    """Tests for communication complexity module."""
    
    def test_log_rank_and_function(self):
        """AND function has rank 1."""
        from boolfunc.analysis.communication_complexity import log_rank_bound
        
        f = bf.AND(4)
        rank, bound = log_rank_bound(f)
        
        # AND has rank 1 (all 0s except one 1)
        assert rank == 1
        assert bound == 0.0
    
    def test_log_rank_parity(self):
        """Parity has full rank."""
        from boolfunc.analysis.communication_complexity import log_rank_bound
        
        f = bf.parity(4)
        rank, bound = log_rank_bound(f)
        
        # Parity has high rank
        assert rank >= 2
        assert bound >= 1.0
    
    def test_fooling_set_bound(self):
        """Fooling set provides valid lower bound."""
        from boolfunc.analysis.communication_complexity import fooling_set_bound
        
        f = bf.majority(5)
        size, bound = fooling_set_bound(f)
        
        assert size >= 1
        assert bound >= 0
    
    def test_communication_matrix(self):
        """Communication matrix has correct structure."""
        from boolfunc.analysis.communication_complexity import CommunicationMatrix
        
        f = bf.AND(4)
        cm = CommunicationMatrix(f)
        
        assert cm.matrix.shape == (4, 4)  # 2^2 x 2^2
        assert cm.density() == 1/16  # Only one 1 in AND
    
    def test_communication_profile(self):
        """Full profile computation works."""
        from boolfunc.analysis.communication_complexity import CommunicationComplexityProfile
        
        f = bf.OR(4)
        profile = CommunicationComplexityProfile(f)
        result = profile.compute()
        
        assert 'rank' in result
        assert 'log_rank_lower_bound' in result
        assert 'fooling_set_size' in result
        assert result['lower_bound'] >= 0


class TestDecisionTreeExport:
    """Tests for decision tree export functionality."""
    
    def test_export_text(self):
        """Text export produces valid output."""
        from boolfunc.visualization.decision_tree_export import export_decision_tree_text
        
        f = bf.AND(3)
        text = export_decision_tree_text(f)
        
        assert len(text) > 0
        assert "?" in text  # Decision nodes have ?
        assert "[" in text  # Leaf nodes have []
    
    def test_export_dot(self):
        """DOT export produces valid Graphviz."""
        from boolfunc.visualization.decision_tree_export import export_decision_tree_dot
        
        f = bf.OR(3)
        dot = export_decision_tree_dot(f)
        
        assert "digraph" in dot
        assert "node" in dot
        assert "->" in dot  # Edges
    
    def test_export_json(self):
        """JSON export produces valid structure."""
        from boolfunc.visualization.decision_tree_export import export_decision_tree_json
        
        f = bf.majority(3)
        data = export_decision_tree_json(f)
        
        assert 'function' in data
        assert 'tree' in data
        assert data['function']['n_vars'] == 3
    
    def test_export_tikz(self):
        """TikZ export produces valid LaTeX."""
        from boolfunc.visualization.decision_tree_export import export_decision_tree_tikz
        
        f = bf.parity(3)
        tikz = export_decision_tree_tikz(f)
        
        assert "tikzpicture" in tikz
        assert "child" in tikz
    
    def test_exporter_class(self):
        """DecisionTreeExporter class works."""
        from boolfunc.visualization.decision_tree_export import DecisionTreeExporter
        
        f = bf.AND(3)
        exporter = DecisionTreeExporter(f, var_names=['a', 'b', 'c'])
        
        assert exporter.depth() >= 1
        assert exporter.size() >= 3
        assert "a" in exporter.to_text()


class TestLaTeXExport:
    """Tests for LaTeX/TikZ export."""
    
    def test_fourier_tikz(self):
        """Fourier spectrum TikZ export works."""
        from boolfunc.visualization.latex_export import export_fourier_tikz
        
        f = bf.majority(3)
        tikz = export_fourier_tikz(f)
        
        assert "tikzpicture" in tikz
        assert "axis" in tikz
        assert "addplot" in tikz
    
    def test_influences_tikz(self):
        """Influence chart TikZ export works."""
        from boolfunc.visualization.latex_export import export_influences_tikz
        
        f = bf.parity(4)
        tikz = export_influences_tikz(f)
        
        assert "tikzpicture" in tikz
        assert "bar" in tikz.lower()
    
    def test_cube_tikz(self):
        """Boolean cube TikZ export works."""
        from boolfunc.visualization.latex_export import export_cube_tikz
        
        for n in [2, 3, 4]:
            tikz = export_cube_tikz(n)
            assert "tikzpicture" in tikz
            assert "draw" in tikz
    
    def test_spectrum_table(self):
        """Spectrum table export works."""
        from boolfunc.visualization.latex_export import export_spectrum_table
        
        f = bf.AND(3)
        table = export_spectrum_table(f)
        
        assert "tabular" in table
        assert "hline" in table
    
    def test_comparison_table(self):
        """Comparison table works."""
        from boolfunc.visualization.latex_export import export_comparison_table
        
        functions = {
            "AND": bf.AND(4),
            "OR": bf.OR(4),
            "Maj": bf.majority(5),
        }
        table = export_comparison_table(functions)
        
        assert "tabular" in table
        assert "AND" in table
        assert "OR" in table
    
    def test_latex_exporter_class(self):
        """LaTeXExporter class works."""
        from boolfunc.visualization.latex_export import LaTeXExporter
        
        f = bf.majority(3)
        exporter = LaTeXExporter(f)
        
        assert len(exporter.fourier_spectrum()) > 100
        assert len(exporter.influences()) > 100
        assert "pgfplots" in exporter.preamble()


class TestNewFamilies:
    """Tests for new function families."""
    
    def test_iterated_majority(self):
        """Iterated majority family works."""
        from boolfunc.families.builtins import IteratedMajorityFamily
        
        family = IteratedMajorityFamily(group_size=3)
        
        # Valid n = 9 (3^2)
        f = family.generate(9)
        assert f.n_vars == 9
        
        # Check monotonicity
        from boolfunc.analysis import PropertyTester
        tester = PropertyTester(f)
        assert tester.monotonicity_test(num_queries=500)
    
    def test_iterated_majority_influence(self):
        """Iterated majority has expected influence scaling."""
        from boolfunc.families.builtins import IteratedMajorityFamily
        
        family = IteratedMajorityFamily(group_size=3)
        
        # Compare n=3 and n=9
        f3 = family.generate(3)
        f9 = family.generate(9)
        
        # Influence should scale as n^{log_3(2)} ≈ n^0.63
        inf3 = f3.total_influence()
        inf9 = f9.total_influence()
        
        # Check rough scaling (allowing tolerance)
        ratio = inf9 / inf3
        expected_ratio = (9/3) ** (np.log(2) / np.log(3))  # ≈ 1.89
        assert 1.4 <= ratio <= 2.5
    
    def test_random_dnf(self):
        """Random DNF family works."""
        from boolfunc.families.builtins import RandomDNFFamily
        
        family = RandomDNFFamily(term_width=3)
        
        # Generate with seed for reproducibility
        f1 = family.generate(5, seed=42)
        f2 = family.generate(5, seed=42)
        
        # Same seed should give same function
        tt1 = f1.get_representation("truth_table")
        tt2 = f2.get_representation("truth_table")
        assert np.array_equal(tt1, tt2)
        
        # Different seed should (likely) give different function
        f3 = family.generate(5, seed=123)
        tt3 = f3.get_representation("truth_table")
        # Note: could be equal by chance, but unlikely
    
    def test_sbox_family(self):
        """S-box family works."""
        from boolfunc.families.builtins import SboxFamily
        
        family = SboxFamily.aes(bit=0)
        f = family.generate()
        
        assert f.n_vars == 8  # AES S-box is 8-bit
        
        # Check balancedness (good S-box components are balanced)
        tt = f.get_representation("truth_table")
        ones = sum(1 for x in tt if x)
        assert 100 < ones < 156  # Should be roughly balanced
    
    def test_sbox_all_components(self):
        """Can get all S-box components."""
        from boolfunc.families.builtins import SboxFamily
        
        family = SboxFamily.aes()
        components = family.all_components()
        
        assert len(components) == 8  # 8-bit S-box has 8 components
        
        for comp in components:
            assert comp.n_vars == 8


class TestQuantumWalks:
    """Tests for quantum walk algorithms."""
    
    def test_quantum_walk_analysis(self):
        """Quantum walk analysis works."""
        from boolfunc.quantum import quantum_walk_analysis
        
        f = bf.AND(4)
        result = quantum_walk_analysis(f)
        
        assert 'spectral_gap' in result
        assert 'mixing_time' in result
        assert 'quantum_walk_complexity' in result
        assert result['speedup_over_classical'] > 1
    
    def test_element_distinctness(self):
        """Element distinctness analysis works."""
        from boolfunc.quantum import element_distinctness_analysis
        
        f = bf.parity(4)
        result = element_distinctness_analysis(f)
        
        assert 'has_collision' in result
        assert 'quantum_complexity' in result
        assert result['speedup'] > 1
    
    def test_quantum_walk_search(self):
        """Quantum walk search simulation works."""
        from boolfunc.quantum import quantum_walk_search
        
        f = bf.OR(4)  # Has 15 marked vertices
        result = quantum_walk_search(f)
        
        assert 'marked_vertices' in result
        assert 'max_success_probability' in result
        assert result['marked_vertices'] == 15
        assert result['max_success_probability'] > 0.5


class TestAutoRepresentation:
    """Tests for auto representation selection."""
    
    def test_recommend_small_n(self):
        """Small n recommends dense."""
        from boolfunc.core.auto_representation import recommend_representation
        
        rec = recommend_representation(8)
        assert rec['representation'] == 'truth_table'
    
    def test_recommend_large_n(self):
        """Large n recommends packed."""
        from boolfunc.core.auto_representation import recommend_representation
        
        rec = recommend_representation(18)
        assert rec['representation'] == 'packed_truth_table'
    
    def test_recommend_sparse(self):
        """Sparse function recommends sparse."""
        from boolfunc.core.auto_representation import recommend_representation
        
        rec = recommend_representation(20, sparsity=0.05)
        assert rec['representation'] == 'sparse_truth_table'
    
    def test_adaptive_function(self):
        """AdaptiveFunction works."""
        from boolfunc.core.auto_representation import AdaptiveFunction
        
        # Create a sparse function (AND)
        f = bf.AND(8)
        tt = np.array([int(f.evaluate(x)) for x in range(256)], dtype=bool)
        
        adaptive = AdaptiveFunction(tt, n_vars=8)
        
        # Should evaluate correctly
        assert adaptive.evaluate(255) == True  # All 1s
        assert adaptive.evaluate(0) == False
        
        # Should report format and sparsity
        assert adaptive.format in ['dense', 'packed', 'sparse']
        assert 0 <= adaptive.sparsity <= 0.5


class TestParallelOptimizations:
    """Tests for parallel and batch operations."""
    
    def test_parallel_batch_influences(self):
        """Parallel influence computation works."""
        from boolfunc.core.optimizations import parallel_batch_influences
        
        functions = [bf.AND(6), bf.OR(6), bf.parity(6), bf.majority(7)]
        results = parallel_batch_influences(functions)
        
        assert len(results) == 4
        assert len(results[0]) == 6
        assert len(results[3]) == 7
    
    def test_compute_cache(self):
        """Compute cache works."""
        from boolfunc.core.optimizations import ComputeCache
        
        cache = ComputeCache(max_size=10)
        
        # Put and get
        cache.put("hash1", "influences", [0.1, 0.2])
        found, val = cache.get("hash1", "influences")
        
        assert found
        assert val == [0.1, 0.2]
        
        # Miss
        found2, _ = cache.get("nonexistent", "influences")
        assert not found2
        
        # Stats
        stats = cache.stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
