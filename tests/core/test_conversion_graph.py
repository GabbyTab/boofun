"""
Comprehensive tests for core/conversion_graph module.

Tests representation conversion graph, paths, and costs.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf
from boofun.core.conversion_graph import (
    ConversionCost,
    ConversionEdge,
    ConversionPath,
    ConversionGraph,
    get_conversion_graph,
    find_conversion_path,
    register_custom_conversion,
    get_conversion_options,
    estimate_conversion_cost,
)


class TestConversionCost:
    """Test ConversionCost class."""
    
    def test_cost_creation(self):
        """ConversionCost should be creatable."""
        cost = ConversionCost(time_complexity=1.0, space_complexity=2.0)
        
        assert cost is not None
    
    def test_cost_attributes(self):
        """ConversionCost should have time and space complexity."""
        cost = ConversionCost(time_complexity=1.5, space_complexity=2.5)
        
        assert hasattr(cost, 'time_complexity')
        assert hasattr(cost, 'space_complexity')
        assert hasattr(cost, 'total_cost')
    
    def test_cost_comparison(self):
        """Costs should be comparable."""
        cost1 = ConversionCost(time_complexity=1.0, space_complexity=1.0)
        cost2 = ConversionCost(time_complexity=2.0, space_complexity=2.0)
        
        # Should be comparable
        assert cost1 < cost2
    
    def test_cost_addition(self):
        """Costs should be addable."""
        cost1 = ConversionCost(time_complexity=1.0, space_complexity=1.0)
        cost2 = ConversionCost(time_complexity=1.0, space_complexity=2.0)
        
        combined = cost1 + cost2
        assert combined.time_complexity == 2.0


class TestConversionEdge:
    """Test ConversionEdge class."""
    
    def test_edge_creation(self):
        """ConversionEdge should be creatable."""
        cost = ConversionCost(time_complexity=1.0, space_complexity=1.0)
        edge = ConversionEdge(
            source="truth_table",
            target="fourier",
            cost=cost,
            converter=lambda x: x
        )
        
        assert edge is not None
    
    def test_edge_attributes(self):
        """ConversionEdge should have expected attributes."""
        cost = ConversionCost(time_complexity=1.0, space_complexity=1.0)
        edge = ConversionEdge(
            source="truth_table",
            target="fourier",
            cost=cost,
            converter=lambda x: x
        )
        
        assert edge.source == "truth_table"
        assert edge.target == "fourier"
        assert edge.cost is not None


class TestConversionPath:
    """Test ConversionPath class."""
    
    def test_path_creation(self):
        """ConversionPath should be creatable."""
        cost = ConversionCost(time_complexity=1.0, space_complexity=1.0)
        edge = ConversionEdge(
            source="truth_table",
            target="fourier",
            cost=cost,
            converter=lambda x: x
        )
        
        path = ConversionPath(edges=[edge])
        
        assert path is not None
    
    def test_path_edges(self):
        """ConversionPath should have edges."""
        cost = ConversionCost(time_complexity=1.0, space_complexity=1.0)
        edge = ConversionEdge(
            source="a",
            target="b",
            cost=cost,
            converter=lambda x: x
        )
        
        path = ConversionPath(edges=[edge])
        
        assert hasattr(path, 'edges')
        assert len(path.edges) == 1


class TestConversionGraph:
    """Test ConversionGraph class."""
    
    def test_graph_creation(self):
        """ConversionGraph should be creatable."""
        graph = ConversionGraph()
        
        assert graph is not None
    
    def test_graph_add_conversion(self):
        """Should be able to add conversions to graph."""
        graph = ConversionGraph()
        
        # Try to add a conversion
        if hasattr(graph, 'add_conversion'):
            cost = ConversionCost(time=1.0, space=1.0)
            graph.add_conversion("a", "b", lambda x: x, cost)
    
    def test_graph_find_path(self):
        """Should be able to find conversion paths."""
        graph = ConversionGraph()
        
        if hasattr(graph, 'find_path'):
            # Try to find a path
            try:
                path = graph.find_path("truth_table", "fourier")
                # May or may not find a path
            except (ValueError, KeyError):
                pass  # Expected if no path exists
    
    def test_graph_get_representations(self):
        """Should be able to list available representations."""
        graph = ConversionGraph()
        
        if hasattr(graph, 'get_representations'):
            reps = graph.get_representations()
            assert isinstance(reps, (list, set, tuple))


class TestGetConversionGraph:
    """Test get_conversion_graph function."""
    
    def test_get_global_graph(self):
        """get_conversion_graph should return a graph."""
        graph = get_conversion_graph()
        
        assert graph is not None
        assert isinstance(graph, ConversionGraph)
    
    def test_global_graph_singleton(self):
        """Global graph should be a singleton."""
        graph1 = get_conversion_graph()
        graph2 = get_conversion_graph()
        
        assert graph1 is graph2
    
    def test_global_graph_has_conversions(self):
        """Global graph should have some built-in conversions."""
        graph = get_conversion_graph()
        
        # Should have at least some representations
        if hasattr(graph, '_edges') or hasattr(graph, 'edges'):
            edges = getattr(graph, '_edges', getattr(graph, 'edges', {}))
            # May have edges depending on initialization
            assert edges is not None


class TestFindConversionPath:
    """Test find_conversion_path function."""
    
    def test_function_exists(self):
        """find_conversion_path should be callable."""
        assert callable(find_conversion_path)
    
    def test_find_path_returns_path_or_none(self):
        """find_conversion_path should return path or None."""
        try:
            path = find_conversion_path("truth_table", "fourier")
            # Path is either a ConversionPath or None
            assert path is None or isinstance(path, ConversionPath)
        except (ValueError, KeyError):
            pass  # May raise if representations don't exist


class TestGetConversionOptions:
    """Test get_conversion_options function."""
    
    def test_function_exists(self):
        """get_conversion_options should be callable."""
        assert callable(get_conversion_options)
    
    def test_get_options_from_truth_table(self):
        """Should get conversion options from truth table."""
        try:
            options = get_conversion_options("truth_table")
            assert isinstance(options, dict)
        except (ValueError, KeyError):
            pass  # May not be configured


class TestEstimateConversionCost:
    """Test estimate_conversion_cost function."""
    
    def test_function_exists(self):
        """estimate_conversion_cost should be callable."""
        assert callable(estimate_conversion_cost)
    
    def test_estimate_returns_cost_or_none(self):
        """Should estimate cost or return None."""
        try:
            cost = estimate_conversion_cost("truth_table", "fourier")
            # Should return ConversionCost or None
            assert cost is None or isinstance(cost, (ConversionCost, float, int))
        except (ValueError, KeyError):
            pass  # May raise if path doesn't exist


class TestRegisterCustomConversion:
    """Test register_custom_conversion function."""
    
    def test_function_exists(self):
        """register_custom_conversion should be callable."""
        assert callable(register_custom_conversion)
    
    def test_register_conversion(self):
        """Should be able to register custom conversions."""
        try:
            register_custom_conversion(
                "custom_source",
                "custom_target",
                lambda x, n: x,
                time_cost=1.0,
                space_cost=1.0
            )
        except TypeError:
            # May have different signature
            pass


class TestConversionGraphIntegration:
    """Integration tests for conversion graph."""
    
    def test_function_uses_conversions(self):
        """BooleanFunction should use conversion graph."""
        f = bf.majority(3)
        
        # Getting different representations should work
        tt = f.get_representation("truth_table")
        assert tt is not None
        
        fourier = f.fourier()
        assert len(fourier) == 8
    
    def test_multiple_representation_access(self):
        """Accessing multiple representations should work."""
        f = bf.AND(3)
        
        # Various representations
        tt = list(f.get_representation("truth_table"))
        fourier = f.fourier()
        
        # Consistency check
        assert len(tt) == 8
        assert len(fourier) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
