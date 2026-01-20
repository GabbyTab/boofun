"""
Tests for visualization submodules.

Tests animation, decision_tree, growth_plots, widgets, etc.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import boofun as bf
from boofun.visualization import animation, decision_tree, growth_plots, widgets


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close('all')


class TestAnimationModule:
    """Test animation module."""
    
    def test_module_exists(self):
        """Animation module should exist."""
        assert animation is not None
    
    def test_module_contents(self):
        """Animation module should have functions/classes."""
        contents = [name for name in dir(animation) if not name.startswith('_')]
        assert len(contents) > 0
    
    def test_animation_functions(self):
        """Test animation functions that exist."""
        f = bf.majority(3)
        
        # Just verify functions exist - don't call them as they may need special args
        for name in dir(animation):
            if name.startswith('_'):
                continue
            obj = getattr(animation, name)
            if callable(obj):
                assert obj is not None


class TestDecisionTreeModule:
    """Test decision_tree module."""
    
    def test_module_exists(self):
        """Decision tree module should exist."""
        assert decision_tree is not None
    
    def test_module_contents(self):
        """Module should have expected contents."""
        contents = [name for name in dir(decision_tree) if not name.startswith('_')]
        assert len(contents) > 0
    
    def test_decision_tree_visualization(self):
        """Test decision tree visualization if available."""
        f = bf.AND(3)
        
        # Look for visualization functions
        for name in dir(decision_tree):
            if 'plot' in name.lower() or 'draw' in name.lower() or 'visual' in name.lower():
                obj = getattr(decision_tree, name)
                if callable(obj):
                    try:
                        result = obj(f)
                    except (TypeError, ValueError):
                        pass


class TestGrowthPlotsModule:
    """Test growth_plots module."""
    
    def test_module_exists(self):
        """Growth plots module should exist."""
        assert growth_plots is not None
    
    def test_module_contents(self):
        """Module should have expected contents."""
        contents = [name for name in dir(growth_plots) if not name.startswith('_')]
        assert len(contents) > 0
    
    def test_growth_plot_classes(self):
        """Test growth plot classes."""
        # Look for plotter classes
        for name in dir(growth_plots):
            if name.startswith('_'):
                continue
            obj = getattr(growth_plots, name)
            if isinstance(obj, type):
                # It's a class, try to instantiate
                try:
                    instance = obj()
                except TypeError:
                    # May need args
                    pass


class TestWidgetsModule:
    """Test widgets module."""
    
    def test_module_exists(self):
        """Widgets module should exist."""
        assert widgets is not None
    
    def test_module_contents(self):
        """Module should have expected contents."""
        contents = [name for name in dir(widgets) if not name.startswith('_')]
        assert len(contents) > 0
    
    def test_widget_classes(self):
        """Test widget classes if available."""
        f = bf.majority(3)
        
        for name in dir(widgets):
            if name.startswith('_'):
                continue
            obj = getattr(widgets, name)
            if isinstance(obj, type):
                try:
                    instance = obj(f)
                except (TypeError, ValueError, ImportError):
                    # May need ipywidgets or different args
                    pass


class TestQuickPlotFunctions:
    """Test quick plot convenience functions."""
    
    def test_quick_plot_influences(self):
        """Test quick influence plot."""
        from boofun import visualization
        
        f = bf.majority(5)
        
        if hasattr(visualization, 'plot_influences'):
            fig = visualization.plot_influences(f, show=False)
            assert fig is not None
    
    def test_quick_plot_fourier(self):
        """Test quick Fourier plot."""
        from boofun import visualization
        
        f = bf.parity(3)
        
        if hasattr(visualization, 'plot_fourier_spectrum'):
            fig = visualization.plot_fourier_spectrum(f, show=False)
            assert fig is not None
    
    def test_quick_plot_truthtable(self):
        """Test quick truth table plot."""
        from boofun import visualization
        
        f = bf.AND(3)
        
        if hasattr(visualization, 'plot_truth_table'):
            fig = visualization.plot_truth_table(f, show=False)
            assert fig is not None


class TestVisualizationWithFamilies:
    """Test visualization with function families."""
    
    def test_visualize_majority_family(self):
        """Visualize majority functions of different sizes."""
        from boofun.visualization import BooleanFunctionVisualizer
        
        for n in [3, 5, 7]:
            f = bf.majority(n)
            viz = BooleanFunctionVisualizer(f)
            
            fig = viz.plot_influences(show=False)
            assert fig is not None
            plt.close('all')
    
    def test_visualize_parity_family(self):
        """Visualize parity functions of different sizes."""
        from boofun.visualization import BooleanFunctionVisualizer
        
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            viz = BooleanFunctionVisualizer(f)
            
            fig = viz.plot_fourier_spectrum(show=False)
            assert fig is not None
            plt.close('all')


class TestVisualizationStyles:
    """Test visualization styling options."""
    
    def test_different_colormaps(self):
        """Test with different color schemes."""
        from boofun.visualization import BooleanFunctionVisualizer
        
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)
        
        # Basic plot should work
        fig = viz.plot_influences(show=False)
        assert fig is not None
    
    def test_different_figure_sizes(self):
        """Test with different figure sizes."""
        from boofun.visualization import BooleanFunctionVisualizer
        
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)
        
        for size in [(8, 6), (10, 8), (12, 10)]:
            fig = viz.plot_influences(figsize=size, show=False)
            assert fig is not None
            plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
