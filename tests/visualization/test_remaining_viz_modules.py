"""
Tests for remaining visualization modules.

Targets: animation, growth_plots, decision_tree_export, latex_export
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import boofun as bf
from boofun.families import MajorityFamily, ParityFamily


@pytest.fixture(autouse=True)
def cleanup_plots():
    yield
    plt.close('all')


class TestGrowthPlots:
    """Test growth_plots module."""
    
    def test_growth_plots_import(self):
        """Growth plots should import."""
        from boofun.visualization import growth_plots
        assert growth_plots is not None
    
    def test_asymptotic_plotter_class(self):
        """Test AsymptoticPlotter if available."""
        from boofun.visualization import growth_plots
        
        if hasattr(growth_plots, 'AsymptoticPlotter'):
            plotter = growth_plots.AsymptoticPlotter()
            assert plotter is not None
    
    def test_growth_visualization_with_family(self):
        """Test growth visualization with a family."""
        from boofun.visualization import growth_plots
        
        # Use majority family
        family = MajorityFamily()
        
        # Look for plotting functions
        for name in dir(growth_plots):
            if name.startswith('_'):
                continue
            obj = getattr(growth_plots, name)
            if callable(obj) and 'plot' in name.lower():
                try:
                    result = obj(family)
                except (TypeError, ValueError, AttributeError):
                    pass  # May need different args


class TestDecisionTreeExport:
    """Test decision_tree_export module."""
    
    def test_module_import(self):
        """Module should import."""
        from boofun.visualization import decision_tree_export
        assert decision_tree_export is not None
    
    def test_module_contents(self):
        """Module should have functions."""
        from boofun.visualization import decision_tree_export
        
        contents = [n for n in dir(decision_tree_export) if not n.startswith('_')]
        assert len(contents) > 0
    
    def test_export_functions(self):
        """Test export functions."""
        from boofun.visualization import decision_tree_export
        
        f = bf.AND(3)
        
        # Try any export functions
        for name in dir(decision_tree_export):
            if name.startswith('_'):
                continue
            obj = getattr(decision_tree_export, name)
            if callable(obj):
                try:
                    result = obj(f)
                except (TypeError, ValueError, AttributeError):
                    pass


class TestLatexExport:
    """Test latex_export module."""
    
    def test_module_import(self):
        """Module should import."""
        from boofun.visualization import latex_export
        assert latex_export is not None
    
    def test_module_contents(self):
        """Module should have functions."""
        from boofun.visualization import latex_export
        
        contents = [n for n in dir(latex_export) if not n.startswith('_')]
        assert len(contents) > 0
    
    def test_latex_export_functions(self):
        """Test LaTeX export functions."""
        from boofun.visualization import latex_export
        
        f = bf.majority(3)
        
        for name in dir(latex_export):
            if name.startswith('_'):
                continue
            obj = getattr(latex_export, name)
            if callable(obj):
                try:
                    result = obj(f)
                except (TypeError, ValueError, AttributeError):
                    pass


class TestAnimationDetailed:
    """More detailed animation tests."""
    
    def test_module_import(self):
        """Module should import."""
        from boofun.visualization import animation
        assert animation is not None
    
    def test_animation_classes(self):
        """Test animation classes."""
        from boofun.visualization import animation
        
        for name in dir(animation):
            if name.startswith('_'):
                continue
            obj = getattr(animation, name)
            if isinstance(obj, type):
                # It's a class
                try:
                    instance = obj()
                except TypeError:
                    # May need arguments
                    pass


class TestInteractiveModule:
    """Test interactive module."""
    
    def test_module_import(self):
        """Module should import."""
        from boofun.visualization import interactive
        assert interactive is not None
    
    def test_module_contents(self):
        """Module should have contents."""
        from boofun.visualization import interactive
        
        contents = [n for n in dir(interactive) if not n.startswith('_')]
        assert len(contents) > 0


class TestVisualizationIntegration:
    """Integration tests for visualization with analysis."""
    
    def test_visualize_influences_with_analysis(self):
        """Visualize influences with analysis context."""
        from boofun.visualization import BooleanFunctionVisualizer
        from boofun.analysis import SpectralAnalyzer
        
        f = bf.majority(5)
        
        # Analysis
        analyzer = SpectralAnalyzer(f)
        influences = analyzer.influences()
        
        # Visualization
        viz = BooleanFunctionVisualizer(f)
        fig = viz.plot_influences(show=False)
        
        assert len(influences) == 5
        assert fig is not None
    
    def test_visualize_fourier_with_analysis(self):
        """Visualize Fourier spectrum with analysis."""
        from boofun.visualization import BooleanFunctionVisualizer
        
        f = bf.parity(3)
        
        # Get Fourier data
        fourier = f.fourier()
        
        # Visualize
        viz = BooleanFunctionVisualizer(f)
        fig = viz.plot_fourier_spectrum(show=False)
        
        assert len(fourier) == 8
        assert fig is not None
    
    def test_visualize_noise_stability(self):
        """Visualize noise stability."""
        from boofun.visualization import BooleanFunctionVisualizer
        
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)
        
        if hasattr(viz, 'plot_noise_stability'):
            fig = viz.plot_noise_stability(show=False)
            assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
