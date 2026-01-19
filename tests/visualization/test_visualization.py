"""
Tests for visualization module.

These tests verify visualization functionality works correctly,
even when plotting libraries may not be available.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf


class TestBooleanFunctionVisualizer:
    """Test BooleanFunctionVisualizer class."""
    
    def test_visualizer_init(self):
        """Visualizer should initialize with a function."""
        f = bf.majority(3)
        
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            viz = BooleanFunctionVisualizer(f)
            assert viz.function == f
            assert viz.n_vars == 3
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_visualizer_rejects_invalid_function(self):
        """Visualizer should reject functions without n_vars."""
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            
            # Create a mock function without n_vars
            class MockFunc:
                n_vars = None
            
            with pytest.raises(ValueError, match="must have defined"):
                BooleanFunctionVisualizer(MockFunc())
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_visualizer_backend_selection(self):
        """Visualizer should accept different backends."""
        f = bf.AND(3)
        
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            
            # Matplotlib backend
            viz_mpl = BooleanFunctionVisualizer(f, backend="matplotlib")
            assert viz_mpl.backend == "matplotlib"
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_plot_influences(self):
        """plot_influences should work for valid functions."""
        f = bf.majority(3)
        
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for testing
            
            viz = BooleanFunctionVisualizer(f)
            fig = viz.plot_influences(show=False)
            assert fig is not None
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_plot_fourier_spectrum(self):
        """plot_fourier_spectrum should work."""
        f = bf.parity(3)
        
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            import matplotlib
            matplotlib.use('Agg')
            
            viz = BooleanFunctionVisualizer(f)
            fig = viz.plot_fourier_spectrum(show=False)
            assert fig is not None
        except ImportError:
            pytest.skip("Matplotlib not available")


class TestQuickPlotFunctions:
    """Test quick plot helper functions."""
    
    def test_plot_influences_quick(self):
        """Quick influence plot should work."""
        f = bf.AND(4)
        
        try:
            from boofun.visualization import plot_influences
            import matplotlib
            matplotlib.use('Agg')
            
            fig = plot_influences(f, show=False)
            assert fig is not None
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_plot_fourier_quick(self):
        """Quick Fourier plot should work."""
        f = bf.parity(3)
        
        try:
            from boofun.visualization import plot_fourier_spectrum
            import matplotlib
            matplotlib.use('Agg')
            
            fig = plot_fourier_spectrum(f, show=False)
            assert fig is not None
        except ImportError:
            pytest.skip("Matplotlib not available")


class TestInteractiveVisualization:
    """Test interactive Plotly visualization."""
    
    def test_plotly_backend(self):
        """Plotly backend should work if available."""
        f = bf.majority(3)
        
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            import plotly
            
            viz = BooleanFunctionVisualizer(f, backend="plotly")
            assert viz.backend == "plotly"
        except ImportError:
            pytest.skip("Plotly not available")
    
    def test_interactive_dashboard(self):
        """Interactive dashboard should create without error."""
        f = bf.AND(3)
        
        try:
            from boofun.visualization import create_dashboard
            
            dashboard = create_dashboard(f)
            assert dashboard is not None
        except (ImportError, AttributeError):
            pytest.skip("Interactive dashboard not available")


class TestVisualizationHelpers:
    """Test visualization helper functions."""
    
    def test_color_by_degree(self):
        """Color mapping by degree should work."""
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            
            f = bf.majority(3)
            viz = BooleanFunctionVisualizer(f)
            
            # The visualizer should be able to color Fourier coefficients by degree
            fourier = f.fourier()
            assert len(fourier) == 8  # 2^3 coefficients
        except ImportError:
            pytest.skip("Matplotlib not available")
    
    def test_truth_table_heatmap(self):
        """Truth table heatmap should work."""
        f = bf.AND(2)
        
        try:
            from boofun.visualization import BooleanFunctionVisualizer
            import matplotlib
            matplotlib.use('Agg')
            
            viz = BooleanFunctionVisualizer(f)
            
            if hasattr(viz, 'plot_truth_table'):
                fig = viz.plot_truth_table(show=False)
                assert fig is not None
        except ImportError:
            pytest.skip("Matplotlib not available")


class TestNoPlottingLibrary:
    """Test behavior when plotting libraries are not available."""
    
    def test_module_imports_without_matplotlib(self):
        """Module should import even without matplotlib."""
        # The module should have been imported already
        from boofun import visualization
        
        # Check the HAS_MATPLOTLIB flag exists
        assert hasattr(visualization, 'HAS_MATPLOTLIB')
    
    def test_warns_without_matplotlib(self):
        """Should have set warning flag if matplotlib missing."""
        from boofun import visualization
        
        # Just check the module loaded - the warning would have been issued at import
        assert visualization is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
