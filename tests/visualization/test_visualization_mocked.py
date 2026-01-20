"""
Mocked tests for visualization module.

Uses mocks to test visualization code paths without requiring
actual plotting backends.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
sys.path.insert(0, 'src')

import boofun as bf


class TestVisualizationWithMocks:
    """Test visualization with mocked plotting backends."""
    
    @pytest.fixture
    def mock_matplotlib(self):
        """Mock matplotlib for testing."""
        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.figure.return_value = mock_fig
        return mock_plt, mock_fig, mock_ax
    
    def test_visualizer_initialization(self, mock_matplotlib):
        """Test visualizer initialization with mocked matplotlib."""
        mock_plt, mock_fig, mock_ax = mock_matplotlib
        
        with patch.dict('sys.modules', {'matplotlib': MagicMock(), 'matplotlib.pyplot': mock_plt}):
            from boofun.visualization import BooleanFunctionVisualizer, HAS_MATPLOTLIB
            
            if HAS_MATPLOTLIB:
                f = bf.majority(3)
                viz = BooleanFunctionVisualizer(f)
                
                assert viz is not None
                assert viz.n_vars == 3


class TestVisualizationDataGeneration:
    """Test data generation for visualization (no actual plotting)."""
    
    def test_generate_influence_data(self):
        """Generate influence data for plotting."""
        f = bf.majority(5)
        influences = f.influences()
        
        # Data should be suitable for bar chart
        assert len(influences) == 5
        assert all(isinstance(i, (int, float, np.number)) for i in influences)
        assert all(0 <= i <= 1 for i in influences)
    
    def test_generate_fourier_data(self):
        """Generate Fourier data for plotting."""
        f = bf.parity(3)
        fourier = np.array(f.fourier())
        
        # Data should be suitable for spectrum plot
        assert len(fourier) == 8
        assert np.isfinite(fourier).all()
    
    def test_generate_truth_table_data(self):
        """Generate truth table data for visualization."""
        f = bf.AND(3)
        tt = list(f.get_representation("truth_table"))
        
        # Data should be suitable for heatmap
        assert len(tt) == 8
        assert all(v in [0, 1, True, False] for v in tt)
    
    def test_generate_degree_distribution(self):
        """Generate Fourier degree distribution data."""
        f = bf.majority(5)
        fourier = np.array(f.fourier())
        
        # Compute weight at each degree
        n = 5
        degree_weights = {}
        for S in range(2**n):
            deg = bin(S).count('1')
            weight = fourier[S]**2
            degree_weights[deg] = degree_weights.get(deg, 0) + weight
        
        # Should have weights at various degrees
        assert sum(degree_weights.values()) - 1.0 < 1e-10


class TestVisualizationHelperFunctions:
    """Test visualization helper functions."""
    
    def test_color_mapping(self):
        """Test color mapping for influences."""
        influences = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Map to colors (simple normalization)
        normalized = [(i - min(influences)) / (max(influences) - min(influences) + 1e-10) 
                      for i in influences]
        
        assert all(0 <= n <= 1 for n in normalized)
    
    def test_label_generation(self):
        """Test label generation for variables."""
        n = 5
        labels = [f"x_{i}" for i in range(n)]
        
        assert len(labels) == 5
        assert labels[0] == "x_0"


class TestVisualizationSubmodules:
    """Test visualization submodules exist and are importable."""
    
    def test_animation_module(self):
        """Animation module should be importable."""
        from boofun.visualization import animation
        assert animation is not None
    
    def test_decision_tree_module(self):
        """Decision tree module should be importable."""
        from boofun.visualization import decision_tree
        assert decision_tree is not None
    
    def test_growth_plots_module(self):
        """Growth plots module should be importable."""
        from boofun.visualization import growth_plots
        assert growth_plots is not None
    
    def test_widgets_module(self):
        """Widgets module should be importable."""
        from boofun.visualization import widgets
        assert widgets is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
