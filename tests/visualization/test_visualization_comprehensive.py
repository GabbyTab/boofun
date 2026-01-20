"""
Comprehensive tests for visualization module.

Tests visualization classes and functions, exercising code paths
even when plotting libraries may not be fully available.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.visualization import HAS_MATPLOTLIB, HAS_PLOTLY, BooleanFunctionVisualizer


class TestVisualizerCreation:
    """Test BooleanFunctionVisualizer creation."""

    def test_visualizer_class_exists(self):
        """Visualizer class should exist."""
        assert BooleanFunctionVisualizer is not None

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
    def test_create_with_matplotlib(self):
        """Create visualizer with matplotlib backend."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f, backend="matplotlib")

        assert viz is not None
        assert viz.backend == "matplotlib"

    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_create_with_plotly(self):
        """Create visualizer with plotly backend."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f, backend="plotly")

        assert viz is not None
        assert viz.backend == "plotly"

    def test_visualizer_stores_function(self):
        """Visualizer should store the function."""
        f = bf.majority(3)

        if HAS_MATPLOTLIB:
            viz = BooleanFunctionVisualizer(f, backend="matplotlib")
            assert viz.function is f
            assert viz.n_vars == 3
        elif HAS_PLOTLY:
            viz = BooleanFunctionVisualizer(f, backend="plotly")
            assert viz.function is f
            assert viz.n_vars == 3


class TestVisualizerMethods:
    """Test visualizer methods."""

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer for testing."""
        f = bf.majority(3)
        if HAS_MATPLOTLIB:
            return BooleanFunctionVisualizer(f, backend="matplotlib")
        elif HAS_PLOTLY:
            return BooleanFunctionVisualizer(f, backend="plotly")
        else:
            pytest.skip("No plotting backend available")

    def test_has_plot_influences(self, visualizer):
        """Visualizer should have plot_influences method."""
        assert hasattr(visualizer, "plot_influences")
        assert callable(visualizer.plot_influences)

    def test_has_plot_fourier_spectrum(self, visualizer):
        """Visualizer should have plot_fourier_spectrum method."""
        assert hasattr(visualizer, "plot_fourier_spectrum")
        assert callable(visualizer.plot_fourier_spectrum)

    def test_has_plot_truth_table(self, visualizer):
        """Visualizer should have plot_truth_table method."""
        assert hasattr(visualizer, "plot_truth_table")
        assert callable(visualizer.plot_truth_table)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
    def test_plot_influences_matplotlib(self):
        """Test plotting influences with matplotlib."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f, backend="matplotlib")

        # Plot without showing
        fig = viz.plot_influences(show=False)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
    def test_plot_fourier_spectrum_matplotlib(self):
        """Test plotting Fourier spectrum with matplotlib."""
        import matplotlib

        matplotlib.use("Agg")

        f = bf.parity(3)
        viz = BooleanFunctionVisualizer(f, backend="matplotlib")

        fig = viz.plot_fourier_spectrum(show=False)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close("all")


class TestQuickPlotFunctions:
    """Test quick plot convenience functions."""

    def test_quick_functions_exist(self):
        """Quick plot functions should exist."""
        from boofun import visualization

        # Check for common quick plot functions
        quick_funcs = [
            "plot_influences",
            "plot_fourier_spectrum",
        ]

        for func_name in quick_funcs:
            if hasattr(visualization, func_name):
                func = getattr(visualization, func_name)
                assert callable(func)


class TestVisualizationHelpers:
    """Test visualization helper functions."""

    def test_has_matplotlib_flag(self):
        """Module should have HAS_MATPLOTLIB flag."""
        from boofun import visualization

        assert hasattr(visualization, "HAS_MATPLOTLIB")
        assert isinstance(visualization.HAS_MATPLOTLIB, bool)

    def test_has_plotly_flag(self):
        """Module should have HAS_PLOTLY flag."""
        from boofun import visualization

        assert hasattr(visualization, "HAS_PLOTLY")
        assert isinstance(visualization.HAS_PLOTLY, bool)


class TestVisualizationWithDifferentFunctions:
    """Test visualization with different Boolean functions."""

    @pytest.fixture
    def backend(self):
        if HAS_MATPLOTLIB:
            return "matplotlib"
        elif HAS_PLOTLY:
            return "plotly"
        else:
            pytest.skip("No plotting backend available")

    @pytest.mark.parametrize(
        "func_factory,n",
        [
            (bf.AND, 3),
            (bf.OR, 3),
            (bf.majority, 3),
            (bf.parity, 4),
        ],
    )
    def test_visualizer_with_various_functions(self, backend, func_factory, n):
        """Visualizer should work with various functions."""
        f = func_factory(n)
        viz = BooleanFunctionVisualizer(f, backend=backend)

        assert viz is not None
        assert viz.n_vars == n


class TestVisualizationEdgeCases:
    """Test edge cases for visualization."""

    def test_backend_case_insensitive(self):
        """Backend name should be case-insensitive."""
        f = bf.majority(3)

        if HAS_MATPLOTLIB:
            viz = BooleanFunctionVisualizer(f, backend="MATPLOTLIB")
            assert viz.backend == "matplotlib"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
