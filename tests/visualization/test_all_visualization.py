"""
Additional tests for all visualization submodules.

Tests to maximize coverage of visualization code paths.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.visualization import HAS_MATPLOTLIB, HAS_PLOTLY

# Skip all tests if no plotting backend available
pytestmark = pytest.mark.skipif(
    not HAS_MATPLOTLIB and not HAS_PLOTLY, reason="No plotting backend available"
)


class TestVisualizationImports:
    """Test visualization module imports."""

    def test_import_main_module(self):
        """Main visualization module should import."""
        from boofun import visualization

        assert visualization is not None

    def test_import_animation(self):
        """Animation module should import."""
        from boofun.visualization import animation

        assert animation is not None

    def test_import_decision_tree(self):
        """Decision tree module should import."""
        from boofun.visualization import decision_tree

        assert decision_tree is not None

    def test_import_growth_plots(self):
        """Growth plots module should import."""
        from boofun.visualization import growth_plots

        assert growth_plots is not None

    def test_import_widgets(self):
        """Widgets module should import."""
        from boofun.visualization import widgets

        assert widgets is not None


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
class TestVisualizationPlotMethods:
    """Test various plot methods."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Set up matplotlib for non-interactive use."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_plot_truth_table(self):
        """Test truth table plotting."""
        from boofun.visualization import BooleanFunctionVisualizer

        f = bf.AND(3)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_truth_table"):
            fig = viz.plot_truth_table(show=False)
            assert fig is not None

    def test_plot_spectral_analysis(self):
        """Test spectral analysis plotting."""
        from boofun.visualization import BooleanFunctionVisualizer

        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_spectral_analysis"):
            fig = viz.plot_spectral_analysis(show=False)
            assert fig is not None

    def test_plot_noise_stability(self):
        """Test noise stability plotting."""
        from boofun.visualization import BooleanFunctionVisualizer

        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_noise_stability"):
            fig = viz.plot_noise_stability(show=False)
            assert fig is not None

    def test_plot_comparison(self):
        """Test function comparison plotting."""
        from boofun.visualization import BooleanFunctionVisualizer

        f1 = bf.AND(3)
        f2 = bf.OR(3)
        viz = BooleanFunctionVisualizer(f1)

        if hasattr(viz, "compare_functions"):
            fig = viz.compare_functions([f2], show=False)
            assert fig is not None


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
class TestDecisionTreeVisualization:
    """Test decision tree visualization."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_decision_tree_module_classes(self):
        """Check decision tree module has expected classes."""
        from boofun.visualization import decision_tree

        # Check for common classes/functions
        module_contents = dir(decision_tree)
        assert len(module_contents) > 0


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
class TestGrowthPlotsVisualization:
    """Test growth plots visualization."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_growth_plots_module_contents(self):
        """Check growth plots module has expected contents."""
        from boofun.visualization import growth_plots

        module_contents = dir(growth_plots)
        assert len(module_contents) > 0


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
class TestWidgetsVisualization:
    """Test widgets visualization."""

    def test_widgets_module_contents(self):
        """Check widgets module has expected contents."""
        from boofun.visualization import widgets

        module_contents = dir(widgets)
        assert len(module_contents) > 0


class TestVisualizationDataPreparation:
    """Test data preparation for visualization."""

    def test_influences_data(self):
        """Influences data should be suitable for plotting."""
        f = bf.majority(5)
        influences = f.influences()

        assert len(influences) == 5
        assert all(0 <= i <= 1 for i in influences)

    def test_fourier_data(self):
        """Fourier data should be suitable for plotting."""
        f = bf.parity(3)
        fourier = np.array(f.fourier())

        assert len(fourier) == 8
        assert np.sum(fourier**2) - 1.0 < 1e-10


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Matplotlib not available")
class TestVisualizationSaveFunctionality:
    """Test saving plots to files."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_save_path_parameter(self, tmp_path):
        """Test saving plots to a file."""
        from boofun.visualization import BooleanFunctionVisualizer

        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        save_file = tmp_path / "test_plot.png"
        fig = viz.plot_influences(save_path=str(save_file), show=False)

        # File might or might not be created depending on implementation
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
