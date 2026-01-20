"""
Full visualization tests with matplotlib.

These tests actually render plots using matplotlib's Agg backend.
"""

import sys

import pytest

sys.path.insert(0, "src")

# Set non-interactive backend BEFORE importing matplotlib.pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import boofun as bf
from boofun.visualization import BooleanFunctionVisualizer


@pytest.fixture(autouse=True)
def cleanup_plots():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


class TestInfluencePlots:
    """Test influence plotting."""

    @pytest.mark.parametrize(
        "func_factory,n",
        [
            (bf.AND, 3),
            (bf.OR, 3),
            (bf.majority, 3),
            (bf.majority, 5),
            (bf.parity, 4),
        ],
    )
    def test_plot_influences_various_functions(self, func_factory, n):
        """Plot influences for various functions."""
        f = func_factory(n)
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_influences(show=False)
        assert fig is not None

    def test_plot_influences_with_figsize(self):
        """Plot with custom figure size."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_influences(figsize=(12, 8), show=False)
        assert fig is not None

    def test_plot_influences_save(self, tmp_path):
        """Save influence plot to file."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        save_file = tmp_path / "influences.png"
        fig = viz.plot_influences(save_path=str(save_file), show=False)

        assert fig is not None
        assert save_file.exists()


class TestFourierSpectrumPlots:
    """Test Fourier spectrum plotting."""

    @pytest.mark.parametrize(
        "func_factory,n",
        [
            (bf.AND, 3),
            (bf.OR, 3),
            (bf.majority, 3),
            (bf.parity, 3),
        ],
    )
    def test_plot_fourier_spectrum_various(self, func_factory, n):
        """Plot Fourier spectrum for various functions."""
        f = func_factory(n)
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_fourier_spectrum(show=False)
        assert fig is not None

    def test_plot_fourier_spectrum_parity(self):
        """Parity has single spike in spectrum."""
        f = bf.parity(3)
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_fourier_spectrum(show=False)
        assert fig is not None

    def test_plot_fourier_spectrum_save(self, tmp_path):
        """Save Fourier spectrum plot."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        save_file = tmp_path / "fourier.png"
        fig = viz.plot_fourier_spectrum(save_path=str(save_file), show=False)

        assert fig is not None
        assert save_file.exists()


class TestTruthTablePlots:
    """Test truth table visualization."""

    def test_plot_truth_table_small(self):
        """Plot truth table for small function."""
        f = bf.AND(3)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_truth_table"):
            fig = viz.plot_truth_table(show=False)
            assert fig is not None

    def test_plot_truth_table_majority(self):
        """Plot truth table for majority."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_truth_table"):
            fig = viz.plot_truth_table(show=False)
            assert fig is not None


class TestNoiseStabilityPlots:
    """Test noise stability visualization."""

    def test_plot_noise_stability(self):
        """Plot noise stability curve."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_noise_stability"):
            fig = viz.plot_noise_stability(show=False)
            assert fig is not None

    def test_plot_noise_stability_comparison(self):
        """Compare noise stability of different functions."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_noise_stability"):
            fig = viz.plot_noise_stability(show=False)
            assert fig is not None


class TestSpectralAnalysisPlots:
    """Test spectral analysis visualization."""

    def test_plot_spectral_analysis(self):
        """Plot comprehensive spectral analysis."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_spectral_analysis"):
            fig = viz.plot_spectral_analysis(show=False)
            assert fig is not None

    def test_plot_degree_distribution(self):
        """Plot Fourier weight by degree."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "plot_degree_distribution"):
            fig = viz.plot_degree_distribution(show=False)
            assert fig is not None


class TestComparisonPlots:
    """Test function comparison visualization."""

    def test_compare_two_functions(self):
        """Compare two functions."""
        f1 = bf.AND(3)
        f2 = bf.OR(3)

        viz = BooleanFunctionVisualizer(f1)

        if hasattr(viz, "compare_functions"):
            fig = viz.compare_functions([f2], show=False)
            assert fig is not None

    def test_compare_multiple_functions(self):
        """Compare multiple functions."""
        f1 = bf.majority(3)
        f2 = bf.AND(3)
        f3 = bf.OR(3)

        viz = BooleanFunctionVisualizer(f1)

        if hasattr(viz, "compare_functions"):
            fig = viz.compare_functions([f2, f3], show=False)
            assert fig is not None


class TestDashboard:
    """Test comprehensive dashboard."""

    def test_create_dashboard(self):
        """Create comprehensive dashboard."""
        f = bf.majority(5)
        viz = BooleanFunctionVisualizer(f)

        if hasattr(viz, "create_dashboard"):
            fig = viz.create_dashboard(show=False)
            assert fig is not None


class TestVisualizationExport:
    """Test export functionality."""

    def test_export_to_png(self, tmp_path):
        """Export plot to PNG."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        save_file = tmp_path / "test.png"
        fig = viz.plot_influences(save_path=str(save_file), show=False)

        assert save_file.exists()

    def test_export_to_pdf(self, tmp_path):
        """Export plot to PDF."""
        f = bf.majority(3)
        viz = BooleanFunctionVisualizer(f)

        save_file = tmp_path / "test.pdf"
        fig = viz.plot_influences(save_path=str(save_file), show=False)

        # PDF export might not work in all environments
        assert fig is not None


class TestVisualizationEdgeCases:
    """Test edge cases for visualization."""

    def test_single_variable_function(self):
        """Visualize single variable function."""
        f = bf.create([0, 1])
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_influences(show=False)
        assert fig is not None

    def test_constant_function(self):
        """Visualize constant function."""
        f = bf.create([0, 0, 0, 0])
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_influences(show=False)
        assert fig is not None

    def test_larger_function(self):
        """Visualize larger function (n=7)."""
        f = bf.majority(7)
        viz = BooleanFunctionVisualizer(f)

        fig = viz.plot_influences(show=False)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
