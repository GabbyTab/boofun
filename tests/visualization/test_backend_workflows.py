"""End-to-end coverage for visualization backend branches."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import boofun as bf
from boofun.visualization import (
    BooleanFunctionVisualizer,
    plot_function_comparison,
    plot_hypercube,
    plot_sensitivity_heatmap,
)

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_matplotlib_backend_complete_workflow(tmp_path: Path) -> None:
    visualizer = BooleanFunctionVisualizer(bf.majority(3), backend="matplotlib")

    assert visualizer.plot_influences(save_path=str(tmp_path / "influences.png"), show=False)
    assert visualizer.plot_fourier_spectrum(
        max_degree=2, save_path=str(tmp_path / "fourier.png"), show=False
    )
    assert visualizer.plot_truth_table(show=False)
    assert visualizer.plot_noise_stability_curve(np.array([-1.0, 0.0, 0.5, 1.0]), show=False)
    dashboard = visualizer.create_dashboard(save_path=str(tmp_path / "dashboard.png"), show=False)
    assert len(dashboard.axes) >= 4

    large_table = BooleanFunctionVisualizer(bf.majority(5)).plot_truth_table(show=False)
    assert large_table is not None

    for filename in ("influences.png", "fourier.png", "dashboard.png"):
        assert (tmp_path / filename).exists()


def test_plotly_backend_complete_workflow(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    visualizer = BooleanFunctionVisualizer(bf.majority(3), backend="plotly")

    influences = visualizer.plot_influences(save_path=str(tmp_path / "influences.html"), show=False)
    assert len(influences.data) == 1

    fourier = visualizer.plot_fourier_spectrum(max_degree=2, show=False)
    assert len(fourier.data) >= 2

    table = visualizer.plot_truth_table(show=False)
    assert table.data[0].type == "table"

    heatmap = BooleanFunctionVisualizer(bf.majority(5), backend="plotly").plot_truth_table(
        show=False
    )
    assert heatmap.data[0].type == "heatmap"

    stability = visualizer.plot_noise_stability_curve(np.array([-1.0, 0.0, 1.0]), show=False)
    assert len(stability.data) == 1

    dashboard = visualizer.create_dashboard(save_path=str(tmp_path / "dashboard.html"), show=False)
    assert len(dashboard.data) == 4
    assert (tmp_path / "influences.html").exists()
    assert (tmp_path / "dashboard.html").exists()


def test_comparison_hypercube_and_sensitivity_workflows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    functions = {"and": bf.AND(3), "parity": bf.parity(4)}

    influence_comparison = plot_function_comparison(functions, metric="influences")
    assert len(influence_comparison.axes[0].patches) == 8

    fourier_comparison = plot_function_comparison(functions, metric="fourier")
    assert len(fourier_comparison.axes[0].lines) == 2

    with pytest.raises(ValueError, match="Unknown metric"):
        plot_function_comparison(functions, metric="unknown")

    small_cube = plot_hypercube(bf.AND(2), show=False)
    assert small_cube is not None
    projected_cube = plot_hypercube(
        bf.AND(4), save_path=str(tmp_path / "hypercube.png"), show=False
    )
    assert projected_cube is not None

    with pytest.raises(ValueError, match="only supports"):
        plot_hypercube(bf.AND(6), show=False)

    bar_heatmap = plot_sensitivity_heatmap(bf.majority(3), show=False)
    assert len(bar_heatmap.axes[0].patches) == 8
    matrix_heatmap = plot_sensitivity_heatmap(
        bf.majority(5), save_path=str(tmp_path / "sensitivity.png"), show=False
    )
    assert len(matrix_heatmap.axes) == 2


def test_unknown_backend_dispatch() -> None:
    visualizer = BooleanFunctionVisualizer(bf.AND(2))
    visualizer.backend = "unknown"

    with pytest.raises(ValueError, match="Unknown backend"):
        visualizer.plot_influences(show=False)
    assert visualizer.plot_fourier_spectrum(show=False) is None
    assert visualizer.plot_truth_table(show=False) is None
    assert visualizer.plot_noise_stability_curve(np.array([0.0]), show=False) is None
    assert visualizer.create_dashboard(show=False) is None
