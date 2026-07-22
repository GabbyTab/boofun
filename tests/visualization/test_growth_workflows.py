"""Behavioral coverage for growth plotting and animation workflows."""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import boofun as bf
from boofun.families import GrowthTracker, MajorityFamily, ParityFamily
from boofun.visualization.animation import (
    GrowthAnimator,
    animate_fourier_spectrum,
    animate_growth,
    animate_influences,
    create_growth_animation,
)
from boofun.visualization.growth_plots import (
    ComplexityVisualizer,
    GrowthVisualizer,
    LTFVisualizer,
    quick_growth_plot,
)

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


@pytest.fixture
def parity_tracker() -> GrowthTracker:
    tracker = GrowthTracker(ParityFamily())
    tracker.mark("total_influence").mark("variance")
    tracker.observe(n_values=[2, 3, 4])
    return tracker


def test_growth_visualizer_matplotlib_workflow(parity_tracker: GrowthTracker) -> None:
    visualizer = GrowthVisualizer("matplotlib")

    figure, axis = visualizer.plot_growth(
        parity_tracker,
        "total_influence",
        log_x=True,
        log_y=True,
        title="Parity growth",
    )
    assert figure is not None
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"

    comparison = visualizer.plot_family_comparison(
        {"parity": parity_tracker}, "total_influence", log_x=True
    )
    assert comparison is not None

    for reference in ("sqrt_n", "n", "log_n", "constant"):
        assert visualizer.plot_convergence_rate(
            parity_tracker, "total_influence", reference=reference
        )

    assert visualizer.plot_multi_property_growth(parity_tracker, ["total_influence"])
    assert visualizer.plot_multi_property_growth(parity_tracker)

    with pytest.raises(ValueError, match="No results"):
        visualizer.plot_growth(parity_tracker, "missing")
    with pytest.raises(ValueError, match="Unknown reference"):
        visualizer.plot_convergence_rate(parity_tracker, "total_influence", reference="bad")


def test_growth_visualizer_plotly_workflow(parity_tracker: GrowthTracker) -> None:
    pytest.importorskip("plotly")
    visualizer = GrowthVisualizer("plotly")

    growth = visualizer.plot_growth(
        parity_tracker,
        "total_influence",
        log_x=True,
        log_y=True,
    )
    assert len(growth.data) >= 1
    assert growth.layout.xaxis.type == "log"

    comparison = visualizer.plot_family_comparison(
        {"parity": parity_tracker}, "total_influence", log_y=True
    )
    assert len(comparison.data) >= 1

    convergence = visualizer.plot_convergence_rate(parity_tracker, "total_influence", reference="n")
    assert len(convergence.data) == 1

    multi = visualizer.plot_multi_property_growth(parity_tracker)
    assert len(multi.data) == 2


def test_specialized_growth_visualizers_and_convenience() -> None:
    weights_figure = LTFVisualizer().plot_weight_distribution(np.array([1.0, -2.0, 3.0]))
    assert len(weights_figure.axes) == 2

    complexity_figure = ComplexityVisualizer().plot_complexity_relations(bf.AND(2))
    assert len(complexity_figure.axes) == 2

    quick_figure = quick_growth_plot("parity", ["total_influence", "variance"], n_values=[2, 3, 4])
    assert quick_figure is not None

    with pytest.raises(ValueError, match="Unknown family"):
        quick_growth_plot("unknown", n_values=[2])


def _exercise_animation(animation: Any) -> None:
    init = getattr(animation, "_init_func", None)
    if init is not None:
        assert init()
    assert animation._func(0)
    animation._draw_was_started = True


def test_growth_animator_callbacks_and_save(monkeypatch: pytest.MonkeyPatch) -> None:
    family = ParityFamily()
    animator = GrowthAnimator(family)

    growth = animator.animate("total_influence", n_range=(2, 5, 1), interval=1)
    _exercise_animation(growth)
    assert growth._func(2)

    influences = animator.animate_influences(n_range=(2, 5, 1), interval=1)
    _exercise_animation(influences)
    assert influences._func(2)

    saved: list[tuple[str, Any, int]] = []

    def fake_save(filename: str, writer: Any, fps: int, **kwargs: Any) -> None:
        saved.append((filename, writer, fps))

    monkeypatch.setattr(animator.anim, "save", fake_save)
    animator.save("growth.gif", fps=3)
    animator.save("growth.mp4")
    animator.save("growth.bin")
    assert [writer for _, writer, _ in saved] == ["pillow", "ffmpeg", None]

    with pytest.raises(RuntimeError, match="No animation"):
        GrowthAnimator(family).save("missing.gif")


def test_animation_convenience_callbacks() -> None:
    family = ParityFamily()
    animations = [
        animate_growth(family, "degree", (2, 5, 1), interval=1),
        animate_influences(family, (2, 5, 1), interval=1),
        animate_fourier_spectrum(family, (2, 5, 1), interval=1),
        create_growth_animation(family, ["total_influence", "degree"], (2, 5, 1), interval=1),
    ]

    for animation in animations:
        _exercise_animation(animation)


def test_growth_tracker_full_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    tracker = GrowthTracker(MajorityFamily())
    tracker.mark("total_influence")
    tracker.mark("influence_99")
    tracker.mark("noise_stability", rho=0.25)
    tracker.mark("custom", name="boom", compute_fn=lambda function: 1 / 0)

    results = tracker.observe(n_values=[2, 3, 4, 5], verbose=True)
    assert results["total_influence"].n_values == [3, 5]
    assert results["influence_99"].computed_values == [0.0, 0.0]
    assert results["boom"].computed_values == [None, None]
    assert "Theory" in tracker.summary()
    assert tracker.get_result("total_influence") is results["total_influence"]
    assert "done" in capsys.readouterr().out

    assert tracker.plot("total_influence", log_scale=True) is not None
    assert tracker.plot_all() is not None
    tracker.clear()
    assert tracker.results == {}

    with pytest.warns(UserWarning, match="No valid n values"):
        assert GrowthTracker(MajorityFamily()).observe(n_values=[2, 4]) == {}
