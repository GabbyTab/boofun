"""Behavioral coverage for optional Jupyter widget workflows."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

import boofun as bf
from boofun.visualization import widgets as widget_module

matplotlib.use("Agg", force=True)

pytestmark = pytest.mark.skipif(
    not widget_module.HAS_WIDGETS,
    reason="ipywidgets is not installed",
)


@pytest.fixture(autouse=True)
def widget_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widget_module, "_clear_output", lambda **kwargs: None)
    monkeypatch.setattr(widget_module, "_display", lambda value: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


def test_function_explorer_views_and_display(capsys: pytest.CaptureFixture[str]) -> None:
    explorer = widget_module.InteractiveFunctionExplorer(bf.AND(2))

    for view in (
        "Truth Table",
        "Fourier Spectrum",
        "Influences",
        "Noise Stability",
        "Summary",
    ):
        explorer.view_dropdown.value = view
        explorer._update_display()

    explorer.display()
    output = capsys.readouterr().out
    assert "Truth Table" in output
    assert "Boolean Function Summary" in output

    large = widget_module.InteractiveFunctionExplorer(bf.AND(7))
    large._show_truth_table()
    assert "too large" in capsys.readouterr().out


def test_growth_explorer_all_properties_and_failure_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def family(n: int) -> bf.BooleanFunction:
        if n == 2:
            raise ValueError("intentional gap")
        return bf.AND(n)

    explorer = widget_module.GrowthExplorer(family, name="AND", n_range=(1, 3))
    explorer.n_slider.value = 3

    for property_name in explorer.property_dropdown.options:
        explorer.property_dropdown.value = property_name
        explorer._update_display()

    explorer.display()
    assert "AND(n=3)" in capsys.readouterr().out


def test_property_dashboard_views_and_convenience_factories(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dashboard = widget_module.PropertyDashboard({"and": bf.AND(2), "or": bf.OR(2)})

    for view in dashboard.property_dropdown.options:
        dashboard.property_dropdown.value = view
        dashboard._update_display()

    dashboard.display()
    assert "Property" in capsys.readouterr().out

    function_explorer = widget_module.create_function_explorer(bf.parity(2))
    growth_explorer = widget_module.create_growth_explorer(bf.OR, name="OR", n_range=(1, 3))
    assert function_explorer.n == 2
    assert growth_explorer.name == "OR"
