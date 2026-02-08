"""
Interactive Jupyter widgets for Boolean function exploration.

This module provides ipywidgets-based interactive tools for
exploring Boolean functions in Jupyter notebooks.

Requires: ipywidgets (pip install ipywidgets)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict

import numpy as np

# Module logger
_logger = logging.getLogger("boofun.visualization.widgets")

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

# Check for widget support
try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

__all__ = [
    "InteractiveFunctionExplorer",
    "GrowthExplorer",
    "PropertyDashboard",
    "create_function_explorer",
    "create_growth_explorer",
    "HAS_WIDGETS",
]


def _check_widgets():
    """Ensure ipywidgets is available."""
    if not HAS_WIDGETS:
        raise ImportError(
            "ipywidgets required for interactive widgets. " "Install with: pip install ipywidgets"
        )


class InteractiveFunctionExplorer:
    """
    Interactive widget for exploring a single Boolean function.

    Features:
    - View truth table
    - View Fourier coefficients
    - View influences
    - Noise stability curve
    """

    def __init__(self, f: "BooleanFunction"):
        """
        Create explorer for a Boolean function.

        Args:
            f: Boolean function to explore
        """
        _check_widgets()
        self.f = f
        n = f.n_vars
        assert n is not None
        self.n: int = n
        self._create_widgets()

    def _create_widgets(self):
        """Build the widget interface."""
        # View selector
        self.view_dropdown = widgets.Dropdown(
            options=["Truth Table", "Fourier Spectrum", "Influences", "Noise Stability", "Summary"],
            value="Summary",
            description="View:",
            style={"description_width": "initial"},
        )
        self.view_dropdown.observe(self._on_view_change, names="value")

        # Output area
        self.output = widgets.Output()

        # Initial display
        self._update_display()

    def _on_view_change(self, change):
        """Handle view change."""
        self._update_display()

    def _update_display(self):
        """Update the output based on selected view."""
        with self.output:
            clear_output(wait=True)
            view = self.view_dropdown.value

            if view == "Truth Table":
                self._show_truth_table()
            elif view == "Fourier Spectrum":
                self._show_fourier()
            elif view == "Influences":
                self._show_influences()
            elif view == "Noise Stability":
                self._show_noise_stability()
            elif view == "Summary":
                self._show_summary()

    def _show_truth_table(self):
        """Display truth table."""
        if self.n > 6:
            print(f"Truth table too large to display (n={self.n})")
            return

        print(f"Truth Table (n={self.n}):")
        print("-" * 30)
        for x in range(2**self.n):
            bits = format(x, f"0{self.n}b")
            val = self.f.evaluate(x)
            print(f"  {bits} → {val}")

    def _show_fourier(self):
        """Display Fourier spectrum."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib required for Fourier plot")
            return

        fourier = self.f.fourier()
        weights = self.f.spectral_weight_by_degree()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Coefficient magnitudes
        ax1.bar(range(len(fourier)), np.abs(fourier), alpha=0.7)
        ax1.set_xlabel("Subset Index")
        ax1.set_ylabel("|Coefficient|")
        ax1.set_title("Fourier Coefficients")

        # Weight by degree
        degrees = list(weights.keys())
        weight_vals = list(weights.values())
        ax2.bar(degrees, weight_vals, alpha=0.7, color="orange")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Spectral Weight")
        ax2.set_title("Weight by Degree")

        plt.tight_layout()
        plt.show()

    def _show_influences(self):
        """Display influences."""
        if not HAS_MATPLOTLIB:
            influences = self.f.influences()
            for i, inf in enumerate(influences):
                print(f"  x_{i}: {inf:.4f}")
            return

        influences = self.f.influences()

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(range(len(influences)), influences, alpha=0.7, color="steelblue")
        ax.set_xlabel("Variable")
        ax.set_ylabel("Influence")
        ax.set_title(f"Variable Influences (Total: {sum(influences):.3f})")
        ax.set_xticks(range(len(influences)))
        ax.set_xticklabels([f"x_{i}" for i in range(len(influences))])

        # Add value labels
        for bar, inf in zip(bars, influences):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{inf:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        plt.show()

    def _show_noise_stability(self):
        """Display noise stability curve."""
        if not HAS_MATPLOTLIB:
            print("Matplotlib required for noise stability plot")
            return

        rho_values = np.linspace(0, 1, 20)
        stabilities = [self.f.noise_stability(rho) for rho in rho_values]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rho_values, stabilities, "b-", linewidth=2)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
        ax.set_xlabel("Correlation ρ")
        ax.set_ylabel("Noise Stability")
        ax.set_title("Noise Stability Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _show_summary(self):
        """Display function summary."""
        print(f"=== Boolean Function Summary ===")
        print(f"Variables: n = {self.n}")
        print(f"")
        print(f"Properties:")
        print(f"  Balanced: {self.f.is_balanced()}")
        print(f"  Monotone: {self.f.is_monotone(50)}")
        print(f"  Degree: {self.f.degree()}")
        print(f"")
        print(f"Spectral:")
        print(f"  Total Influence: {self.f.total_influence():.4f}")
        print(f"  Max Influence: {self.f.max_influence():.4f}")
        print(f"  Expectation: {self.f.fourier()[0]:.4f}")
        print(f"  Variance: {self.f.variance():.4f}")
        print(f"")
        print(f"Complexity:")
        print(f"  Sensitivity: {self.f.sensitivity()}")
        print(f"  Support Size: {self.f.hamming_weight()}")

    def display(self):
        """Show the interactive widget."""
        display(widgets.VBox([self.view_dropdown, self.output]))


class GrowthExplorer:
    """
    Interactive widget for exploring how functions grow with n.

    Features:
    - Slider to change n
    - Real-time plots of properties
    - Compare with theoretical bounds
    """

    def __init__(
        self,
        family_func: Callable[[int], "BooleanFunction"],
        name: str = "Function",
        n_range: tuple = (1, 12),
    ):
        """
        Create growth explorer.

        Args:
            family_func: Function that takes n and returns BooleanFunction
            name: Name of the function family
            n_range: (min_n, max_n) range for slider
        """
        _check_widgets()
        self.family_func = family_func
        self.name = name
        self.n_range = n_range
        self._create_widgets()

    def _create_widgets(self):
        """Build the widget interface."""
        # N slider
        self.n_slider = widgets.IntSlider(
            value=5,
            min=self.n_range[0],
            max=self.n_range[1],
            step=1,
            description="n:",
            continuous_update=False,
        )
        self.n_slider.observe(self._on_n_change, names="value")

        # Property selector
        self.property_dropdown = widgets.Dropdown(
            options=["Total Influence", "Max Influence", "Degree", "Sensitivity", "Support Size"],
            value="Total Influence",
            description="Property:",
        )
        self.property_dropdown.observe(self._on_property_change, names="value")

        # Output area
        self.output = widgets.Output()

        # Initial display
        self._update_display()

    def _on_n_change(self, change):
        """Handle n change."""
        self._update_display()

    def _on_property_change(self, change):
        """Handle property change."""
        self._update_display()

    def _update_display(self):
        """Update the display."""
        with self.output:
            clear_output(wait=True)

            n = self.n_slider.value
            prop = self.property_dropdown.value

            # Compute values for range of n
            n_values = list(range(self.n_range[0], n + 1))
            prop_values = []

            for n_val in n_values:
                try:
                    f = self.family_func(n_val)
                    if prop == "Total Influence":
                        val = f.total_influence()
                    elif prop == "Max Influence":
                        val = f.max_influence()
                    elif prop == "Degree":
                        val = f.degree()
                    elif prop == "Sensitivity":
                        val = f.sensitivity()
                    elif prop == "Support Size":
                        val = f.hamming_weight()
                    prop_values.append(val)
                except Exception as e:
                    _logger.debug(f"Property '{prop}' computation failed for n={n}: {e}")
                    prop_values.append(np.nan)

            if HAS_MATPLOTLIB:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(n_values, prop_values, "bo-", markersize=8, linewidth=2)
                ax.set_xlabel("n (number of variables)")
                ax.set_ylabel(prop)
                ax.set_title(f"{self.name}: {prop} vs n")
                ax.grid(True, alpha=0.3)

                # Highlight current n
                ax.axvline(x=n, color="red", linestyle="--", alpha=0.5)

                plt.tight_layout()
                plt.show()

            # Print current values
            self.family_func(n)
            print(f"\n{self.name}(n={n}):")
            print(f"  {prop}: {prop_values[-1]:.4f}" if prop_values else "  Error computing")

    def display(self):
        """Show the interactive widget."""
        controls = widgets.HBox([self.n_slider, self.property_dropdown])
        display(widgets.VBox([controls, self.output]))


class PropertyDashboard:
    """
    Dashboard comparing multiple functions side by side.
    """

    def __init__(self, functions: Dict[str, "BooleanFunction"]):
        """
        Create dashboard for multiple functions.

        Args:
            functions: Dict mapping names to BooleanFunction objects
        """
        _check_widgets()
        self.functions = functions
        self._create_widgets()

    def _create_widgets(self):
        """Build the widget interface."""
        # Property selector
        self.property_dropdown = widgets.Dropdown(
            options=["Influences", "Fourier Spectrum", "Properties"],
            value="Influences",
            description="Compare:",
        )
        self.property_dropdown.observe(self._on_change, names="value")

        # Output area
        self.output = widgets.Output()

        # Initial display
        self._update_display()

    def _on_change(self, change):
        """Handle change."""
        self._update_display()

    def _update_display(self):
        """Update the display."""
        with self.output:
            clear_output(wait=True)

            view = self.property_dropdown.value

            if view == "Influences":
                self._compare_influences()
            elif view == "Fourier Spectrum":
                self._compare_fourier()
            elif view == "Properties":
                self._compare_properties()

    def _compare_influences(self):
        """Compare influences across functions."""
        if not HAS_MATPLOTLIB:
            return

        num_funcs = len(self.functions)
        fig, axes = plt.subplots(1, num_funcs, figsize=(4 * num_funcs, 4))
        if num_funcs == 1:
            axes = [axes]

        for ax, (name, f) in zip(axes, self.functions.items()):
            influences = f.influences()
            ax.bar(range(len(influences)), influences, alpha=0.7)
            ax.set_title(f"{name}")
            ax.set_xlabel("Variable")
            ax.set_ylabel("Influence")

        plt.tight_layout()
        plt.show()

    def _compare_fourier(self):
        """Compare Fourier spectra."""
        if not HAS_MATPLOTLIB:
            return

        num_funcs = len(self.functions)
        fig, axes = plt.subplots(1, num_funcs, figsize=(4 * num_funcs, 4))
        if num_funcs == 1:
            axes = [axes]

        for ax, (name, f) in zip(axes, self.functions.items()):
            weights = f.spectral_weight_by_degree()
            ax.bar(weights.keys(), weights.values(), alpha=0.7)
            ax.set_title(f"{name}")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Weight")

        plt.tight_layout()
        plt.show()

    def _compare_properties(self):
        """Compare properties in a table."""
        print("=" * 60)
        print(f"{'Property':<20}", end="")
        for name in self.functions.keys():
            print(f"{name:<15}", end="")
        print()
        print("=" * 60)

        properties = [
            ("n", lambda f: f.n_vars),
            ("Balanced", lambda f: f.is_balanced()),
            ("Degree", lambda f: f.degree()),
            ("Total Influence", lambda f: f"{f.total_influence():.3f}"),
            ("Max Influence", lambda f: f"{f.max_influence():.3f}"),
            ("Sensitivity", lambda f: f.sensitivity()),
            ("Support Size", lambda f: f.hamming_weight()),
        ]

        for prop_name, prop_func in properties:
            print(f"{prop_name:<20}", end="")
            for f in self.functions.values():
                try:
                    val = prop_func(f)
                    print(f"{str(val):<15}", end="")
                except Exception as e:
                    _logger.debug(f"Property '{prop_name}' computation failed: {e}")
                    print(f"{'N/A':<15}", end="")
            print()

    def display(self):
        """Show the dashboard."""
        display(widgets.VBox([self.property_dropdown, self.output]))


def create_function_explorer(f: "BooleanFunction") -> InteractiveFunctionExplorer:
    """
    Convenience function to create and display a function explorer.

    Args:
        f: Boolean function to explore

    Returns:
        InteractiveFunctionExplorer instance
    """
    explorer = InteractiveFunctionExplorer(f)
    explorer.display()
    return explorer


def create_growth_explorer(
    family_func: Callable[[int], "BooleanFunction"],
    name: str = "Function",
    n_range: tuple = (1, 12),
) -> GrowthExplorer:
    """
    Convenience function to create and display a growth explorer.

    Args:
        family_func: Function that takes n and returns BooleanFunction
        name: Name of the function family
        n_range: (min_n, max_n) range

    Returns:
        GrowthExplorer instance
    """
    explorer = GrowthExplorer(family_func, name, n_range)
    explorer.display()
    return explorer
