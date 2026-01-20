"""
Growth and asymptotic visualization for Boolean function families.

This module provides specialized plotting tools for:
- Asymptotic behavior as n grows
- Family comparison plots
- Theoretical vs computed overlays
- Phase transition visualization
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

# Module logger
_logger = logging.getLogger("boofun.visualization.growth_plots")

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    from ..families.tracker import GrowthTracker


class GrowthVisualizer:
    """
    Specialized visualizer for asymptotic behavior of Boolean function families.

    Features:
    - Plot properties vs n with theoretical overlays
    - Compare multiple families
    - Visualize convergence rates
    - Show phase transitions
    """

    def __init__(self, backend: str = "matplotlib"):
        """
        Initialize growth visualizer.

        Args:
            backend: "matplotlib" or "plotly"
        """
        self.backend = backend.lower()

        if self.backend == "matplotlib" and not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")
        if self.backend == "plotly" and not HAS_PLOTLY:
            raise ImportError("Plotly not available")

        # Default styling
        self.colors = (
            plt.cm.tab10.colors
            if HAS_MATPLOTLIB
            else [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
        )

    def plot_growth(
        self,
        tracker: "GrowthTracker",
        marker_name: str,
        show_theory: bool = True,
        show_error: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        ax=None,
        **kwargs,
    ):
        """
        Plot a single tracked property vs n.

        Args:
            tracker: GrowthTracker with computed results
            marker_name: Which marker to plot
            show_theory: Show theoretical prediction
            show_error: Show error bars/ribbon (if available)
            log_x, log_y: Use log scale
            figsize: Figure size
            title: Custom title
            ax: Matplotlib axes (optional)

        Returns:
            Figure/axes
        """
        if marker_name not in tracker.results:
            raise ValueError(f"No results for marker: {marker_name}")

        result = tracker.results[marker_name]
        n_arr, computed_arr, theory_arr = result.to_arrays()

        if self.backend == "matplotlib":
            return self._plot_growth_matplotlib(
                n_arr,
                computed_arr,
                theory_arr,
                marker_name,
                tracker.family.metadata.name,
                show_theory,
                log_x,
                log_y,
                figsize,
                title,
                ax,
                **kwargs,
            )
        else:
            return self._plot_growth_plotly(
                n_arr,
                computed_arr,
                theory_arr,
                marker_name,
                tracker.family.metadata.name,
                show_theory,
                log_x,
                log_y,
                title,
                **kwargs,
            )

    def _plot_growth_matplotlib(
        self,
        n_arr,
        computed_arr,
        theory_arr,
        marker_name,
        family_name,
        show_theory,
        log_x,
        log_y,
        figsize,
        title,
        ax,
        **kwargs,
    ):
        """Plot growth using matplotlib."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        color = kwargs.get("color", self.colors[0])

        # Plot computed values
        ax.plot(
            n_arr,
            computed_arr,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{marker_name} (computed)",
        )

        # Plot theoretical values
        if show_theory and theory_arr is not None:
            valid_theory = ~np.isnan(theory_arr.astype(float))
            if np.any(valid_theory):
                ax.plot(
                    n_arr[valid_theory],
                    theory_arr[valid_theory],
                    "--",
                    color="gray",
                    linewidth=2,
                    alpha=0.7,
                    label=f"{marker_name} (theory)",
                )

        # Log scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        # Labels
        ax.set_xlabel("n (number of variables)", fontsize=12)
        ax.set_ylabel(marker_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(title or f"{family_name}: {marker_name} vs n", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_growth_plotly(
        self,
        n_arr,
        computed_arr,
        theory_arr,
        marker_name,
        family_name,
        show_theory,
        log_x,
        log_y,
        title,
        **kwargs,
    ):
        """Plot growth using plotly."""
        fig = go.Figure()

        # Computed values
        fig.add_trace(
            go.Scatter(
                x=n_arr,
                y=computed_arr,
                mode="lines+markers",
                name=f"{marker_name} (computed)",
                line=dict(width=3),
                marker=dict(size=10),
            )
        )

        # Theoretical values
        if show_theory and theory_arr is not None:
            valid_theory = ~np.isnan(theory_arr.astype(float))
            if np.any(valid_theory):
                fig.add_trace(
                    go.Scatter(
                        x=n_arr[valid_theory],
                        y=theory_arr[valid_theory],
                        mode="lines",
                        name=f"{marker_name} (theory)",
                        line=dict(dash="dash", color="gray"),
                    )
                )

        fig.update_layout(
            title=title or f"{family_name}: {marker_name} vs n",
            xaxis_title="n (number of variables)",
            yaxis_title=marker_name.replace("_", " ").title(),
            xaxis_type="log" if log_x else "linear",
            yaxis_type="log" if log_y else "linear",
        )

        return fig

    def plot_family_comparison(
        self,
        trackers: Dict[str, "GrowthTracker"],
        marker_name: str,
        show_theory: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ):
        """
        Compare a property across multiple function families.

        Args:
            trackers: Dict mapping family names to GrowthTrackers
            marker_name: Property to compare
            show_theory: Show theoretical predictions
            log_x, log_y: Use log scale
            figsize: Figure size
            title: Custom title

        Returns:
            Figure
        """
        if self.backend == "matplotlib":
            return self._compare_families_matplotlib(
                trackers, marker_name, show_theory, log_x, log_y, figsize, title
            )
        else:
            return self._compare_families_plotly(
                trackers, marker_name, show_theory, log_x, log_y, title
            )

    def _compare_families_matplotlib(
        self, trackers, marker_name, show_theory, log_x, log_y, figsize, title
    ):
        """Compare families using matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)

        for i, (name, tracker) in enumerate(trackers.items()):
            if marker_name not in tracker.results:
                continue

            result = tracker.results[marker_name]
            n_arr, computed_arr, theory_arr = result.to_arrays()

            color = self.colors[i % len(self.colors)]

            # Plot computed
            ax.plot(n_arr, computed_arr, "o-", color=color, linewidth=2, markersize=6, label=name)

            # Plot theory (dashed, same color)
            if show_theory and theory_arr is not None:
                valid = ~np.isnan(theory_arr.astype(float))
                if np.any(valid):
                    ax.plot(
                        n_arr[valid], theory_arr[valid], "--", color=color, alpha=0.5, linewidth=1.5
                    )

        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        ax.set_xlabel("n (number of variables)", fontsize=12)
        ax.set_ylabel(marker_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(title or f"Family Comparison: {marker_name}", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _compare_families_plotly(self, trackers, marker_name, show_theory, log_x, log_y, title):
        """Compare families using plotly."""
        fig = go.Figure()

        colors = self.colors

        for i, (name, tracker) in enumerate(trackers.items()):
            if marker_name not in tracker.results:
                continue

            result = tracker.results[marker_name]
            n_arr, computed_arr, theory_arr = result.to_arrays()

            color = colors[i % len(colors)]

            # Computed
            fig.add_trace(
                go.Scatter(
                    x=n_arr,
                    y=computed_arr,
                    mode="lines+markers",
                    name=name,
                    line=dict(color=color, width=2),
                    marker=dict(size=8),
                )
            )

            # Theory
            if show_theory and theory_arr is not None:
                valid = ~np.isnan(theory_arr.astype(float))
                if np.any(valid):
                    fig.add_trace(
                        go.Scatter(
                            x=n_arr[valid],
                            y=theory_arr[valid],
                            mode="lines",
                            name=f"{name} (theory)",
                            line=dict(color=color, dash="dash", width=1),
                            opacity=0.5,
                        )
                    )

        fig.update_layout(
            title=title or f"Family Comparison: {marker_name}",
            xaxis_title="n (number of variables)",
            yaxis_title=marker_name.replace("_", " ").title(),
            xaxis_type="log" if log_x else "linear",
            yaxis_type="log" if log_y else "linear",
        )

        return fig

    def plot_convergence_rate(
        self,
        tracker: "GrowthTracker",
        marker_name: str,
        reference: str = "sqrt_n",
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Plot ratio of computed value to theoretical reference.

        Useful for seeing convergence rates and constant factors.

        Args:
            tracker: GrowthTracker with results
            marker_name: Property to analyze
            reference: Reference function ("sqrt_n", "n", "log_n", "constant")
            figsize: Figure size

        Returns:
            Figure
        """
        if marker_name not in tracker.results:
            raise ValueError(f"No results for marker: {marker_name}")

        result = tracker.results[marker_name]
        n_arr, computed_arr, _ = result.to_arrays()

        # Compute reference
        if reference == "sqrt_n":
            ref_arr = np.sqrt(n_arr)
            ref_label = "√n"
        elif reference == "n":
            ref_arr = n_arr.astype(float)
            ref_label = "n"
        elif reference == "log_n":
            ref_arr = np.log(n_arr)
            ref_label = "log(n)"
        elif reference == "constant":
            ref_arr = np.ones_like(n_arr, dtype=float)
            ref_label = "1"
        else:
            raise ValueError(f"Unknown reference: {reference}")

        ratio = computed_arr / ref_arr

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(n_arr, ratio, "o-", linewidth=2, markersize=8, color=self.colors[0])

            # Fit a horizontal line (constant) if converging
            if len(ratio) > 2:
                mean_ratio = np.mean(ratio[-3:])  # Last 3 points
                ax.axhline(
                    y=mean_ratio,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Limiting value ≈ {mean_ratio:.4f}",
                )

            ax.set_xlabel("n", fontsize=12)
            ax.set_ylabel(f"{marker_name} / {ref_label}", fontsize=12)
            ax.set_title(f"Convergence: {marker_name} / {ref_label}", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=n_arr, y=ratio, mode="lines+markers", name=f"{marker_name} / {ref_label}"
                )
            )

            if len(ratio) > 2:
                mean_ratio = np.mean(ratio[-3:])
                fig.add_hline(y=mean_ratio, line_dash="dash", annotation_text=f"≈ {mean_ratio:.4f}")

            fig.update_layout(
                title=f"Convergence: {marker_name} / {ref_label}",
                xaxis_title="n",
                yaxis_title=f"{marker_name} / {ref_label}",
            )

            return fig

    def plot_multi_property_growth(
        self,
        tracker: "GrowthTracker",
        marker_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 4),
    ):
        """
        Plot multiple properties side by side.

        Args:
            tracker: GrowthTracker with results
            marker_names: Which markers to plot (None = all)
            figsize: Figure size (per subplot)

        Returns:
            Figure
        """
        if marker_names is None:
            marker_names = list(tracker.results.keys())

        n_markers = len(marker_names)

        if self.backend == "matplotlib":
            fig, axes = plt.subplots(1, n_markers, figsize=(figsize[0], figsize[1]))
            if n_markers == 1:
                axes = [axes]

            for ax, marker_name in zip(axes, marker_names):
                self.plot_growth(tracker, marker_name, ax=ax, show_theory=True, title=marker_name)

            fig.suptitle(f"{tracker.family.metadata.name} Family Properties", fontsize=14)
            plt.tight_layout()
            return fig

        else:
            fig = make_subplots(rows=1, cols=n_markers, subplot_titles=marker_names)

            for i, marker_name in enumerate(marker_names, 1):
                if marker_name in tracker.results:
                    result = tracker.results[marker_name]
                    n_arr, computed_arr, theory_arr = result.to_arrays()

                    fig.add_trace(
                        go.Scatter(x=n_arr, y=computed_arr, mode="lines+markers", name=marker_name),
                        row=1,
                        col=i,
                    )

            fig.update_layout(
                title=f"{tracker.family.metadata.name} Family Properties", showlegend=False
            )

            return fig


class LTFVisualizer:
    """
    Specialized visualizations for Linear Threshold Functions.
    """

    def __init__(self, backend: str = "matplotlib"):
        self.backend = backend

    def plot_weight_distribution(
        self,
        weights: np.ndarray,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ):
        """
        Plot weight distribution of an LTF.

        Args:
            weights: LTF weight vector
            figsize: Figure size
            title: Custom title

        Returns:
            Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar plot of weights
        ax1.bar(range(len(weights)), weights, alpha=0.7, color="steelblue")
        ax1.set_xlabel("Variable Index")
        ax1.set_ylabel("Weight")
        ax1.set_title("Weight Values")
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # Sorted cumulative weight squared
        sorted_sq = np.sort(weights**2)[::-1]
        cumsum = np.cumsum(sorted_sq) / np.sum(sorted_sq)

        ax2.plot(range(1, len(cumsum) + 1), cumsum, "o-", linewidth=2)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Critical index threshold")
        ax2.set_xlabel("Number of Top Variables")
        ax2.set_ylabel("Cumulative Weight Fraction")
        ax2.set_title("Weight Concentration")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title or "LTF Weight Analysis", fontsize=14)
        plt.tight_layout()

        return fig

    def plot_influence_vs_weight(
        self,
        f: "BooleanFunction",
        figsize: Tuple[int, int] = (8, 8),
    ):
        """
        Scatter plot comparing |weight| vs influence.

        Args:
            f: Boolean function (should be LTF)
            figsize: Figure size

        Returns:
            Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")

        from ..analysis import SpectralAnalyzer
        from ..analysis.ltf_analysis import find_ltf_weights, is_ltf

        if not is_ltf(f):
            raise ValueError("Function is not an LTF")

        weights, _ = find_ltf_weights(f)
        influences = SpectralAnalyzer(f).influences()

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(np.abs(weights), influences, s=100, alpha=0.7)

        # Add variable labels
        for i, (w, inf) in enumerate(zip(np.abs(weights), influences)):
            ax.annotate(f"x_{i}", (w, inf), xytext=(5, 5), textcoords="offset points")

        # Fit line (influence should scale with |weight|)
        if len(weights) > 2:
            z = np.polyfit(np.abs(weights), influences, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, max(np.abs(weights)), 100)
            ax.plot(x_line, p(x_line), "--", color="red", alpha=0.7, label=f"Linear fit")

        ax.set_xlabel("|Weight|", fontsize=12)
        ax.set_ylabel("Influence", fontsize=12)
        ax.set_title("Influence vs |Weight| for LTF", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class ComplexityVisualizer:
    """
    Visualize query complexity measures and their relationships.
    """

    def __init__(self, backend: str = "matplotlib"):
        self.backend = backend

    def plot_complexity_relations(
        self,
        f: "BooleanFunction",
        figsize: Tuple[int, int] = (12, 5),
    ):
        """
        Plot relationships between different complexity measures.

        Shows s(f), bs(f), deg(f), C(f) and their bounds.

        Args:
            f: Boolean function
            figsize: Figure size

        Returns:
            Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required")

        from ..analysis.block_sensitivity import block_sensitivity
        from ..analysis.certificates import certificate_complexity
        from ..analysis.fourier import fourier_degree
        from ..analysis.sensitivity import sensitivity

        # Compute measures
        s = sensitivity(f)
        bs = block_sensitivity(f)
        deg = fourier_degree(f)

        try:
            c0, c1 = certificate_complexity(f)
            C = max(c0, c1)
        except Exception as e:
            _logger.debug(f"Certificate complexity computation failed: {e}")
            C = None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar chart of measures
        measures = {"s(f)": s, "bs(f)": bs, "deg(f)": deg}
        if C is not None:
            measures["C(f)"] = C

        ax1.bar(measures.keys(), measures.values(), alpha=0.7, color="steelblue")
        ax1.set_ylabel("Value")
        ax1.set_title("Complexity Measures")
        ax1.grid(True, alpha=0.3)

        # Known bounds visualization
        bounds_data = [
            ("s ≤ bs", s, bs),
            ("s² ≥ deg", s**2, deg),
            ("bs ≤ s⁴", bs, min(s**4, max(measures.values()) * 1.2)),
        ]

        ax2.set_xlim(0, 4)
        ax2.set_ylim(0, max(measures.values()) * 1.3)

        for i, (label, val1, val2) in enumerate(bounds_data):
            y = i * 0.3 + 0.1
            ax2.plot([val1, val2], [y, y], "o-", linewidth=2, markersize=10)
            ax2.annotate(label, (max(val1, val2) + 0.1, y))

        ax2.set_yticks([])
        ax2.set_xlabel("Value")
        ax2.set_title("Known Inequalities (Sensitivity Theorem)")

        fig.suptitle(f"Query Complexity Analysis (n={f.n_vars})", fontsize=14)
        plt.tight_layout()

        return fig


# Convenience function to quickly visualize growth
def quick_growth_plot(
    family_name: str,
    properties: List[str] = ["total_influence", "noise_stability"],
    n_values: Optional[List[int]] = None,
    **kwargs,
):
    """
    Quick way to visualize asymptotic behavior of a built-in family.

    Args:
        family_name: "majority", "parity", "tribes", "and", "or", "dictator"
        properties: Which properties to track
        n_values: List of n values (default: odd 3-15)

    Returns:
        Figure
    """
    from ..families import (
        ANDFamily,
        DictatorFamily,
        GrowthTracker,
        MajorityFamily,
        ORFamily,
        ParityFamily,
        TribesFamily,
    )

    families = {
        "majority": MajorityFamily,
        "parity": ParityFamily,
        "tribes": TribesFamily,
        "and": ANDFamily,
        "or": ORFamily,
        "dictator": DictatorFamily,
    }

    if family_name.lower() not in families:
        raise ValueError(f"Unknown family: {family_name}")

    family = families[family_name.lower()]()
    tracker = GrowthTracker(family)

    for prop in properties:
        tracker.mark(prop, **kwargs)

    if n_values is None:
        n_values = [n for n in range(3, 16, 2) if family.validate_n(n)]

    tracker.observe(n_values=n_values)

    viz = GrowthVisualizer()
    return viz.plot_multi_property_growth(tracker, properties)


__all__ = [
    "GrowthVisualizer",
    "LTFVisualizer",
    "ComplexityVisualizer",
    "quick_growth_plot",
]
