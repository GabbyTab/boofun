"""
Interactive visualizations using Plotly.

This module provides interactive visualizations for Boolean function analysis,
including Fourier spectrum exploration, influence heatmaps, and comparison tools.
"""

import logging
from typing import TYPE_CHECKING, List, Tuple

# Module logger
_logger = logging.getLogger("boofun.visualization.interactive")

# Check for Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

if TYPE_CHECKING:
    from ..core.base import BooleanFunction
    from ..families.base import FunctionFamily

__all__ = [
    "interactive_fourier_spectrum",
    "interactive_influence_heatmap",
    "interactive_complexity_comparison",
    "interactive_growth_explorer",
    "FourierExplorer",
]


def _check_plotly():
    """Ensure Plotly is available."""
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly required for interactive visualizations. " "Install with: pip install plotly"
        )


def interactive_fourier_spectrum(
    f: "BooleanFunction",
    show_labels: bool = True,
    highlight_threshold: float = 0.1,
    color_by_degree: bool = True,
    height: int = 600,
    width: int = 900,
) -> "go.Figure":
    """
    Create an interactive Fourier spectrum visualization.

    Features:
    - Hover to see coefficient details
    - Click to highlight specific coefficients
    - Zoom and pan
    - Color by degree or magnitude

    Args:
        f: BooleanFunction to visualize
        show_labels: Show subset labels on bars
        highlight_threshold: Highlight coefficients above this magnitude
        color_by_degree: Color bars by Fourier degree
        height: Figure height
        width: Figure width

    Returns:
        Plotly figure
    """
    _check_plotly()

    n = f.n_vars
    fourier = f.fourier()

    # Prepare data
    indices = list(range(len(fourier)))
    coefficients = [float(fourier[i]) for i in indices]

    # Compute degrees and labels
    degrees = [bin(i).count("1") for i in indices]
    labels = [format(i, f"0{n}b") for i in indices]
    subset_labels = []
    for i in indices:
        if i == 0:
            subset_labels.append("∅")
        else:
            bits = [str(j) for j in range(n) if (i >> (n - 1 - j)) & 1]
            subset_labels.append("{" + ",".join(bits) + "}")

    # Create hover text
    hover_text = [
        f"S = {subset_labels[i]}<br>"
        f"Binary: {labels[i]}<br>"
        f"Degree: {degrees[i]}<br>"
        f"f̂(S) = {coefficients[i]:.4f}<br>"
        f"|f̂(S)|² = {coefficients[i]**2:.4f}"
        for i in range(len(indices))
    ]

    # Colors
    if color_by_degree:
        colors = degrees
        colorscale = "Viridis"
        colorbar_title = "Degree"
    else:
        colors = [abs(c) for c in coefficients]
        colorscale = "Blues"
        colorbar_title = "|f̂(S)|"

    # Create figure
    fig = go.Figure()

    # Add bars
    fig.add_trace(
        go.Bar(
            x=indices,
            y=coefficients,
            marker=dict(
                color=colors,
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                line=dict(width=0.5, color="black"),
            ),
            hovertext=hover_text,
            hoverinfo="text",
            name="Fourier coefficients",
        )
    )

    # Highlight large coefficients
    large_indices = [i for i, c in enumerate(coefficients) if abs(c) >= highlight_threshold]
    if large_indices:
        fig.add_trace(
            go.Scatter(
                x=large_indices,
                y=[coefficients[i] for i in large_indices],
                mode="markers",
                marker=dict(
                    size=15, color="red", symbol="star", line=dict(width=2, color="darkred")
                ),
                hoverinfo="skip",
                name=f"|f̂(S)| ≥ {highlight_threshold}",
            )
        )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Layout
    fig.update_layout(
        title=dict(text=f"Fourier Spectrum (n={n})", font=dict(size=18)),
        xaxis=dict(
            title="Subset index",
            tickmode="array" if n <= 4 else "auto",
            tickvals=indices if n <= 4 else None,
            ticktext=subset_labels if n <= 4 else None,
        ),
        yaxis=dict(
            title="Fourier coefficient f̂(S)",
            zeroline=True,
        ),
        height=height,
        width=width,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="closest",
    )

    # Add annotations for stats
    variance = sum(c**2 for c in coefficients if i > 0 for i, c in enumerate(coefficients))
    total_inf = f.total_influence()

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=f"Var[f] = {variance:.4f}<br>I[f] = {total_inf:.4f}",
        showarrow=False,
        font=dict(size=12),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
    )

    return fig


def interactive_influence_heatmap(
    f: "BooleanFunction",
    height: int = 400,
    width: int = 600,
) -> "go.Figure":
    """
    Create an interactive heatmap of variable influences.

    Args:
        f: BooleanFunction to visualize
        height: Figure height
        width: Figure width

    Returns:
        Plotly figure
    """
    _check_plotly()

    n = f.n_vars
    influences = f.influences()

    # Create single-row heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=[influences],
            x=[f"x_{i}" for i in range(n)],
            y=["Influence"],
            colorscale="Reds",
            hovertemplate="Variable: %{x}<br>Influence: %{z:.4f}<extra></extra>",
            colorbar=dict(title="Inf_i[f]"),
        )
    )

    # Add text annotations
    for i, inf in enumerate(influences):
        fig.add_annotation(
            x=f"x_{i}",
            y="Influence",
            text=f"{inf:.3f}",
            showarrow=False,
            font=dict(color="white" if inf > 0.3 else "black", size=12),
        )

    fig.update_layout(
        title=f"Variable Influences (n={n}, I[f]={sum(influences):.3f})",
        height=height,
        width=width,
    )

    return fig


def interactive_complexity_comparison(
    f: "BooleanFunction",
    height: int = 500,
    width: int = 800,
) -> "go.Figure":
    """
    Create interactive comparison of complexity measures.

    Args:
        f: BooleanFunction to analyze
        height: Figure height
        width: Figure width

    Returns:
        Plotly figure with radar chart of complexity measures
    """
    _check_plotly()

    from ..analysis.query_complexity import QueryComplexityProfile

    profile = QueryComplexityProfile(f)
    measures = profile.compute()

    # Select key measures for radar chart
    radar_measures = {
        "D(f)": measures.get("D", 0),
        "s(f)": measures.get("s", 0),
        "bs(f)": measures.get("bs", 0),
        "C(f)": measures.get("C", 0),
        "deg(f)": measures.get("deg", 0),
        "Q(f)": measures.get("Q2", 0),
    }

    # Normalize to [0, n] for visualization
    n = f.n_vars
    max_val = max(max(radar_measures.values()), n)

    categories = list(radar_measures.keys())
    values = [v / max_val for v in radar_measures.values()]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Complexity measures",
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=[
                    f"{int(max_val*0.25)}",
                    f"{int(max_val*0.5)}",
                    f"{int(max_val*0.75)}",
                    f"{int(max_val)}",
                ],
            )
        ),
        title=f"Query Complexity Profile (n={n})",
        height=height,
        width=width,
        showlegend=True,
    )

    return fig


def interactive_growth_explorer(
    family: "FunctionFamily",
    n_range: Tuple[int, int, int] = (3, 15, 2),
    properties: List[str] = None,
    height: int = 600,
    width: int = 1000,
) -> "go.Figure":
    """
    Create interactive explorer for function family growth.

    Features:
    - Slider to animate through n values
    - Multiple property plots
    - Theoretical overlays

    Args:
        family: Function family to explore
        n_range: (start, stop, step) for n values
        properties: Properties to plot (default: influence, degree)
        height: Figure height
        width: Figure width

    Returns:
        Plotly figure with animation
    """
    _check_plotly()

    if properties is None:
        properties = ["total_influence", "degree", "variance"]

    n_values = list(range(*n_range))
    if hasattr(family, "validate_n"):
        n_values = [n for n in n_values if family.validate_n(n)]

    # Compute data for each n
    data = {prop: [] for prop in properties}
    data["n"] = n_values

    for n in n_values:
        f = family.generate(n)
        for prop in properties:
            if prop == "total_influence":
                data[prop].append(f.total_influence())
            elif prop == "max_influence":
                data[prop].append(f.max_influence())
            elif prop == "degree":
                data[prop].append(f.degree())
            elif prop == "variance":
                data[prop].append(f.variance())
            elif prop == "sensitivity":
                data[prop].append(f.sensitivity())
            else:
                try:
                    data[prop].append(getattr(f, prop)())
                except Exception as e:
                    _logger.debug(f"Property '{prop}' computation failed for n={n}: {e}")
                    data[prop].append(0)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=len(properties),
        subplot_titles=[p.replace("_", " ").title() for p in properties],
    )

    for i, prop in enumerate(properties, 1):
        fig.add_trace(
            go.Scatter(
                x=data["n"],
                y=data[prop],
                mode="lines+markers",
                name=prop,
                hovertemplate=f"{prop}=%{{y:.4f}}<br>n=%{{x}}<extra></extra>",
            ),
            row=1,
            col=i,
        )

        fig.update_xaxes(title_text="n", row=1, col=i)
        fig.update_yaxes(title_text=prop.replace("_", " ").title(), row=1, col=i)

    fig.update_layout(
        title=f"{family.metadata.name} Family Growth", height=height, width=width, showlegend=False
    )

    return fig


class FourierExplorer:
    """
    Interactive Fourier coefficient explorer.

    Provides a comprehensive interface for exploring the Fourier
    structure of Boolean functions.
    """

    def __init__(self, f: "BooleanFunction"):
        """
        Initialize explorer with a Boolean function.

        Args:
            f: BooleanFunction to explore
        """
        _check_plotly()
        self.function = f
        self.n_vars = f.n_vars
        self.fourier = f.fourier()
        self._cache = {}

    def spectrum_plot(self, **kwargs) -> "go.Figure":
        """Get interactive spectrum plot."""
        return interactive_fourier_spectrum(self.function, **kwargs)

    def influence_plot(self, **kwargs) -> "go.Figure":
        """Get interactive influence heatmap."""
        return interactive_influence_heatmap(self.function, **kwargs)

    def complexity_plot(self, **kwargs) -> "go.Figure":
        """Get interactive complexity comparison."""
        return interactive_complexity_comparison(self.function, **kwargs)

    def degree_distribution(self, height: int = 400, width: int = 600) -> "go.Figure":
        """
        Plot spectral weight by degree.

        Args:
            height: Figure height
            width: Figure width

        Returns:
            Plotly figure
        """
        weights = self.function.spectral_weight_by_degree()

        degrees = sorted(weights.keys())
        values = [weights[d] for d in degrees]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=degrees,
                    y=values,
                    marker_color="steelblue",
                    hovertemplate="Degree %{x}<br>Weight: %{y:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=f"Spectral Weight by Degree (n={self.n_vars})",
            xaxis_title="Fourier Degree",
            yaxis_title="Spectral Weight W^{=k}[f]",
            height=height,
            width=width,
        )

        return fig

    def top_coefficients(self, k: int = 10, height: int = 400, width: int = 600) -> "go.Figure":
        """
        Plot the top k Fourier coefficients by magnitude.

        Args:
            k: Number of top coefficients to show
            height: Figure height
            width: Figure width

        Returns:
            Plotly figure
        """
        n = self.n_vars

        # Get top k by magnitude
        indexed = [(i, self.fourier[i]) for i in range(len(self.fourier))]
        sorted_by_mag = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)[:k]

        indices, values = zip(*sorted_by_mag)

        # Create labels
        labels = []
        for i in indices:
            if i == 0:
                labels.append("∅")
            else:
                bits = [str(j) for j in range(n) if (i >> (n - 1 - j)) & 1]
                labels.append("{" + ",".join(bits) + "}")

        colors = ["green" if v >= 0 else "red" for v in values]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(range(k)),
                    y=list(values),
                    marker_color=colors,
                    text=labels,
                    textposition="outside",
                    hovertemplate="S = %{text}<br>f̂(S) = %{y:.4f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=f"Top {k} Fourier Coefficients",
            xaxis_title="Rank",
            yaxis_title="f̂(S)",
            height=height,
            width=width,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )

        return fig

    def dashboard(self, height: int = 800, width: int = 1200) -> "go.Figure":
        """
        Create a comprehensive dashboard with multiple views.

        Args:
            height: Total figure height
            width: Total figure width

        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Fourier Spectrum",
                "Spectral Weight by Degree",
                "Variable Influences",
                "Top Coefficients",
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "heatmap"}, {"type": "bar"}]],
        )

        n = self.n_vars

        # 1. Spectrum (simplified)
        fig.add_trace(
            go.Bar(
                x=list(range(min(32, len(self.fourier)))),
                y=[float(self.fourier[i]) for i in range(min(32, len(self.fourier)))],
                marker_color="steelblue",
                name="f̂(S)",
            ),
            row=1,
            col=1,
        )

        # 2. Weight by degree
        weights = self.function.spectral_weight_by_degree()
        fig.add_trace(
            go.Bar(
                x=list(weights.keys()),
                y=list(weights.values()),
                marker_color="orange",
                name="W^{=k}",
            ),
            row=1,
            col=2,
        )

        # 3. Influences
        influences = self.function.influences()
        fig.add_trace(go.Heatmap(z=[influences], colorscale="Reds", showscale=False), row=2, col=1)

        # 4. Top coefficients
        indexed = [(i, self.fourier[i]) for i in range(len(self.fourier))]
        top5 = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)[:5]
        fig.add_trace(
            go.Bar(
                x=list(range(5)),
                y=[t[1] for t in top5],
                marker_color=["green" if t[1] >= 0 else "red" for t in top5],
                name="Top 5",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"Boolean Function Dashboard (n={n})",
            height=height,
            width=width,
            showlegend=False,
        )

        return fig
