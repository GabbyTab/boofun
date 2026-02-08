"""
Animation utilities for Boolean function visualization.

This module provides tools for creating animations showing how
properties of Boolean functions change as n increases.

Supports:
- Matplotlib animations (GIF, MP4)
- Plotly animations (HTML)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..families.base import FunctionFamily

# Check for dependencies
try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    pass

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

__all__ = [
    "animate_growth",
    "animate_influences",
    "animate_fourier_spectrum",
    "create_growth_animation",
    "GrowthAnimator",
]


def _check_matplotlib():
    """Ensure matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for animations")


class GrowthAnimator:
    """
    Create animations showing function properties as n grows.

    Example:
        >>> animator = GrowthAnimator(MajorityFamily())
        >>> animator.animate("total_influence", n_range=(3, 15, 2))
        >>> animator.save("majority_growth.gif")
    """

    def __init__(self, family: "FunctionFamily"):
        """
        Initialize animator with a function family.

        Args:
            family: BooleanFamily to animate
        """
        _check_matplotlib()
        self.family = family
        self.fig: Any = None
        self.anim: Any = None
        self._frames_data: List[Dict[str, Any]] = []

    def animate(
        self,
        property_name: str,
        n_range: Tuple[int, int, int] = (3, 15, 2),
        figsize: Tuple[int, int] = (10, 6),
        interval: int = 500,
        property_func: Optional[Callable] = None,
    ) -> animation.FuncAnimation:
        """
        Create animation of property growth.

        Args:
            property_name: Name of property to animate
            n_range: (start, stop, step) for n values
            figsize: Figure size
            interval: Milliseconds between frames
            property_func: Custom function to compute property (default uses builtin)

        Returns:
            Matplotlib animation object
        """
        n_values = list(range(*n_range))

        # Compute property values and cache functions
        self._frames_data = []
        for n in n_values:
            f = self.family.generate(n)

            if property_func:
                val = property_func(f)
            elif property_name == "total_influence":
                val = f.total_influence()
            elif property_name == "max_influence":
                val = f.max_influence()
            elif property_name == "degree":
                val = f.degree()
            elif property_name == "sensitivity":
                val = f.sensitivity()
            elif property_name == "variance":
                val = f.variance()
            elif property_name == "hamming_weight":
                val = f.hamming_weight()
            else:
                val = getattr(f, property_name)()

            self._frames_data.append(
                {
                    "n": n,
                    "value": val,
                    "function": f,
                }
            )

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Initialize plot elements
        (self.line,) = self.ax.plot([], [], "bo-", markersize=10, linewidth=2)
        (self.point,) = self.ax.plot([], [], "ro", markersize=15)
        self.title = self.ax.set_title("")

        # Set axis limits
        all_n = [d["n"] for d in self._frames_data]
        all_vals = [d["value"] for d in self._frames_data]
        self.ax.set_xlim(min(all_n) - 1, max(all_n) + 1)
        self.ax.set_ylim(0, max(all_vals) * 1.2)
        self.ax.set_xlabel("n (number of variables)")
        self.ax.set_ylabel(property_name.replace("_", " ").title())
        self.ax.grid(True, alpha=0.3)

        def init():
            self.line.set_data([], [])
            self.point.set_data([], [])
            return self.line, self.point

        def update(frame):
            # Show all points up to current frame
            ns = [d["n"] for d in self._frames_data[: frame + 1]]
            vals = [d["value"] for d in self._frames_data[: frame + 1]]

            self.line.set_data(ns, vals)
            self.point.set_data([ns[-1]], [vals[-1]])
            self.title.set_text(
                f"{self.family.metadata.name}: {property_name} (n={ns[-1]})\nValue: {vals[-1]:.4f}"
            )

            return self.line, self.point, self.title

        self.anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=len(self._frames_data),
            init_func=init,
            blit=False,
            interval=interval,
            repeat=True,
        )

        return self.anim

    def animate_influences(
        self,
        n_range: Tuple[int, int, int] = (3, 15, 2),
        figsize: Tuple[int, int] = (12, 6),
        interval: int = 700,
    ) -> animation.FuncAnimation:
        """
        Animate influence distribution as n grows.

        Shows bar chart of influences with bars growing/shrinking.

        Args:
            n_range: (start, stop, step) for n values
            figsize: Figure size
            interval: Milliseconds between frames

        Returns:
            Matplotlib animation object
        """
        n_values = list(range(*n_range))

        # Compute all influences
        self._frames_data = []
        max_n = max(n_values)

        for n in n_values:
            f = self.family.generate(n)
            influences = f.influences()
            # Pad to max_n for consistent bar positions
            padded = np.zeros(max_n)
            padded[: len(influences)] = influences

            self._frames_data.append(
                {
                    "n": n,
                    "influences": padded,
                    "actual_n": len(influences),
                }
            )

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Initialize bars
        x = np.arange(max_n)
        self.bars = self.ax.bar(x, np.zeros(max_n), alpha=0.7, color="steelblue")
        self.title = self.ax.set_title("")

        # Set axis limits
        max_inf = max(max(d["influences"]) for d in self._frames_data)
        self.ax.set_ylim(0, max_inf * 1.2)
        self.ax.set_xlabel("Variable Index")
        self.ax.set_ylabel("Influence")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([f"x_{i}" for i in range(max_n)])

        def update(frame):
            data = self._frames_data[frame]

            for bar, h in zip(self.bars, data["influences"]):
                bar.set_height(h)
                # Fade out unused bars
                if bar.get_x() >= data["actual_n"]:
                    bar.set_alpha(0.1)
                else:
                    bar.set_alpha(0.7)

            total_inf = sum(data["influences"][: data["actual_n"]])
            self.title.set_text(
                f'{self.family.metadata.name}: Influences (n={data["n"]})\n'
                f"Total Influence: {total_inf:.3f}"
            )

            return list(self.bars) + [self.title]

        self.anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=len(self._frames_data),
            interval=interval,
            repeat=True,
            blit=False,
        )

        return self.anim

    def save(self, filename: str, fps: int = 2, **kwargs):
        """
        Save animation to file.

        Args:
            filename: Output filename (supports .gif, .mp4)
            fps: Frames per second
            **kwargs: Additional arguments to animation.save()
        """
        if self.anim is None:
            raise RuntimeError("No animation created. Call animate() first.")

        if filename.endswith(".gif"):
            writer = "pillow"
        elif filename.endswith(".mp4"):
            writer = "ffmpeg"
        else:
            writer = None

        self.anim.save(filename, writer=writer, fps=fps, **kwargs)

    def show(self):
        """Display the animation."""
        plt.show()


def animate_growth(
    family: "FunctionFamily",
    property_name: str = "total_influence",
    n_range: Tuple[int, int, int] = (3, 15, 2),
    **kwargs,
) -> animation.FuncAnimation:
    """
    Convenience function to create a growth animation.

    Args:
        family: BooleanFamily to animate
        property_name: Property to track
        n_range: (start, stop, step) for n values
        **kwargs: Additional arguments to GrowthAnimator.animate()

    Returns:
        Matplotlib animation object
    """
    animator = GrowthAnimator(family)
    return animator.animate(property_name, n_range, **kwargs)


def animate_influences(
    family: "FunctionFamily", n_range: Tuple[int, int, int] = (3, 15, 2), **kwargs
) -> animation.FuncAnimation:
    """
    Convenience function to animate influence distribution.

    Args:
        family: BooleanFamily to animate
        n_range: (start, stop, step) for n values
        **kwargs: Additional arguments

    Returns:
        Matplotlib animation object
    """
    animator = GrowthAnimator(family)
    return animator.animate_influences(n_range, **kwargs)


def animate_fourier_spectrum(
    family: "FunctionFamily",
    n_range: Tuple[int, int, int] = (3, 9, 2),
    figsize: Tuple[int, int] = (12, 6),
    interval: int = 800,
) -> animation.FuncAnimation:
    """
    Animate Fourier spectrum (spectral weight by degree) as n grows.

    Args:
        family: BooleanFamily to animate
        n_range: (start, stop, step) for n values
        figsize: Figure size
        interval: Milliseconds between frames

    Returns:
        Matplotlib animation object
    """
    _check_matplotlib()

    n_values = list(range(*n_range))
    max_n = max(n_values)

    # Compute all spectra
    frames_data = []
    for n in n_values:
        f = family.generate(n)
        weights = f.spectral_weight_by_degree()

        # Pad to max_n+1 degrees
        padded = np.zeros(max_n + 1)
        for deg, w in weights.items():
            if deg <= max_n:
                padded[deg] = w

        frames_data.append(
            {
                "n": n,
                "weights": padded,
                "max_degree": max(weights.keys()),
            }
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initialize bars
    x = np.arange(max_n + 1)
    bars = ax.bar(x, np.zeros(max_n + 1), alpha=0.7, color="orange")
    title = ax.set_title("")

    # Set axis limits
    max_weight = max(max(d["weights"]) for d in frames_data)
    ax.set_ylim(0, max_weight * 1.2)
    ax.set_xlabel("Fourier Degree")
    ax.set_ylabel("Spectral Weight")
    ax.set_xticks(x)

    def update(frame):
        data = frames_data[frame]

        for bar, h in zip(bars, data["weights"]):
            bar.set_height(h)

        title.set_text(
            f'{family.metadata.name}: Spectral Weight by Degree (n={data["n"]})\n'
            f'Max degree with weight: {data["max_degree"]}'
        )

        return list(bars) + [title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_data), interval=interval, repeat=True, blit=False
    )

    return anim


def create_growth_animation(
    family: "FunctionFamily",
    properties: Optional[List[str]] = None,
    n_range: Tuple[int, int, int] = (3, 15, 2),
    figsize: Tuple[int, int] = (14, 8),
    interval: int = 600,
) -> animation.FuncAnimation:
    """
    Create multi-panel animation showing multiple properties.

    Args:
        family: BooleanFamily to animate
        properties: List of properties to show (default: influence, degree, variance)
        n_range: (start, stop, step) for n values
        figsize: Figure size
        interval: Milliseconds between frames

    Returns:
        Matplotlib animation object
    """
    _check_matplotlib()

    if properties is None:
        properties = ["total_influence", "degree", "variance"]

    n_values = list(range(*n_range))
    num_props = len(properties)

    # Compute all data
    frames_data: List[Dict[str, Any]] = []
    for n in n_values:
        f = family.generate(n)
        frame: Dict[str, Any] = {"n": n}

        for prop in properties:
            if prop == "total_influence":
                frame[prop] = f.total_influence()
            elif prop == "max_influence":
                frame[prop] = f.max_influence()
            elif prop == "degree":
                frame[prop] = f.degree()
            elif prop == "variance":
                frame[prop] = f.variance()
            elif prop == "sensitivity":
                frame[prop] = f.sensitivity()
            else:
                frame[prop] = getattr(f, prop)()

        frames_data.append(frame)

    # Create figure with subplots
    fig, axes = plt.subplots(1, num_props, figsize=figsize)
    if num_props == 1:
        axes = [axes]

    # Initialize plots
    lines = []
    points = []

    for ax, prop in zip(axes, properties):
        (line,) = ax.plot([], [], "bo-", markersize=8, linewidth=2)
        (point,) = ax.plot([], [], "ro", markersize=12)
        lines.append(line)
        points.append(point)

        # Set limits
        all_vals = [d[prop] for d in frames_data]
        ax.set_xlim(min(n_values) - 1, max(n_values) + 1)
        ax.set_ylim(0, max(all_vals) * 1.2)
        ax.set_xlabel("n")
        ax.set_ylabel(prop.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.set_title(prop.replace("_", " ").title())

    fig.suptitle(f"{family.metadata.name} Growth", fontsize=14)

    def update(frame):
        for line, point, prop in zip(lines, points, properties):
            ns = [d["n"] for d in frames_data[: frame + 1]]
            vals = [d[prop] for d in frames_data[: frame + 1]]

            line.set_data(ns, vals)
            point.set_data([ns[-1]], [vals[-1]])

        return lines + points

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_data), interval=interval, repeat=True, blit=False
    )

    plt.tight_layout()

    return anim
