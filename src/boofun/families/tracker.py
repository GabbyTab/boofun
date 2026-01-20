"""
Growth tracking for Boolean function families.

The GrowthTracker observes and records properties of a function family
as n grows, enabling:
- Visualization of asymptotic behavior
- Comparison with theoretical predictions
- Discovery of patterns and phase transitions
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Module logger
_logger = logging.getLogger("boofun.families.tracker")

if TYPE_CHECKING:
    from ..core.base import BooleanFunction
    from .base import FunctionFamily


class MarkerType(Enum):
    """Types of properties that can be tracked."""

    SCALAR = "scalar"  # Single number (total influence, expectation)
    VECTOR = "vector"  # One value per variable (influences)
    MATRIX = "matrix"  # n x n values (interactions)
    BOOLEAN = "boolean"  # True/False property
    FOURIER = "fourier"  # Fourier spectrum related
    COMPLEXITY = "complexity"  # Query/complexity measures


@dataclass
class Marker:
    """
    A property to track as n grows.

    Attributes:
        name: Identifier for this marker
        compute_fn: Function(BooleanFunction) -> value
        marker_type: Type of the value
        description: Human-readable description
        theoretical_fn: Optional function(n) -> theoretical value
        params: Additional parameters for computation
    """

    name: str
    compute_fn: Callable[["BooleanFunction"], Any]
    marker_type: MarkerType = MarkerType.SCALAR
    description: str = ""
    theoretical_fn: Optional[Callable[[int], Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def compute(self, f: "BooleanFunction") -> Any:
        """Compute the marked value for a function."""
        return self.compute_fn(f)

    def theoretical(self, n: int) -> Optional[Any]:
        """Get theoretical value if available."""
        if self.theoretical_fn is not None:
            return self.theoretical_fn(n)
        return None


class PropertyMarker:
    """Factory for common property markers."""

    @staticmethod
    def total_influence(theoretical: Optional[Callable[[int], float]] = None) -> Marker:
        """Track total influence I[f]."""

        def compute(f):
            from ..analysis import SpectralAnalyzer

            return SpectralAnalyzer(f).total_influence()

        return Marker(
            name="total_influence",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description="Total influence I[f] = Σ Inf_i[f]",
            theoretical_fn=theoretical,
        )

    @staticmethod
    def influences(variable: Optional[int] = None) -> Marker:
        """Track variable influences."""

        def compute(f):
            from ..analysis import SpectralAnalyzer

            infs = SpectralAnalyzer(f).influences()
            if variable is not None:
                return infs[variable] if variable < len(infs) else 0.0
            return infs

        marker_type = MarkerType.SCALAR if variable is not None else MarkerType.VECTOR
        name = f"influence_{variable}" if variable is not None else "influences"

        return Marker(
            name=name,
            compute_fn=compute,
            marker_type=marker_type,
            description=f"Variable influence{'s' if variable is None else f' for x_{variable}'}",
            params={"variable": variable},
        )

    @staticmethod
    def noise_stability(
        rho: float = 0.5, theoretical: Optional[Callable[[int, float], float]] = None
    ) -> Marker:
        """Track noise stability Stab_ρ[f]."""

        def compute(f):
            from ..analysis import SpectralAnalyzer

            return SpectralAnalyzer(f).noise_stability(rho)

        def theory(n):
            if theoretical:
                return theoretical(n, rho)
            return None

        return Marker(
            name=f"noise_stability_{rho}",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description=f"Noise stability Stab_{rho}[f]",
            theoretical_fn=theory if theoretical else None,
            params={"rho": rho},
        )

    @staticmethod
    def fourier_degree() -> Marker:
        """Track Fourier degree."""

        def compute(f):
            from ..analysis.fourier import fourier_degree

            return fourier_degree(f)

        return Marker(
            name="fourier_degree",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description="Fourier degree (max |S| with f̂(S) ≠ 0)",
        )

    @staticmethod
    def spectral_concentration(k: int) -> Marker:
        """Track weight on coefficients of degree ≤ k."""

        def compute(f):
            from ..analysis import SpectralAnalyzer

            return SpectralAnalyzer(f).spectral_concentration(k)

        return Marker(
            name=f"spectral_concentration_{k}",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description=f"Σ|S|≤{k} f̂(S)²",
            params={"k": k},
        )

    @staticmethod
    def expectation() -> Marker:
        """Track E[f] = f̂(∅)."""

        def compute(f):
            from ..analysis import SpectralAnalyzer

            fourier = SpectralAnalyzer(f).fourier_expansion()
            return fourier[0]

        return Marker(
            name="expectation",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description="Expectation E[f] = f̂(∅)",
        )

    @staticmethod
    def variance() -> Marker:
        """Track Var[f] = 1 - f̂(∅)²."""

        def compute(f):
            from ..analysis import SpectralAnalyzer

            fourier = SpectralAnalyzer(f).fourier_expansion()
            return 1 - fourier[0] ** 2

        return Marker(
            name="variance",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description="Variance Var[f] = 1 - E[f]²",
        )

    @staticmethod
    def is_property(property_name: str) -> Marker:
        """Track whether a property holds."""

        def compute(f):
            method = getattr(f, f"is_{property_name}", None)
            if method is not None:
                return method()

            # Try property tester
            from ..analysis import PropertyTester

            tester = PropertyTester(f)
            test_method = getattr(tester, f"{property_name}_test", None)
            if test_method is not None:
                return test_method()

            return None

        return Marker(
            name=f"is_{property_name}",
            compute_fn=compute,
            marker_type=MarkerType.BOOLEAN,
            description=f"Is function {property_name}?",
        )

    @staticmethod
    def sensitivity() -> Marker:
        """Track sensitivity s(f)."""

        def compute(f):
            from ..analysis.sensitivity import sensitivity

            return sensitivity(f)

        return Marker(
            name="sensitivity",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description="Sensitivity s(f)",
        )

    @staticmethod
    def block_sensitivity() -> Marker:
        """Track block sensitivity bs(f)."""

        def compute(f):
            from ..analysis.block_sensitivity import block_sensitivity

            return block_sensitivity(f)

        return Marker(
            name="block_sensitivity",
            compute_fn=compute,
            marker_type=MarkerType.SCALAR,
            description="Block sensitivity bs(f)",
        )

    @staticmethod
    def custom(
        name: str,
        compute_fn: Callable[["BooleanFunction"], Any],
        marker_type: MarkerType = MarkerType.SCALAR,
        description: str = "",
        theoretical_fn: Optional[Callable[[int], Any]] = None,
    ) -> Marker:
        """Create a custom marker."""
        return Marker(
            name=name,
            compute_fn=compute_fn,
            marker_type=marker_type,
            description=description,
            theoretical_fn=theoretical_fn,
        )


@dataclass
class TrackingResult:
    """Results of tracking a property across n values."""

    marker: Marker
    n_values: List[int]
    computed_values: List[Any]
    theoretical_values: Optional[List[Any]] = None
    computation_times: Optional[List[float]] = None

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Convert to numpy arrays for plotting."""
        n_arr = np.array(self.n_values)
        computed_arr = np.array(self.computed_values)
        theory_arr = None
        if self.theoretical_values is not None:
            theory_arr = np.array(self.theoretical_values)
        return n_arr, computed_arr, theory_arr


class GrowthTracker:
    """
    Track properties of a function family as n grows.

    Usage:
        tracker = GrowthTracker(MajorityFamily())
        tracker.mark("total_influence")
        tracker.mark("noise_stability", rho=0.5)

        results = tracker.observe([5, 7, 9, 11, 13, 15])
        tracker.plot("total_influence", show_theory=True)
    """

    def __init__(self, family: "FunctionFamily"):
        """
        Initialize tracker for a function family.

        Args:
            family: The function family to track
        """
        self.family = family
        self.markers: Dict[str, Marker] = {}
        self.results: Dict[str, TrackingResult] = {}
        self._functions_cache: Dict[int, "BooleanFunction"] = {}

    def mark(self, property_name: str, **kwargs) -> "GrowthTracker":
        """
        Add a property to track.

        Args:
            property_name: Built-in property name or "custom"
            **kwargs: Parameters for the marker

        Returns:
            self (for chaining)

        Built-in properties:
            - "total_influence"
            - "influences" or "influence_i" (with variable=i)
            - "noise_stability" (with rho=0.5)
            - "fourier_degree"
            - "spectral_concentration" (with k=degree)
            - "expectation"
            - "variance"
            - "is_<property>" (e.g., "is_monotone", "is_balanced")
            - "sensitivity"
            - "block_sensitivity"
        """
        # Try to get theoretical formula from family
        theoretical = None
        family_asymptotics = self.family.metadata.asymptotics

        if property_name == "total_influence":
            theoretical = family_asymptotics.get("total_influence")
            marker = PropertyMarker.total_influence(theoretical)

        elif property_name == "influences":
            marker = PropertyMarker.influences(variable=kwargs.get("variable"))

        elif property_name.startswith("influence_"):
            var = int(property_name.split("_")[1])
            marker = PropertyMarker.influences(variable=var)

        elif property_name == "noise_stability":
            rho = kwargs.get("rho", 0.5)
            theoretical = family_asymptotics.get("noise_stability")
            marker = PropertyMarker.noise_stability(rho, theoretical)

        elif property_name == "fourier_degree":
            marker = PropertyMarker.fourier_degree()

        elif property_name == "spectral_concentration":
            k = kwargs.get("k", 1)
            marker = PropertyMarker.spectral_concentration(k)

        elif property_name == "expectation":
            marker = PropertyMarker.expectation()

        elif property_name == "variance":
            marker = PropertyMarker.variance()

        elif property_name.startswith("is_"):
            prop = property_name[3:]
            marker = PropertyMarker.is_property(prop)

        elif property_name == "sensitivity":
            marker = PropertyMarker.sensitivity()

        elif property_name == "block_sensitivity":
            marker = PropertyMarker.block_sensitivity()

        elif property_name == "custom":
            marker = PropertyMarker.custom(**kwargs)

        else:
            # Try as a custom callable
            if "compute_fn" in kwargs:
                marker = PropertyMarker.custom(property_name, **kwargs)
            else:
                raise ValueError(f"Unknown property: {property_name}")

        self.markers[marker.name] = marker
        return self

    def observe(
        self,
        n_values: Optional[List[int]] = None,
        n_range: Optional[range] = None,
        n_min: int = 3,
        n_max: int = 15,
        step: int = 2,
        verbose: bool = False,
    ) -> Dict[str, TrackingResult]:
        """
        Observe the family over a range of n values.

        Args:
            n_values: Explicit list of n values
            n_range: Range object for n values
            n_min, n_max, step: Parameters for default range
            verbose: Print progress

        Returns:
            Dictionary mapping marker name -> TrackingResult
        """
        import time

        # Determine n values
        if n_values is not None:
            ns = list(n_values)
        elif n_range is not None:
            ns = list(n_range)
        else:
            ns = list(range(n_min, n_max + 1, step))

        # Filter valid n values
        ns = [n for n in ns if self.family.validate_n(n)]

        if not ns:
            warnings.warn("No valid n values for this family")
            return {}

        # Initialize results
        for marker_name in self.markers:
            self.results[marker_name] = TrackingResult(
                marker=self.markers[marker_name],
                n_values=[],
                computed_values=[],
                theoretical_values=[],
                computation_times=[],
            )

        # Compute for each n
        for n in ns:
            if verbose:
                print(f"Computing n={n}...", end=" ")

            # Generate or get cached function
            if n not in self._functions_cache:
                self._functions_cache[n] = self.family.generate(n)
            f = self._functions_cache[n]

            # Compute each marker
            for marker_name, marker in self.markers.items():
                result = self.results[marker_name]

                start = time.time()
                try:
                    value = marker.compute(f)
                except Exception as e:
                    if verbose:
                        print(f"Error computing {marker_name}: {e}")
                    value = None
                elapsed = time.time() - start

                result.n_values.append(n)
                result.computed_values.append(value)
                result.computation_times.append(elapsed)

                # Get theoretical value
                theory = marker.theoretical(n)
                if theory is None and marker.name in self.family.metadata.asymptotics:
                    theory_fn = self.family.metadata.asymptotics[marker.name]
                    if callable(theory_fn):
                        try:
                            theory = theory_fn(n, **marker.params)
                        except Exception as e:
                            _logger.debug(
                                f"Theoretical value computation failed for {marker.name} at n={n}: {e}"
                            )
                            theory = None
                result.theoretical_values.append(theory)

            if verbose:
                print("done")

        return self.results

    def get_result(self, marker_name: str) -> Optional[TrackingResult]:
        """Get tracking result for a specific marker."""
        return self.results.get(marker_name)

    def plot(
        self,
        marker_name: str,
        show_theory: bool = True,
        log_scale: bool = False,
        ax=None,
        **plot_kwargs,
    ):
        """
        Plot a tracked property vs n.

        Args:
            marker_name: Which marker to plot
            show_theory: Show theoretical prediction line
            log_scale: Use log scale for y-axis
            ax: Matplotlib axes (creates new figure if None)
            **plot_kwargs: Additional arguments for plt.plot

        Returns:
            Matplotlib axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return None

        if marker_name not in self.results:
            raise ValueError(f"No results for marker: {marker_name}")

        result = self.results[marker_name]
        n_arr, computed_arr, theory_arr = result.to_arrays()

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot computed values
        label = plot_kwargs.pop("label", f"{marker_name} (computed)")
        ax.plot(n_arr, computed_arr, "o-", label=label, **plot_kwargs)

        # Plot theoretical values
        if show_theory and theory_arr is not None and not np.any(np.isnan(theory_arr)):
            ax.plot(
                n_arr, theory_arr, "--", color="gray", label=f"{marker_name} (theory)", alpha=0.7
            )

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel("n (number of variables)")
        ax.set_ylabel(result.marker.description or marker_name)
        ax.set_title(f"{self.family.metadata.name}: {marker_name} vs n")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_all(self, show_theory: bool = True, figsize: Tuple[int, int] = (15, 4)):
        """Plot all tracked markers in subplots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
            return None

        n_markers = len(self.results)
        if n_markers == 0:
            warnings.warn("No results to plot")
            return None

        fig, axes = plt.subplots(1, n_markers, figsize=(figsize[0], figsize[1]))
        if n_markers == 1:
            axes = [axes]

        for ax, marker_name in zip(axes, self.results.keys()):
            self.plot(marker_name, show_theory=show_theory, ax=ax)

        plt.tight_layout()
        return fig

    def summary(self) -> str:
        """Generate text summary of tracking results."""
        lines = [
            f"Growth Tracking Summary: {self.family.metadata.name}",
            "=" * 50,
        ]

        for marker_name, result in self.results.items():
            n_arr, computed_arr, theory_arr = result.to_arrays()

            lines.append(f"\n{marker_name}:")
            lines.append(f"  n range: {n_arr[0]} to {n_arr[-1]}")

            if result.marker.marker_type == MarkerType.SCALAR:
                lines.append(f"  Computed: {computed_arr[0]:.4f} → {computed_arr[-1]:.4f}")
                if theory_arr is not None and not np.any([t is None for t in theory_arr]):
                    lines.append(f"  Theory:   {theory_arr[0]:.4f} → {theory_arr[-1]:.4f}")
                    # Compute relative error
                    rel_errors = np.abs(computed_arr - theory_arr) / (np.abs(theory_arr) + 1e-10)
                    lines.append(f"  Max relative error: {np.max(rel_errors):.2%}")

        return "\n".join(lines)

    def clear(self):
        """Clear all tracking data."""
        self.results.clear()
        self._functions_cache.clear()


__all__ = [
    "MarkerType",
    "Marker",
    "PropertyMarker",
    "TrackingResult",
    "GrowthTracker",
]
