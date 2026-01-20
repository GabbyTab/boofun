"""
Boolean Function Families - Parameterized families that grow with n.

This module provides tools for:
1. Defining function families (Majority_n, Tribes_{k,n}, LTF families)
2. Tracking properties as n grows
3. Visualizing asymptotic behavior

Key Concepts:
- FunctionFamily: A parameterized family f_n for varying n
- GrowthTracker: Observes and records properties as n increases
- Marker: A property to track (influences, noise stability, etc.)
- Theoretical bounds: Known asymptotic formulas for comparison

Usage:
    from boofun.families import MajorityFamily, GrowthTracker

    # Create a family
    maj = MajorityFamily()

    # Track properties
    tracker = GrowthTracker(maj)
    tracker.mark("total_influence")
    tracker.mark("noise_stability", rho=0.5)

    # Observe growth
    results = tracker.observe(n_values=[5, 7, 9, 11, 13, 15])

    # Plot with theoretical comparison
    tracker.plot("total_influence", show_theory=True)
"""

from .base import FunctionFamily, InductiveFamily
from .builtins import (
    ANDFamily,
    DictatorFamily,
    LTFFamily,
    MajorityFamily,
    ORFamily,
    ParityFamily,
    ThresholdFamily,
    TribesFamily,
)
from .theoretical import TheoreticalBounds
from .tracker import GrowthTracker, Marker, PropertyMarker

__all__ = [
    # Base classes
    "FunctionFamily",
    "InductiveFamily",
    # Built-in families
    "MajorityFamily",
    "ParityFamily",
    "TribesFamily",
    "ThresholdFamily",
    "ANDFamily",
    "ORFamily",
    "DictatorFamily",
    "LTFFamily",
    # Tracking
    "GrowthTracker",
    "Marker",
    "PropertyMarker",
    # Theory
    "TheoreticalBounds",
]
