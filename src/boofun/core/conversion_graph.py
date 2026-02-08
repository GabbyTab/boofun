"""
Conversion graph system for Boolean function representations.

Implements two-level dispatch for converting between representations:
every representation converts through truth_table as the universal hub.

    source -> truth_table -> target

This is simpler and more predictable than the previous Dijkstra-based
pathfinding (removed in v1.3.0), with no loss of functionality since
truth_table is the one representation every other type can convert to/from.
"""

import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .representations.registry import get_strategy
from .spaces import Space

# ---------------------------------------------------------------------------
# Large-n safety: controls what happens when a conversion would materialise
# a 2^n truth table for large n.
#
# Options:
#   "warn"  -- emit a warning but proceed (default)
#   "raise" -- raise ValueError
#   "off"   -- no check at all
#   int     -- override the threshold (default 25)
# ---------------------------------------------------------------------------
_LARGE_N_POLICY: str = "warn"
_LARGE_N_THRESHOLD: int = 25


def set_large_n_policy(policy: str = "warn", threshold: int = 25) -> None:
    """Configure what happens when a conversion would create a huge truth table.

    Args:
        policy: One of ``"warn"`` (default), ``"raise"``, or ``"off"``.
        threshold: Number of variables above which the policy applies
                   (default 25, meaning 2^25 = 33 million entries).

    Example::

        import boofun
        from boofun.core.conversion_graph import set_large_n_policy

        # I know what I'm doing -- let me materialise up to n=28
        set_large_n_policy("warn", threshold=28)

        # Hard error for safety in automated pipelines
        set_large_n_policy("raise", threshold=22)

        # Disable the check entirely
        set_large_n_policy("off")
    """
    global _LARGE_N_POLICY, _LARGE_N_THRESHOLD
    if policy not in ("warn", "raise", "off"):
        raise ValueError(f"policy must be 'warn', 'raise', or 'off', got {policy!r}")
    _LARGE_N_POLICY = policy
    _LARGE_N_THRESHOLD = threshold


class ConversionCost:
    """
    Represents the cost of converting between two representations.

    Includes time complexity, space complexity, and accuracy considerations.
    """

    def __init__(
        self,
        time_complexity: float,
        space_complexity: float,
        accuracy_loss: float = 0.0,
        is_exact: bool = True,
    ):
        self.time_complexity = time_complexity
        self.space_complexity = space_complexity
        self.accuracy_loss = accuracy_loss
        self.is_exact = is_exact
        self.total_cost = 0.6 * time_complexity + 0.3 * space_complexity + 0.1 * accuracy_loss

    def __lt__(self, other: "ConversionCost") -> bool:
        return self.total_cost < other.total_cost

    def __add__(self, other: "ConversionCost") -> "ConversionCost":
        return ConversionCost(
            time_complexity=self.time_complexity + other.time_complexity,
            space_complexity=max(self.space_complexity, other.space_complexity),
            accuracy_loss=min(1.0, self.accuracy_loss + other.accuracy_loss),
            is_exact=self.is_exact and other.is_exact,
        )

    def __repr__(self) -> str:
        return (
            f"ConversionCost(time={self.time_complexity:.2f}, "
            f"space={self.space_complexity:.2f}, loss={self.accuracy_loss:.3f})"
        )


class ConversionEdge:
    """Represents an edge in the conversion graph."""

    def __init__(
        self,
        source: str,
        target: str,
        cost: ConversionCost,
        converter: Optional[Callable] = None,
    ):
        self.source = source
        self.target = target
        self.cost = cost
        self.converter = converter

    def __repr__(self) -> str:
        return f"ConversionEdge({self.source} -> {self.target}, cost={self.cost})"


class ConversionPath:
    """Represents a complete conversion path between representations."""

    def __init__(self, edges: List[ConversionEdge]):
        self.edges = edges
        self.source = edges[0].source if edges else None
        self.target = edges[-1].target if edges else None

        self.total_cost = ConversionCost(0, 0, 0, True)
        for edge in edges:
            self.total_cost += edge.cost

    def execute(self, data: Any, space: Space, n_vars: int) -> Any:
        """Execute the conversion path step by step.

        If the path goes through truth_table and n_vars exceeds the
        configured threshold, behaviour depends on the policy set via
        :func:`set_large_n_policy` (default: warn).
        """
        if _LARGE_N_POLICY != "off" and n_vars > _LARGE_N_THRESHOLD:
            hub_involved = any(
                e.source == "truth_table" or e.target == "truth_table" for e in self.edges
            )
            if hub_involved:
                size_mb = 2**n_vars * 8 / (1024 * 1024)
                msg = (
                    f"This will materialise a 2^{n_vars} truth table "
                    f"({2**n_vars:,} entries, ~{size_mb:.0f} MB). "
                    f"For large n, consider approximate methods that don't "
                    f"need the full table:\n"
                    f"\n"
                    f"  from boofun.analysis.sampling import (\n"
                    f"      estimate_fourier_coefficient,  # f_hat(S) via Monte Carlo\n"
                    f"      estimate_influence,            # Inf_i via sampling\n"
                    f"  )\n"
                    f"  from boofun.analysis import PropertyTester  # query-based tests\n"
                    f"\n"
                    f"To silence this warning:\n"
                    f"  from boofun.core.conversion_graph import set_large_n_policy\n"
                    f'  set_large_n_policy("off")       # disable check\n'
                    f'  set_large_n_policy("warn", 30)  # raise threshold to 30'
                )
                if _LARGE_N_POLICY == "raise":
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, stacklevel=2)
        current_data = data
        for edge in self.edges:
            if edge.converter:
                current_data = edge.converter(current_data, space, n_vars)
            else:
                source_strategy = get_strategy(edge.source)
                target_strategy = get_strategy(edge.target)
                current_data = source_strategy.convert_to(
                    target_strategy, current_data, space, n_vars
                )
        return current_data

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        path_str = " -> ".join(
            [s for s in [self.source] + [edge.target for edge in self.edges] if s is not None]
        )
        return f"ConversionPath({path_str}, cost={self.total_cost})"


# ---------------------------------------------------------------------------
# Known representation types and their conversion costs to/from truth_table
# ---------------------------------------------------------------------------

_HUB = "truth_table"

# (source, target) -> ConversionCost for direct edges only
_DEFAULT_EDGES: Dict[Tuple[str, str], ConversionCost] = {
    # truth_table -> X
    (_HUB, "fourier_expansion"): ConversionCost(100, 50, 0.0, True),
    (_HUB, "anf"): ConversionCost(30, 40, 0.0, True),
    (_HUB, "polynomial"): ConversionCost(60, 30, 0.0, True),
    (_HUB, "symbolic"): ConversionCost(40, 20, 0.0, True),
    (_HUB, "distribution"): ConversionCost(150, 80, 0.0, True),
    (_HUB, "circuit"): ConversionCost(300, 200, 0.2, False),
    (_HUB, "bdd"): ConversionCost(250, 150, 0.0, True),
    (_HUB, "cnf"): ConversionCost(200, 150, 0.0, True),
    (_HUB, "dnf"): ConversionCost(200, 150, 0.0, True),
    (_HUB, "ltf"): ConversionCost(500, 100, 0.4, False),
    # X -> truth_table
    ("fourier_expansion", _HUB): ConversionCost(120, 100, 0.0, True),
    ("anf", _HUB): ConversionCost(90, 100, 0.0, True),
    ("polynomial", _HUB): ConversionCost(70, 100, 0.0, True),
    ("symbolic", _HUB): ConversionCost(50, 100, 0.1, False),
    ("distribution", _HUB): ConversionCost(200, 100, 0.3, False),
    ("circuit", _HUB): ConversionCost(100, 100, 0.0, True),
    ("bdd", _HUB): ConversionCost(80, 100, 0.0, True),
    ("cnf", _HUB): ConversionCost(120, 100, 0.0, True),
    ("dnf", _HUB): ConversionCost(120, 100, 0.0, True),
    ("ltf", _HUB): ConversionCost(60, 100, 0.0, True),
}


class ConversionGraph:
    """
    Manages conversions between Boolean function representations.

    Uses two-level dispatch through truth_table as the universal hub:
        source -> truth_table -> target

    Direct edges (source -> target) are used when available;
    otherwise the path goes through truth_table.
    """

    def __init__(self) -> None:
        self.edges: Dict[str, List[ConversionEdge]] = defaultdict(list)
        self.path_cache: Dict[Tuple[str, str], Optional[ConversionPath]] = {}
        self._build_default_graph()

    def _build_default_graph(self) -> None:
        for (source, target), cost in _DEFAULT_EDGES.items():
            self._add_edge_internal(source, target, cost)

    def _add_edge_internal(
        self,
        source: str,
        target: str,
        cost: ConversionCost,
        converter: Optional[Callable] = None,
    ) -> None:
        edge = ConversionEdge(source, target, cost, converter)
        self.edges[source].append(edge)

    def add_edge(
        self,
        source: str,
        target: str,
        cost: ConversionCost,
        converter: Optional[Callable] = None,
    ) -> None:
        """Add a conversion edge to the graph."""
        self._add_edge_internal(source, target, cost, converter)
        self.path_cache.clear()

    # ------------------------------------------------------------------
    # Path finding: two-level dispatch through truth_table
    # ------------------------------------------------------------------

    def _find_direct_edge(self, source: str, target: str) -> Optional[ConversionEdge]:
        """Return the cheapest direct edge from source to target, if any."""
        best: Optional[ConversionEdge] = None
        for edge in self.edges.get(source, []):
            if edge.target == target:
                if best is None or edge.cost < best.cost:
                    best = edge
        return best

    def find_optimal_path(
        self, source: str, target: str, n_vars: Optional[int] = None
    ) -> Optional[ConversionPath]:
        """
        Find a conversion path from source to target.

        Strategy:
          1. If source == target, return None (no conversion needed).
          2. If a direct edge exists, use it.
          3. Otherwise route through truth_table: source -> truth_table -> target.
        """
        if source == target:
            return None

        cache_key = (source, target)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        # Try direct edge
        direct = self._find_direct_edge(source, target)
        if direct is not None:
            path = ConversionPath([direct])
            self.path_cache[cache_key] = path
            return path

        # Route through truth_table hub
        if source != _HUB and target != _HUB:
            to_hub = self._find_direct_edge(source, _HUB)
            from_hub = self._find_direct_edge(_HUB, target)
            if to_hub is not None and from_hub is not None:
                path = ConversionPath([to_hub, from_hub])
                self.path_cache[cache_key] = path
                return path

        # No path found
        self.path_cache[cache_key] = None
        return None

    def get_conversion_options(
        self, source: str, max_cost: Optional[float] = None
    ) -> Dict[str, ConversionPath]:
        """Get all possible conversion targets from a source representation."""
        options: Dict[str, ConversionPath] = {}
        all_targets = self._get_all_nodes() - {source}

        for target in all_targets:
            path = self.find_optimal_path(source, target)
            if path and (max_cost is None or path.total_cost.total_cost <= max_cost):
                options[target] = path

        return options

    def estimate_conversion_cost(
        self, source: str, target: str, n_vars: Optional[int] = None
    ) -> Optional[ConversionCost]:
        """Estimate conversion cost without executing the path."""
        path = self.find_optimal_path(source, target, n_vars)
        return path.total_cost if path else None

    def _get_all_nodes(self) -> Set[str]:
        nodes: Set[str] = set()
        nodes.update(self.edges.keys())
        for edge_list in self.edges.values():
            nodes.update(edge.target for edge in edge_list)
        return nodes

    def clear_cache(self) -> None:
        self.path_cache.clear()

    def get_graph_stats(self) -> Dict[str, Any]:
        all_nodes = self._get_all_nodes()
        total_edges = sum(len(edge_list) for edge_list in self.edges.values())

        reachable_pairs = 0
        for source in all_nodes:
            for target in all_nodes:
                if source != target and self.find_optimal_path(source, target) is not None:
                    reachable_pairs += 1

        total_pairs = len(all_nodes) * (len(all_nodes) - 1)
        connectivity = reachable_pairs / total_pairs if total_pairs > 0 else 0

        return {
            "num_nodes": len(all_nodes),
            "num_edges": total_edges,
            "connectivity": connectivity,
            "cached_paths": len(self.path_cache),
            "nodes": sorted(all_nodes),
        }

    def visualize_graph(self, output_format: str = "text") -> str:
        if output_format == "dot":
            return self._generate_dot_graph()
        return self._generate_text_graph()

    def _generate_text_graph(self) -> str:
        lines = ["Conversion Graph:", "=" * 50]
        for node in sorted(self._get_all_nodes()):
            edges = self.edges.get(node, [])
            if edges:
                lines.append(f"\n{node}:")
                for edge in sorted(edges, key=lambda e: e.cost.total_cost):
                    lines.append(f"  -> {edge.target} (cost: {edge.cost.total_cost:.2f})")
            else:
                lines.append(f"\n{node}: (no outgoing edges)")
        return "\n".join(lines)

    def _generate_dot_graph(self) -> str:
        lines = [
            "digraph ConversionGraph {",
            "  rankdir=LR;",
            "  node [shape=box];",
        ]
        for node in sorted(self._get_all_nodes()):
            lines.append(f'  "{node}";')
        for source, edge_list in self.edges.items():
            for edge in edge_list:
                cost = edge.cost.total_cost
                color = "green" if edge.cost.is_exact else "orange"
                lines.append(
                    f'  "{source}" -> "{edge.target}" ' f'[label="{cost:.1f}", color={color}];'
                )
        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level API (unchanged from v1.2.0)
# ---------------------------------------------------------------------------

_conversion_graph = ConversionGraph()


def get_conversion_graph() -> ConversionGraph:
    """Get the global conversion graph instance."""
    return _conversion_graph


def find_conversion_path(
    source: str, target: str, n_vars: Optional[int] = None
) -> Optional[ConversionPath]:
    """Find optimal conversion path between representations."""
    return _conversion_graph.find_optimal_path(source, target, n_vars)


def register_custom_conversion(
    source: str, target: str, cost: ConversionCost, converter: Callable
) -> None:
    """Register a custom conversion between representations."""
    _conversion_graph.add_edge(source, target, cost, converter)


def get_conversion_options(
    source: str, max_cost: Optional[float] = None
) -> Dict[str, ConversionPath]:
    """Get all conversion options from a source representation."""
    return _conversion_graph.get_conversion_options(source, max_cost)


def estimate_conversion_cost(
    source: str, target: str, n_vars: Optional[int] = None
) -> Optional[ConversionCost]:
    """Estimate the cost of converting between representations."""
    return _conversion_graph.estimate_conversion_cost(source, target, n_vars)
