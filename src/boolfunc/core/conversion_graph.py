"""
Lazy conversion graph system for Boolean function representations.

This module implements the conversion graph mentioned in the design document,
enabling intelligent conversion between representations with optimal paths
and caching for efficiency.
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
import heapq
from abc import ABC, abstractmethod
import warnings

from .representations.base import BooleanFunctionRepresentation
from .representations.registry import get_strategy, STRATEGY_REGISTRY
from .spaces import Space


class ConversionCost:
    """
    Represents the cost of converting between two representations.
    
    Includes time complexity, space complexity, and accuracy considerations.
    """
    
    def __init__(self, 
                 time_complexity: float,
                 space_complexity: float, 
                 accuracy_loss: float = 0.0,
                 is_exact: bool = True):
        """
        Initialize conversion cost.
        
        Args:
            time_complexity: Time cost (higher = more expensive)
            space_complexity: Space cost (higher = more expensive) 
            accuracy_loss: Information loss (0 = lossless, 1 = complete loss)
            is_exact: Whether conversion is mathematically exact
        """
        self.time_complexity = time_complexity
        self.space_complexity = space_complexity
        self.accuracy_loss = accuracy_loss
        self.is_exact = is_exact
        
        # Combined cost for pathfinding (weighted sum)
        self.total_cost = (
            0.6 * time_complexity + 
            0.3 * space_complexity + 
            0.1 * accuracy_loss
        )
    
    def __lt__(self, other: 'ConversionCost') -> bool:
        """Compare costs for priority queue."""
        return self.total_cost < other.total_cost
    
    def __add__(self, other: 'ConversionCost') -> 'ConversionCost':
        """Combine costs along a path."""
        return ConversionCost(
            time_complexity=self.time_complexity + other.time_complexity,
            space_complexity=max(self.space_complexity, other.space_complexity),  # Peak memory
            accuracy_loss=min(1.0, self.accuracy_loss + other.accuracy_loss),  # Cumulative loss
            is_exact=self.is_exact and other.is_exact
        )
    
    def __repr__(self) -> str:
        return f"ConversionCost(time={self.time_complexity:.2f}, space={self.space_complexity:.2f}, loss={self.accuracy_loss:.3f})"


class ConversionEdge:
    """Represents an edge in the conversion graph."""
    
    def __init__(self, 
                 source: str, 
                 target: str, 
                 cost: ConversionCost,
                 converter: Optional[Callable] = None):
        """
        Initialize conversion edge.
        
        Args:
            source: Source representation name
            target: Target representation name
            cost: Conversion cost
            converter: Optional custom conversion function
        """
        self.source = source
        self.target = target
        self.cost = cost
        self.converter = converter
        
    def __repr__(self) -> str:
        return f"ConversionEdge({self.source} -> {self.target}, cost={self.cost})"


class ConversionPath:
    """Represents a complete conversion path between representations."""
    
    def __init__(self, edges: List[ConversionEdge]):
        """
        Initialize conversion path.
        
        Args:
            edges: List of conversion edges forming the path
        """
        self.edges = edges
        self.source = edges[0].source if edges else None
        self.target = edges[-1].target if edges else None
        
        # Calculate total cost
        self.total_cost = ConversionCost(0, 0, 0, True)
        for edge in edges:
            self.total_cost += edge.cost
    
    def execute(self, data: Any, space: Space, n_vars: int) -> Any:
        """
        Execute the conversion path.
        
        Args:
            data: Source data
            space: Mathematical space
            n_vars: Number of variables
            
        Returns:
            Converted data
        """
        current_data = data
        current_repr = self.source
        
        for edge in self.edges:
            if edge.converter:
                # Use custom converter
                current_data = edge.converter(current_data, space, n_vars)
            else:
                # Use standard representation conversion
                source_strategy = get_strategy(edge.source)
                target_strategy = get_strategy(edge.target)
                current_data = source_strategy.convert_to(
                    target_strategy, current_data, space, n_vars
                )
            current_repr = edge.target
        
        return current_data
    
    def __len__(self) -> int:
        return len(self.edges)
    
    def __repr__(self) -> str:
        path_str = " -> ".join([self.source] + [edge.target for edge in self.edges])
        return f"ConversionPath({path_str}, cost={self.total_cost})"


class ConversionGraph:
    """
    Manages the graph of possible conversions between representations.
    
    Implements Dijkstra's algorithm to find optimal conversion paths and
    caches results for efficiency.
    """
    
    def __init__(self):
        """Initialize conversion graph."""
        self.edges: Dict[str, List[ConversionEdge]] = defaultdict(list)
        self.path_cache: Dict[Tuple[str, str], Optional[ConversionPath]] = {}
        self.cost_estimates: Dict[str, Dict[str, ConversionCost]] = {}
        self._build_default_graph()
    
    def _build_default_graph(self):
        """Build the default conversion graph with known conversions."""
        
        # Define conversion costs for different representation pairs
        conversion_costs = {
            # Truth table is the universal converter (high cost but always works)
            ('truth_table', 'fourier_expansion'): ConversionCost(100, 50, 0.0, True),
            ('truth_table', 'anf'): ConversionCost(30, 40, 0.0, True),
            ('truth_table', 'polynomial'): ConversionCost(60, 30, 0.0, True),
            ('truth_table', 'symbolic'): ConversionCost(40, 20, 0.0, True),
            
            # ANF conversions
            ('anf', 'truth_table'): ConversionCost(90, 100, 0.0, True),
            ('anf', 'polynomial'): ConversionCost(20, 10, 0.0, True),
            ('anf', 'symbolic'): ConversionCost(30, 15, 0.0, True),
            
            # Fourier conversions (expensive but mathematically rich)
            ('fourier_expansion', 'truth_table'): ConversionCost(120, 100, 0.0, True),
            ('fourier_expansion', 'anf'): ConversionCost(110, 80, 0.0, True),
            
            # Polynomial conversions
            ('polynomial', 'anf'): ConversionCost(25, 15, 0.0, True),
            ('polynomial', 'truth_table'): ConversionCost(70, 100, 0.0, True),
            ('polynomial', 'symbolic'): ConversionCost(15, 10, 0.0, True),
            
            # Symbolic conversions (potentially lossy for complex expressions)
            ('symbolic', 'truth_table'): ConversionCost(50, 100, 0.1, False),
            ('symbolic', 'anf'): ConversionCost(40, 30, 0.05, True),
            ('symbolic', 'polynomial'): ConversionCost(20, 15, 0.0, True),
            
            # Distribution conversions (statistical, potentially lossy)
            ('distribution', 'truth_table'): ConversionCost(200, 100, 0.3, False),
            ('truth_table', 'distribution'): ConversionCost(150, 80, 0.0, True),
            
            # Circuit conversions (structural, may be lossy)
            ('circuit', 'truth_table'): ConversionCost(100, 100, 0.0, True),
            ('truth_table', 'circuit'): ConversionCost(300, 200, 0.2, False),
            
            # BDD conversions (structure-dependent costs)
            ('bdd', 'truth_table'): ConversionCost(80, 100, 0.0, True),
            ('truth_table', 'bdd'): ConversionCost(250, 150, 0.0, True),
            
            # CNF/DNF conversions (registered as 'cnf' and 'dnf')
            ('cnf', 'truth_table'): ConversionCost(120, 100, 0.0, True),
            ('dnf', 'truth_table'): ConversionCost(120, 100, 0.0, True),
            ('truth_table', 'cnf'): ConversionCost(200, 150, 0.0, True),
            ('truth_table', 'dnf'): ConversionCost(200, 150, 0.0, True),
            
            # LTF conversions (approximation-based)
            ('ltf', 'truth_table'): ConversionCost(60, 100, 0.0, True),
            ('truth_table', 'ltf'): ConversionCost(500, 100, 0.4, False),  # High cost, lossy
        }
        
        # Add edges to the graph
        for (source, target), cost in conversion_costs.items():
            self.add_edge(source, target, cost)
    
    def add_edge(self, source: str, target: str, cost: ConversionCost, 
                 converter: Optional[Callable] = None):
        """
        Add a conversion edge to the graph.
        
        Args:
            source: Source representation name
            target: Target representation name
            cost: Conversion cost
            converter: Optional custom conversion function
        """
        edge = ConversionEdge(source, target, cost, converter)
        self.edges[source].append(edge)
        
        # Clear cached paths that might be affected
        self._invalidate_cache_for_node(source)
        self._invalidate_cache_for_node(target)
    
    def find_optimal_path(self, source: str, target: str, 
                         n_vars: Optional[int] = None) -> Optional[ConversionPath]:
        """
        Find the optimal conversion path using Dijkstra's algorithm.
        
        Args:
            source: Source representation name
            target: Target representation name
            n_vars: Number of variables (affects cost calculations)
            
        Returns:
            Optimal conversion path or None if no path exists
        """
        # Check cache first
        cache_key = (source, target)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Dijkstra's algorithm
        distances = {node: ConversionCost(float('inf'), float('inf'), 1.0, False) 
                    for node in self._get_all_nodes()}
        distances[source] = ConversionCost(0, 0, 0, True)
        
        previous = {}
        edge_map = {}  # Track which edge was used to reach each node
        visited = set()
        
        # Priority queue: (cost, node)
        pq = [(ConversionCost(0, 0, 0, True), source)]
        
        while pq:
            current_cost, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            if current_node == target:
                # Reconstruct path
                path_edges = []
                node = target
                while node in previous:
                    edge = edge_map[node]
                    path_edges.append(edge)
                    node = previous[node]
                
                path_edges.reverse()
                path = ConversionPath(path_edges) if path_edges else None
                self.path_cache[cache_key] = path
                return path
            
            # Explore neighbors
            for edge in self.edges.get(current_node, []):
                neighbor = edge.target
                if neighbor in visited:
                    continue
                
                # Adjust cost based on problem size
                adjusted_cost = self._adjust_cost_for_size(edge.cost, n_vars)
                new_cost = current_cost + adjusted_cost
                
                if new_cost < distances[neighbor]:
                    distances[neighbor] = new_cost
                    previous[neighbor] = current_node
                    edge_map[neighbor] = edge
                    heapq.heappush(pq, (new_cost, neighbor))
        
        # No path found
        self.path_cache[cache_key] = None
        return None
    
    def get_conversion_options(self, source: str, 
                             max_cost: Optional[float] = None) -> Dict[str, ConversionPath]:
        """
        Get all possible conversion targets from a source representation.
        
        Args:
            source: Source representation name
            max_cost: Maximum acceptable cost (optional)
            
        Returns:
            Dictionary mapping target representations to optimal paths
        """
        options = {}
        all_targets = self._get_all_nodes() - {source}
        
        for target in all_targets:
            path = self.find_optimal_path(source, target)
            if path and (max_cost is None or path.total_cost.total_cost <= max_cost):
                options[target] = path
        
        return options
    
    def estimate_conversion_cost(self, source: str, target: str, 
                               n_vars: Optional[int] = None) -> Optional[ConversionCost]:
        """
        Estimate conversion cost without finding the full path.
        
        Args:
            source: Source representation name
            target: Target representation name
            n_vars: Number of variables
            
        Returns:
            Estimated conversion cost or None if no path exists
        """
        path = self.find_optimal_path(source, target, n_vars)
        return path.total_cost if path else None
    
    def _adjust_cost_for_size(self, base_cost: ConversionCost, 
                             n_vars: Optional[int]) -> ConversionCost:
        """Adjust conversion cost based on problem size."""
        if n_vars is None:
            return base_cost
        
        # Exponential scaling for time complexity
        size_factor = 2 ** min(n_vars, 10)  # Cap at 2^10 to prevent overflow
        time_scaling = np.log2(size_factor + 1)
        space_scaling = size_factor / 1024.0  # Normalize space scaling
        
        return ConversionCost(
            time_complexity=base_cost.time_complexity * time_scaling,
            space_complexity=base_cost.space_complexity * space_scaling,
            accuracy_loss=base_cost.accuracy_loss,
            is_exact=base_cost.is_exact
        )
    
    def _get_all_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        nodes = set()
        nodes.update(self.edges.keys())
        for edge_list in self.edges.values():
            nodes.update(edge.target for edge in edge_list)
        return nodes
    
    def _invalidate_cache_for_node(self, node: str):
        """Invalidate cached paths involving a specific node."""
        keys_to_remove = []
        for (source, target) in self.path_cache:
            if source == node or target == node:
                keys_to_remove.append((source, target))
        
        for key in keys_to_remove:
            del self.path_cache[key]
    
    def clear_cache(self):
        """Clear the path cache."""
        self.path_cache.clear()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversion graph."""
        all_nodes = self._get_all_nodes()
        total_edges = sum(len(edge_list) for edge_list in self.edges.values())
        
        # Calculate connectivity
        reachable_pairs = 0
        for source in all_nodes:
            for target in all_nodes:
                if source != target:
                    path = self.find_optimal_path(source, target)
                    if path:
                        reachable_pairs += 1
        
        total_pairs = len(all_nodes) * (len(all_nodes) - 1)
        connectivity = reachable_pairs / total_pairs if total_pairs > 0 else 0
        
        return {
            'num_nodes': len(all_nodes),
            'num_edges': total_edges,
            'connectivity': connectivity,
            'cached_paths': len(self.path_cache),
            'nodes': sorted(all_nodes)
        }
    
    def visualize_graph(self, output_format: str = 'text') -> str:
        """
        Create a text visualization of the conversion graph.
        
        Args:
            output_format: Output format ('text' or 'dot')
            
        Returns:
            String representation of the graph
        """
        if output_format == 'dot':
            return self._generate_dot_graph()
        else:
            return self._generate_text_graph()
    
    def _generate_text_graph(self) -> str:
        """Generate text representation of the graph."""
        lines = ["Conversion Graph:"]
        lines.append("=" * 50)
        
        all_nodes = sorted(self._get_all_nodes())
        
        for node in all_nodes:
            edges = self.edges.get(node, [])
            if edges:
                lines.append(f"\n{node}:")
                for edge in sorted(edges, key=lambda e: e.cost.total_cost):
                    lines.append(f"  -> {edge.target} (cost: {edge.cost.total_cost:.2f})")
            else:
                lines.append(f"\n{node}: (no outgoing edges)")
        
        return "\n".join(lines)
    
    def _generate_dot_graph(self) -> str:
        """Generate DOT format for graph visualization tools."""
        lines = ["digraph ConversionGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        
        # Add nodes
        for node in sorted(self._get_all_nodes()):
            lines.append(f'  "{node}";')
        
        # Add edges
        for source, edge_list in self.edges.items():
            for edge in edge_list:
                cost = edge.cost.total_cost
                color = "green" if edge.cost.is_exact else "orange"
                lines.append(
                    f'  "{source}" -> "{edge.target}" '
                    f'[label="{cost:.1f}", color={color}];'
                )
        
        lines.append("}")
        return "\n".join(lines)


# Global conversion graph instance
_conversion_graph = ConversionGraph()


def get_conversion_graph() -> ConversionGraph:
    """Get the global conversion graph instance."""
    return _conversion_graph


def find_conversion_path(source: str, target: str, 
                        n_vars: Optional[int] = None) -> Optional[ConversionPath]:
    """
    Find optimal conversion path between representations.
    
    Args:
        source: Source representation name
        target: Target representation name
        n_vars: Number of variables
        
    Returns:
        Optimal conversion path or None
    """
    return _conversion_graph.find_optimal_path(source, target, n_vars)


def register_custom_conversion(source: str, target: str, 
                             cost: ConversionCost,
                             converter: Callable):
    """
    Register a custom conversion between representations.
    
    Args:
        source: Source representation name
        target: Target representation name
        cost: Conversion cost
        converter: Conversion function
    """
    _conversion_graph.add_edge(source, target, cost, converter)


def get_conversion_options(source: str, max_cost: Optional[float] = None) -> Dict[str, ConversionPath]:
    """
    Get all conversion options from a source representation.
    
    Args:
        source: Source representation name
        max_cost: Maximum acceptable cost
        
    Returns:
        Dictionary of conversion options
    """
    return _conversion_graph.get_conversion_options(source, max_cost)


def estimate_conversion_cost(source: str, target: str, 
                           n_vars: Optional[int] = None) -> Optional[ConversionCost]:
    """
    Estimate the cost of converting between representations.
    
    Args:
        source: Source representation name
        target: Target representation name
        n_vars: Number of variables
        
    Returns:
        Estimated conversion cost
    """
    return _conversion_graph.estimate_conversion_cost(source, target, n_vars)
