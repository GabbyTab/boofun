"""
Decision tree visualization for Boolean functions.

This module provides tools to visualize decision trees that compute
Boolean functions, including optimal trees and user-specified trees.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "DecisionTreeNode",
    "build_optimal_decision_tree",
    "plot_decision_tree",
    "decision_tree_to_dict",
]


class DecisionTreeNode:
    """
    Node in a decision tree for a Boolean function.
    
    Can be either:
    - Internal node: queries a variable, has left (0) and right (1) children
    - Leaf node: outputs a constant value (0 or 1)
    """
    
    def __init__(
        self,
        is_leaf: bool = False,
        value: Optional[int] = None,
        variable: Optional[int] = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        depth: int = 0,
    ):
        """
        Create a decision tree node.
        
        Args:
            is_leaf: True if this is a leaf node
            value: Output value for leaf nodes (0 or 1)
            variable: Variable index to query for internal nodes
            left: Child for variable=0
            right: Child for variable=1
            depth: Depth of this node in the tree
        """
        self.is_leaf = is_leaf
        self.value = value
        self.variable = variable
        self.left = left
        self.right = right
        self.depth = depth
    
    def evaluate(self, x: int) -> int:
        """Evaluate the decision tree on input x."""
        if self.is_leaf:
            return self.value
        
        bit = (x >> self.variable) & 1
        if bit == 0:
            return self.left.evaluate(x)
        else:
            return self.right.evaluate(x)
    
    def __repr__(self) -> str:
        if self.is_leaf:
            return f"Leaf({self.value})"
        return f"Query(x_{self.variable})"


def build_optimal_decision_tree(
    f: "BooleanFunction",
    available_vars: Optional[List[int]] = None,
    inputs: Optional[List[int]] = None,
    depth: int = 0,
    max_depth: int = 20,
) -> DecisionTreeNode:
    """
    Build an optimal (or near-optimal) decision tree for f.
    
    Uses a greedy algorithm based on influence to select variables.
    
    Args:
        f: BooleanFunction to build tree for
        available_vars: Variables not yet queried (None = all)
        inputs: Current set of inputs to distinguish (None = all)
        depth: Current depth in the tree
        max_depth: Maximum allowed depth
        
    Returns:
        Root of the decision tree
    """
    n = f.n_vars
    
    if available_vars is None:
        available_vars = list(range(n))
    
    if inputs is None:
        inputs = list(range(2**n))
    
    # Get function values for current inputs
    values = [f.evaluate(x) for x in inputs]
    
    # Check if all values are the same (leaf node)
    if len(set(values)) == 1:
        return DecisionTreeNode(is_leaf=True, value=values[0], depth=depth)
    
    # Check max depth
    if depth >= max_depth or not available_vars:
        # Return most common value
        most_common = 1 if sum(values) > len(values) // 2 else 0
        return DecisionTreeNode(is_leaf=True, value=most_common, depth=depth)
    
    # Choose best variable to split on (maximizes information gain / influence)
    best_var = None
    best_score = -1
    
    for var in available_vars:
        # Split inputs by variable value
        left_inputs = [x for x in inputs if not ((x >> var) & 1)]
        right_inputs = [x for x in inputs if (x >> var) & 1]
        
        # Score: prefer balanced splits with homogeneous children
        left_vals = [f.evaluate(x) for x in left_inputs] if left_inputs else []
        right_vals = [f.evaluate(x) for x in right_inputs] if right_inputs else []
        
        # Entropy reduction
        def entropy(vals):
            if not vals:
                return 0
            p1 = sum(vals) / len(vals)
            if p1 == 0 or p1 == 1:
                return 0
            return -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
        
        before = entropy(values)
        after = 0
        if left_inputs:
            after += len(left_inputs) / len(inputs) * entropy(left_vals)
        if right_inputs:
            after += len(right_inputs) / len(inputs) * entropy(right_vals)
        
        info_gain = before - after
        
        if info_gain > best_score:
            best_score = info_gain
            best_var = var
    
    if best_var is None:
        best_var = available_vars[0]
    
    # Split and recurse
    left_inputs = [x for x in inputs if not ((x >> best_var) & 1)]
    right_inputs = [x for x in inputs if (x >> best_var) & 1]
    remaining_vars = [v for v in available_vars if v != best_var]
    
    left_child = build_optimal_decision_tree(
        f, remaining_vars, left_inputs, depth + 1, max_depth
    )
    right_child = build_optimal_decision_tree(
        f, remaining_vars, right_inputs, depth + 1, max_depth
    )
    
    return DecisionTreeNode(
        is_leaf=False,
        variable=best_var,
        left=left_child,
        right=right_child,
        depth=depth,
    )


def decision_tree_to_dict(node: DecisionTreeNode) -> Dict[str, Any]:
    """Convert decision tree to dictionary for serialization."""
    if node.is_leaf:
        return {"type": "leaf", "value": node.value}
    
    return {
        "type": "internal",
        "variable": node.variable,
        "left": decision_tree_to_dict(node.left),
        "right": decision_tree_to_dict(node.right),
    }


def _count_nodes(node: DecisionTreeNode) -> int:
    """Count total nodes in tree."""
    if node.is_leaf:
        return 1
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


def _tree_depth(node: DecisionTreeNode) -> int:
    """Get maximum depth of tree."""
    if node.is_leaf:
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


def plot_decision_tree(
    f: "BooleanFunction",
    tree: Optional[DecisionTreeNode] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    node_size: int = 2000,
    font_size: int = 10,
) -> Any:
    """
    Plot a decision tree for a Boolean function.
    
    Args:
        f: BooleanFunction (used to build tree if not provided)
        tree: Pre-built decision tree (None to build optimal)
        figsize: Figure size
        save_path: Path to save the plot
        show: Whether to display
        node_size: Size of tree nodes
        font_size: Font size for labels
        
    Returns:
        Figure object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        raise ImportError("Matplotlib required for decision tree visualization")
    
    # Build tree if not provided
    if tree is None:
        tree = build_optimal_decision_tree(f)
    
    # Get tree statistics
    depth = _tree_depth(tree)
    num_nodes = _count_nodes(tree)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position nodes using BFS
    positions = {}
    
    def assign_positions(node, x, y, x_offset, level=0):
        """Recursively assign positions to nodes."""
        positions[id(node)] = (x, y)
        
        if not node.is_leaf:
            new_offset = x_offset / 2
            assign_positions(node.left, x - x_offset, y - 1, new_offset, level + 1)
            assign_positions(node.right, x + x_offset, y - 1, new_offset, level + 1)
    
    # Start with root at top center
    initial_offset = 2 ** (depth - 1) if depth > 1 else 1
    assign_positions(tree, 0, 0, initial_offset)
    
    # Draw edges
    def draw_edges(node):
        if node.is_leaf:
            return
        
        pos = positions[id(node)]
        left_pos = positions[id(node.left)]
        right_pos = positions[id(node.right)]
        
        # Edge to left child (x=0)
        ax.plot([pos[0], left_pos[0]], [pos[1], left_pos[1]], 
               'b-', linewidth=1.5, alpha=0.7)
        # Label
        mid_x = (pos[0] + left_pos[0]) / 2
        mid_y = (pos[1] + left_pos[1]) / 2
        ax.text(mid_x - 0.2, mid_y, '0', fontsize=font_size-2, color='blue')
        
        # Edge to right child (x=1)
        ax.plot([pos[0], right_pos[0]], [pos[1], right_pos[1]], 
               'r-', linewidth=1.5, alpha=0.7)
        mid_x = (pos[0] + right_pos[0]) / 2
        mid_y = (pos[1] + right_pos[1]) / 2
        ax.text(mid_x + 0.1, mid_y, '1', fontsize=font_size-2, color='red')
        
        draw_edges(node.left)
        draw_edges(node.right)
    
    draw_edges(tree)
    
    # Draw nodes
    def draw_nodes(node):
        pos = positions[id(node)]
        
        if node.is_leaf:
            # Leaf: circle with output value
            color = 'lightgreen' if node.value else 'lightcoral'
            circle = plt.Circle(pos, 0.3, color=color, ec='black', linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(node.value), 
                   ha='center', va='center', fontsize=font_size, fontweight='bold', zorder=6)
        else:
            # Internal: rectangle with variable
            rect = mpatches.FancyBboxPatch(
                (pos[0] - 0.4, pos[1] - 0.25), 0.8, 0.5,
                boxstyle="round,pad=0.05", facecolor='lightyellow',
                edgecolor='black', linewidth=2, zorder=5
            )
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], f'x_{node.variable}', 
                   ha='center', va='center', fontsize=font_size, fontweight='bold', zorder=6)
        
        if not node.is_leaf:
            draw_nodes(node.left)
            draw_nodes(node.right)
    
    draw_nodes(tree)
    
    # Set axis properties
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title and legend
    ax.set_title(f'Decision Tree (depth={depth}, nodes={num_nodes})', fontsize=14)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Query variable'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Output 1'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='Output 0'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig
