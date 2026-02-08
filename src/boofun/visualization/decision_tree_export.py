"""
Decision Tree Export Utilities.

This module provides utilities for exporting Boolean function decision trees
to various formats including:
- Text/ASCII representation
- Graphviz DOT format
- JSON structure
- LaTeX/TikZ code
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "export_decision_tree_text",
    "export_decision_tree_dot",
    "export_decision_tree_json",
    "export_decision_tree_tikz",
    "DecisionTreeExporter",
]


class DecisionTreeNode:
    """Represents a node in a decision tree."""

    def __init__(
        self,
        variable: Optional[int] = None,
        value: Optional[bool] = None,
        low: Optional["DecisionTreeNode"] = None,
        high: Optional["DecisionTreeNode"] = None,
        depth: int = 0,
    ):
        self.variable = variable  # None for leaf nodes
        self.value = value  # Boolean value for leaf nodes
        self.low = low  # Child for variable = 0
        self.high = high  # Child for variable = 1
        self.depth = depth
        self.is_leaf = variable is None

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.value})"
        return f"Node(x_{self.variable})"


def build_decision_tree(f: "BooleanFunction", max_depth: Optional[int] = None) -> DecisionTreeNode:
    """
    Build a decision tree for a Boolean function.

    Uses Shannon expansion to build the tree.

    Args:
        f: Boolean function
        max_depth: Maximum depth (default: n_vars)

    Returns:
        Root node of decision tree
    """
    n = f.n_vars
    assert n is not None
    if max_depth is None:
        max_depth = n

    def build_subtree(assignment: Dict[int, int], depth: int) -> DecisionTreeNode:
        # Check if we can determine the value
        assert max_depth is not None
        if depth >= max_depth or len(assignment) == n:
            # Evaluate with current assignment
            x = sum(v << i for i, v in assignment.items())
            # Fill in remaining bits with 0
            val = f.evaluate(x)
            return DecisionTreeNode(value=bool(val), depth=depth)

        # Check if function is constant on remaining variables
        remaining_vars = [i for i in range(n) if i not in assignment]
        if not remaining_vars:
            x = sum(v << i for i, v in assignment.items())
            val = f.evaluate(x)
            return DecisionTreeNode(value=bool(val), depth=depth)

        # Try to find a variable to split on
        # Use next available variable
        var = remaining_vars[0]

        # Build children
        low_assign = assignment.copy()
        low_assign[var] = 0
        low_child = build_subtree(low_assign, depth + 1)

        high_assign = assignment.copy()
        high_assign[var] = 1
        high_child = build_subtree(high_assign, depth + 1)

        # Check if we can collapse
        if low_child.is_leaf and high_child.is_leaf and low_child.value == high_child.value:
            return DecisionTreeNode(value=low_child.value, depth=depth)

        return DecisionTreeNode(variable=var, low=low_child, high=high_child, depth=depth)

    return build_subtree({}, 0)


def export_decision_tree_text(
    f: "BooleanFunction", var_names: Optional[List[str]] = None, max_depth: Optional[int] = None
) -> str:
    """
    Export decision tree as ASCII text.

    Args:
        f: Boolean function
        var_names: Variable names (default: x_0, x_1, ...)
        max_depth: Maximum depth

    Returns:
        ASCII representation
    """
    n = f.n_vars
    assert n is not None
    if var_names is None:
        var_names = [f"x_{i}" for i in range(n)]

    tree = build_decision_tree(f, max_depth)

    lines: List[str] = []

    def print_tree(
        node: DecisionTreeNode, prefix: str = "", is_left: bool = True, edge_label: str = ""
    ):
        pass  # node is always DecisionTreeNode

        connector = "├── " if is_left else "└── "

        if node.depth == 0:
            if node.is_leaf:
                lines.append(f"[{1 if node.value else 0}]")
            else:
                assert node.variable is not None
                assert node.low is not None
                assert node.high is not None
                lines.append(f"{var_names[node.variable]}?")
                print_tree(node.low, "", True, "=0")
                print_tree(node.high, "", False, "=1")
        else:
            if node.is_leaf:
                lines.append(f"{prefix}{connector}{edge_label} → [{1 if node.value else 0}]")
            else:
                assert node.variable is not None
                assert node.low is not None
                assert node.high is not None
                lines.append(f"{prefix}{connector}{edge_label} → {var_names[node.variable]}?")
                new_prefix = prefix + ("│   " if is_left else "    ")
                print_tree(node.low, new_prefix, True, "=0")
                print_tree(node.high, new_prefix, False, "=1")

    print_tree(tree)
    return "\n".join(lines)


def export_decision_tree_dot(
    f: "BooleanFunction",
    var_names: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
    graph_name: str = "DecisionTree",
) -> str:
    """
    Export decision tree as Graphviz DOT format.

    Args:
        f: Boolean function
        var_names: Variable names
        max_depth: Maximum depth
        graph_name: Name for the graph

    Returns:
        DOT format string
    """
    n = f.n_vars
    assert n is not None
    if var_names is None:
        var_names = [f"x_{i}" for i in range(n)]

    tree = build_decision_tree(f, max_depth)

    lines = [f"digraph {graph_name} {{", "    rankdir=TB;", "    node [shape=ellipse];", ""]

    node_id = [0]  # Mutable counter

    def add_node(
        node: DecisionTreeNode, parent_id: Optional[int] = None, edge_label: str = ""
    ) -> int:
        current_id = node_id[0]
        node_id[0] += 1

        if node.is_leaf:
            label = "1" if node.value else "0"
            shape = "box"
            color = "lightgreen" if node.value else "lightcoral"
            lines.append(
                f'    node{current_id} [label="{label}", shape={shape}, style=filled, fillcolor="{color}"];'
            )
        else:
            assert node.variable is not None
            label = var_names[node.variable]
            lines.append(f'    node{current_id} [label="{label}"];')

        if parent_id is not None:
            lines.append(f'    node{parent_id} -> node{current_id} [label="{edge_label}"];')

        if not node.is_leaf:
            assert node.low is not None
            assert node.high is not None
            add_node(node.low, current_id, "0")
            add_node(node.high, current_id, "1")

        return current_id

    add_node(tree)
    lines.append("}")

    return "\n".join(lines)


def export_decision_tree_json(
    f: "BooleanFunction", var_names: Optional[List[str]] = None, max_depth: Optional[int] = None
) -> Dict[str, Any]:
    """
    Export decision tree as JSON structure.

    Args:
        f: Boolean function
        var_names: Variable names
        max_depth: Maximum depth

    Returns:
        JSON-serializable dictionary
    """
    n = f.n_vars
    assert n is not None
    if var_names is None:
        var_names = [f"x_{i}" for i in range(n)]

    tree = build_decision_tree(f, max_depth)

    def node_to_dict(node: DecisionTreeNode) -> Dict[str, Any]:
        if node.is_leaf:
            return {"type": "leaf", "value": node.value, "depth": node.depth}
        else:
            assert node.variable is not None
            assert node.low is not None
            assert node.high is not None
            return {
                "type": "internal",
                "variable": node.variable,
                "variable_name": var_names[node.variable],
                "depth": node.depth,
                "low": node_to_dict(node.low),
                "high": node_to_dict(node.high),
            }

    return {"function": {"n_vars": n, "variable_names": var_names}, "tree": node_to_dict(tree)}


def export_decision_tree_tikz(
    f: "BooleanFunction", var_names: Optional[List[str]] = None, max_depth: Optional[int] = None
) -> str:
    """
    Export decision tree as LaTeX TikZ code.

    Args:
        f: Boolean function
        var_names: Variable names
        max_depth: Maximum depth

    Returns:
        TikZ code string
    """
    n = f.n_vars
    assert n is not None
    if var_names is None:
        var_names = [f"$x_{{{i}}}$" for i in range(n)]

    tree = build_decision_tree(f, max_depth)

    lines = [
        "\\begin{tikzpicture}[",
        "    level distance=1.5cm,",
        "    sibling distance=3cm,",
        "    every node/.style={circle, draw, minimum size=0.8cm},",
        "    leaf/.style={rectangle, draw, minimum size=0.6cm},",
        "    edge from parent/.style={draw, -latex}",
        "]",
        "",
    ]

    def node_to_tikz(node: DecisionTreeNode, indent: str = "    ") -> str:
        if node.is_leaf:
            val = "1" if node.value else "0"
            return f"node[leaf] {{{val}}}"
        else:
            assert node.variable is not None
            assert node.low is not None
            assert node.high is not None
            label = var_names[node.variable]
            low_tikz = node_to_tikz(node.low, indent + "    ")
            high_tikz = node_to_tikz(node.high, indent + "    ")

            return (
                f"node {{{label}}}\n"
                f"{indent}child {{ {low_tikz} edge from parent node[left] {{0}} }}\n"
                f"{indent}child {{ {high_tikz} edge from parent node[right] {{1}} }}"
            )

    lines.append(f"\\{node_to_tikz(tree)};")
    lines.append("\\end{tikzpicture}")

    return "\n".join(lines)


class DecisionTreeExporter:
    """
    Class for exporting decision trees in multiple formats.

    Provides a unified interface for exporting Boolean function
    decision trees to various formats.
    """

    def __init__(
        self,
        f: "BooleanFunction",
        var_names: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ):
        """
        Initialize exporter.

        Args:
            f: Boolean function
            var_names: Variable names
            max_depth: Maximum depth for tree
        """
        self.function = f
        n = f.n_vars
        assert n is not None
        self.var_names = var_names or [f"x_{i}" for i in range(n)]
        self.max_depth = max_depth
        self._tree: Optional[DecisionTreeNode] = None

    @property
    def tree(self) -> DecisionTreeNode:
        """Get the decision tree (built lazily)."""
        if self._tree is None:
            self._tree = build_decision_tree(self.function, self.max_depth)
        return self._tree

    def to_text(self) -> str:
        """Export to ASCII text."""
        return export_decision_tree_text(self.function, self.var_names, self.max_depth)

    def to_dot(self, graph_name: str = "DecisionTree") -> str:
        """Export to Graphviz DOT format."""
        return export_decision_tree_dot(self.function, self.var_names, self.max_depth, graph_name)

    def to_json(self) -> Dict[str, Any]:
        """Export to JSON structure."""
        return export_decision_tree_json(self.function, self.var_names, self.max_depth)

    def to_tikz(self) -> str:
        """Export to LaTeX TikZ."""
        return export_decision_tree_tikz(self.function, self.var_names, self.max_depth)

    def save_dot(self, filename: str) -> None:
        """Save DOT to file."""
        with open(filename, "w") as f:
            f.write(self.to_dot())

    def depth(self) -> int:
        """Get tree depth."""

        def get_depth(node: DecisionTreeNode) -> int:
            if node.is_leaf:
                return 0
            assert node.low is not None
            assert node.high is not None
            return 1 + max(get_depth(node.low), get_depth(node.high))

        return get_depth(self.tree)

    def size(self) -> int:
        """Get number of nodes."""

        def count_nodes(node: DecisionTreeNode) -> int:
            if node.is_leaf:
                return 1
            assert node.low is not None
            assert node.high is not None
            return 1 + count_nodes(node.low) + count_nodes(node.high)

        return count_nodes(self.tree)

    def summary(self) -> str:
        """Get tree summary."""
        return (
            f"Decision Tree for {self.function.n_vars}-variable function\n"
            f"  Depth: {self.depth()}\n"
            f"  Nodes: {self.size()}\n"
            f"  Variables: {', '.join(self.var_names)}"
        )
