"""
LaTeX/TikZ Export for Boolean Function Visualizations.

This module provides comprehensive LaTeX export for:
- Fourier spectrum diagrams
- Influence bar charts
- Decision trees
- Function family comparison tables
- Boolean cube visualizations
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

# Module logger
_logger = logging.getLogger("boofun.visualization.latex_export")

if TYPE_CHECKING:
    from ..core.base import BooleanFunction

__all__ = [
    "export_fourier_tikz",
    "export_influences_tikz",
    "export_cube_tikz",
    "export_comparison_table",
    "export_spectrum_table",
    "LaTeXExporter",
]


def export_fourier_tikz(
    f: "BooleanFunction",
    var_names: Optional[List[str]] = None,
    width: float = 10,
    height: float = 5,
    highlight_threshold: float = 0.1,
    show_labels: bool = True,
) -> str:
    """
    Export Fourier spectrum as TikZ bar chart.

    Args:
        f: Boolean function
        var_names: Variable names for subset labels
        width: Chart width in cm
        height: Chart height in cm
        highlight_threshold: Highlight coefficients above this
        show_labels: Show subset labels on x-axis

    Returns:
        TikZ code string
    """
    n = f.n_vars
    assert n is not None
    fourier = f.fourier()

    if var_names is None:
        var_names = [str(i) for i in range(n)]

    # Limit for reasonable LaTeX output
    max_show = min(len(fourier), 32)

    lines = [
        "\\begin{tikzpicture}",
        f"\\begin{{axis}}[",
        f"    width={width}cm,",
        f"    height={height}cm,",
        f"    ybar,",
        f"    bar width=0.5,",
        f"    xlabel={{Subset $S$}},",
        f"    ylabel={{$\\hat{{f}}(S)$}},",
        f"    ymin=-1, ymax=1,",
        f"    xtick=data,",
    ]

    if show_labels and n is not None and n <= 4:
        # Create subset labels
        labels = []
        for i in range(max_show):
            if i == 0:
                labels.append("$\\emptyset$")
            else:
                bits = [var_names[j] for j in range(n) if (i >> j) & 1]
                labels.append("$\\{" + ",".join(bits) + "\\}$")
        lines.append(f"    xticklabels={{{', '.join(labels)}}},")
        lines.append(f"    x tick label style={{rotate=45, anchor=east}},")
    else:
        lines.append(f"    xticklabels={{}},")

    lines.append("]")

    # Add bars
    coords = []
    for i in range(max_show):
        val = float(fourier[i])
        "blue" if abs(val) < highlight_threshold else "red"
        coords.append(f"({i}, {val:.4f})")

    lines.append(f"\\addplot[fill=blue!50] coordinates {{{' '.join(coords)}}};")

    # Add zero line
    lines.append(f"\\draw[dashed, gray] (axis cs:0,0) -- (axis cs:{max_show-1},0);")

    lines.append("\\end{axis}")
    lines.append("\\end{tikzpicture}")

    return "\n".join(lines)


def export_influences_tikz(
    f: "BooleanFunction",
    var_names: Optional[List[str]] = None,
    width: float = 8,
    height: float = 5,
    horizontal: bool = True,
) -> str:
    """
    Export influences as TikZ bar chart.

    Args:
        f: Boolean function
        var_names: Variable names
        width: Chart width in cm
        height: Chart height in cm
        horizontal: Use horizontal bars

    Returns:
        TikZ code string
    """
    n = f.n_vars
    assert n is not None
    influences = f.influences()

    if var_names is None:
        var_names = [f"$x_{{{i}}}$" for i in range(n)]

    bar_type = "xbar" if horizontal else "ybar"

    lines = [
        "\\begin{tikzpicture}",
        f"\\begin{{axis}}[",
        f"    width={width}cm,",
        f"    height={height}cm,",
        f"    {bar_type},",
        f"    bar width=0.4cm,",
    ]

    if horizontal:
        lines.extend(
            [
                f"    xlabel={{Influence $\\mathrm{{Inf}}_i[f]$}},",
                f"    symbolic y coords={{{', '.join(var_names)}}},",
                f"    ytick=data,",
                f"    xmin=0,",
            ]
        )
    else:
        lines.extend(
            [
                f"    ylabel={{Influence $\\mathrm{{Inf}}_i[f]$}},",
                f"    symbolic x coords={{{', '.join(var_names)}}},",
                f"    xtick=data,",
                f"    ymin=0,",
            ]
        )

    lines.append("]")

    # Add bars
    coords = []
    for i, inf in enumerate(influences):
        if horizontal:
            coords.append(f"({inf:.4f}, {var_names[i]})")
        else:
            coords.append(f"({var_names[i]}, {inf:.4f})")

    lines.append(f"\\addplot[fill=blue!60] coordinates {{{' '.join(coords)}}};")

    lines.append("\\end{axis}")
    lines.append("\\end{tikzpicture}")

    return "\n".join(lines)


def export_cube_tikz(n: int = 3, labels: bool = True) -> str:
    """
    Export Boolean cube visualization as TikZ.

    Args:
        n: Number of dimensions (2, 3, or 4)
        labels: Show vertex labels

    Returns:
        TikZ code string
    """
    if n not in [2, 3, 4]:
        raise ValueError("Cube visualization only supports n=2,3,4")

    lines = [
        "\\begin{tikzpicture}[",
        "    scale=2,",
        "    vertex/.style={circle, draw, fill=white, minimum size=5pt, inner sep=1pt},",
        "    edge/.style={draw, thick}",
        "]",
    ]

    positions: Dict[str, Tuple[float, float]] = {}
    edges: List[Tuple[str, str]] = []

    if n == 2:
        # Square
        positions = {
            "00": (0, 0),
            "01": (1, 0),
            "10": (0, 1),
            "11": (1, 1),
        }
        edges = [("00", "01"), ("00", "10"), ("01", "11"), ("10", "11")]  # noqa: E501
    elif n == 3:
        # Cube
        positions = {
            "000": (0, 0),
            "001": (1, 0),
            "010": (0, 1),
            "011": (1, 1),
            "100": (0.4, 0.4),
            "101": (1.4, 0.4),
            "110": (0.4, 1.4),
            "111": (1.4, 1.4),
        }
        edges = [
            ("000", "001"),
            ("000", "010"),
            ("000", "100"),
            ("001", "011"),
            ("001", "101"),
            ("010", "011"),
            ("010", "110"),
            ("011", "111"),
            ("100", "101"),
            ("100", "110"),
            ("101", "111"),
            ("110", "111"),
        ]
    else:  # n == 4
        # Tesseract projection
        inner_offset = 0.3
        for i in range(16):
            bits = format(i, "04b")
            x: float = int(bits[3]) + inner_offset * int(bits[1])
            y: float = int(bits[2]) + inner_offset * int(bits[0])
            positions[bits] = (x, y)

        for i in range(16):
            for b in range(4):
                j = i ^ (1 << b)
                if j > i:
                    edges.append((format(i, "04b"), format(j, "04b")))

    # Draw edges
    for v1, v2 in edges:
        x1, y1 = positions[v1]
        x2, y2 = positions[v2]
        lines.append(f"\\draw[edge] ({x1:.2f}, {y1:.2f}) -- ({x2:.2f}, {y2:.2f});")

    # Draw vertices
    for vertex, (x, y) in positions.items():
        label = vertex if labels else ""
        lines.append(f"\\node[vertex] at ({x:.2f}, {y:.2f}) {{{label}}};")

    lines.append("\\end{tikzpicture}")

    return "\n".join(lines)


def export_spectrum_table(
    f: "BooleanFunction", var_names: Optional[List[str]] = None, max_rows: int = 16
) -> str:
    """
    Export Fourier spectrum as LaTeX table.

    Args:
        f: Boolean function
        var_names: Variable names
        max_rows: Maximum rows to show

    Returns:
        LaTeX table string
    """
    n = f.n_vars
    assert n is not None
    fourier = f.fourier()

    if var_names is None:
        var_names = [str(i) for i in range(n)]

    lines = [
        "\\begin{tabular}{|c|c|c|c|}",
        "\\hline",
        "$S$ & Binary & $|S|$ & $\\hat{f}(S)$ \\\\",
        "\\hline",
    ]

    # Sort by magnitude
    indexed = [(i, fourier[i]) for i in range(len(fourier))]
    sorted_by_mag = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)[:max_rows]

    for idx, coef in sorted_by_mag:
        # Subset label
        if idx == 0:
            subset = "$\\emptyset$"
        else:
            bits = [var_names[j] for j in range(n) if (idx >> j) & 1]
            subset = "$\\{" + ", ".join(bits) + "\\}$"

        binary = format(idx, f"0{n}b")
        degree = bin(idx).count("1")

        lines.append(f"{subset} & {binary} & {degree} & {coef:.4f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def export_comparison_table(
    functions: Dict[str, "BooleanFunction"], properties: Optional[List[str]] = None
) -> str:
    """
    Export comparison table of multiple functions.

    Args:
        functions: Dict mapping names to BooleanFunction objects
        properties: List of properties to compare

    Returns:
        LaTeX table string
    """
    if properties is None:
        properties = ["n_vars", "total_influence", "max_influence", "variance", "degree"]

    prop_labels = {
        "n_vars": "$n$",
        "total_influence": "$I[f]$",
        "max_influence": "$\\max_i \\mathrm{Inf}_i$",
        "variance": "$\\mathrm{Var}[f]$",
        "degree": "$\\deg(f)$",
        "sensitivity": "$s(f)$",
        "noise_stability": "$\\mathrm{Stab}_{0.5}[f]$",
    }

    names = list(functions.keys())

    # Header
    header = " & ".join(["Property"] + names) + " \\\\"

    lines = [
        "\\begin{tabular}{|l|" + "c|" * len(names) + "}",
        "\\hline",
        header,
        "\\hline",
    ]

    for prop in properties:
        label = prop_labels.get(prop, prop)
        values = []

        for name, f in functions.items():
            try:
                val: Any
                if prop == "n_vars":
                    val = f.n_vars
                elif prop == "total_influence":
                    val = f.total_influence()
                elif prop == "max_influence":
                    val = f.max_influence()
                elif prop == "variance":
                    val = f.variance()
                elif prop == "degree":
                    val = f.degree()
                elif prop == "sensitivity":
                    val = f.sensitivity()
                elif prop == "noise_stability":
                    val = f.noise_stability(0.5)
                else:
                    val = getattr(f, prop)()

                if isinstance(val, float):
                    values.append(f"{val:.3f}")
                else:
                    values.append(str(val))
            except Exception as e:
                _logger.debug(f"Property '{prop}' computation failed: {e}")
                values.append("--")

        row = " & ".join([label] + values) + " \\\\"
        lines.append(row)

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


class LaTeXExporter:
    """
    Unified LaTeX exporter for Boolean function visualizations.
    """

    def __init__(self, f: "BooleanFunction", var_names: Optional[List[str]] = None):
        """
        Initialize exporter.

        Args:
            f: Boolean function
            var_names: Variable names
        """
        self.function = f
        n = f.n_vars
        assert n is not None
        self.var_names = var_names or [str(i) for i in range(n)]

    def fourier_spectrum(self, **kwargs) -> str:
        """Export Fourier spectrum as TikZ."""
        return export_fourier_tikz(self.function, self.var_names, **kwargs)

    def influences(self, **kwargs) -> str:
        """Export influences as TikZ."""
        return export_influences_tikz(self.function, self.var_names, **kwargs)

    def spectrum_table(self, **kwargs) -> str:
        """Export spectrum as LaTeX table."""
        return export_spectrum_table(self.function, self.var_names, **kwargs)

    def preamble(self) -> str:
        """Get required LaTeX preamble."""
        return """% Required packages for BooFun visualizations
\\usepackage{tikz}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}
\\usetikzlibrary{positioning, shapes, arrows}
"""

    def full_document(self, *contents: str) -> str:
        """
        Wrap content in a complete LaTeX document.

        Args:
            *contents: TikZ/table content strings

        Returns:
            Complete LaTeX document
        """
        body = "\n\n".join(contents)

        return f"""\\documentclass{{article}}
{self.preamble()}
\\begin{{document}}

{body}

\\end{{document}}
"""

    def save(self, filename: str, *contents: str, full_doc: bool = True):
        """
        Save LaTeX to file.

        Args:
            filename: Output filename
            *contents: Content strings
            full_doc: Wrap in complete document
        """
        if full_doc:
            output = self.full_document(*contents)
        else:
            output = "\n\n".join(contents)

        with open(filename, "w") as f:
            f.write(output)
