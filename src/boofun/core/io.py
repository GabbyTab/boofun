"""
File I/O for Boolean functions.

Supports multiple formats:
- JSON: Full representation with metadata
- .bf: Scott Aaronson's Boolean Function Wizard format
- DIMACS CNF: Standard SAT solver format

Usage:
    from boofun.core.io import load, save

    # Load from file (format auto-detected)
    f = load("function.json")
    f = load("function.bf")
    f = load("function.cnf")

    # Save to file
    save(f, "output.json")
    save(f, "output.bf")
    save(f, "output.cnf")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..utils.exceptions import BooleanFunctionError

if TYPE_CHECKING:
    from .base import BooleanFunction

__all__ = [
    "load",
    "save",
    "load_json",
    "save_json",
    "load_bf",
    "save_bf",
    "load_dimacs_cnf",
    "save_dimacs_cnf",
    "detect_format",
]


class FileIOError(BooleanFunctionError):
    """Error during file I/O operations."""

    def __init__(self, message: str, path: Optional[str] = None, **kwargs):
        self.path = path
        super().__init__(message, **kwargs)


def detect_format(path: Union[str, Path]) -> str:
    """
    Detect file format from extension or content.

    Args:
        path: Path to the file

    Returns:
        Format string: "json", "bf", or "dimacs_cnf"

    Raises:
        FileIOError: If format cannot be detected
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return "json"
    elif suffix == ".bf":
        return "bf"
    elif suffix in (".cnf", ".dimacs"):
        return "dimacs_cnf"

    # Try to detect from content
    if path.exists():
        with open(path, "r") as f:
            first_line = f.readline().strip()

        if first_line.startswith("{"):
            return "json"
        elif first_line.startswith("p cnf") or first_line.startswith("c "):
            return "dimacs_cnf"
        elif first_line.isdigit():
            return "bf"

    raise FileIOError(
        f"Cannot detect format for '{path}'",
        path=str(path),
        suggestion="Use explicit format or standard extension (.json, .bf, .cnf)",
    )


def load(
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> "BooleanFunction":
    """
    Load a Boolean function from file.

    Args:
        path: Path to the file
        format: Optional format override ("json", "bf", "dimacs_cnf")
        **kwargs: Format-specific options

    Returns:
        BooleanFunction instance

    Raises:
        FileIOError: If file cannot be loaded
        FileNotFoundError: If file does not exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format is None:
        format = detect_format(path)

    if format == "json":
        return load_json(path, **kwargs)
    elif format == "bf":
        return load_bf(path, **kwargs)
    elif format == "dimacs_cnf":
        return load_dimacs_cnf(path, **kwargs)
    else:
        raise FileIOError(
            f"Unknown format: '{format}'",
            path=str(path),
            suggestion="Supported formats: json, bf, dimacs_cnf",
        )


def save(
    func: "BooleanFunction",
    path: Union[str, Path],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a Boolean function to file.

    Args:
        func: BooleanFunction to save
        path: Destination path
        format: Optional format override ("json", "bf", "dimacs_cnf")
        **kwargs: Format-specific options

    Raises:
        FileIOError: If file cannot be saved
    """
    path = Path(path)

    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix == ".bf":
            format = "bf"
        elif suffix in (".cnf", ".dimacs"):
            format = "dimacs_cnf"
        else:
            format = "json"  # Default

    if format == "json":
        save_json(func, path, **kwargs)
    elif format == "bf":
        save_bf(func, path, **kwargs)
    elif format == "dimacs_cnf":
        save_dimacs_cnf(func, path, **kwargs)
    else:
        raise FileIOError(
            f"Unknown format: '{format}'",
            path=str(path),
            suggestion="Supported formats: json, bf, dimacs_cnf",
        )


# =============================================================================
# JSON Format
# =============================================================================


def load_json(path: Union[str, Path], **kwargs) -> "BooleanFunction":
    """
    Load Boolean function from JSON file.

    JSON format stores the full representation including metadata.

    Args:
        path: Path to JSON file

    Returns:
        BooleanFunction instance
    """
    from .base import BooleanFunction
    from .factory import BooleanFunctionFactory

    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    # Extract representation type and data
    rep_type = data.get("type", "truth_table")
    n_vars = data.get("n") or data.get("n_vars")

    if rep_type == "truth_table":
        values = data.get("values", data.get("table", []))
        return BooleanFunctionFactory.from_truth_table(BooleanFunction, values, n=n_vars)
    elif rep_type == "fourier_expansion":
        coeffs = np.array(data.get("coefficients", []))
        return BooleanFunctionFactory.from_multilinear(BooleanFunction, coeffs, n=n_vars)
    elif rep_type == "anf":
        # Reconstruct monomials
        monomials = {}
        for term in data.get("monomials", []):
            key = frozenset(term.get("variables", []))
            monomials[key] = term.get("coefficient", 1)
        return BooleanFunctionFactory.from_polynomial(BooleanFunction, monomials, n=n_vars)
    elif rep_type == "dnf":
        from .representations.dnf_form import DNFFormula

        dnf = DNFFormula.from_dict(data)
        return BooleanFunctionFactory.from_dnf(BooleanFunction, dnf, n=n_vars)
    elif rep_type == "cnf":
        from .representations.cnf_form import CNFFormula

        cnf = CNFFormula.from_dict(data)
        return BooleanFunctionFactory.from_cnf(BooleanFunction, cnf, n=n_vars)
    else:
        # Try truth table as fallback
        if "values" in data:
            return BooleanFunctionFactory.from_truth_table(
                BooleanFunction, data["values"], n=n_vars
            )
        raise FileIOError(
            f"Unknown representation type in JSON: '{rep_type}'",
            path=str(path),
        )


def save_json(
    func: "BooleanFunction",
    path: Union[str, Path],
    representation: str = "truth_table",
    pretty: bool = True,
    **kwargs,
) -> None:
    """
    Save Boolean function to JSON file.

    Args:
        func: BooleanFunction to save
        path: Destination path
        representation: Which representation to save
        pretty: Whether to format with indentation
    """
    from .representations.registry import get_strategy

    path = Path(path)

    # Get the representation data
    if func.has_rep(representation):
        data = func.representations[representation]
    else:
        # Convert to requested representation
        data = func.get_representation(representation)

    # Convert to numpy array if it's a list (truth table often stored as list)
    if isinstance(data, list):
        data = np.array(data)

    # Use the strategy's dump method
    strategy = get_strategy(representation)
    export_data = strategy.dump(data, space=func.space, n_vars=func.n_vars)

    # Add metadata
    export_data["type"] = representation
    export_data["n_vars"] = func.n_vars

    with open(path, "w") as f:
        if pretty:
            json.dump(export_data, f, indent=2, default=_json_serializer)
        else:
            json.dump(export_data, f, default=_json_serializer)


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# .bf Format (Scott Aaronson's Boolean Function Wizard)
# =============================================================================


def load_bf(path: Union[str, Path], **kwargs) -> "BooleanFunction":
    """
    Load Boolean function from .bf file (Aaronson format).

    Format:
        n           (number of variables)
        00...0 v₀   (optional input string, then output value)
        00...1 v₁
        ...
        11...1 v_{2^n-1}

    Values can be 0, 1, or -1 (partial function, undefined).

    Args:
        path: Path to .bf file

    Returns:
        BooleanFunction instance
    """
    from .base import BooleanFunction
    from .factory import BooleanFunctionFactory

    path = Path(path)

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise FileIOError("Empty .bf file", path=str(path))

    # First line is number of variables
    try:
        n_vars = int(lines[0])
    except ValueError:
        raise FileIOError(
            f"First line must be number of variables, got: '{lines[0]}'",
            path=str(path),
        )

    size = 1 << n_vars
    truth_table = np.zeros(size, dtype=bool)
    known_mask = np.ones(size, dtype=bool)  # Track which values are known

    for i, line in enumerate(lines[1:]):
        if i >= size:
            break

        parts = line.split()

        if len(parts) == 1:
            # Just the output value
            value = int(parts[0])
            idx = i
        elif len(parts) == 2:
            # Input string and output value
            input_str, value_str = parts
            value = int(value_str)
            # Parse input string to index
            idx = int(input_str, 2)
        else:
            raise FileIOError(
                f"Invalid line format at line {i + 2}: '{line}'",
                path=str(path),
            )

        if value == -1:
            # Undefined (partial function)
            known_mask[idx] = False
        else:
            truth_table[idx] = bool(value)

    # Create the function
    func = BooleanFunctionFactory.from_truth_table(BooleanFunction, truth_table, n=n_vars)

    # If partial, store the mask in metadata
    if not np.all(known_mask):
        func._metadata["partial"] = True
        func._metadata["known_mask"] = known_mask

    return func


def save_bf(
    func: "BooleanFunction",
    path: Union[str, Path],
    include_inputs: bool = True,
    **kwargs,
) -> None:
    """
    Save Boolean function to .bf file (Aaronson format).

    Args:
        func: BooleanFunction to save
        path: Destination path
        include_inputs: Whether to include input bit strings
    """
    path = Path(path)
    n_vars = func.n_vars

    if n_vars is None:
        raise FileIOError("Cannot save function without defined n_vars")

    truth_table = np.asarray(func.get_representation("truth_table"), dtype=int)

    with open(path, "w") as f:
        f.write(f"{n_vars}\n")

        for i in range(len(truth_table)):
            if include_inputs:
                input_str = format(i, f"0{n_vars}b")
                f.write(f"{input_str} {truth_table[i]}\n")
            else:
                f.write(f"{truth_table[i]}\n")


# =============================================================================
# DIMACS CNF Format
# =============================================================================


def load_dimacs_cnf(path: Union[str, Path], **kwargs) -> "BooleanFunction":
    """
    Load Boolean function from DIMACS CNF file.

    DIMACS CNF is the standard format for SAT solvers:
        c comment line
        p cnf <n_vars> <n_clauses>
        1 -2 3 0     (clause: x1 OR NOT x2 OR x3)
        -1 2 0       (clause: NOT x1 OR x2)
        ...

    Positive integers are positive literals, negative are negated.
    Each clause ends with 0.

    Args:
        path: Path to .cnf file

    Returns:
        BooleanFunction with CNF representation
    """
    from .base import BooleanFunction
    from .factory import BooleanFunctionFactory
    from .representations.cnf_form import CNFClause, CNFFormula

    path = Path(path)

    clauses = []
    n_vars = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("c"):
                # Comment or empty line
                continue

            if line.startswith("p cnf"):
                # Problem line
                parts = line.split()
                if len(parts) >= 4:
                    n_vars = int(parts[2])
                    # n_clauses is parts[3] but we don't validate against it
                continue

            if line.startswith("%") or line.startswith("0"):
                # End of file marker
                break

            # Parse clause
            literals = list(map(int, line.split()))

            # Remove trailing 0
            if literals and literals[-1] == 0:
                literals = literals[:-1]

            if not literals:
                continue

            positive_vars = set()
            negative_vars = set()

            for lit in literals:
                if lit > 0:
                    positive_vars.add(lit - 1)  # Convert to 0-indexed
                elif lit < 0:
                    negative_vars.add(-lit - 1)  # Convert to 0-indexed

            clauses.append(CNFClause(positive_vars, negative_vars))

    if n_vars is None:
        # Infer from clauses
        all_vars = set()
        for clause in clauses:
            all_vars.update(clause.get_variables())
        n_vars = max(all_vars) + 1 if all_vars else 0

    cnf = CNFFormula(clauses, n_vars)
    return BooleanFunctionFactory.from_cnf(BooleanFunction, cnf, n=n_vars)


def save_dimacs_cnf(
    func: "BooleanFunction",
    path: Union[str, Path],
    comment: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save Boolean function to DIMACS CNF file.

    Note: Converts function to CNF if not already in that representation.

    Args:
        func: BooleanFunction to save
        path: Destination path
        comment: Optional comment to include
    """
    from .representations.cnf_form import CNFFormula

    path = Path(path)

    # Get or convert to CNF
    if func.has_rep("cnf"):
        cnf = func.representations["cnf"]
    else:
        cnf = func.get_representation("cnf")

    if not isinstance(cnf, CNFFormula):
        raise FileIOError(
            "Could not convert function to CNF format",
            path=str(path),
            suggestion="Function may not be representable as CNF",
        )

    with open(path, "w") as f:
        # Comment
        if comment:
            for line in comment.split("\n"):
                f.write(f"c {line}\n")
        f.write("c Generated by BooFun\n")

        # Problem line
        f.write(f"p cnf {cnf.n_vars} {len(cnf.clauses)}\n")

        # Clauses
        for clause in cnf.clauses:
            literals = []
            for var in sorted(clause.positive_vars):
                literals.append(var + 1)  # Convert to 1-indexed
            for var in sorted(clause.negative_vars):
                literals.append(-(var + 1))  # Convert to 1-indexed, negated

            f.write(" ".join(map(str, literals)) + " 0\n")
