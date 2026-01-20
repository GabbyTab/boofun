"""
Comprehensive tests for file I/O operations.

Tests cover:
- JSON format (load/save)
- .bf format (Aaronson's Boolean Function Wizard)
- DIMACS CNF format
- Format detection
- Error handling
- Round-trip consistency
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import boofun as bf
from boofun.core.io import (
    FileIOError,
    detect_format,
    load,
    load_bf,
    load_dimacs_cnf,
    load_json,
    save,
    save_bf,
    save_dimacs_cnf,
    save_json,
)


class TestFormatDetection:
    """Tests for automatic format detection."""

    def test_detect_json_by_extension(self, tmp_path):
        """Detect JSON format from .json extension."""
        path = tmp_path / "test.json"
        path.write_text("{}")
        assert detect_format(path) == "json"

    def test_detect_bf_by_extension(self, tmp_path):
        """Detect .bf format from extension."""
        path = tmp_path / "test.bf"
        path.write_text("2\n0\n1\n1\n0")
        assert detect_format(path) == "bf"

    def test_detect_cnf_by_extension(self, tmp_path):
        """Detect DIMACS CNF format from .cnf extension."""
        path = tmp_path / "test.cnf"
        path.write_text("p cnf 2 1\n1 2 0")
        assert detect_format(path) == "dimacs_cnf"

    def test_detect_dimacs_by_extension(self, tmp_path):
        """Detect DIMACS format from .dimacs extension."""
        path = tmp_path / "test.dimacs"
        path.write_text("p cnf 2 1\n1 2 0")
        assert detect_format(path) == "dimacs_cnf"

    def test_detect_json_by_content(self, tmp_path):
        """Detect JSON format from content (starts with {)."""
        path = tmp_path / "test.txt"
        path.write_text('{"type": "truth_table"}')
        assert detect_format(path) == "json"

    def test_detect_cnf_by_content(self, tmp_path):
        """Detect DIMACS CNF from content (starts with p cnf or c)."""
        path = tmp_path / "test.txt"
        path.write_text("c comment\np cnf 2 1\n1 2 0")
        assert detect_format(path) == "dimacs_cnf"

    def test_detect_bf_by_content(self, tmp_path):
        """Detect .bf format from content (starts with digit)."""
        path = tmp_path / "test.txt"
        path.write_text("2\n0\n1\n1\n0")
        assert detect_format(path) == "bf"

    def test_detect_unknown_raises(self, tmp_path):
        """Unknown format raises FileIOError."""
        path = tmp_path / "test.xyz"
        # File doesn't exist, extension unknown
        with pytest.raises(FileIOError):
            detect_format(path)


class TestJSONFormat:
    """Tests for JSON file format."""

    def test_save_load_truth_table(self, tmp_path):
        """Round-trip for truth table representation."""
        f = bf.create([0, 1, 1, 0])  # XOR
        path = tmp_path / "xor.json"
        
        save_json(f, path)
        loaded = load_json(path)
        
        assert loaded.n_vars == 2
        for i in range(4):
            assert loaded.evaluate(i) == f.evaluate(i)

    def test_save_load_majority(self, tmp_path):
        """Round-trip for majority function."""
        f = bf.majority(3)
        path = tmp_path / "maj3.json"
        
        save(f, path)
        loaded = load(path)
        
        assert loaded.n_vars == 3
        for i in range(8):
            assert loaded.evaluate(i) == f.evaluate(i)

    def test_save_pretty_format(self, tmp_path):
        """Test pretty-printed JSON output."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "pretty.json"
        
        save_json(f, path, pretty=True)
        
        content = path.read_text()
        # Pretty format should have newlines
        assert "\n" in content
        assert "  " in content  # Indentation

    def test_save_compact_format(self, tmp_path):
        """Test compact JSON output."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "compact.json"
        
        save_json(f, path, pretty=False)
        
        content = path.read_text()
        # Compact format is single line
        lines = content.strip().split("\n")
        assert len(lines) == 1

    def test_load_with_metadata(self, tmp_path):
        """Load JSON with explicit metadata."""
        data = {
            "type": "truth_table",
            "n": 2,
            "values": [False, True, True, False],
        }
        path = tmp_path / "with_meta.json"
        path.write_text(json.dumps(data))
        
        loaded = load_json(path)
        assert loaded.n_vars == 2

    def test_load_minimal_json(self, tmp_path):
        """Load JSON with minimal data (values only)."""
        data = {
            "type": "truth_table",
            "values": [0, 1, 1, 0],
        }
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps(data))
        
        loaded = load_json(path)
        assert loaded.n_vars == 2

    def test_json_preserves_function_values(self, tmp_path):
        """Verify all function values are preserved."""
        # Create a random function
        f = bf.random(4, seed=42)
        path = tmp_path / "random.json"
        
        save(f, path)
        loaded = load(path)
        
        # Check all 16 values
        for i in range(16):
            assert loaded.evaluate(i) == f.evaluate(i), f"Mismatch at input {i}"


class TestBFFormat:
    """Tests for .bf format (Aaronson's Boolean Function Wizard)."""

    def test_save_load_xor(self, tmp_path):
        """Round-trip for XOR function."""
        f = bf.parity(2)
        path = tmp_path / "xor.bf"
        
        save_bf(f, path)
        loaded = load_bf(path)
        
        assert loaded.n_vars == 2
        for i in range(4):
            assert loaded.evaluate(i) == f.evaluate(i)

    def test_save_with_inputs(self, tmp_path):
        """Save with input bit strings."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "with_inputs.bf"
        
        save_bf(f, path, include_inputs=True)
        
        content = path.read_text()
        assert "00 0" in content or "00 1" in content
        assert "01" in content
        assert "10" in content
        assert "11" in content

    def test_save_without_inputs(self, tmp_path):
        """Save without input bit strings (compact)."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "no_inputs.bf"
        
        save_bf(f, path, include_inputs=False)
        
        content = path.read_text()
        lines = [l for l in content.strip().split("\n") if l]
        # First line is n_vars, rest are values only
        assert lines[0] == "2"
        assert len(lines) == 5  # n_vars + 4 values

    def test_load_bf_format_with_inputs(self, tmp_path):
        """Load .bf file with explicit input strings."""
        content = """2
00 0
01 1
10 1
11 0"""
        path = tmp_path / "xor.bf"
        path.write_text(content)
        
        loaded = load_bf(path)
        
        assert loaded.n_vars == 2
        assert loaded.evaluate(0) == False  # 00
        assert loaded.evaluate(1) == True   # 01
        assert loaded.evaluate(2) == True   # 10
        assert loaded.evaluate(3) == False  # 11

    def test_load_bf_format_values_only(self, tmp_path):
        """Load .bf file with values only (no input strings)."""
        content = """2
0
1
1
0"""
        path = tmp_path / "xor.bf"
        path.write_text(content)
        
        loaded = load_bf(path)
        
        assert loaded.n_vars == 2
        assert loaded.evaluate(0) == False
        assert loaded.evaluate(1) == True

    def test_load_partial_function(self, tmp_path):
        """Load .bf file with undefined values (-1)."""
        content = """2
00 0
01 -1
10 1
11 -1"""
        path = tmp_path / "partial.bf"
        path.write_text(content)
        
        loaded = load_bf(path)
        
        # Check metadata indicates partial
        assert loaded._metadata.get("partial") == True
        assert "known_mask" in loaded._metadata

    def test_roundtrip_majority(self, tmp_path):
        """Round-trip test for majority function."""
        f = bf.majority(5)
        path = tmp_path / "maj5.bf"
        
        save(f, path, format="bf")
        loaded = load(path)
        
        assert loaded.n_vars == 5
        for i in range(32):
            assert loaded.evaluate(i) == f.evaluate(i)


class TestDIMACSCNF:
    """Tests for DIMACS CNF format."""

    def test_load_simple_cnf(self, tmp_path):
        """Load simple DIMACS CNF file."""
        # (x1 OR x2) AND (NOT x1 OR x2)
        content = """c Simple CNF example
p cnf 2 2
1 2 0
-1 2 0"""
        path = tmp_path / "simple.cnf"
        path.write_text(content)
        
        loaded = load_dimacs_cnf(path)
        
        assert loaded.n_vars == 2
        # x1=0, x2=0: (0 OR 0) AND (1 OR 0) = False
        # x1=0, x2=1: (0 OR 1) AND (1 OR 1) = True
        # x1=1, x2=0: (1 OR 0) AND (0 OR 0) = False
        # x1=1, x2=1: (1 OR 1) AND (0 OR 1) = True
        assert loaded.evaluate([0, 0]) == False
        assert loaded.evaluate([0, 1]) == True
        assert loaded.evaluate([1, 0]) == False
        assert loaded.evaluate([1, 1]) == True

    def test_load_cnf_with_comments(self, tmp_path):
        """Load CNF with multiple comment lines."""
        content = """c Comment 1
c Comment 2
c Yet another comment
p cnf 2 1
1 2 0"""
        path = tmp_path / "comments.cnf"
        path.write_text(content)
        
        loaded = load_dimacs_cnf(path)
        assert loaded.n_vars == 2

    def test_save_cnf(self, tmp_path):
        """Save function to DIMACS CNF format."""
        # Create AND function (only satisfiable when all inputs are 1)
        f = bf.AND(2)
        path = tmp_path / "and.cnf"
        
        save_dimacs_cnf(f, path, comment="AND function")
        
        content = path.read_text()
        assert "p cnf" in content
        assert "c AND function" in content

    def test_roundtrip_cnf(self, tmp_path):
        """Round-trip test for CNF format."""
        # Create a function, save as CNF, reload
        # Note: CNF conversion may not preserve exact structure
        # but should preserve function values
        f = bf.create([0, 0, 0, 1])  # AND
        path = tmp_path / "test.cnf"
        
        save(f, path, format="dimacs_cnf")
        loaded = load(path)
        
        # Verify function values match
        for i in range(4):
            assert loaded.evaluate(i) == f.evaluate(i)


class TestGenericLoadSave:
    """Tests for generic load/save functions."""

    def test_load_autodetect_json(self, tmp_path):
        """Load auto-detects JSON format."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "test.json"
        save(f, path)
        
        loaded = load(path)
        assert loaded.n_vars == 2

    def test_load_autodetect_bf(self, tmp_path):
        """Load auto-detects .bf format."""
        path = tmp_path / "test.bf"
        path.write_text("2\n0\n1\n1\n0")
        
        loaded = load(path)
        assert loaded.n_vars == 2

    def test_load_explicit_format(self, tmp_path):
        """Load with explicit format override."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "test.txt"
        save_json(f, path)
        
        loaded = load(path, format="json")
        assert loaded.n_vars == 2

    def test_save_default_json(self, tmp_path):
        """Save defaults to JSON for unknown extension."""
        f = bf.create([0, 1, 1, 0])
        path = tmp_path / "test.xyz"
        
        save(f, path)  # Should default to JSON
        
        content = path.read_text()
        assert content.startswith("{")

    def test_load_nonexistent_raises(self, tmp_path):
        """Load nonexistent file raises FileNotFoundError."""
        path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load(path)

    def test_load_unknown_format_raises(self, tmp_path):
        """Load with unknown format raises FileIOError."""
        path = tmp_path / "test.json"
        path.write_text("{}")
        
        with pytest.raises(FileIOError):
            load(path, format="unknown_format")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_bf_file_raises(self, tmp_path):
        """Empty .bf file raises FileIOError."""
        path = tmp_path / "empty.bf"
        path.write_text("")
        
        with pytest.raises(FileIOError):
            load_bf(path)

    def test_invalid_bf_first_line_raises(self, tmp_path):
        """Invalid first line in .bf file raises FileIOError."""
        path = tmp_path / "invalid.bf"
        path.write_text("not_a_number\n0\n1\n1\n0")
        
        with pytest.raises(FileIOError):
            load_bf(path)

    def test_large_function_roundtrip(self, tmp_path):
        """Round-trip works for larger functions."""
        f = bf.random(8, seed=123)  # 256 values
        path = tmp_path / "large.json"
        
        save(f, path)
        loaded = load(path)
        
        assert loaded.n_vars == 8
        # Spot check some values
        for i in [0, 127, 255]:
            assert loaded.evaluate(i) == f.evaluate(i)

    def test_save_function_without_nvars_raises(self, tmp_path):
        """Saving function without n_vars raises error for .bf format."""
        f = bf.BooleanFunction()
        f.n_vars = None
        path = tmp_path / "no_nvars.bf"
        
        with pytest.raises(FileIOError):
            save_bf(f, path)


class TestTopLevelAPI:
    """Tests for top-level bf.load and bf.save."""

    def test_bf_load_available(self):
        """bf.load is available at top level."""
        assert hasattr(bf, "load")

    def test_bf_save_available(self):
        """bf.save is available at top level."""
        assert hasattr(bf, "save")

    def test_bf_load_save_roundtrip(self, tmp_path):
        """Round-trip using bf.load and bf.save."""
        f = bf.majority(3)
        path = tmp_path / "maj.json"
        
        bf.save(f, path)
        loaded = bf.load(path)
        
        assert loaded.n_vars == 3
        for i in range(8):
            assert loaded.evaluate(i) == f.evaluate(i)
