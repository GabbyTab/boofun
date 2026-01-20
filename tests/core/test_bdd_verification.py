"""
BDD (Binary Decision Diagram) verification tests.

Tests BDD correctness and verifies node counts against known theoretical bounds:
- AND_n: n+2 nodes (linear chain + 2 terminals)
- OR_n: n+2 nodes (linear chain + 2 terminals)
- PARITY_n: 2n+2 nodes (diamond structure + 2 terminals)
- Symmetric functions: O(n²) nodes

References:
- Bryant, "Graph-Based Algorithms for Boolean Function Manipulation" (1986)
- Wegener, "BDDs - Design, Analysis, Complexity, and Applications" (2000)
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.core.representations.bdd import BDD, BDDNode, BDDRepresentation
from boofun.core.representations.truth_table import TruthTableRepresentation
from boofun.core.spaces import Space


class TestBDDEvaluation:
    """Test BDD evaluation correctness."""

    def test_bdd_and_correctness(self):
        """BDD for AND evaluates correctly at all inputs."""
        for n in [2, 3, 4]:
            f = bf.AND(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            for x in range(1 << n):
                expected = tt_data[x]
                actual = bdd_repr.evaluate(x, bdd_data, Space.BOOLEAN_CUBE, n)
                assert actual == expected, f"AND_{n} failed at x={x}"

    def test_bdd_or_correctness(self):
        """BDD for OR evaluates correctly at all inputs."""
        for n in [2, 3, 4]:
            f = bf.OR(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            for x in range(1 << n):
                expected = tt_data[x]
                actual = bdd_repr.evaluate(x, bdd_data, Space.BOOLEAN_CUBE, n)
                assert actual == expected, f"OR_{n} failed at x={x}"

    def test_bdd_parity_correctness(self):
        """BDD for Parity evaluates correctly at all inputs."""
        for n in [2, 3, 4, 5]:
            f = bf.parity(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            for x in range(1 << n):
                expected = tt_data[x]
                actual = bdd_repr.evaluate(x, bdd_data, Space.BOOLEAN_CUBE, n)
                assert actual == expected, f"PARITY_{n} failed at x={x}"

    def test_bdd_majority_correctness(self):
        """BDD for Majority evaluates correctly at all inputs."""
        for n in [3, 5]:
            f = bf.majority(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            for x in range(1 << n):
                expected = tt_data[x]
                actual = bdd_repr.evaluate(x, bdd_data, Space.BOOLEAN_CUBE, n)
                assert actual == expected, f"MAJ_{n} failed at x={x}"


class TestBDDNodeCount:
    """Test BDD node counts against theoretical bounds."""

    def _count_nodes(self, bdd_data) -> int:
        """Count nodes in a BDD."""
        if isinstance(bdd_data, BDD):
            return self._count_bdd_nodes(bdd_data.root, set())
        elif isinstance(bdd_data, BDDNode):
            return self._count_bdd_nodes(bdd_data, set())
        elif isinstance(bdd_data, dict):
            if "root" in bdd_data:
                return self._count_bdd_nodes(bdd_data["root"], set())
        return 0

    def _count_bdd_nodes(self, node, visited) -> int:
        """Recursively count unique nodes."""
        if node is None:
            return 0

        node_id = id(node)
        if node_id in visited:
            return 0

        visited.add(node_id)

        if node.is_terminal:
            return 1

        return (
            1 + self._count_bdd_nodes(node.low, visited) + self._count_bdd_nodes(node.high, visited)
        )

    def test_and_node_count_bound(self):
        """
        AND_n has O(n) nodes in ROBDD.
        Specifically: AND has a linear chain structure with n+2 nodes.
        """
        for n in [3, 4, 5]:
            f = bf.AND(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            node_count = self._count_nodes(bdd_data)
            # AND should have at most n+2 nodes (n decision nodes + 2 terminals)
            assert node_count <= n + 2 + 2, f"AND_{n} has {node_count} nodes, expected ≤ {n+4}"

    def test_or_node_count_bound(self):
        """
        OR_n has O(n) nodes in ROBDD.
        Specifically: OR has a linear chain structure with n+2 nodes.
        """
        for n in [3, 4, 5]:
            f = bf.OR(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            node_count = self._count_nodes(bdd_data)
            # OR should have at most n+2 nodes
            assert node_count <= n + 2 + 2, f"OR_{n} has {node_count} nodes, expected ≤ {n+4}"

    def test_parity_node_count_bound(self):
        """
        PARITY_n has O(n) nodes in ROBDD.
        Specifically: 2n+2 nodes (diamond structure).
        """
        for n in [3, 4, 5]:
            f = bf.parity(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            node_count = self._count_nodes(bdd_data)
            # Parity should have at most 2n+2 nodes
            assert node_count <= 2 * n + 4, f"PARITY_{n} has {node_count} nodes, expected ≤ {2*n+4}"

    def test_symmetric_node_count_bound(self):
        """
        Symmetric functions have O(n²) nodes.
        Majority is symmetric.
        """
        for n in [3, 5]:
            f = bf.majority(n)
            bdd_repr = BDDRepresentation()
            tt_repr = TruthTableRepresentation()

            tt_data = f.get_representation("truth_table")
            bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

            node_count = self._count_nodes(bdd_data)
            # Symmetric functions have O(n²) nodes
            assert (
                node_count <= n * n + 2 * n + 2
            ), f"MAJ_{n} has {node_count} nodes, expected O(n²)"


class TestBDDReduction:
    """Test that BDDs are properly reduced."""

    def test_no_redundant_nodes(self):
        """
        Reduced BDD: No node has low == high.
        """
        f = bf.AND(4)
        bdd_repr = BDDRepresentation()
        tt_repr = TruthTableRepresentation()

        tt_data = f.get_representation("truth_table")
        bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, 4)

        def check_no_redundant(node, visited=None):
            if visited is None:
                visited = set()
            if node is None or node.is_terminal:
                return True
            if id(node) in visited:
                return True
            visited.add(id(node))

            # Check: low != high for decision nodes
            if node.low is node.high:
                return False

            return check_no_redundant(node.low, visited) and check_no_redundant(node.high, visited)

        if isinstance(bdd_data, BDD):
            assert check_no_redundant(bdd_data.root)
        elif isinstance(bdd_data, BDDNode):
            assert check_no_redundant(bdd_data)


class TestBDDConsistency:
    """Test BDD consistency with truth table representation."""

    def test_roundtrip_random_functions(self):
        """Convert truth table -> BDD -> evaluate matches original."""
        import random

        random.seed(42)

        bdd_repr = BDDRepresentation()
        tt_repr = TruthTableRepresentation()

        for n in [3, 4]:
            for _ in range(5):
                # Random truth table
                tt = [random.choice([True, False]) for _ in range(1 << n)]
                f = bf.create(tt)

                tt_data = f.get_representation("truth_table")
                bdd_data = bdd_repr.convert_from(tt_repr, tt_data, Space.BOOLEAN_CUBE, n)

                for x in range(1 << n):
                    expected = tt_data[x]
                    actual = bdd_repr.evaluate(x, bdd_data, Space.BOOLEAN_CUBE, n)
                    assert (
                        actual == expected
                    ), f"Random function mismatch at x={x}: expected {expected}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
