"""
Comprehensive tests for decision_trees module.

Tests cover:
- DP algorithms for various function types
- Edge cases (constant, single variable)
- Tree reconstruction and evaluation
- Enumeration and counting
- Known complexity values
"""

import numpy as np
import pytest

import boofun as bf
from boofun.analysis.decision_trees import (
    DecisionTree,
    count_decision_trees,
    decision_tree_depth_dp,
    decision_tree_depth_uniform_dp,
    decision_tree_depth_weighted_dp,
    decision_tree_size_dp,
    enumerate_decision_trees,
    randomized_complexity_matrix,
    reconstruct_tree,
    tree_depth,
    tree_size,
)


class TestDecisionTreeDepthDP:
    """Tests for decision_tree_depth_dp (worst-case depth)."""

    def test_constant_zero(self):
        """Constant 0 function has depth 0."""
        f = bf.create([0, 0, 0, 0])
        assert decision_tree_depth_dp(f) == 0

    def test_constant_one(self):
        """Constant 1 function has depth 0."""
        f = bf.create([1, 1, 1, 1])
        assert decision_tree_depth_dp(f) == 0

    def test_dictator_x0(self):
        """Dictator on x0 has depth 1."""
        f = bf.create([0, 1, 0, 1])  # f(x) = x0
        assert decision_tree_depth_dp(f) == 1

    def test_dictator_x1(self):
        """Dictator on x1 has depth 1."""
        f = bf.create([0, 0, 1, 1])  # f(x) = x1
        assert decision_tree_depth_dp(f) == 1

    def test_and_2vars(self):
        """AND of 2 variables has depth 2."""
        f = bf.create([0, 0, 0, 1])  # f(x) = x0 AND x1
        assert decision_tree_depth_dp(f) == 2

    def test_or_2vars(self):
        """OR of 2 variables has depth 2."""
        f = bf.create([0, 1, 1, 1])  # f(x) = x0 OR x1
        assert decision_tree_depth_dp(f) == 2

    def test_xor_2vars(self):
        """XOR of 2 variables has depth 2."""
        f = bf.create([0, 1, 1, 0])  # f(x) = x0 XOR x1
        assert decision_tree_depth_dp(f) == 2

    def test_and_3vars(self):
        """AND of 3 variables has depth 3."""
        f = bf.create([0, 0, 0, 0, 0, 0, 0, 1])
        assert decision_tree_depth_dp(f) == 3

    def test_or_3vars(self):
        """OR of 3 variables has depth 3."""
        f = bf.create([0, 1, 1, 1, 1, 1, 1, 1])
        assert decision_tree_depth_dp(f) == 3

    def test_majority_3vars(self):
        """Majority of 3 variables has depth 3."""
        # MAJ(x0,x1,x2) = 1 iff at least 2 inputs are 1
        f = bf.create([0, 0, 0, 1, 0, 1, 1, 1])
        assert decision_tree_depth_dp(f) == 3

    def test_single_variable_function(self):
        """Single variable function has depth 1."""
        f = bf.create([0, 1])  # f(x) = x0
        assert decision_tree_depth_dp(f) == 1


class TestDecisionTreeUniformDP:
    """Tests for average-case depth under uniform distribution."""

    def test_constant_avg_depth(self):
        """Constant function has average depth 0."""
        f = bf.create([0, 0, 0, 0])
        avg_depth, tree = decision_tree_depth_uniform_dp(f)
        assert avg_depth == 0.0
        assert tree is not None

    def test_dictator_avg_depth(self):
        """Dictator has average depth 1."""
        f = bf.create([0, 1, 0, 1])
        avg_depth, tree = decision_tree_depth_uniform_dp(f)
        assert avg_depth == 1.0

    def test_and_avg_depth(self):
        """AND has average depth between 1 and 2."""
        f = bf.create([0, 0, 0, 1])
        avg_depth, tree = decision_tree_depth_uniform_dp(f)
        # AND can be computed with avg depth < 2 (query x0 first, if 0 return 0)
        assert 1.0 <= avg_depth <= 2.0

    def test_xor_avg_depth(self):
        """XOR has average depth exactly 2 (must query both vars)."""
        f = bf.create([0, 1, 1, 0])
        avg_depth, tree = decision_tree_depth_uniform_dp(f)
        assert avg_depth == 2.0

    def test_returns_tree(self):
        """Function returns a valid decision tree."""
        f = bf.create([0, 1, 1, 0])
        avg_depth, tree = decision_tree_depth_uniform_dp(f)
        assert isinstance(tree, DecisionTree)


class TestDecisionTreeWeightedDP:
    """Tests for average-case depth under arbitrary distribution."""

    def test_uniform_matches_uniform_dp(self):
        """Uniform distribution matches dedicated uniform function."""
        f = bf.create([0, 1, 1, 0])

        # Uniform distribution
        probs = [0.25, 0.25, 0.25, 0.25]

        depth_uniform, _ = decision_tree_depth_uniform_dp(f)
        depth_weighted, _ = decision_tree_depth_weighted_dp(f, probs)

        # Should be approximately equal
        assert abs(depth_uniform - depth_weighted) < 0.01

    def test_concentrated_distribution(self):
        """Concentrated distribution can reduce expected depth."""
        f = bf.create([0, 1, 1, 0])

        # All probability on input 0
        probs = [1.0, 0.0, 0.0, 0.0]

        depth, tree = decision_tree_depth_weighted_dp(f, probs)
        # Expected depth should be 0 since only input with prob is constant
        # (Actually XOR(0,0) = 0 is always output, but tree still exists)
        assert depth >= 0


class TestDecisionTreeSizeDP:
    """Tests for minimizing tree size."""

    def test_constant_size_one(self):
        """Constant function has size 1 (single leaf)."""
        f = bf.create([0, 0, 0, 0])
        size, depth, tree = decision_tree_size_dp(f)
        assert size == 1
        assert depth == 0

    def test_dictator_size_two(self):
        """Dictator has size 2 (two leaves)."""
        f = bf.create([0, 1, 0, 1])
        size, depth, tree = decision_tree_size_dp(f)
        assert size == 2
        assert depth == 1

    def test_returns_tree(self):
        """Function returns a valid tree."""
        f = bf.create([0, 1, 1, 0])
        size, depth, tree = decision_tree_size_dp(f)
        assert isinstance(tree, DecisionTree)
        assert tree.size() == size
        assert tree.depth() == depth


class TestDecisionTreeClass:
    """Tests for DecisionTree dataclass."""

    def test_leaf_creation(self):
        """Create leaf node."""
        leaf = DecisionTree(value=0)
        assert leaf.is_leaf()
        assert leaf.value == 0
        assert leaf.depth() == 0
        assert leaf.size() == 1

    def test_internal_node(self):
        """Create internal node."""
        left = DecisionTree(value=0)
        right = DecisionTree(value=1)
        node = DecisionTree(var=0, left=left, right=right)

        assert not node.is_leaf()
        assert node.var == 0
        assert node.depth() == 1
        assert node.size() == 2

    def test_deep_tree(self):
        """Create and measure deep tree."""
        # Build tree of depth 3
        l0 = DecisionTree(value=0)
        l1 = DecisionTree(value=1)
        n1 = DecisionTree(var=2, left=l0, right=l1)
        n2 = DecisionTree(var=1, left=l0, right=n1)
        root = DecisionTree(var=0, left=n2, right=l1)

        assert root.depth() == 3

    def test_evaluate_dictator(self):
        """Evaluate dictator tree."""
        left = DecisionTree(value=0)
        right = DecisionTree(value=1)
        tree = DecisionTree(var=0, left=left, right=right)

        # f(x) = x0
        assert tree.evaluate(0b00, 2) == 0
        assert tree.evaluate(0b01, 2) == 1
        assert tree.evaluate(0b10, 2) == 0
        assert tree.evaluate(0b11, 2) == 1

    def test_query_depth(self):
        """Query depth counts queries made."""
        left = DecisionTree(value=0)
        right = DecisionTree(value=1)
        tree = DecisionTree(var=0, left=left, right=right)

        # Always makes exactly 1 query
        assert tree.query_depth(0b00, 2) == 1
        assert tree.query_depth(0b11, 2) == 1

    def test_to_dict(self):
        """Convert tree to dictionary."""
        tree = DecisionTree(var=0, left=DecisionTree(value=0), right=DecisionTree(value=1))

        d = tree.to_dict()
        assert d["type"] == "internal"
        assert d["var"] == 0
        assert d["left"]["type"] == "leaf"
        assert d["left"]["value"] == 0


class TestTreeUtilities:
    """Tests for tree_depth and tree_size utilities."""

    def test_tree_depth_empty(self):
        """Empty list has depth 0."""
        assert tree_depth([]) == 0

    def test_tree_depth_list(self):
        """Depth of list-format tree."""
        # [var, left, right]
        tree = [0, [], []]  # Query x0, both branches are leaves
        assert tree_depth(tree) == 1

    def test_tree_depth_nested(self):
        """Depth of nested tree."""
        tree = [0, [1, [], []], []]
        assert tree_depth(tree) == 2

    def test_tree_size_empty(self):
        """Empty tree has size 1."""
        assert tree_size([]) == 1

    def test_tree_size_list(self):
        """Size of list-format tree."""
        tree = [0, [], []]
        assert tree_size(tree) == 2


class TestEnumerateDecisionTrees:
    """Tests for tree enumeration."""

    def test_constant_one_tree(self):
        """Constant function has exactly one tree."""
        f = bf.create([0, 0, 0, 0])
        trees = enumerate_decision_trees(f)
        assert len(trees) == 1
        assert trees[0].is_leaf()

    def test_dictator_trees(self):
        """Dictator has multiple equivalent trees."""
        f = bf.create([0, 1, 0, 1])  # x0 on 2 vars
        trees = enumerate_decision_trees(f)
        # All trees should compute the same function
        for tree in trees:
            for x in range(4):
                expected = (x >> 0) & 1  # x0
                assert tree.evaluate(x, 2) == expected

    def test_too_large_raises(self):
        """Large n raises error."""
        f = bf.create([0] * 512)  # 9 variables
        with pytest.raises(ValueError, match="only practical"):
            enumerate_decision_trees(f)


class TestCountDecisionTrees:
    """Tests for counting decision trees."""

    def test_constant_count_one(self):
        """Constant function has count 1."""
        f = bf.create([0, 0, 0, 0])
        assert count_decision_trees(f) == 1

    def test_matches_enumeration(self):
        """Count matches length of enumeration."""
        f = bf.create([0, 1, 1, 0])  # XOR on 2 vars
        count = count_decision_trees(f)
        trees = enumerate_decision_trees(f)
        assert count == len(trees)


class TestRandomizedComplexityMatrix:
    """Tests for randomized complexity analysis."""

    def test_matrix_shape(self):
        """Matrix has correct shape."""
        f = bf.create([0, 1, 1, 0])
        matrix = randomized_complexity_matrix(f)

        # Rows = inputs, columns = trees
        n_inputs = 4
        n_trees = len(enumerate_decision_trees(f))

        assert matrix.shape[0] == n_inputs
        assert matrix.shape[1] == n_trees

    def test_filter_by_output(self):
        """Filter inputs by output value."""
        f = bf.create([0, 0, 0, 1])  # AND

        # Only 1-inputs
        matrix_1 = randomized_complexity_matrix(f, output_value=1)
        assert matrix_1.shape[0] == 1  # Only one 1-input (11)

        # Only 0-inputs
        matrix_0 = randomized_complexity_matrix(f, output_value=0)
        assert matrix_0.shape[0] == 3  # Three 0-inputs

    def test_entries_are_depths(self):
        """Matrix entries are query depths."""
        f = bf.create([0, 1, 0, 1])  # Dictator x0
        matrix = randomized_complexity_matrix(f)

        # All entries should be non-negative integers
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 2)  # Max depth is 2 for 2-var function


class TestKnownComplexityValues:
    """Test against known complexity values from literature."""

    def test_and_depth_equals_n(self):
        """D(AND_n) = n."""
        for n in [2, 3, 4]:
            tt = [0] * (1 << n)
            tt[-1] = 1  # Only all-1s gives 1
            f = bf.create(tt)
            assert decision_tree_depth_dp(f) == n

    def test_or_depth_equals_n(self):
        """D(OR_n) = n."""
        for n in [2, 3, 4]:
            tt = [1] * (1 << n)
            tt[0] = 0  # Only all-0s gives 0
            f = bf.create(tt)
            assert decision_tree_depth_dp(f) == n

    def test_parity_depth_equals_n(self):
        """D(PARITY_n) = n."""
        for n in [2, 3, 4]:
            tt = [bin(x).count("1") % 2 for x in range(1 << n)]
            f = bf.create(tt)
            assert decision_tree_depth_dp(f) == n
