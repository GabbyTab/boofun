"""
Tests for core/auto_representation module.

Tests automatic representation selection for Boolean functions:
- estimate_sparsity
- recommend_representation
- auto_select_representation
- AdaptiveFunction class
- optimize_representation
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.core.auto_representation import (
    DENSE_THRESHOLD,
    PACKED_THRESHOLD,
    SPARSE_RATIO_THRESHOLD,
    AdaptiveFunction,
    auto_select_representation,
    estimate_sparsity,
    optimize_representation,
    recommend_representation,
)


# ---------------------------------------------------------------------------
# Tests: estimate_sparsity
# ---------------------------------------------------------------------------


class TestEstimateSparsity:
    """Test estimate_sparsity function."""

    def test_sparsity_bounded(self):
        """Sparsity is in [0, 0.5]."""
        for tt in [
            np.array([0, 0, 0, 0]),
            np.array([1, 1, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0, 0, 0, 1]),
        ]:
            sparsity = estimate_sparsity(tt)
            assert 0.0 <= sparsity <= 0.5

    def test_balanced_is_half(self):
        """Balanced truth table (equal 0s and 1s) has sparsity 0.5."""
        tt = np.array([0, 1, 1, 0])
        assert abs(estimate_sparsity(tt) - 0.5) < 1e-10

    def test_all_zeros(self):
        """All-zero truth table has sparsity 0."""
        tt = np.array([0, 0, 0, 0])
        assert abs(estimate_sparsity(tt)) < 1e-10

    def test_all_ones(self):
        """All-one truth table has sparsity 0."""
        tt = np.array([1, 1, 1, 1])
        assert abs(estimate_sparsity(tt)) < 1e-10

    def test_single_one(self):
        """Truth table with single 1 has sparsity 1/size."""
        tt = np.array([0, 0, 0, 1])
        assert abs(estimate_sparsity(tt) - 0.25) < 1e-10

    def test_returns_numeric(self):
        """Sparsity returns a numeric value."""
        tt = np.array([0, 0, 0, 1])
        sparsity = estimate_sparsity(tt)
        assert isinstance(sparsity, (int, float, np.number))


# ---------------------------------------------------------------------------
# Tests: recommend_representation
# ---------------------------------------------------------------------------


class TestRecommendRepresentation:
    """Test recommend_representation function."""

    def test_small_n_recommends_dense(self):
        """n <= DENSE_THRESHOLD recommends dense truth table."""
        rec = recommend_representation(n_vars=3)
        assert rec["representation"] == "truth_table"

    def test_medium_n_recommends_packed(self):
        """DENSE_THRESHOLD < n <= PACKED_THRESHOLD recommends packed."""
        n = DENSE_THRESHOLD + 1
        rec = recommend_representation(n_vars=n)
        assert rec["representation"] == "packed_truth_table"

    def test_large_n_recommends_packed(self):
        """n > PACKED_THRESHOLD recommends packed (default access)."""
        n = PACKED_THRESHOLD + 1
        rec = recommend_representation(n_vars=n)
        assert rec["representation"] == "packed_truth_table"

    def test_sparse_function_recommends_sparse(self):
        """Low sparsity triggers sparse recommendation."""
        n = DENSE_THRESHOLD + 1
        rec = recommend_representation(n_vars=n, sparsity=0.01)
        assert rec["representation"] == "sparse_truth_table"

    def test_memory_limit_triggers_sparse(self):
        """When memory limit is too tight for packed, recommends sparse."""
        n = 30  # 2^30 bits = 128 MB packed
        rec = recommend_representation(n_vars=n, memory_limit_mb=0.001)
        assert rec["representation"] == "sparse_truth_table"

    def test_sparse_queries_pattern(self):
        """access_pattern='sparse_queries' favors sparse for large n."""
        n = PACKED_THRESHOLD + 1
        rec = recommend_representation(n_vars=n, access_pattern="sparse_queries")
        assert rec["representation"] == "sparse_truth_table"

    def test_result_has_required_keys(self):
        """Recommendation dict has expected keys."""
        rec = recommend_representation(n_vars=3)
        assert "representation" in rec
        assert "reason" in rec
        assert "n_vars" in rec

    def test_threshold_boundary(self):
        """At the exact DENSE_THRESHOLD boundary, dense is recommended."""
        rec = recommend_representation(n_vars=DENSE_THRESHOLD)
        assert rec["representation"] == "truth_table"


# ---------------------------------------------------------------------------
# Tests: auto_select_representation
# ---------------------------------------------------------------------------


class TestAutoSelectRepresentation:
    """Test auto_select_representation function."""

    def test_small_selects_dense(self):
        """Small truth table auto-selects dense."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        result = auto_select_representation(tt)
        assert result["recommendation"]["representation"] == "truth_table"
        assert result["format"] == "dense"

    def test_result_has_sparsity(self):
        """Result includes computed sparsity."""
        tt = np.array([0, 0, 0, 1], dtype=bool)
        result = auto_select_representation(tt)
        assert "sparsity" in result

    def test_data_preserves_values(self):
        """Auto-selected data evaluates correctly."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        result = auto_select_representation(tt)
        if result["format"] == "dense":
            assert np.array_equal(result["data"], tt)

    def test_n_vars_auto_detected(self):
        """n_vars is auto-detected from truth table length."""
        tt = np.zeros(16, dtype=bool)
        result = auto_select_representation(tt)
        assert result["recommendation"]["n_vars"] == 4

    def test_n_vars_explicit(self):
        """Explicit n_vars is used when provided."""
        tt = np.zeros(16, dtype=bool)
        result = auto_select_representation(tt, n_vars=4)
        assert result["recommendation"]["n_vars"] == 4


# ---------------------------------------------------------------------------
# Tests: AdaptiveFunction class
# ---------------------------------------------------------------------------


class TestAdaptiveFunction:
    """Test AdaptiveFunction class."""

    def test_dense_evaluation(self):
        """Dense format evaluates correctly."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        af = AdaptiveFunction(tt, n_vars=2)
        assert af.evaluate(0) is False
        assert af.evaluate(1) is True
        assert af.evaluate(2) is True
        assert af.evaluate(3) is False

    def test_forced_dense(self):
        """Forced 'dense' representation works."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        af = AdaptiveFunction(tt, force_representation="dense")
        assert af.format == "dense"
        assert af.evaluate(1) is True

    def test_forced_sparse(self):
        """Forced 'sparse' representation works."""
        tt = np.array([0, 0, 0, 1], dtype=bool)
        af = AdaptiveFunction(tt, force_representation="sparse")
        assert af.format == "sparse"
        assert af.evaluate(3) is True
        assert af.evaluate(0) is False

    def test_forced_invalid_raises(self):
        """Invalid forced representation raises ValueError."""
        tt = np.array([0, 1], dtype=bool)
        with pytest.raises(ValueError, match="Unknown representation"):
            AdaptiveFunction(tt, force_representation="unknown")

    def test_to_dense_roundtrip(self):
        """to_dense recovers the original truth table."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        for repr_type in ["dense", "sparse"]:
            af = AdaptiveFunction(tt, force_representation=repr_type)
            recovered = af.to_dense()
            assert np.array_equal(recovered, tt)

    def test_sparsity_property(self):
        """Sparsity property returns correct value."""
        tt = np.array([0, 0, 0, 1], dtype=bool)
        af = AdaptiveFunction(tt)
        assert abs(af.sparsity - 0.25) < 1e-10

    def test_memory_usage_returns_dict(self):
        """memory_usage returns dict with format key."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        af = AdaptiveFunction(tt, force_representation="dense")
        mem = af.memory_usage()
        assert mem["format"] == "dense"
        assert "bytes" in mem

    def test_memory_usage_sparse(self):
        """memory_usage for sparse format includes num_exceptions."""
        tt = np.array([0, 0, 0, 1], dtype=bool)
        af = AdaptiveFunction(tt, force_representation="sparse")
        mem = af.memory_usage()
        assert mem["format"] == "sparse"
        assert "num_exceptions" in mem

    def test_summary_string(self):
        """summary returns a descriptive string."""
        tt = np.array([0, 1, 1, 0], dtype=bool)
        af = AdaptiveFunction(tt)
        s = af.summary()
        assert "AdaptiveFunction" in s
        assert "n=2" in s

    def test_n_vars_detection(self):
        """n_vars is correctly auto-detected."""
        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        af = AdaptiveFunction(tt)
        assert af.n_vars == 3
        assert af.size == 8


# ---------------------------------------------------------------------------
# Tests: optimize_representation
# ---------------------------------------------------------------------------


class TestOptimizeRepresentation:
    """Test optimize_representation function."""

    def test_on_boolean_function(self):
        """optimize_representation works on a BooleanFunction."""
        f = bf.parity(3)
        result = optimize_representation(f)
        assert "recommended_representation" in result
        assert "sparsity" in result
        assert result["n_vars"] == 3

    def test_small_function_stays_dense(self):
        """Small functions should recommend staying dense."""
        f = bf.AND(3)
        result = optimize_representation(f)
        assert result["recommended_representation"] == "truth_table"

    def test_result_keys(self):
        """Result has all expected keys."""
        f = bf.majority(3)
        result = optimize_representation(f)
        expected_keys = {
            "current_representation",
            "recommended_representation",
            "reason",
            "sparsity",
            "n_vars",
            "would_save_memory",
        }
        assert expected_keys.issubset(set(result.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
