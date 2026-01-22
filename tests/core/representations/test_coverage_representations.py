"""
Comprehensive tests for representation modules to increase coverage.

Tests cover:
- packed_truth_table.py
- ltf.py
- polynomial.py
- sparse_truth_table.py
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.core.spaces import Space


# =============================================================================
# Packed Truth Table Tests (target: lines 65-88, 97, 106-114, 119, 123-136, etc.)
# =============================================================================
class TestPackedTruthTableEvaluation:
    """Test evaluation methods of PackedTruthTableRepresentation."""

    def test_evaluate_scalar_index(self):
        """Test evaluation with scalar index."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        # Test evaluation at various indices
        for i in range(8):
            result = repr_obj.evaluate(np.array(i), packed, space, 3)
            assert result == bool(tt[i])

    def test_evaluate_binary_vector(self):
        """Test evaluation with binary vector input."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        # Binary input [0, 0, 0] -> index 0
        result = repr_obj.evaluate(np.array([0, 0, 0]), packed, space, 3)
        assert result == bool(tt[0])

        # Binary input [1, 0, 0] -> index 1 (LSB first)
        result = repr_obj.evaluate(np.array([1, 0, 0]), packed, space, 3)
        assert result == bool(tt[1])

    def test_evaluate_array_of_indices(self):
        """Test evaluation with array of indices."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        indices = np.array([0, 1, 2, 7])
        results = repr_obj.evaluate(indices, packed, space, 3)
        assert len(results) == 4
        assert results[0] == bool(tt[0])
        assert results[3] == bool(tt[7])

    def test_evaluate_batch_binary_vectors(self):
        """Test evaluation with 2D array of binary vectors."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        # Batch of binary vectors
        batch = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
        results = repr_obj.evaluate(batch, packed, space, 3)
        assert len(results) == 4

    def test_evaluate_index_out_of_range(self):
        """Test that out of range index raises error."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        with pytest.raises(IndexError):
            repr_obj.evaluate(np.array(100), packed, space, 2)

    def test_evaluate_unsupported_shape(self):
        """Test that unsupported input shape raises error."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        # 3D array should raise
        with pytest.raises(ValueError):
            repr_obj.evaluate(np.zeros((2, 2, 2)), packed, space, 2)

    def test_binary_to_index(self):
        """Test _binary_to_index helper method."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
        )

        repr_obj = PackedTruthTableRepresentation()

        # LSB first: [1, 0, 0] -> 1, [0, 1, 0] -> 2, [1, 1, 0] -> 3
        assert repr_obj._binary_to_index(np.array([0, 0, 0])) == 0
        assert repr_obj._binary_to_index(np.array([1, 0, 0])) == 1
        assert repr_obj._binary_to_index(np.array([0, 1, 0])) == 2
        assert repr_obj._binary_to_index(np.array([1, 1, 0])) == 3
        assert repr_obj._binary_to_index(np.array([1, 1, 1])) == 7


class TestPackedTruthTableDump:
    """Test dump/export methods."""

    def test_dump_returns_dict(self):
        """Test dump returns serializable dictionary."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed = create_packed_truth_table(tt)
        repr_obj = PackedTruthTableRepresentation()

        result = repr_obj.dump(packed)
        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "packed_truth_table"
        assert "n_vars" in result
        assert "size" in result


class TestPackedTruthTableConversion:
    """Test conversion methods."""

    def test_convert_from_truth_table(self):
        """Test conversion from truth table to packed."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
        )
        from boofun.core.representations.truth_table import TruthTableRepresentation

        tt_repr = TruthTableRepresentation()
        packed_repr = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        tt_data = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed_data = packed_repr.convert_from(tt_repr, tt_data, space, 3)

        assert isinstance(packed_data, dict)

    def test_convert_to_truth_table(self):
        """Test conversion from packed to truth table."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )
        from boofun.core.representations.truth_table import TruthTableRepresentation

        tt_repr = TruthTableRepresentation()
        packed_repr = PackedTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        tt_original = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed_data = create_packed_truth_table(tt_original)

        # Convert back
        tt_result = packed_repr.convert_to(tt_repr, packed_data, space, 3)
        assert tt_result is not None


class TestPackedTruthTableUtilities:
    """Test utility methods."""

    def test_create_empty(self):
        """Test create_empty method."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
        )

        repr_obj = PackedTruthTableRepresentation()
        empty = repr_obj.create_empty(4)

        assert isinstance(empty, dict)
        assert empty["n_vars"] == 4

    def test_is_complete(self):
        """Test is_complete method."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        repr_obj = PackedTruthTableRepresentation()

        tt = np.array([0, 1, 1, 0], dtype=bool)
        packed = create_packed_truth_table(tt)

        assert repr_obj.is_complete(packed)
        assert repr_obj.is_complete(None) is False

    def test_to_numpy(self):
        """Test to_numpy conversion."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
            create_packed_truth_table,
        )

        repr_obj = PackedTruthTableRepresentation()

        tt = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)
        packed = create_packed_truth_table(tt)

        result = repr_obj.to_numpy(packed)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, tt)

    def test_time_complexity_rank(self):
        """Test time_complexity_rank method."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
        )

        repr_obj = PackedTruthTableRepresentation()
        rank = repr_obj.time_complexity_rank(10)

        assert isinstance(rank, dict)
        assert "evaluation" in rank

    def test_get_storage_requirements(self):
        """Test get_storage_requirements method."""
        from boofun.core.representations.packed_truth_table import (
            PackedTruthTableRepresentation,
        )

        repr_obj = PackedTruthTableRepresentation()
        req = repr_obj.get_storage_requirements(10)

        assert isinstance(req, dict)
        assert "packed_bytes" in req


# =============================================================================
# LTF Representation Tests (target: lines 38, 52-56, 60, 69, 93-120, etc.)
# =============================================================================
class TestLTFParameters:
    """Test LTFParameters dataclass."""

    def test_creation(self):
        """Test LTFParameters creation."""
        from boofun.core.representations.ltf import LTFParameters

        params = LTFParameters(
            weights=np.array([1.0, 2.0, 3.0]), threshold=2.5, n_vars=3
        )
        assert params.n_vars == 3
        assert params.threshold == 2.5

    def test_weights_length_mismatch(self):
        """Test validation when weights length doesn't match n_vars."""
        from boofun.core.representations.ltf import LTFParameters

        with pytest.raises(ValueError):
            LTFParameters(weights=np.array([1.0, 2.0]), threshold=1.0, n_vars=3)

    def test_evaluate(self):
        """Test LTFParameters.evaluate method."""
        from boofun.core.representations.ltf import LTFParameters

        # Majority(3): weights = [1, 1, 1], threshold = 2
        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)

        # 0+0+0 = 0 < 2 -> False
        assert params.evaluate([0, 0, 0]) == False  # noqa: E712
        # 1+1+0 = 2 >= 2 -> True
        assert params.evaluate([1, 1, 0]) == True  # noqa: E712
        # 1+1+1 = 3 >= 2 -> True
        assert params.evaluate([1, 1, 1]) == True  # noqa: E712

    def test_evaluate_wrong_length(self):
        """Test evaluate with wrong input length."""
        from boofun.core.representations.ltf import LTFParameters

        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)

        with pytest.raises(ValueError):
            params.evaluate([0, 0])  # Too few

    def test_to_dict(self):
        """Test LTFParameters.to_dict method."""
        from boofun.core.representations.ltf import LTFParameters

        params = LTFParameters(weights=np.array([1.0, 2.0]), threshold=1.5, n_vars=2)
        d = params.to_dict()

        assert d["weights"] == [1.0, 2.0]
        assert d["threshold"] == 1.5
        assert d["n_vars"] == 2

    def test_from_dict(self):
        """Test LTFParameters.from_dict method."""
        from boofun.core.representations.ltf import LTFParameters

        d = {"weights": [1.0, 2.0, 3.0], "threshold": 2.0, "n_vars": 3}
        params = LTFParameters.from_dict(d)

        assert params.n_vars == 3
        assert params.threshold == 2.0


class TestLTFRepresentationEvaluation:
    """Test LTFRepresentation evaluation."""

    def test_evaluate_integer_index(self):
        """Test evaluation at integer index."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        # Create majority(3) LTF
        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)
        repr_obj = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        # Index 0 = [0,0,0] -> False
        result = repr_obj.evaluate(np.array(0), params, space, 3)
        assert result == False  # noqa: E712

        # Index 7 = [1,1,1] -> True
        result = repr_obj.evaluate(np.array(7), params, space, 3)
        assert result == True  # noqa: E712

    def test_evaluate_binary_vector(self):
        """Test evaluation with binary vector."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)
        repr_obj = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        result = repr_obj.evaluate(np.array([1, 1, 0]), params, space, 3)
        assert result == True  # noqa: E712

    def test_evaluate_array_of_indices(self):
        """Test evaluation with array of indices."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)
        repr_obj = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        indices = np.array([0, 3, 5, 7])
        results = repr_obj.evaluate(indices, params, space, 3)
        assert len(results) == 4

    def test_evaluate_batch_binary(self):
        """Test evaluation with batch of binary vectors."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)
        repr_obj = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        batch = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]])
        results = repr_obj.evaluate(batch, params, space, 3)
        assert len(results) == 3
        assert results[0] == False  # noqa: E712
        assert results[1] == True  # noqa: E712

    def test_evaluate_unsupported_shape(self):
        """Test that unsupported shape raises error."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)
        repr_obj = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        with pytest.raises(ValueError):
            repr_obj.evaluate(np.zeros((2, 2, 2)), params, space, 3)


class TestLTFRepresentationConversion:
    """Test LTF conversion methods."""

    def test_convert_from_majority(self):
        """Test converting majority to LTF."""
        from boofun.core.representations.ltf import LTFRepresentation
        from boofun.core.representations.truth_table import TruthTableRepresentation

        maj = bf.majority(3)
        tt = np.array(list(maj.get_representation("truth_table")), dtype=bool)

        tt_repr = TruthTableRepresentation()
        ltf_repr = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        params = ltf_repr.convert_from(tt_repr, tt, space, 3)
        assert params is not None
        assert params.n_vars == 3

    def test_convert_from_non_ltf_raises(self):
        """Test that converting non-LTF raises error."""
        from boofun.core.representations.ltf import LTFRepresentation
        from boofun.core.representations.truth_table import TruthTableRepresentation

        parity = bf.parity(3)
        tt = np.array(list(parity.get_representation("truth_table")), dtype=bool)

        tt_repr = TruthTableRepresentation()
        ltf_repr = LTFRepresentation()
        space = Space.BOOLEAN_CUBE

        with pytest.raises(ValueError):
            ltf_repr.convert_from(tt_repr, tt, space, 3)


class TestLTFRepresentationUtilities:
    """Test LTF utility methods."""

    def test_dump(self):
        """Test dump method."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)
        repr_obj = LTFRepresentation()

        result = repr_obj.dump(params)
        assert isinstance(result, dict)
        assert result["type"] == "ltf"

    def test_create_empty(self):
        """Test create_empty method."""
        from boofun.core.representations.ltf import LTFRepresentation

        repr_obj = LTFRepresentation()
        empty = repr_obj.create_empty(4)

        assert empty.n_vars == 4
        # Empty is constant False: all weights=0, threshold=1
        assert empty.evaluate([0, 0, 0, 0]) == False  # noqa: E712

    def test_is_complete(self):
        """Test is_complete method."""
        from boofun.core.representations.ltf import LTFParameters, LTFRepresentation

        repr_obj = LTFRepresentation()
        params = LTFParameters(weights=np.ones(3), threshold=2.0, n_vars=3)

        assert repr_obj.is_complete(params)

    def test_time_complexity_rank(self):
        """Test time_complexity_rank method."""
        from boofun.core.representations.ltf import LTFRepresentation

        repr_obj = LTFRepresentation()
        rank = repr_obj.time_complexity_rank(10)

        assert isinstance(rank, dict)
        assert "evaluation" in rank

    def test_get_storage_requirements(self):
        """Test get_storage_requirements method."""
        from boofun.core.representations.ltf import LTFRepresentation

        repr_obj = LTFRepresentation()
        req = repr_obj.get_storage_requirements(10)

        assert isinstance(req, dict)
        assert req["total_parameters"] == 11


class TestLTFUtilityFunctions:
    """Test LTF module utility functions."""

    def test_is_ltf_majority(self):
        """Test is_ltf with majority."""
        from boofun.core.representations.ltf import is_ltf

        maj_tt = list(bf.majority(3).get_representation("truth_table"))
        assert is_ltf(maj_tt, 3)

    def test_is_ltf_parity(self):
        """Test is_ltf with parity (not LTF)."""
        from boofun.core.representations.ltf import is_ltf

        parity_tt = list(bf.parity(3).get_representation("truth_table"))
        assert not is_ltf(parity_tt, 3)

    def test_create_majority_ltf(self):
        """Test create_majority_ltf function."""
        from boofun.core.representations.ltf import create_majority_ltf

        params = create_majority_ltf(5)
        assert params.n_vars == 5
        # Should have equal weights
        assert np.allclose(params.weights, np.ones(5))

    def test_create_majority_ltf_even_raises(self):
        """Test create_majority_ltf with even n raises."""
        from boofun.core.representations.ltf import create_majority_ltf

        with pytest.raises(ValueError):
            create_majority_ltf(4)

    def test_create_threshold_ltf(self):
        """Test create_threshold_ltf function."""
        from boofun.core.representations.ltf import create_threshold_ltf

        params = create_threshold_ltf(4, 2)
        assert params.n_vars == 4
        assert params.threshold == 2

    def test_create_threshold_ltf_invalid_k(self):
        """Test create_threshold_ltf with invalid k."""
        from boofun.core.representations.ltf import create_threshold_ltf

        with pytest.raises(ValueError):
            create_threshold_ltf(4, 0)  # k too small

        with pytest.raises(ValueError):
            create_threshold_ltf(4, 5)  # k too large


# =============================================================================
# Polynomial Representation Tests (target: lines 49-62, 89, 98-103, etc.)
# =============================================================================
class TestPolynomialRepresentation:
    """Test polynomial representation."""

    def test_evaluate_from_coefficients(self):
        """Test polynomial evaluation."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        space = Space.BOOLEAN_CUBE

        # Parity has all coefficients = 1 for singleton sets
        f = bf.parity(3)
        poly_data = f.get_representation("polynomial")

        # Evaluate at some inputs
        result = repr_obj.evaluate(np.array(0), poly_data, space, 3)
        assert isinstance(result, (bool, np.bool_))

    def test_evaluate_binary_vector(self):
        """Test evaluation with binary vector."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        space = Space.BOOLEAN_CUBE

        f = bf.parity(3)
        poly_data = f.get_representation("polynomial")

        # Evaluate with binary vector
        result = repr_obj.evaluate(np.array([1, 0, 0]), poly_data, space, 3)
        assert isinstance(result, (bool, np.bool_))

    def test_evaluate_array_indices(self):
        """Test evaluation with array of indices."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        space = Space.BOOLEAN_CUBE

        f = bf.parity(3)
        poly_data = f.get_representation("polynomial")

        # Evaluate with array of indices (must be different length than n_vars)
        indices = np.array([0, 1, 7, 4])
        results = repr_obj.evaluate(indices, poly_data, space, 3)
        assert len(results) == 4

    def test_evaluate_batch_binary(self):
        """Test evaluation with batch of binary vectors."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        space = Space.BOOLEAN_CUBE

        f = bf.parity(3)
        poly_data = f.get_representation("polynomial")

        # Batch of binary vectors
        batch = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]])
        results = repr_obj.evaluate(batch, poly_data, space, 3)
        assert len(results) == 3

    def test_evaluate_unsupported_shape(self):
        """Test that unsupported shape raises error."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        space = Space.BOOLEAN_CUBE

        f = bf.parity(3)
        poly_data = f.get_representation("polynomial")

        with pytest.raises(ValueError):
            repr_obj.evaluate(np.zeros((2, 2, 2)), poly_data, space, 3)

    def test_dump(self):
        """Test dump method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()

        f = bf.parity(3)
        poly_data = f.get_representation("polynomial")

        result = repr_obj.dump(poly_data)
        assert result["type"] == "polynomial"
        assert "monomials" in result

    def test_create_empty(self):
        """Test create_empty method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        empty = repr_obj.create_empty(3)

        assert empty == {}

    def test_is_complete(self):
        """Test is_complete method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()

        # Empty polynomial (constant 0) is not complete
        assert not repr_obj.is_complete({})

        # Polynomial with non-zero term is complete
        assert repr_obj.is_complete({frozenset([0]): 1})

    def test_get_degree(self):
        """Test get_degree method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()

        # Empty polynomial has degree 0
        assert repr_obj.get_degree({}) == 0

        # Single variable has degree 1
        assert repr_obj.get_degree({frozenset([0]): 1}) == 1

        # Product of 3 variables has degree 3
        assert repr_obj.get_degree({frozenset([0, 1, 2]): 1}) == 3

    def test_get_monomials(self):
        """Test get_monomials method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()

        poly = {frozenset([0]): 1, frozenset([1, 2]): 1, frozenset([0, 1]): 0}
        monomials = repr_obj.get_monomials(poly)

        assert frozenset([0]) in monomials
        assert frozenset([1, 2]) in monomials
        assert frozenset([0, 1]) not in monomials  # coefficient 0

    def test_add_polynomials(self):
        """Test polynomial addition."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()

        poly1 = {frozenset([0]): 1, frozenset([1]): 1}
        poly2 = {frozenset([1]): 1, frozenset([2]): 1}

        result = repr_obj.add_polynomials(poly1, poly2)

        # x_0 + x_1 + x_1 + x_2 = x_0 + x_2 (x_1 cancels)
        assert frozenset([0]) in result
        assert frozenset([1]) not in result
        assert frozenset([2]) in result

    def test_multiply_polynomials(self):
        """Test polynomial multiplication."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()

        poly1 = {frozenset([0]): 1}  # x_0
        poly2 = {frozenset([1]): 1}  # x_1

        result = repr_obj.multiply_polynomials(poly1, poly2)

        # x_0 * x_1 = x_0 x_1
        assert frozenset([0, 1]) in result

    def test_time_complexity_rank(self):
        """Test time_complexity_rank method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        rank = repr_obj.time_complexity_rank(10)

        assert isinstance(rank, dict)

    def test_get_storage_requirements(self):
        """Test get_storage_requirements method."""
        from boofun.core.representations.polynomial import PolynomialRepresentation

        repr_obj = PolynomialRepresentation()
        req = repr_obj.get_storage_requirements(10)

        assert "max_monomials" in req


class TestPolynomialUtilityFunctions:
    """Test polynomial utility functions."""

    def test_create_monomial(self):
        """Test create_monomial function."""
        from boofun.core.representations.polynomial import create_monomial

        mono = create_monomial([0, 2])
        assert frozenset([0, 2]) in mono
        assert mono[frozenset([0, 2])] == 1

    def test_create_constant(self):
        """Test create_constant function."""
        from boofun.core.representations.polynomial import create_constant

        const_true = create_constant(True)
        assert frozenset() in const_true

        const_false = create_constant(False)
        assert const_false == {}

    def test_create_variable(self):
        """Test create_variable function."""
        from boofun.core.representations.polynomial import create_variable

        var = create_variable(2)
        assert frozenset([2]) in var


class TestSparseTruthTableRepresentation:
    """Test sparse truth table representation."""

    def test_create_sparse(self):
        """Test creating sparse representation."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()

        # Create a mostly-zero function
        empty = repr_obj.create_empty(3)
        assert empty is not None
        assert empty["default_value"] == False  # noqa: E712
        assert empty["exceptions"] == {}

    def test_sparse_evaluate_scalar(self):
        """Test evaluating sparse representation at scalar index."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        # Create sparse data manually
        sparse_data = {
            "default_value": False,
            "exceptions": {7: True},
            "n_vars": 3,
            "size": 8,
        }

        # Test evaluation
        assert repr_obj.evaluate(np.array(0), sparse_data, space, 3) == False  # noqa
        assert repr_obj.evaluate(np.array(7), sparse_data, space, 3) == True  # noqa

    def test_sparse_evaluate_binary_vector(self):
        """Test evaluating sparse representation at binary vector."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        sparse_data = {
            "default_value": False,
            "exceptions": {7: True},
            "n_vars": 3,
            "size": 8,
        }

        # [1,1,1] -> index 7
        result = repr_obj.evaluate(np.array([1, 1, 1]), sparse_data, space, 3)
        assert result == True  # noqa: E712

    def test_sparse_evaluate_array_indices(self):
        """Test evaluating sparse representation at array of indices."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        sparse_data = {
            "default_value": False,
            "exceptions": {7: True},
            "n_vars": 3,
            "size": 8,
        }

        # Must use different length than n_vars=3 to avoid binary vector interpretation
        indices = np.array([0, 3, 7, 5])
        results = repr_obj.evaluate(indices, sparse_data, space, 3)
        assert len(results) == 4
        assert results[2] == True  # noqa: E712

    def test_sparse_evaluate_batch_binary(self):
        """Test evaluating sparse representation at batch of binary vectors."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        sparse_data = {
            "default_value": False,
            "exceptions": {7: True},
            "n_vars": 3,
            "size": 8,
        }

        batch = np.array([[0, 0, 0], [1, 1, 1]])
        results = repr_obj.evaluate(batch, sparse_data, space, 3)
        assert len(results) == 2

    def test_sparse_evaluate_index_out_of_range(self):
        """Test that out of range index raises error."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        sparse_data = {
            "default_value": False,
            "exceptions": {},
            "n_vars": 3,
            "size": 8,
        }

        with pytest.raises(IndexError):
            repr_obj.evaluate(np.array(100), sparse_data, space, 3)

    def test_sparse_evaluate_unsupported_shape(self):
        """Test that unsupported shape raises error."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        sparse_data = {
            "default_value": False,
            "exceptions": {},
            "n_vars": 3,
            "size": 8,
        }

        with pytest.raises(ValueError):
            repr_obj.evaluate(np.zeros((2, 2, 2)), sparse_data, space, 3)

    def test_sparse_dump(self):
        """Test dump method."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()

        sparse_data = {
            "default_value": False,
            "exceptions": {7: True},
            "n_vars": 3,
            "size": 8,
        }

        result = repr_obj.dump(sparse_data)
        assert result["type"] == "sparse_truth_table"
        assert "compression_ratio" in result

    def test_sparse_is_complete(self):
        """Test is_complete method."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()

        sparse_data = {
            "default_value": False,
            "exceptions": {},
            "n_vars": 3,
            "size": 8,
        }

        assert repr_obj.is_complete(sparse_data)
        assert not repr_obj.is_complete({})

    def test_sparse_time_complexity_rank(self):
        """Test time_complexity_rank method."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        rank = repr_obj.time_complexity_rank(10)

        assert isinstance(rank, dict)

    def test_sparse_get_storage_requirements(self):
        """Test get_storage_requirements method."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()
        req = repr_obj.get_storage_requirements(10)

        assert "best_case_bytes" in req

    def test_sparse_get_compression_stats(self):
        """Test get_compression_stats method."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )

        repr_obj = SparseTruthTableRepresentation()

        sparse_data = {
            "default_value": False,
            "exceptions": {7: True},
            "n_vars": 3,
            "size": 8,
        }

        stats = repr_obj.get_compression_stats(sparse_data)
        assert "compression_ratio" in stats
        assert "memory_saved" in stats

    def test_sparse_convert_from(self):
        """Test conversion from truth table to sparse."""
        from boofun.core.representations.sparse_truth_table import (
            SparseTruthTableRepresentation,
        )
        from boofun.core.representations.truth_table import TruthTableRepresentation

        tt_repr = TruthTableRepresentation()
        sparse_repr = SparseTruthTableRepresentation()
        space = Space.BOOLEAN_CUBE

        # AND function - only 1 at index 7
        tt_data = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
        sparse_data = sparse_repr.convert_from(tt_repr, tt_data, space, 3)

        assert isinstance(sparse_data, dict)
        assert "default_value" in sparse_data


class TestAdaptiveTruthTableRepresentation:
    """Test adaptive truth table representation."""

    def test_create_adaptive(self):
        """Test creating adaptive representation."""
        from boofun.core.representations.sparse_truth_table import (
            AdaptiveTruthTableRepresentation,
        )

        repr_obj = AdaptiveTruthTableRepresentation()
        empty = repr_obj.create_empty(3)

        assert empty["format"] == "sparse"

    def test_adaptive_is_complete(self):
        """Test is_complete method."""
        from boofun.core.representations.sparse_truth_table import (
            AdaptiveTruthTableRepresentation,
        )

        repr_obj = AdaptiveTruthTableRepresentation()
        empty = repr_obj.create_empty(3)

        assert repr_obj.is_complete(empty)

    def test_adaptive_time_complexity_rank(self):
        """Test time_complexity_rank method."""
        from boofun.core.representations.sparse_truth_table import (
            AdaptiveTruthTableRepresentation,
        )

        repr_obj = AdaptiveTruthTableRepresentation()
        rank = repr_obj.time_complexity_rank(10)

        assert isinstance(rank, dict)

    def test_adaptive_get_storage_requirements(self):
        """Test get_storage_requirements method."""
        from boofun.core.representations.sparse_truth_table import (
            AdaptiveTruthTableRepresentation,
        )

        repr_obj = AdaptiveTruthTableRepresentation()
        req = repr_obj.get_storage_requirements(10)

        assert "optimal" in req


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
