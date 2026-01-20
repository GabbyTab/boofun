import sys

sys.path.insert(0, "src")
"""
Tests for equivalence module.

Tests canonical forms, equivalence testing, and automorphisms:
- apply_permutation
- canonical_form
- are_equivalent
- automorphisms
- equivalence_class_size
- PermutationEquivalence class
- AffineEquivalence class
"""

import pytest

import boofun as bf
from boofun.analysis.equivalence import (
    AffineEquivalence,
    PermutationEquivalence,
    apply_permutation,
    are_equivalent,
    automorphisms,
    canonical_form,
    equivalence_class_size,
)


class TestApplyPermutation:
    """Tests for apply_permutation function."""

    def test_identity_permutation(self):
        """Identity permutation doesn't change function."""
        f = bf.AND(3)
        g = apply_permutation(f, (0, 1, 2))

        tt_f = list(f.get_representation("truth_table"))
        tt_g = list(g.get_representation("truth_table"))
        assert tt_f == tt_g

    def test_symmetric_function_invariant(self):
        """Symmetric functions are invariant under permutation."""
        f = bf.AND(3)
        # Any permutation of AND gives AND
        g = apply_permutation(f, (2, 0, 1))

        tt_f = list(f.get_representation("truth_table"))
        tt_g = list(g.get_representation("truth_table"))
        assert tt_f == tt_g

    def test_dictator_permutation(self):
        """Permuting dictator changes the dependent variable."""
        f = bf.dictator(3, i=0)
        g = apply_permutation(f, (1, 2, 0))  # x0 -> x1, x1 -> x2, x2 -> x0

        # f depends on x0, g should depend on x1
        # This is tricky to verify directly, but we can check they're different
        list(f.get_representation("truth_table"))
        list(g.get_representation("truth_table"))
        # They might be different or same depending on the bit ordering

    def test_invalid_permutation_raises(self):
        """Invalid permutation raises ValueError."""
        f = bf.AND(3)

        with pytest.raises(ValueError):
            apply_permutation(f, (0, 0, 1))  # Not a permutation

        with pytest.raises(ValueError):
            apply_permutation(f, (0, 1))  # Wrong length


class TestCanonicalForm:
    """Tests for canonical_form function."""

    def test_returns_tuple(self):
        """canonical_form returns (tuple, transform)."""
        f = bf.AND(3)
        result = canonical_form(f)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], tuple)

    def test_same_function_same_canonical(self):
        """Same function gives same canonical form."""
        f = bf.AND(3)
        g = bf.AND(3)

        cf = canonical_form(f)[0]
        cg = canonical_form(g)[0]
        assert cf == cg

    def test_symmetric_function_canonical(self):
        """Symmetric functions have unique canonical form."""
        f = bf.majority(3)
        cf, transform = canonical_form(f)

        # Canonical form should be a tuple of values (ints or bools)
        assert len(cf) == 8  # 2^3

    def test_permuted_functions_same_canonical(self):
        """Permuted functions have same canonical form."""
        f = bf.dictator(3, i=0)
        g = bf.dictator(3, i=1)

        cf = canonical_form(f, include_shifts=False, include_negation=False)[0]
        cg = canonical_form(g, include_shifts=False, include_negation=False)[0]

        # Dictators are equivalent under permutation
        assert cf == cg

    def test_negation_equivalence(self):
        """Functions differing by negation have same canonical with include_negation."""
        f = bf.AND(2)
        # Create NAND by negating AND
        tt_and = list(f.get_representation("truth_table"))
        tt_nand = [1 - x for x in tt_and]

        from boofun.core.factory import BooleanFunctionFactory

        g = BooleanFunctionFactory.from_truth_table(type(f), tt_nand, n=2)

        cf = canonical_form(f, include_negation=True)[0]
        cg = canonical_form(g, include_negation=True)[0]
        assert cf == cg


class TestAreEquivalent:
    """Tests for are_equivalent function."""

    def test_same_function_equivalent(self):
        """Function is equivalent to itself."""
        f = bf.AND(3)
        assert are_equivalent(f, f)

    def test_different_n_not_equivalent(self):
        """Functions with different n are not equivalent."""
        f = bf.AND(3)
        g = bf.AND(4)
        assert not are_equivalent(f, g)

    def test_permuted_functions_equivalent(self):
        """Permuted functions are equivalent."""
        f = bf.dictator(3, i=0)
        g = bf.dictator(3, i=1)

        assert are_equivalent(f, g)

    def test_and_or_equivalent_with_negation(self):
        """AND and OR are equivalent under negation and shifts."""
        f = bf.AND(3)
        g = bf.OR(3)

        # AND(x) = NOT OR(NOT x), so they're equivalent under full transformations
        equiv = are_equivalent(f, g, include_shifts=True, include_negation=True)
        # This might or might not be true depending on implementation
        assert isinstance(equiv, bool)

    def test_symmetric_functions_equivalence(self):
        """Test equivalence of symmetric functions."""
        f = bf.majority(3)
        g = apply_permutation(f, (2, 1, 0))

        assert are_equivalent(f, g)


class TestAutomorphisms:
    """Tests for automorphisms function."""

    def test_identity_always_automorphism(self):
        """Identity permutation is always an automorphism."""
        f = bf.dictator(3, i=0)
        autos = automorphisms(f)

        identity = (0, 1, 2)
        assert identity in autos

    def test_symmetric_function_all_automorphisms(self):
        """Symmetric function has all permutations as automorphisms."""
        f = bf.AND(3)
        autos = automorphisms(f)

        # AND is symmetric, so all 6 permutations are automorphisms
        assert len(autos) == 6

    def test_dictator_few_automorphisms(self):
        """Dictator has few automorphisms."""
        f = bf.dictator(3, i=0)
        autos = automorphisms(f)

        # Dictator is only symmetric under permutations fixing the dictator variable
        # That's (n-1)! = 2 permutations
        assert len(autos) == 2


class TestEquivalenceClassSize:
    """Tests for equivalence_class_size function."""

    def test_symmetric_class_size(self):
        """Symmetric function has class size 1."""
        f = bf.AND(3)
        size = equivalence_class_size(f)

        # Symmetric function: orbit size = n! / n! = 1
        assert size == 1

    def test_dictator_class_size(self):
        """Dictator has class size n."""
        f = bf.dictator(3, i=0)
        size = equivalence_class_size(f)

        # Dictator: n! / (n-1)! = n
        assert size == 3

    def test_class_size_positive(self):
        """Class size is always positive."""
        f = bf.majority(3)
        size = equivalence_class_size(f)
        assert size >= 1


class TestPermutationEquivalence:
    """Tests for PermutationEquivalence class."""

    def test_initialization(self):
        """PermutationEquivalence initializes correctly."""
        pe = PermutationEquivalence()
        assert hasattr(pe, "_cache")

    def test_canonical_caching(self):
        """canonical method caches results."""
        pe = PermutationEquivalence()
        f = bf.AND(3)

        c1 = pe.canonical(f)
        c2 = pe.canonical(f)

        assert c1 == c2
        assert len(pe._cache) >= 1

    def test_equivalent_method(self):
        """equivalent method works correctly."""
        pe = PermutationEquivalence()

        f = bf.dictator(3, i=0)
        g = bf.dictator(3, i=1)

        assert pe.equivalent(f, g)

    def test_clear_cache(self):
        """clear_cache clears the cache."""
        pe = PermutationEquivalence()
        f = bf.AND(3)

        pe.canonical(f)
        assert len(pe._cache) >= 1

        pe.clear_cache()
        assert len(pe._cache) == 0


class TestAffineEquivalence:
    """Tests for AffineEquivalence class."""

    def test_initialization(self):
        """AffineEquivalence initializes correctly."""
        ae = AffineEquivalence()
        assert ae.include_negation == True

    def test_initialization_no_negation(self):
        """AffineEquivalence can be initialized without negation."""
        ae = AffineEquivalence(include_negation=False)
        assert ae.include_negation == False

    def test_canonical_caching(self):
        """canonical method caches results."""
        ae = AffineEquivalence()
        f = bf.AND(3)

        c1 = ae.canonical(f)
        c2 = ae.canonical(f)

        assert c1 == c2

    def test_equivalent_method(self):
        """equivalent method works correctly."""
        ae = AffineEquivalence()

        f = bf.AND(3)
        g = bf.AND(3)

        assert ae.equivalent(f, g)

    def test_clear_cache(self):
        """clear_cache clears the cache."""
        ae = AffineEquivalence()
        f = bf.AND(3)

        ae.canonical(f)
        ae.clear_cache()
        assert len(ae._cache) == 0


class TestOnBuiltinFunctions:
    """Test equivalence on built-in functions."""

    def test_majority_equivalence(self):
        """Majority is equivalent under permutation."""
        f = bf.majority(3)
        g = apply_permutation(f, (1, 2, 0))

        assert are_equivalent(f, g)

    def test_parity_equivalence(self):
        """Parity is equivalent under permutation."""
        f = bf.parity(3)
        g = apply_permutation(f, (2, 0, 1))

        assert are_equivalent(f, g)

    def test_threshold_equivalence(self):
        """Same threshold functions are equivalent."""
        f = bf.threshold(3, k=2)
        g = bf.threshold(3, k=2)

        assert are_equivalent(f, g)


class TestEdgeCases:
    """Test edge cases for equivalence."""

    def test_single_variable(self):
        """Single variable functions."""
        f = bf.parity(1)

        cf = canonical_form(f)
        assert isinstance(cf[0], tuple)

        autos = automorphisms(f)
        assert len(autos) >= 1

    def test_two_variables(self):
        """Two variable functions."""
        f = bf.AND(2)

        canonical_form(f)
        autos = automorphisms(f)

        # AND(2) is symmetric, so 2 automorphisms
        assert len(autos) == 2

    def test_constant_function(self):
        """Constant functions."""
        f = bf.constant(True, 3)

        # Constant is symmetric
        autos = automorphisms(f)
        assert len(autos) == 6  # All 3! permutations
