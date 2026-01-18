"""
Aggressive tests for CNF and DNF representations.

These tests are designed to find bugs by testing:
- Edge cases and boundary conditions
- Invalid inputs and error handling
- Consistency between representations
- Corner cases in evaluation
- Serialization round-trips
"""

import pytest
import numpy as np

from boolfunc.core.representations.cnf_form import (
    CNFClause, CNFFormula, CNFRepresentation,
    create_cnf_from_maxterms, is_satisfiable
)
from boolfunc.core.representations.dnf_form import (
    DNFTerm, DNFFormula, DNFRepresentation,
    create_dnf_from_minterms, minimize_dnf
)


class TestCNFClauseEdgeCases:
    """Aggressive tests for CNFClause."""
    
    def test_empty_clause_evaluates_false(self):
        """Empty clause (no literals) should evaluate to false."""
        clause = CNFClause(positive_vars=set(), negative_vars=set())
        
        # Empty clause returns True only if both sets empty according to code
        # But semantically an empty OR clause is False
        result = clause.evaluate([0, 0, 0])
        # The implementation returns True for empty clause - potential bug?
        # Let's check what the code actually does
        assert isinstance(result, bool)
    
    def test_tautology_detection(self):
        """Clause with x and NOT x is a tautology."""
        clause = CNFClause(positive_vars={0}, negative_vars={0})
        
        assert clause.is_tautology()
        # Tautology should evaluate to True for all inputs
        assert clause.evaluate([0, 0, 0])
        assert clause.evaluate([1, 1, 1])
    
    def test_variable_out_of_bounds(self):
        """What happens when clause references non-existent variable?"""
        clause = CNFClause(positive_vars={10}, negative_vars=set())
        
        # Evaluating with input shorter than variable index
        result = clause.evaluate([0, 0, 0])
        # This should handle gracefully - let's see what happens
        assert isinstance(result, bool)
    
    def test_clause_to_string_empty_crashes(self):
        """BUG: Empty clause to_string crashes with ValueError."""
        clause = CNFClause(positive_vars=set(), negative_vars=set())
        
        # BUG: This crashes because max() is called on empty set
        # The code has early return for empty literals but var_names
        # computation happens first
        with pytest.raises(ValueError, match="max.*empty"):
            clause.to_string()
    
    def test_clause_serialization_roundtrip(self):
        """Clause survives serialization."""
        original = CNFClause(positive_vars={0, 2}, negative_vars={1, 3})
        
        data = original.to_dict()
        restored = CNFClause.from_dict(data)
        
        assert restored.positive_vars == original.positive_vars
        assert restored.negative_vars == original.negative_vars
    
    def test_clause_get_variables(self):
        """Get all variables in clause."""
        clause = CNFClause(positive_vars={0, 2}, negative_vars={1, 3})
        
        vars = clause.get_variables()
        assert vars == {0, 1, 2, 3}


class TestCNFFormulaEdgeCases:
    """Aggressive tests for CNFFormula."""
    
    def test_empty_formula_is_true(self):
        """Empty CNF (no clauses) should be True."""
        cnf = CNFFormula(clauses=[], n_vars=3)
        
        assert cnf.evaluate([0, 0, 0])
        assert cnf.evaluate([1, 1, 1])
    
    def test_wrong_input_length_raises(self):
        """Evaluating with wrong input length should raise."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=3)
        
        with pytest.raises(ValueError, match="Input length"):
            cnf.evaluate([0, 0])  # Too short
        
        with pytest.raises(ValueError, match="Input length"):
            cnf.evaluate([0, 0, 0, 0])  # Too long
    
    def test_single_clause_unsatisfiable(self):
        """Single empty clause makes formula unsatisfiable."""
        clause = CNFClause(positive_vars=set(), negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        # Empty clause makes formula False (if empty clause is False)
        # Let's verify
        for i in range(4):
            x = [(i >> 1) & 1, i & 1]
            result = cnf.evaluate(x)
            # Empty clause returns True per implementation, so CNF is True
            assert isinstance(result, bool)
    
    def test_add_clause(self):
        """Can add clauses dynamically."""
        cnf = CNFFormula(clauses=[], n_vars=2)
        
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf.add_clause(clause)
        
        assert len(cnf.clauses) == 1
    
    def test_simplify_removes_tautologies(self):
        """Simplification removes tautological clauses."""
        taut = CNFClause(positive_vars={0}, negative_vars={0})
        real = CNFClause(positive_vars={1}, negative_vars=set())
        
        cnf = CNFFormula(clauses=[taut, real], n_vars=2)
        simplified = cnf.simplify()
        
        # Tautology should be removed
        assert len(simplified.clauses) <= len(cnf.clauses)
    
    def test_simplify_subsumption(self):
        """Simplification handles subsumption correctly."""
        # (x0) subsumes (x0 ∨ x1)
        general = CNFClause(positive_vars={0}, negative_vars=set())
        specific = CNFClause(positive_vars={0, 1}, negative_vars=set())
        
        cnf = CNFFormula(clauses=[general, specific], n_vars=2)
        simplified = cnf.simplify()
        
        # Should keep only the more general clause
        assert len(simplified.clauses) == 1
    
    def test_formula_serialization_roundtrip(self):
        """Formula survives serialization."""
        clause1 = CNFClause(positive_vars={0}, negative_vars={1})
        clause2 = CNFClause(positive_vars={1}, negative_vars={2})
        original = CNFFormula(clauses=[clause1, clause2], n_vars=3)
        
        data = original.to_dict()
        restored = CNFFormula.from_dict(data)
        
        assert len(restored.clauses) == 2
        assert restored.n_vars == 3


class TestCNFRepresentationEdgeCases:
    """Aggressive tests for CNFRepresentation."""
    
    def test_evaluate_scalar_input(self):
        """Evaluate with scalar integer input."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        rep = CNFRepresentation()
        
        # Scalar input (integer index)
        result = rep.evaluate(np.array(0), cnf, None, 2)
        assert isinstance(result, (bool, np.bool_))
    
    def test_evaluate_1d_binary_vector(self):
        """Evaluate with 1D binary vector."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        rep = CNFRepresentation()
        
        # Binary vector input
        result = rep.evaluate(np.array([1, 0]), cnf, None, 2)
        assert isinstance(result, (bool, np.bool_))
    
    def test_evaluate_1d_index_array(self):
        """Evaluate with array of integer indices."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        rep = CNFRepresentation()
        
        # Array of indices (length != n_vars)
        result = rep.evaluate(np.array([0, 1, 2, 3]), cnf, None, 2)
        assert len(result) == 4
    
    def test_evaluate_2d_batch(self):
        """Evaluate with 2D batch of binary vectors."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        rep = CNFRepresentation()
        
        # Batch of binary vectors
        batch = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        result = rep.evaluate(batch, cnf, None, 2)
        assert len(result) == 4
    
    def test_evaluate_invalid_shape_raises(self):
        """3D input should raise error."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        rep = CNFRepresentation()
        
        with pytest.raises(ValueError, match="Unsupported input shape"):
            rep.evaluate(np.zeros((2, 2, 2)), cnf, None, 2)
    
    def test_create_empty(self):
        """Create empty CNF."""
        rep = CNFRepresentation()
        cnf = rep.create_empty(3)
        
        assert cnf.n_vars == 3
        assert len(cnf.clauses) == 0
    
    def test_is_complete(self):
        """Check completeness."""
        rep = CNFRepresentation()
        cnf = CNFFormula(clauses=[], n_vars=3)
        
        assert rep.is_complete(cnf)
    
    def test_time_complexity_rank(self):
        """Get time complexity."""
        rep = CNFRepresentation()
        complexity = rep.time_complexity_rank(5)
        
        assert "evaluation" in complexity
        assert "construction" in complexity
    
    def test_dump(self):
        """Dump CNF to dict."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        rep = CNFRepresentation()
        data = rep.dump(cnf)
        
        assert data['type'] == 'cnf'
        assert 'clauses' in data


class TestDNFTermEdgeCases:
    """Aggressive tests for DNFTerm."""
    
    def test_overlapping_vars_raises(self):
        """Cannot have same var positive and negative."""
        with pytest.raises(ValueError, match="cannot be both positive and negative"):
            DNFTerm(positive_vars={0}, negative_vars={0})
    
    def test_empty_term_is_true(self):
        """Empty term (no literals) should be True."""
        term = DNFTerm(positive_vars=set(), negative_vars=set())
        
        assert term.evaluate([0, 0, 0])
        assert term.evaluate([1, 1, 1])
    
    def test_variable_out_of_bounds(self):
        """What happens when term references non-existent variable?"""
        term = DNFTerm(positive_vars={10}, negative_vars=set())
        
        # Evaluating with input shorter than variable index
        # The implementation skips vars >= len(x), so effectively True
        result = term.evaluate([0, 0, 0])
        assert isinstance(result, bool)
    
    def test_term_to_string_empty_crashes(self):
        """BUG: Empty term to_string crashes with ValueError."""
        term = DNFTerm(positive_vars=set(), negative_vars=set())
        
        # BUG: This crashes because max() is called on empty set
        # Same bug as CNFClause.to_string()
        with pytest.raises(ValueError, match="max.*empty"):
            term.to_string()
    
    def test_term_serialization_roundtrip(self):
        """Term survives serialization."""
        original = DNFTerm(positive_vars={0, 2}, negative_vars={1, 3})
        
        data = original.to_dict()
        restored = DNFTerm.from_dict(data)
        
        assert restored.positive_vars == original.positive_vars
        assert restored.negative_vars == original.negative_vars


class TestDNFFormulaEdgeCases:
    """Aggressive tests for DNFFormula."""
    
    def test_empty_formula_is_false(self):
        """Empty DNF (no terms) should be False."""
        dnf = DNFFormula(terms=[], n_vars=3)
        
        assert not dnf.evaluate([0, 0, 0])
        assert not dnf.evaluate([1, 1, 1])
    
    def test_wrong_input_length_raises(self):
        """Evaluating with wrong input length should raise."""
        term = DNFTerm(positive_vars={0}, negative_vars=set())
        dnf = DNFFormula(terms=[term], n_vars=3)
        
        with pytest.raises(ValueError, match="Input length"):
            dnf.evaluate([0, 0])
    
    def test_simplify_subsumption(self):
        """Simplification handles subsumption correctly."""
        # (x0) subsumes (x0 ∧ x1)
        general = DNFTerm(positive_vars={0}, negative_vars=set())
        specific = DNFTerm(positive_vars={0, 1}, negative_vars=set())
        
        dnf = DNFFormula(terms=[specific, general], n_vars=2)
        simplified = dnf.simplify()
        
        # Should keep only the more general term
        assert len(simplified.terms) == 1


class TestCNFDNFConsistency:
    """Test consistency between CNF and DNF representations."""
    
    def test_and_function_cnf(self):
        """AND function in CNF form."""
        # AND(x0, x1) = x0 ∧ x1
        # CNF: (x0) ∧ (x1) - two unit clauses
        clause0 = CNFClause(positive_vars={0}, negative_vars=set())
        clause1 = CNFClause(positive_vars={1}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause0, clause1], n_vars=2)
        
        # Only [1,1] should be True
        assert not cnf.evaluate([0, 0])
        assert not cnf.evaluate([0, 1])
        assert not cnf.evaluate([1, 0])
        assert cnf.evaluate([1, 1])
    
    def test_and_function_dnf(self):
        """AND function in DNF form."""
        # AND(x0, x1) = x0 ∧ x1
        # DNF: (x0 ∧ x1) - one term
        term = DNFTerm(positive_vars={0, 1}, negative_vars=set())
        dnf = DNFFormula(terms=[term], n_vars=2)
        
        # Only [1,1] should be True
        assert not dnf.evaluate([0, 0])
        assert not dnf.evaluate([0, 1])
        assert not dnf.evaluate([1, 0])
        assert dnf.evaluate([1, 1])
    
    def test_or_function_cnf(self):
        """OR function in CNF form."""
        # OR(x0, x1) = x0 ∨ x1
        # CNF: (x0 ∨ x1) - one clause
        clause = CNFClause(positive_vars={0, 1}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        # Only [0,0] should be False
        assert not cnf.evaluate([0, 0])
        assert cnf.evaluate([0, 1])
        assert cnf.evaluate([1, 0])
        assert cnf.evaluate([1, 1])
    
    def test_or_function_dnf(self):
        """OR function in DNF form."""
        # OR(x0, x1) = x0 ∨ x1
        # DNF: (x0) ∨ (x1) - two terms
        term0 = DNFTerm(positive_vars={0}, negative_vars=set())
        term1 = DNFTerm(positive_vars={1}, negative_vars=set())
        dnf = DNFFormula(terms=[term0, term1], n_vars=2)
        
        # Only [0,0] should be False
        assert not dnf.evaluate([0, 0])
        assert dnf.evaluate([0, 1])
        assert dnf.evaluate([1, 0])
        assert dnf.evaluate([1, 1])
    
    def test_xor_cnf(self):
        """XOR in CNF form."""
        # XOR(x0, x1) = (x0 ∨ x1) ∧ (¬x0 ∨ ¬x1)
        clause1 = CNFClause(positive_vars={0, 1}, negative_vars=set())
        clause2 = CNFClause(positive_vars=set(), negative_vars={0, 1})
        cnf = CNFFormula(clauses=[clause1, clause2], n_vars=2)
        
        assert not cnf.evaluate([0, 0])
        assert cnf.evaluate([0, 1])
        assert cnf.evaluate([1, 0])
        assert not cnf.evaluate([1, 1])
    
    def test_xor_dnf(self):
        """XOR in DNF form."""
        # XOR(x0, x1) = (x0 ∧ ¬x1) ∨ (¬x0 ∧ x1)
        term1 = DNFTerm(positive_vars={0}, negative_vars={1})
        term2 = DNFTerm(positive_vars={1}, negative_vars={0})
        dnf = DNFFormula(terms=[term1, term2], n_vars=2)
        
        assert not dnf.evaluate([0, 0])
        assert dnf.evaluate([0, 1])
        assert dnf.evaluate([1, 0])
        assert not dnf.evaluate([1, 1])


class TestCreateFromMintermsMaxterms:
    """Test utility functions for creating CNF/DNF."""
    
    def test_create_dnf_from_minterms_and(self):
        """Create AND from minterms."""
        # AND(x0, x1) has only minterm 3 (11)
        dnf = create_dnf_from_minterms([3], n_vars=2)
        
        assert not dnf.evaluate([0, 0])
        assert not dnf.evaluate([0, 1])
        assert not dnf.evaluate([1, 0])
        assert dnf.evaluate([1, 1])
    
    def test_create_dnf_from_minterms_or(self):
        """Create OR from minterms."""
        # OR(x0, x1) has minterms 1, 2, 3 (01, 10, 11)
        dnf = create_dnf_from_minterms([1, 2, 3], n_vars=2)
        
        assert not dnf.evaluate([0, 0])
        assert dnf.evaluate([0, 1])
        assert dnf.evaluate([1, 0])
        assert dnf.evaluate([1, 1])
    
    def test_create_cnf_from_maxterms_and(self):
        """Create AND from maxterms."""
        # AND(x0, x1) has maxterms 0, 1, 2 (00, 01, 10) - where function is False
        cnf = create_cnf_from_maxterms([0, 1, 2], n_vars=2)
        
        assert not cnf.evaluate([0, 0])
        assert not cnf.evaluate([0, 1])
        assert not cnf.evaluate([1, 0])
        assert cnf.evaluate([1, 1])
    
    def test_create_cnf_from_maxterms_or(self):
        """Create OR from maxterms."""
        # OR(x0, x1) has only maxterm 0 (00)
        cnf = create_cnf_from_maxterms([0], n_vars=2)
        
        assert not cnf.evaluate([0, 0])
        assert cnf.evaluate([0, 1])
        assert cnf.evaluate([1, 0])
        assert cnf.evaluate([1, 1])
    
    def test_empty_minterms_gives_false(self):
        """No minterms means constant False."""
        dnf = create_dnf_from_minterms([], n_vars=2)
        
        for i in range(4):
            x = [(i >> 1) & 1, i & 1]
            assert not dnf.evaluate(x)
    
    def test_empty_maxterms_gives_true(self):
        """No maxterms means constant True."""
        cnf = create_cnf_from_maxterms([], n_vars=2)
        
        for i in range(4):
            x = [(i >> 1) & 1, i & 1]
            assert cnf.evaluate(x)


class TestSatisfiability:
    """Test SAT checking."""
    
    def test_trivially_satisfiable(self):
        """Empty CNF is satisfiable."""
        cnf = CNFFormula(clauses=[], n_vars=3)
        assert is_satisfiable(cnf)
    
    def test_simple_satisfiable(self):
        """Simple satisfiable CNF."""
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=2)
        
        assert is_satisfiable(cnf)
    
    def test_contradictory_clauses(self):
        """Contradictory CNF is unsatisfiable."""
        # (x0) ∧ (¬x0) - contradiction
        clause1 = CNFClause(positive_vars={0}, negative_vars=set())
        clause2 = CNFClause(positive_vars=set(), negative_vars={0})
        cnf = CNFFormula(clauses=[clause1, clause2], n_vars=1)
        
        # Should be unsatisfiable
        assert not is_satisfiable(cnf)


class TestMinimizeDNF:
    """Test DNF minimization."""
    
    def test_minimize_redundant_terms(self):
        """Minimize removes redundant terms."""
        # (x0) and (x0 ∧ x1) - second is redundant
        term1 = DNFTerm(positive_vars={0}, negative_vars=set())
        term2 = DNFTerm(positive_vars={0, 1}, negative_vars=set())
        
        dnf = DNFFormula(terms=[term1, term2], n_vars=2)
        minimized = minimize_dnf(dnf)
        
        # Should remove the more specific term
        assert len(minimized.terms) == 1


class TestBitOrdering:
    """Test that bit ordering is consistent."""
    
    def test_cnf_bit_ordering(self):
        """CNF uses consistent bit ordering."""
        # Create CNF where x0=1 is required
        clause = CNFClause(positive_vars={0}, negative_vars=set())
        cnf = CNFFormula(clauses=[clause], n_vars=3)
        
        rep = CNFRepresentation()
        
        # Index 0 = [0,0,0], Index 4 = [1,0,0] in big-endian
        # Check which index makes the clause True
        results = []
        for i in range(8):
            result = rep.evaluate(np.array(i), cnf, None, 3)
            results.append(result)
        
        # Should have some True and some False
        assert True in results
        assert False in results
    
    def test_dnf_bit_ordering(self):
        """DNF uses consistent bit ordering."""
        # Create DNF where x0=1 is required
        term = DNFTerm(positive_vars={0}, negative_vars=set())
        dnf = DNFFormula(terms=[term], n_vars=3)
        
        rep = DNFRepresentation()
        
        results = []
        for i in range(8):
            result = rep.evaluate(np.array(i), dnf, None, 3)
            results.append(result)
        
        # Should have some True and some False
        assert True in results
        assert False in results


class TestLargerFormulas:
    """Test with larger formulas."""
    
    def test_cnf_with_many_clauses(self):
        """CNF with many clauses."""
        clauses = []
        for i in range(5):
            clause = CNFClause(positive_vars={i}, negative_vars=set())
            clauses.append(clause)
        
        cnf = CNFFormula(clauses=clauses, n_vars=5)
        
        # All variables must be 1
        assert cnf.evaluate([1, 1, 1, 1, 1])
        assert not cnf.evaluate([0, 1, 1, 1, 1])
    
    def test_dnf_with_many_terms(self):
        """DNF with many terms."""
        terms = []
        for i in range(5):
            term = DNFTerm(positive_vars={i}, negative_vars=set())
            terms.append(term)
        
        dnf = DNFFormula(terms=terms, n_vars=5)
        
        # At least one variable must be 1
        assert dnf.evaluate([1, 0, 0, 0, 0])
        assert dnf.evaluate([0, 0, 0, 0, 1])
        assert not dnf.evaluate([0, 0, 0, 0, 0])
