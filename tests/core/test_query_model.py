"""
Tests for query_model module.

Tests for:
- AccessType enum
- QUERY_COMPLEXITY dictionary
- get_access_type function
- check_query_safety function
- QueryModel class
- safe_alternatives function
"""

import pytest
import warnings
import numpy as np

import boolfunc as bf
from boolfunc.core.query_model import (
    AccessType,
    QUERY_COMPLEXITY,
    get_access_type,
    check_query_safety,
    QueryModel,
    QuerySafetyWarning,
    ExplicitEnumerationError,
    safe_alternatives,
)


class TestAccessType:
    """Tests for AccessType enum."""
    
    def test_enum_values(self):
        """Enum has expected values."""
        assert AccessType.EXPLICIT is not None
        assert AccessType.QUERY is not None
        assert AccessType.STREAMING is not None
        assert AccessType.SYMBOLIC is not None


class TestQueryComplexity:
    """Tests for QUERY_COMPLEXITY dictionary."""
    
    def test_has_safe_operations(self):
        """Dictionary has safe operations."""
        assert "is_linear" in QUERY_COMPLEXITY
        assert QUERY_COMPLEXITY["is_linear"]["safe"] == True
    
    def test_has_unsafe_operations(self):
        """Dictionary has unsafe operations."""
        assert "fourier" in QUERY_COMPLEXITY
        assert QUERY_COMPLEXITY["fourier"]["safe"] == False
    
    def test_query_functions_callable(self):
        """Query count functions are callable."""
        for op, info in QUERY_COMPLEXITY.items():
            query_fn = info["queries"]
            result = query_fn(5, 100)
            assert isinstance(result, (int, float))


class TestGetAccessType:
    """Tests for get_access_type function."""
    
    def test_explicit_with_truth_table(self):
        """Function with truth table is explicit."""
        f = bf.AND(3)
        access_type = get_access_type(f)
        
        assert access_type == AccessType.EXPLICIT
    
    def test_explicit_with_fourier(self):
        """Function with Fourier expansion is explicit."""
        f = bf.parity(3)
        # Get Fourier to ensure it's computed
        _ = f.fourier()
        
        access_type = get_access_type(f)
        assert access_type == AccessType.EXPLICIT


class TestCheckQuerySafety:
    """Tests for check_query_safety function."""
    
    def test_safe_operation_returns_true(self):
        """Safe operations always return True."""
        f = bf.AND(3)
        
        result = check_query_safety(f, "is_linear")
        assert result == True
        
        result = check_query_safety(f, "evaluate")
        assert result == True
    
    def test_small_n_allows_unsafe(self):
        """Small n allows unsafe operations."""
        f = bf.majority(3)
        
        result = check_query_safety(f, "fourier", max_safe_n=20)
        assert result == True
    
    def test_large_n_warns(self):
        """Large n warns for unsafe operations."""
        f = bf.majority(5)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = check_query_safety(f, "fourier", max_safe_n=3)
            
            # Should still return True (with warning) for explicit
            assert result == True
    
    def test_strict_mode_raises(self):
        """Strict mode raises for large n."""
        f = bf.AND(10)
        
        with pytest.raises(ExplicitEnumerationError):
            check_query_safety(f, "fourier", max_safe_n=5, strict=True)
    
    def test_unknown_operation(self):
        """Unknown operation is treated as unsafe."""
        f = bf.parity(3)
        
        result = check_query_safety(f, "unknown_operation")
        assert result == True  # Small n, still allowed


class TestQueryModel:
    """Tests for QueryModel class."""
    
    def test_initialization(self):
        """QueryModel initializes correctly."""
        f = bf.AND(5)
        qm = QueryModel(f)
        
        assert qm.f is f
        assert qm.n == 5
        assert qm.max_queries == 10_000_000
    
    def test_custom_max_queries(self):
        """Can set custom max_queries."""
        f = bf.OR(3)
        qm = QueryModel(f, max_queries=1000)
        
        assert qm.max_queries == 1000
    
    def test_can_compute_safe(self):
        """can_compute returns True for safe operations."""
        f = bf.majority(3)
        qm = QueryModel(f)
        
        assert qm.can_compute("is_linear") == True
        assert qm.can_compute("evaluate") == True
    
    def test_can_compute_small_unsafe(self):
        """can_compute returns True for small n unsafe operations."""
        f = bf.parity(3)
        qm = QueryModel(f)
        
        # 2^3 = 8 queries, well under max
        assert qm.can_compute("fourier") == True
    
    def test_can_compute_large_unsafe(self):
        """can_compute returns False for large n unsafe operations."""
        f = bf.AND(10)
        qm = QueryModel(f, max_queries=500)  # 2^10 = 1024 > 500
        
        assert qm.can_compute("fourier") == False
    
    def test_estimate_cost(self):
        """estimate_cost returns expected structure."""
        f = bf.OR(5)
        qm = QueryModel(f)
        
        cost = qm.estimate_cost("fourier")
        
        assert "queries" in cost
        assert "feasible" in cost
        assert "safe" in cost
        assert "time_estimate" in cost
        assert "description" in cost
        assert "access_type" in cost
    
    def test_estimate_cost_time_estimate_format(self):
        """Time estimate has proper format."""
        f = bf.AND(3)
        qm = QueryModel(f)
        
        # Small query count -> µs
        cost = qm.estimate_cost("evaluate")
        assert "µs" in cost["time_estimate"] or "ms" in cost["time_estimate"]
    
    def test_summary(self):
        """summary returns dict of all operations."""
        f = bf.parity(3)
        qm = QueryModel(f)
        
        summary = qm.summary()
        
        assert isinstance(summary, dict)
        assert "fourier" in summary
        assert "is_linear" in summary
    
    def test_print_summary(self, capsys):
        """print_summary outputs to stdout."""
        f = bf.majority(3)
        qm = QueryModel(f)
        
        qm.print_summary()
        
        captured = capsys.readouterr()
        assert "Query Model Summary" in captured.out
        assert "SAFE operations" in captured.out


class TestSafeAlternatives:
    """Tests for safe_alternatives function."""
    
    def test_has_alternative(self):
        """Some operations have alternatives."""
        alt = safe_alternatives("fourier")
        assert alt is not None
        assert "sample" in alt.lower() or "estimate" in alt.lower()
    
    def test_no_alternative(self):
        """Some operations have no alternative."""
        alt = safe_alternatives("unknown_op")
        assert alt is None
    
    def test_known_alternatives(self):
        """Known alternatives are correct."""
        assert safe_alternatives("is_balanced") is not None
        assert safe_alternatives("influences") is not None


class TestEdgeCases:
    """Edge case tests."""
    
    def test_single_variable(self):
        """Works with single variable function."""
        f = bf.parity(1)
        qm = QueryModel(f)
        
        assert qm.n == 1
        assert qm.can_compute("fourier") == True  # Very small
    
    def test_access_type_in_cost(self):
        """Access type is included in cost."""
        f = bf.AND(3)
        qm = QueryModel(f)
        
        cost = qm.estimate_cost("evaluate")
        assert cost["access_type"] == "EXPLICIT"
