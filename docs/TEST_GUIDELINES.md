# Test Guidelines

How to write tests that actually catch bugs.

## The Convolution Bug: A Case Study

### What Happened
```python
# The bug: silently thresholded real values to Boolean
result_tt = (conv_values < 0).astype(bool)  # Lost magnitude!
```

### Why Tests Didn't Catch It
```python
# ORIGINAL TESTS (BAD)
def test_convolution_same_function(self):
    result = convolution(f, f)
    assert result is not None     # Only checks existence
    assert result.n_vars == 2     # Only checks structure
```

### What Tests Should Have Done
```python
# FIXED TESTS (GOOD)
def test_convolution_theorem_holds(self):
    """Verify (f*g)^(S) = f̂(S)·ĝ(S)."""
    conv_coeffs = convolution(f, g)
    expected = f.fourier() * g.fourier()

    assert np.allclose(conv_coeffs, expected)  # Verifies math!
```

---

## Anti-Patterns to Avoid

### 1. Existence Tests
```python
# BAD: Only checks something returned
assert result is not None

# GOOD: Verify the actual value
assert np.allclose(result, expected_value)
```

### 2. Structure-Only Tests
```python
# BAD: Only checks shape/type
assert result.n_vars == 3
assert isinstance(result, BooleanFunction)

# GOOD: Also verify correctness
assert result.evaluate([0,0,0]) == expected_output
```

### 3. Missing Contract Tests
```python
# BAD: Doesn't verify the mathematical claim
def test_parseval():
    f = bf.parity(3)
    result = parseval_verify(f)
    assert result  # What does this even mean?

# GOOD: Explicitly verify Parseval's identity
def test_parseval_identity():
    f = bf.parity(3)
    sum_squared = sum(c**2 for c in f.fourier())
    assert np.isclose(sum_squared, 1.0)  # Σ f̂(S)² = 1
```

---

## Test Checklist

Before writing a test, answer:

1. **What is the mathematical contract?**
   - What theorem/property should hold?
   - Write the equation in the docstring

2. **What are the edge cases?**
   - Empty input, n=0, n=1
   - Constant functions (all 0s, all 1s)
   - Dictator functions (single variable)

3. **What should fail?**
   - Test that bad inputs raise exceptions
   - Verify error messages are helpful

4. **Does this test the implementation or the math?**
   - Implementation tests: "does it run?"
   - Math tests: "does it compute correctly?"
   - **You need both, but math tests catch more bugs**

---

## Test Template

```python
class TestFunctionName:
    """Tests for function_name following [mathematical reference]."""

    def test_mathematical_property(self):
        """Verify [specific mathematical property].

        Theorem: [state the theorem being tested]
        """
        # Arrange
        f = bf.some_function(3)

        # Act
        result = function_name(f)

        # Assert - verify the MATH, not just existence
        expected = compute_expected_value(f)
        assert np.allclose(result, expected), f"Expected {expected}, got {result}"

    def test_edge_case_constant(self):
        """Edge case: constant function."""
        f = bf.create([0, 0, 0, 0])  # All zeros
        result = function_name(f)
        assert result == expected_for_constant

    def test_invalid_input_raises(self):
        """Invalid input should fail loud."""
        with pytest.raises(ValueError, match="helpful error message"):
            function_name(invalid_input)
```

---

## Files Status

### Fixed (Jan 2026)

| File | Issue | Fix |
|------|-------|-----|
| test_pac_learning.py | 24 weak assertions | Added error rate verification |
| test_builtins.py | 14 weak assertions | Added truth table verification |
| test_module_exploration.py | 20 weak assertions | Added correctness checks |
| test_quantum.py | 13 weak assertions | Added mathematical property tests |
| test_circuit.py | 10 weak assertions | Added circuit evaluation tests |
| test_property_tester.py | (new) | Comprehensive PropertyTester tests |

### Remaining Low Priority

| File | Count | Reason |
|------|-------|--------|
| test_visualization_*.py | 70+ | Visual output, hard to verify |
| test_gpu_*.py | 21 | Infrastructure tests |
| test_benchmarks.py | 9 | Performance tests |

---

## Library Gaps Found

### Local Correction (not in library)

The notebook `lecture2_linearity_testing.ipynb` demonstrates local correction,
but the library doesn't expose this as a function. Consider adding:

```python
# Proposed addition to PropertyTester
def local_correct(self, x: int, repetitions: int = 10) -> int:
    """
    Local correction for functions close to linear.

    If f is ε-close to χ_S, then for random y:
    f(y) ⊕ f(x⊕y) = χ_S(x) with probability ≥ 1-2ε

    Args:
        x: Input to correct
        repetitions: Number of random samples (use majority vote)

    Returns:
        Corrected value in {-1, +1}
    """
```

This would enable testing the BLR self-correction property.
