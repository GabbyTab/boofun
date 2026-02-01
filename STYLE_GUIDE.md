# BooFun Internal Style Guide

Programming principles and patterns for contributors. Formatting is handled by CI (black, isort, flake8).

## Core Principles

1. **Explicit Domain/Codomain** - Always document what goes in and what comes out
2. **Fail Loud** - Never silently hide errors or lose information
3. **Functional** - Prefer pure functions, immutability, composition
4. **Test-Driven** - Write tests first, code second
5. **Concise** - Simple is better than clever

---

## 1. Domain and Codomain Documentation

Every function must explicitly state its domain (input types/constraints) and codomain (output type/guarantees).

### Good

```python
def convolution(f: "BooleanFunction", g: "BooleanFunction") -> np.ndarray:
    """
    Compute Fourier coefficients of the convolution of two Boolean functions.

    Domain:
        f, g: BooleanFunction with same n_vars
        
    Codomain:
        np.ndarray of shape (2^n,) containing real values in [-1, 1]
        
    Mathematical definition:
        (f * g)(x) = E_y[f(y)g(x ⊕ y)]
        Returns (f * g)^(S) = f̂(S) · ĝ(S) for all S
    """
```

### Bad

```python
def convolution(f, g):
    """Compute the convolution."""  # What types? What does it return?
```

### Template

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    One-line summary of what this does.

    Domain:
        arg1: Description and constraints
        arg2: Description and constraints
        
    Codomain:
        Description of return value and guarantees
        
    Raises:
        ValueError: When domain constraints are violated
        
    Example:
        >>> result = function_name(valid_input)
    """
```

---

## 2. Fail Loud (Offensive Programming)

### Principle
If something is wrong, fail immediately with a clear error. Never silently:
- Coerce types
- Truncate data
- Return partial results
- Hide exceptions

### Good

```python
def convolution(f: "BooleanFunction", g: "BooleanFunction") -> np.ndarray:
    if f.n_vars != g.n_vars:
        raise ValueError(
            f"Functions must have same number of variables: {f.n_vars} vs {g.n_vars}"
        )
    # ... proceed with valid inputs
```

### Bad

```python
def convolution(f, g):
    # Silently truncate to smaller function
    n = min(f.n_vars, g.n_vars)  # NO! This hides a bug
    
    # Silently threshold real values to Boolean
    result_tt = (conv_values < 0).astype(bool)  # NO! Information loss
```

### Guidelines

1. **Validate inputs at function entry** - Check constraints immediately
2. **Use specific exception types** - `ValueError` for bad args, `TypeError` for wrong types
3. **Include context in error messages** - Show what was expected vs received
4. **Never catch and ignore exceptions** - If you catch, handle or re-raise
5. **Track type/space changes** - If an operation changes the mathematical space, make it explicit

```python
# If we leave Boolean space, be explicit about it
def some_operation(f: BooleanFunction) -> RealValuedFunction:
    """Returns a real-valued function (NOT Boolean)."""
    # Clear that the output type is different
```

---

## 3. Functional Programming

### Principles
- **Pure functions** - Same input → same output, no side effects
- **Immutability** - Don't modify inputs, return new objects
- **Composition** - Small functions that combine well

### Good

```python
def negate(f: BooleanFunction) -> BooleanFunction:
    """Return a new function g where g(x) = ¬f(x)."""
    # Creates new function, doesn't modify f
    new_tt = [1 - v for v in f.truth_table]
    return BooleanFunction.from_truth_table(new_tt)

# Composable
h = negate(AND(f, g))
```

### Bad

```python
def negate(f: BooleanFunction) -> None:
    """Negate f in place."""
    f.truth_table = [1 - v for v in f.truth_table]  # Mutates input!
```

### Guidelines

1. **Return new objects** instead of mutating
2. **Avoid global state** - Pass dependencies as arguments
3. **Use comprehensions** over loops when clear
4. **Prefer `map`/`filter`** for transformations
5. **Keep functions small** - One responsibility per function

---

## 4. Test-Driven Development

### Workflow
1. Write a failing test that defines desired behavior
2. Write minimal code to pass the test
3. Refactor while keeping tests green

### Test Structure

```python
class TestConvolution:
    """Tests for convolution following the Convolution Theorem."""

    def test_convolution_theorem_holds(self):
        """Verify (f*g)^(S) = f̂(S)·ĝ(S)."""
        f = bf.majority(3)
        g = bf.parity(3)
        
        conv_coeffs = convolution(f, g)
        expected = f.fourier() * g.fourier()
        
        assert np.allclose(conv_coeffs, expected)

    def test_convolution_mismatched_vars_raises(self):
        """Convolution with different n_vars should fail loud."""
        f = bf.AND(2)
        g = bf.OR(3)
        
        with pytest.raises(ValueError, match="same number of variables"):
            convolution(f, g)
```

### Guidelines

1. **Test the contract** - Test domain constraints and codomain guarantees
2. **Test edge cases** - Empty inputs, boundary values, degenerate cases
3. **Test error conditions** - Verify proper exceptions are raised
4. **Use descriptive names** - `test_convolution_theorem_holds` not `test_conv_1`
5. **One assertion per concept** - Multiple asserts OK if testing same property

---

## 5. Concise and Simple

### Principles
- **Readable over clever** - Future you will thank present you
- **Fewer abstractions** - Add complexity only when needed
- **Clear naming** - Names should explain purpose

### Good

```python
# Clear, direct, no unnecessary abstraction
def fourier_coefficient(f: BooleanFunction, S: frozenset) -> float:
    """Compute f̂(S) = E[f(x)·χ_S(x)]."""
    n = f.n_vars
    total = 0.0
    for x in range(1 << n):
        fx = 1 - 2 * f.evaluate(x)
        chi_S = 1 - 2 * (bin(x & set_to_index(S)).count('1') % 2)
        total += fx * chi_S
    return total / (1 << n)
```

### Bad

```python
# Over-engineered
class FourierCoefficientCalculatorFactory:
    def __init__(self, strategy: FourierStrategy):
        self.strategy = strategy
    
    def create_calculator(self) -> FourierCalculator:
        return self.strategy.get_calculator()
```

### Guidelines

1. **YAGNI** - You Aren't Gonna Need It. Don't add features "just in case"
2. **Rule of three** - Only abstract after three concrete instances
3. **Flat is better than nested** - Avoid deep inheritance hierarchies
4. **Use standard library** - Don't reinvent NumPy, itertools, etc.
5. **Comments explain why, code explains what**

---

## Quick Reference

| Principle | Do | Don't |
|-----------|-----|-------|
| Domain/Codomain | Document types, constraints, guarantees | Leave readers guessing |
| Fail Loud | `raise ValueError("n must be positive")` | `n = max(1, n)` |
| Functional | `return new_function(...)` | `f.mutate_in_place()` |
| Test-Driven | Write test first | Test as afterthought |
| Concise | Simple, direct code | Premature abstraction |

---

## Examples of Good Style

### Function with clear contract

```python
def total_influence(f: "BooleanFunction") -> float:
    """
    Compute the total influence I[f] = Σᵢ Infᵢ[f].

    Domain:
        f: BooleanFunction with n_vars ≥ 0
        
    Codomain:
        float in [0, n] where n = f.n_vars
        For Boolean functions, this equals Σ_S |S|·f̂(S)²
        
    Raises:
        TypeError: If f is not a BooleanFunction
        
    Example:
        >>> f = bf.majority(3)
        >>> total_influence(f)
        1.5
    """
    if not isinstance(f, BooleanFunction):
        raise TypeError(f"Expected BooleanFunction, got {type(f)}")
    
    coeffs = f.fourier()
    return sum(
        bin(s).count('1') * c**2 
        for s, c in enumerate(coeffs)
    )
```

### Test with clear intent

```python
def test_total_influence_bounds():
    """Total influence is bounded by [0, n]."""
    for n in range(1, 6):
        f = bf.random(n)
        inf = total_influence(f)
        
        assert 0 <= inf <= n, f"Influence {inf} out of bounds for n={n}"
```
