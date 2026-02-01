# BooFun Style Guide

Formatting is handled by CI. This is about *how* we write code.

## Principles

| Principle | Meaning |
|-----------|---------|
| **KISS** | Keep It Simple. No cleverness. |
| **DRY** | Don't Repeat Yourself. One source of truth. |
| **Fail Loud** | Errors should scream, not whisper. |
| **Functional** | Pure functions. No mutation. |
| **Domain/Codomain** | Always document what goes in and out. |

---

## KISS - Keep It Simple

```python
# Good: Direct and obvious
def is_balanced(f):
    return f.fourier()[0] == 0

# Bad: Over-engineered
class BalanceCheckerFactory:
    def create_checker(self, strategy): ...
```

**Rule**: If a junior dev can't understand it in 30 seconds, simplify.

---

## DRY - Don't Repeat Yourself

```python
# Bad: Logic duplicated
def test_majority_3():
    f = bf.majority(3)
    assert sum(f.fourier()**2) == 1

def test_majority_5():
    f = bf.majority(5)
    assert sum(f.fourier()**2) == 1  # Same check!

# Good: One function, parameterized
@pytest.mark.parametrize("n", [3, 5, 7])
def test_parseval(n):
    f = bf.majority(n)
    assert np.isclose(sum(f.fourier()**2), 1)
```

**Rule**: If you copy-paste, you're doing it wrong.

---

## Fail Loud

```python
# Good: Immediate, clear error
if f.n_vars != g.n_vars:
    raise ValueError(f"Mismatched vars: {f.n_vars} vs {g.n_vars}")

# Bad: Silent "fix"
n = min(f.n_vars, g.n_vars)  # Hides bug!

# Bad: Silent data loss
result = (values < 0).astype(bool)  # Destroys magnitude!
```

**Rule**: Never silently coerce, truncate, or threshold.

---

## Functional

```python
# Good: Returns new object
def negate(f):
    return bf.create([1-v for v in f.truth_table])

# Bad: Mutates input
def negate(f):
    f.truth_table = [1-v for v in f.truth_table]  # Side effect!
```

**Rule**: Same input → same output. No surprises.

---

## Domain/Codomain

Every function documents its contract:

```python
def convolution(f: BooleanFunction, g: BooleanFunction) -> np.ndarray:
    """
    Fourier coefficients of f*g.
    
    Domain: f, g with same n_vars
    Codomain: array of reals (NOT a BooleanFunction!)
    
    Raises: ValueError if n_vars mismatch
    """
```

**Rule**: Reader should know types without reading the code.

---

## Quick Test

Before committing, ask:

1. **KISS**: Can I explain this to a rubber duck?
2. **DRY**: Did I copy-paste anything?
3. **Fail Loud**: What happens with bad input?
4. **Functional**: Does this mutate anything?
5. **Domain/Codomain**: Are types obvious from the signature?

If any answer is wrong, fix it.
