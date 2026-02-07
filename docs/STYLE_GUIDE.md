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
| **Deterministic** | Same inputs → same outputs. Always. |

### When Principles Conflict

Use this precedence:

**Correctness → Determinism → Clarity (KISS) → DRY → Performance**

Example: a cache improves performance and avoids recomputation (DRY), but if it makes behavior harder to reason about (KISS) or introduces nondeterminism, skip it.

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

### assert vs raise

In **library code**, use explicit `raise ValueError(...)` for validation. Python's `-O` flag strips `assert` statements, so `assert` is not a reliable guard in production.

In **test code**, use plain `assert` freely — pytest rewrites assertions for helpful error output.

```python
# Good: library validation
if n <= 0:
    raise ValueError(f"n must be positive, got {n}")

# Good: test assertion
assert np.isclose(result, expected), f"Got {result}, expected {expected}"

# Bad: library validation via assert (stripped by -O)
assert n > 0  # Disappears in optimized mode!
```

### Exception chaining

When translating low-level exceptions into domain exceptions, always chain with `from`:

```python
# Good: preserves causal context
try:
    truth_table = f.get_representation("truth_table")
except KeyError as e:
    raise ConversionError(f"Cannot get truth table for {f}") from e

# Bad: loses the original traceback
except KeyError:
    raise ConversionError("Cannot get truth table")
```

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

**Rule**: Public functions MUST have type hints on all parameters and return values.

```python
# Good: types are part of the interface
def noise_stability(f: BooleanFunction, rho: float) -> float: ...

# Bad: reader has to guess
def noise_stability(f, rho): ...
```

---

## Determinism

Monte Carlo results must be reproducible. Randomness is a controlled input, not an accident.

```python
# Good: caller controls the seed
def estimate_influence(f, i, n_samples, rng=None):
    rng = rng or np.random.default_rng()
    ...

# Good: notebook sets seed once at the top
np.random.seed(42)

# Bad: hidden nondeterminism
def estimate_influence(f, i, n_samples):
    samples = np.random.randint(...)  # Which seed? Unknown!
```

**Rule**: Any function that uses randomness SHOULD accept an `rng` parameter (or seed). Notebooks MUST set a seed at the top.

---

## Bit-Ordering Convention

BooFun uses **LSB = x₀** everywhere. This must be consistent across all code.

```python
# Index i maps to bits via: x_j = (i >> j) & 1
# Index 5 = 0b101 means x₀=1, x₁=0, x₂=1

# Good: LSB-first
x = [(i >> j) & 1 for j in range(n)]

# Bad: MSB-first (common bug!)
x = [int(b) for b in format(i, f"0{n}b")]  # WRONG ORDER
```

**Rule**: When building truth tables from index → bit vectors, always use `(i >> j) & 1`.

See `CONTRIBUTING.md` for the full specification.

---

## Quick Test

Before committing, ask:

1. **KISS**: Can I explain this to a rubber duck?
2. **DRY**: Did I copy-paste anything?
3. **Fail Loud**: What happens with bad input?
4. **Functional**: Does this mutate anything?
5. **Domain/Codomain**: Are types obvious from the signature?

If any answer is wrong, fix it.

---

## Documentation & Notebooks

**Tone**: Clear, mathematical, concise. Let examples speak for themselves.

```markdown
# Bad: Self-promotion
"The POWER of the library: test ANY function!"
"This incredibly useful feature..."

# Good: Just show it
# Testing user-defined functions
def my_hash(x): ...
f = bf.create(my_hash, n=5)
tester.blr_linearity_test(f)  # Works on any function
```

**Rules**:
- No emojis in technical content
- No "power of" / "incredibly" / "amazing" language
- Quality is self-evident from examples, not claims
- Prefer equations and code over prose
- Keep notebooks short — one concept per section
