# Function Operations Guide

BooFun supports building complex functions from simpler ones using Boolean operators and other operations.

## Boolean Operators

Combine functions using Python operators:

| Operator | Name | Result |
|----------|------|--------|
| `f & g` | AND | (f ∧ g)(x) = f(x) ∧ g(x) |
| `f \| g` | OR | (f ∨ g)(x) = f(x) ∨ g(x) |
| `f ^ g` | XOR | (f ⊕ g)(x) = f(x) ⊕ g(x) |
| `~f` | NOT | (¬f)(x) = ¬f(x) |

### Basic Examples

```python
import boofun as bf

# Built-in functions
and3 = bf.AND(3)
or3 = bf.OR(3)
parity3 = bf.parity(3)
maj3 = bf.majority(3)

# Combine with operators
f1 = and3 ^ parity3        # AND XOR PARITY
f2 = maj3 & ~and3          # MAJORITY AND (NOT AND)
f3 = (and3 | or3) ^ parity3  # Complex composition

# All results are BooleanFunction objects
print(f1.n_vars)  # 3
print(f1.evaluate(5))  # Evaluate at input 5
```

### Operator Precedence

Python's operator precedence applies:
1. `~` (NOT) - highest
2. `&` (AND)
3. `^` (XOR)
4. `|` (OR) - lowest

Use parentheses for clarity:

```python
# These are different!
f1 = a | b & c    # Same as: a | (b & c)
f2 = (a | b) & c  # Explicit grouping
```

### Composite Functions and Fourier

Composite functions work seamlessly with analysis:

```python
import numpy as np

f = bf.AND(3) ^ bf.parity(3)

# Fourier coefficients (NumPy array)
fourier = f.fourier()
print(f"Type: {type(fourier)}")  # numpy.ndarray

# Parseval's identity still holds
print(f"‖f̂‖² = {fourier @ fourier}")  # 1.0

# All analysis methods work
print(f"Degree: {f.degree()}")
print(f"Influences: {f.influences()}")
```

## Input Negation

Negate inputs (flip all bits) using unary minus:

```python
f = bf.majority(3)

# Input negation: g(x) = f(-x) where -x flips all bits
g = -f

# In Fourier: ĝ(S) = (-1)^|S| · f̂(S)
# (odd-degree coefficients flip sign)
```

Note: `-f` negates *inputs*, while `~f` negates *output*.

```python
f = bf.AND(3)

neg_input = -f   # g(x) = f(flip all bits of x)
neg_output = ~f  # g(x) = NOT f(x)
```

## Chainable Methods

Alternative syntax using method chaining:

```python
f = bf.AND(3)
g = bf.OR(3)

# These pairs are equivalent:
f & g  ==  f.and_(g)
f | g  ==  f.or_(g)
f ^ g  ==  f.xor(g)
~f     ==  f.not_()
```

## Variable Restriction

Fix variables to constant values:

```python
f = bf.majority(5)

# Fix variable 0 to value 1
g = f.fix(0, 1)
print(g.n_vars)  # 4

# Fix multiple variables
h = f.fix([0, 2], [1, 0])  # x₀=1, x₂=0
print(h.n_vars)  # 3

# Shannon cofactor
cofactor_1 = f.cofactor(0, 1)  # f|_{x₀=1}
cofactor_0 = f.cofactor(0, 0)  # f|_{x₀=0}
```

## Variable Permutation

Reorder input variables:

```python
f = bf.create([0, 0, 0, 1, 0, 1, 1, 1])  # Some 3-variable function

# Permute: new variable i gets old variable perm[i]
g = f.permute_variables([2, 0, 1])  # Cycle: x₀→x₂→x₁→x₀

# Swap two variables
h = f.permute_variables([1, 0, 2])  # Swap x₀ and x₁
```

## Dual Function

The dual swaps AND and OR (De Morgan dual for monotone functions):

```python
f = bf.AND(3)
f_dual = f.dual()  # f*(x) = NOT f(NOT x)

# For AND: dual is OR
# For monotone: f* swaps roles of AND/OR
```

## Extending Functions

Add dummy variables:

```python
f = bf.majority(3)

# Extend to 5 variables (new vars are dummy)
g = f.extend(5, method="dummy")
print(g.n_vars)  # 5

# New variables don't affect output
for x in range(8):
    assert f.evaluate(x) == g.evaluate(x)  # Same for first 8 inputs
```

## Composing with Noise

Apply noise operator (sampling-based):

```python
f = bf.parity(5)

# Apply noise: each input bit flips with prob (1-ρ)/2
noisy_f = f.apply_noise(rho=0.9, samples=100)

# Get exact noise expectations
expectations = f.noise_expectation(rho=0.9)
# Returns E[f(y)|x] for all x, where y is ρ-correlated with x
```

## Building Custom Functions

Combine operations to build complex functions:

```python
# Threshold-2 function: at least 2 of 4 variables are 1
x0 = bf.dictator(4, 0)
x1 = bf.dictator(4, 1)
x2 = bf.dictator(4, 2)
x3 = bf.dictator(4, 3)

threshold_2 = (
    (x0 & x1) | (x0 & x2) | (x0 & x3) |
    (x1 & x2) | (x1 & x3) | (x2 & x3)
)

# Verify
print(threshold_2.evaluate(0b0011))  # 1 (two bits set)
print(threshold_2.evaluate(0b0001))  # 0 (one bit set)
```

## NumPy Integration

Fourier coefficients are NumPy arrays:

```python
import numpy as np

f = bf.majority(5)
fourier = f.fourier()

# All NumPy operations work
print(f"Shape: {fourier.shape}")
print(f"Dtype: {fourier.dtype}")
print(f"L2 norm: {np.linalg.norm(fourier)}")
print(f"Dot product: {fourier @ fourier}")
print(f"Max coefficient: {np.max(np.abs(fourier))}")

# Find heavy coefficients
heavy = np.where(np.abs(fourier) > 0.1)[0]
print(f"Heavy coefficient indices: {heavy}")
```

## Performance Notes

- Boolean operations create new truth tables (O(2^n) time and space)
- For n > 14, consider lazy evaluation or symbolic representations
- Composite functions cache their representations after first computation

## See Also

- [Representations Guide](representations.md) - Creating functions
- [Spectral Analysis Guide](spectral_analysis.md) - Fourier analysis
- [Query Complexity Guide](query_complexity.md) - Sensitivity and certificates
