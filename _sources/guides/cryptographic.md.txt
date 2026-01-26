# Cryptographic Analysis Guide

Tools for analyzing Boolean functions in cryptographic contexts, including S-box analysis.

## Overview

BooFun provides comprehensive cryptographic analysis tools:

- **Nonlinearity** - Distance to nearest affine function
- **Bent functions** - Maximum nonlinearity detection
- **Walsh transform** - Linear correlation analysis
- **Algebraic properties** - ANF, degree, algebraic immunity
- **Correlation immunity** - Resistance to correlation attacks
- **Avalanche criteria** - SAC, propagation criterion
- **S-box analysis** - LAT, DDT, comprehensive analyzer

## Basic Cryptographic Measures

### Nonlinearity

The distance from f to the nearest affine function.

| Task | Function | Description |
|------|----------|-------------|
| Nonlinearity | `crypto.nonlinearity(f)` | Distance to nearest affine |
| Bent detection | `crypto.is_bent(f)` | Maximum nonlinearity |
| Balance | `f.is_balanced()` | Equal 0s and 1s |

```python
import boofun as bf
from boofun.analysis import cryptographic as crypto

# Check nonlinearity of a function
f = bf.create([0, 1, 1, 0, 1, 0, 0, 1])  # 3-variable function

nl = crypto.nonlinearity(f)
print(f"Nonlinearity: {nl}")

# Maximum nonlinearity for n=3 is 2^(n-1) - 2^((n-1)/2) = 2
is_bent = crypto.is_bent(f)
print(f"Is bent: {is_bent}")
```

### Bent Functions

Bent functions achieve maximum nonlinearity (only exist for even n).

```python
# Create a bent function (n=4)
# Bent functions have nonlinearity = 2^(n-1) - 2^(n/2-1) = 6 for n=4
bent_tt = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
f = bf.create(bent_tt)

print(f"Is bent: {crypto.is_bent(f)}")
print(f"Nonlinearity: {crypto.nonlinearity(f)}")  # Should be 6
```

## Walsh/Fourier Analysis

### Walsh Transform

The Walsh-Hadamard transform for cryptographic analysis.

| Task | Function |
|------|----------|
| Walsh transform | `crypto.walsh_transform(f)` |
| Walsh spectrum | `crypto.walsh_spectrum(f)` |

```python
f = bf.majority(5)

# Get Walsh transform (all coefficients)
walsh = crypto.walsh_transform(f)
print(f"Walsh transform shape: {walsh.shape}")

# Get Walsh spectrum (unique absolute values and counts)
spectrum = crypto.walsh_spectrum(f)
print("Walsh spectrum:")
for value, count in sorted(spectrum.items()):
    print(f"  |W| = {value}: {count} coefficients")
```

### Algebraic Normal Form

The ANF representation over GF(2).

| Task | Function |
|------|----------|
| ANF | `crypto.algebraic_normal_form(f)` |
| Algebraic degree | `crypto.algebraic_degree(f)` |

```python
f = bf.create([0, 1, 1, 0])  # XOR function

anf = crypto.algebraic_normal_form(f)
print(f"ANF: {anf}")

deg = crypto.algebraic_degree(f)
print(f"Algebraic degree: {deg}")
```

## Correlation Properties

### Correlation Immunity

Resistance to correlation attacks.

| Task | Function | Description |
|------|----------|-------------|
| Correlation immunity | `crypto.correlation_immunity(f)` | CI order |
| Resiliency | `crypto.resiliency(f)` | Balanced + CI |

A function is t-th order correlation immune if its output is statistically
independent of any t input variables.

```python
# XOR is maximally correlation immune
f = bf.parity(4)
ci = crypto.correlation_immunity(f)
print(f"Correlation immunity of XOR_4: {ci}")  # = 4

# Resiliency = CI for balanced functions
res = crypto.resiliency(f)
print(f"Resiliency: {res}")
```

### Avalanche Criteria

How output bits change when input bits flip.

| Task | Function | Description |
|------|----------|-------------|
| SAC | `crypto.strict_avalanche_criterion(f)` | Bit flip propagation |
| Propagation criterion | `crypto.propagation_criterion(f, k)` | PC(k) |

**Strict Avalanche Criterion (SAC):** Flipping any input bit changes
the output with probability 1/2.

```python
f = bf.majority(5)

# Check SAC
sac_satisfied = crypto.strict_avalanche_criterion(f)
print(f"SAC satisfied: {sac_satisfied}")

# Check propagation criterion of order k
for k in [1, 2, 3]:
    pc_k = crypto.propagation_criterion(f, k)
    print(f"PC({k}) satisfied: {pc_k}")
```

## S-Box Analysis

For analyzing substitution boxes used in block ciphers.

### Linear Approximation Table (LAT)

Measures linear correlations between input/output bits.

```python
# AES S-box (first 16 entries for demo)
sbox = [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
        0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76]

lat = crypto.linear_approximation_table(sbox)
print(f"LAT shape: {lat.shape}")

# Maximum absolute bias (excluding (0,0))
max_bias = max(abs(lat[i,j]) for i in range(len(lat))
               for j in range(len(lat[0])) if i > 0 or j > 0)
print(f"Max linear bias: {max_bias}")
```

### Difference Distribution Table (DDT)

Measures differential propagation through the S-box.

```python
ddt = crypto.difference_distribution_table(sbox)
print(f"DDT shape: {ddt.shape}")

# Maximum differential probability (excluding 0 -> 0)
max_diff = max(ddt[i,j] for i in range(len(ddt))
               for j in range(len(ddt[0])) if i > 0)
print(f"Max differential: {max_diff}")
```

### Complete S-Box Analysis

Use SBoxAnalyzer for comprehensive analysis.

```python
from boofun.analysis.cryptographic import SBoxAnalyzer

# Standard 4-bit S-box
sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
        0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]

analyzer = SBoxAnalyzer(sbox)

# Get complete summary
print(analyzer.summary())

# Access individual properties
print(f"Is bijective: {analyzer.is_bijective()}")
print(f"Max LAT entry: {analyzer.max_lat_entry()}")
print(f"Max DDT entry: {analyzer.max_ddt_entry()}")
print(f"Nonlinearity: {analyzer.nonlinearity()}")
```

## Cryptographic Design Criteria

### Good S-Box Properties

| Property | Requirement | Why |
|----------|-------------|-----|
| Bijective | Yes | Invertibility |
| High nonlinearity | Close to bent bound | Linear attack resistance |
| Low max DDT | ≤ 4 for 8-bit | Differential attack resistance |
| Low max LAT | ≤ 16 for 8-bit | Linear attack resistance |
| SAC satisfied | Yes | Avalanche effect |
| High algebraic degree | n-1 | Algebraic attack resistance |

### Example: Evaluating an S-Box

```python
def evaluate_sbox(sbox):
    """Evaluate cryptographic strength of an S-box."""
    analyzer = SBoxAnalyzer(sbox)
    n = int(np.log2(len(sbox)))

    print(f"=== S-Box Analysis (n={n}) ===")
    print(f"Bijective: {analyzer.is_bijective()}")

    # Nonlinearity
    nl = analyzer.nonlinearity()
    bent_bound = 2**(n-1) - 2**(n//2 - 1) if n % 2 == 0 else None
    print(f"Nonlinearity: {nl}", end="")
    if bent_bound:
        print(f" (bent bound: {bent_bound})")
    else:
        print()

    # Linear properties
    max_lat = analyzer.max_lat_entry()
    print(f"Max LAT entry: {max_lat}")

    # Differential properties
    max_ddt = analyzer.max_ddt_entry()
    print(f"Max DDT entry: {max_ddt}")

    # Algebraic degree
    deg = analyzer.algebraic_degree()
    print(f"Algebraic degree: {deg} (max: {n})")

    return analyzer

# Evaluate the example S-box
sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
        0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
evaluate_sbox(sbox)
```

## Boolean Functions from Integers

Create functions from hexadecimal specifications (common in crypto):

```python
# Specify truth table as hex
f = bf.create(0xAC90, n=4)  # 4-variable function from hex

print(f"Is balanced: {f.is_balanced()}")
print(f"Nonlinearity: {crypto.nonlinearity(f)}")
print(f"Is bent: {crypto.is_bent(f)}")
```

## Mathematical Background

### Nonlinearity

The nonlinearity of f is:

$$nl(f) = 2^{n-1} - \frac{1}{2} \max_{\omega} |W_f(\omega)|$$

where W_f is the Walsh transform.

### Bent Functions

f is bent if |W_f(ω)| = 2^(n/2) for all ω. Bent functions:
- Only exist for even n
- Are maximally nonlinear
- Are never balanced
- Have flat Walsh spectrum

### Correlation Immunity

f is t-th order correlation immune iff:

$$\hat{f}(S) = 0 \text{ for all } 1 \leq |S| \leq t$$

### LAT and DDT

**LAT entry:**
$$LAT[a,b] = \sum_x (-1)^{a \cdot x \oplus b \cdot S(x)}$$

**DDT entry:**
$$DDT[\Delta_x, \Delta_y] = |\{x : S(x) \oplus S(x \oplus \Delta_x) = \Delta_y\}|$$

## See Also

- [Spectral Analysis Guide](spectral_analysis.md) - Walsh-Hadamard basics
- [Representations Guide](representations.md) - ANF representation
- Carlet, *Boolean Functions for Cryptography and Coding Theory*
