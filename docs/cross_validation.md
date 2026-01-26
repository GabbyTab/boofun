# Cross-Validation

This document describes how to cross-validate BooFun results with other libraries and established mathematical software.

## Overview

Cross-validation ensures our implementations are mathematically correct by comparing results with:
1. **BoolForge** - Boolean function/network library for systems biology
2. **SageMath** - Open-source mathematics software
3. **Mathematica** - Wolfram's computational mathematics platform

## BoolForge Cross-Validation

[BoolForge](https://github.com/ckadelka/BoolForge) (Kadelka & Coberly, 2025) is a Python library for Boolean functions and networks, focused on canalization for systems biology.

### What We Validate

| Property | BooFun | BoolForge | Status |
|----------|--------|-----------|--------|
| is_canalizing | `is_canalizing(f)` | `f.is_canalizing()` | ✓ Matches |
| canalizing_depth | `get_canalizing_depth(f)` | `f.get_canalizing_depth()` | ✓ Matches |
| essential_variables | `get_essential_variables(f)` | `f.get_number_of_essential_variables()` | ✓ Matches |
| is_monotonic | `f.is_monotone()` | `f.is_monotonic()` | ✓ Matches |
| symmetry_groups | `get_symmetry_groups(f)` | `f.get_symmetry_groups()` | ✓ Matches |

### Running the Tests

```bash
# Install BoolForge
pip install git+https://github.com/ckadelka/BoolForge

# Run cross-validation tests
pytest tests/cross_validation/test_boolforge.py -v
```

### Example Validation

```python
import boolforge
import boofun as bf
from boofun.analysis.canalization import is_canalizing, get_canalizing_depth

# Create same function in both libraries
bf_and = bf.AND(3)
boolforge_and = boolforge.BooleanFunction([0, 0, 0, 0, 0, 0, 0, 1])

# Compare canalization
assert is_canalizing(bf_and) == boolforge_and.is_canalizing()  # Both True
assert get_canalizing_depth(bf_and) == boolforge_and.get_canalizing_depth()  # Both 3
```

### Notes on BoolForge

- BoolForge uses **Monte Carlo** for some measures (like `get_average_sensitivity()`), so values may vary slightly from BooFun's exact computations.
- BoolForge focuses on **Boolean networks** (interconnected functions) which BooFun does not support.
- BoolForge has features BooFun lacks: `get_layer_structure()`, `get_canalizing_strength()`, random generators with constraints.
- BooFun has features BoolForge lacks: Fourier analysis, query complexity, property testing, hypercontractivity.

## Key Functions to Validate

### 1. Fourier Transform (Walsh-Hadamard)

The core of Boolean function analysis. Verify that:
- Parseval's identity holds: Σ f̂(S)² = 1 for Boolean functions
- Specific coefficients match known theoretical values

#### SageMath Validation

```python
# SageMath code for comparison
from sage.all import *

def sage_fourier_coefficients(f, n):
    """Compute Fourier coefficients in Sage for comparison."""
    size = 2^n
    coeffs = []

    for S in range(size):
        # Compute f̂(S) = E[f(x) * χ_S(x)]
        total = 0
        for x in range(size):
            # χ_S(x) = (-1)^(popcount(x & S))
            parity = Integer(x & S).popcount() % 2
            char_val = (-1)^parity
            # f(x) in {-1, +1}
            f_val = 1 - 2*f(x)
            total += f_val * char_val
        coeffs.append(total / size)

    return coeffs
```

#### Mathematica Validation

```mathematica
(* Mathematica code for comparison *)
FourierCoefficient[f_, n_, S_] := Module[{size, total},
    size = 2^n;
    total = Sum[
        With[{parity = Mod[DigitCount[BitAnd[x, S], 2, 1], 2]},
            (1 - 2*f[x]) * (-1)^parity
        ],
        {x, 0, size - 1}
    ];
    total / size
]
```

### 2. Influence Computation

Verify that influences match for known functions:

| Function | Expected Influence per Variable |
|----------|--------------------------------|
| Majority_n (odd n) | 2/π * √(2/(π*n)) ≈ 0.798/√n |
| Parity_n | 1 for all variables |
| Dictator | 1 for dictator variable, 0 otherwise |
| AND_n | 2^(1-n) for all variables |

#### SageMath Test

```python
# Verify majority influence against theory
from sage.all import *

def majority_influence_theory(n):
    """Theoretical influence for majority function."""
    # For large n: Inf_i[MAJ_n] ≈ sqrt(2/(π*n))
    return sqrt(2 / (pi * n))

# Compare with computed values
for n in [5, 7, 9, 11, 13]:
    theoretical = float(majority_influence_theory(n))
    print(f"n={n}: theoretical ≈ {theoretical:.4f}")
```

### 3. Noise Stability

Verify noise stability formulas:

| Function | Noise Stability Formula |
|----------|------------------------|
| Dictator | Stab_ρ = ρ |
| Parity_n | Stab_ρ = ρ^n |
| Majority (limit) | (1/2) + (1/π)*arcsin(ρ) |

#### Mathematica Verification

```mathematica
(* Verify noise stability of Majority approaches Sheppard's formula *)
SheppardsFormula[rho_] := 1/2 + ArcSin[rho]/Pi

(* For large n, majority noise stability should approach this *)
MajorityNoiseStability[n_, rho_] := (* computed value *)

(* Compare *)
Table[
    {n, SheppardsFormula[0.5], MajorityNoiseStability[n, 0.5]},
    {n, 5, 21, 2}
]
```

### 4. Total Influence

Verify total influence bounds:

| Function | Total Influence |
|----------|----------------|
| Majority_n | Θ(√n) |
| Parity_n | n |
| AND_n | n * 2^(1-n) |
| Tribes_n | Θ(√n) |

## Automated Cross-Validation

### Test Script

```python
#!/usr/bin/env python3
"""
Cross-validate BooFun with Sage (requires SageMath installation).
"""

import subprocess
import json
import numpy as np

def run_sage_comparison(n, function_type):
    """Run Sage computation and compare with BooFun."""
    sage_script = f'''
from sage.all import *
import json

n = {n}
function_type = "{function_type}"

# Compute in Sage
if function_type == "parity":
    def f(x):
        return Integer(x).popcount() % 2
elif function_type == "majority":
    def f(x):
        return 1 if Integer(x).popcount() > n // 2 else 0
elif function_type == "and":
    def f(x):
        return 1 if x == 2^n - 1 else 0

# Compute Fourier coefficients
coeffs = []
size = 2^n
for S in range(size):
    total = 0
    for x in range(size):
        parity = Integer(x & S).popcount() % 2
        char_val = (-1)^parity
        f_val = 1 - 2*f(x)
        total += f_val * char_val
    coeffs.append(float(total / size))

print(json.dumps({{"coefficients": coeffs}}))
'''

    result = subprocess.run(
        ["sage", "-c", sage_script],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise RuntimeError(f"Sage failed: {result.stderr}")


def compare_with_boofun(n, function_type):
    """Compare Sage results with BooFun."""
    import boofun as bf

    # Get BooFun result
    if function_type == "parity":
        f = bf.parity(n)
    elif function_type == "majority":
        f = bf.majority(n)
    elif function_type == "and":
        f = bf.AND(n)

    bf_coeffs = f.fourier()

    # Get Sage result
    try:
        sage_result = run_sage_comparison(n, function_type)
        sage_coeffs = np.array(sage_result["coefficients"])

        # Compare
        max_diff = np.max(np.abs(bf_coeffs - sage_coeffs))
        print(f"{function_type}(n={n}): max difference = {max_diff:.2e}")

        if max_diff < 1e-10:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")

    except FileNotFoundError:
        print("  (Sage not installed - skipping)")
```

## Known Theoretical Results

### Parity Function
- f̂(S) = 0 for S ≠ [n]
- f̂([n]) = ±1 (depending on convention)
- Total influence = n
- Noise stability = ρ^n

### Majority Function
- All variables have equal influence (symmetric)
- Degree = n (for odd n)
- Fourier weight concentrated on odd-sized subsets
- For large n: Inf_i ≈ √(2/(πn))

### Threshold Functions
- Chow parameters uniquely determine threshold functions
- LTFs have low noise sensitivity (stable)

## Integration with CI

Add to CI workflow:

```yaml
# .github/workflows/ci.yml
cross-validation:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v3
    - name: Install Sage
      run: |
        sudo apt-get update
        sudo apt-get install -y sagemath
    - name: Run cross-validation
      run: |
        python scripts/cross_validate_sage.py
```

## References

1. O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge University Press.
2. SageMath Documentation: https://doc.sagemath.org/
3. Wolfram Language Documentation: https://reference.wolfram.com/language/
