# Cryptographic S-box Analysis

This example demonstrates how to use BoolFunc to analyze cryptographic S-boxes, a critical component of block ciphers like AES.

## Background

S-boxes (Substitution boxes) are nonlinear components in symmetric ciphers. Their cryptographic strength depends on several Boolean function properties:

- **Nonlinearity**: Distance from all affine functions (higher is better)
- **Algebraic degree**: Complexity of the polynomial representation
- **Balancedness**: Equal number of 0 and 1 outputs
- **Differential uniformity**: Resistance to differential cryptanalysis

## Setup

```python
import numpy as np
import boolfunc as bf
from boolfunc.analysis import PropertyTester

# AES S-box (first 32 values shown)
AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    # ... full 256 entries
]
```

## Extracting Component Functions

An 8-bit S-box can be viewed as 8 separate Boolean functions, one for each output bit:

```python
def get_sbox_component(sbox: list, bit: int) -> bf.BooleanFunction:
    """Extract the bit-th component function from an S-box."""
    truth_table = [(sbox[x] >> bit) & 1 for x in range(256)]
    return bf.BooleanFunction.from_truth_table(truth_table, n_vars=8)

# Get all 8 component functions
components = [get_sbox_component(AES_SBOX, i) for i in range(8)]
```

## Nonlinearity Analysis

Nonlinearity is the minimum Hamming distance to all affine functions:

```python
def compute_nonlinearity(f: bf.BooleanFunction) -> int:
    """Compute nonlinearity using Walsh-Hadamard transform."""
    fourier = f.fourier()
    n = f.n_vars
    N = 2**n
    
    # Convert Fourier coefficients to Walsh spectrum
    # Walsh(a) = sum_x (-1)^{f(x) + <a,x>}
    walsh = fourier * N
    
    # Nonlinearity = (N - max|Walsh|) / 2
    max_walsh = max(abs(w) for w in walsh)
    return int((N - max_walsh) / 2)

# AES S-box components have nonlinearity 112 (optimal for 8-bit)
for i, comp in enumerate(components):
    nl = compute_nonlinearity(comp)
    print(f"Component {i}: nonlinearity = {nl}")
```

## Balancedness Check

A balanced function outputs 0 and 1 equally often:

```python
for i, comp in enumerate(components):
    tester = PropertyTester(comp)
    balanced = tester.balanced_test()
    print(f"Component {i}: balanced = {balanced}")
```

## Linearity Testing

Good S-boxes should be far from linear:

```python
for i, comp in enumerate(components):
    tester = PropertyTester(comp)
    
    # Should fail - S-box is highly nonlinear
    is_linear = tester.blr_linearity_test(num_queries=1000)
    print(f"Component {i}: linear = {is_linear}")  # Should be False
```

## Fourier Analysis

Examine the spectral structure:

```python
for i, comp in enumerate(components):
    # Get Fourier degree (highest level with non-zero weight)
    degree = comp.degree()
    
    # Get spectral weight distribution
    weights = comp.spectral_weight_by_degree()
    
    # Total influence (measure of complexity)
    total_inf = comp.total_influence()
    
    print(f"Component {i}:")
    print(f"  Fourier degree: {degree}")
    print(f"  Total influence: {total_inf:.2f}")
    print(f"  Weight at degree 1: {weights.get(1, 0):.4f}")
```

## Interactive Visualization

```python
from boolfunc.visualization.interactive import FourierExplorer

# Explore the Fourier spectrum interactively
explorer = FourierExplorer(components[0])

# In Jupyter notebook:
# fig = explorer.spectrum_plot()
# fig.show()

# Get top coefficients
fig = explorer.top_coefficients(k=10)
```

## Comparing S-boxes

Compare different S-box designs:

```python
# Simple (weak) S-box: just bit rotation
weak_sbox = [(x << 3 | x >> 5) & 0xFF for x in range(256)]
weak_comp = get_sbox_component(weak_sbox, 0)

# Compare
aes_nl = compute_nonlinearity(components[0])
weak_nl = compute_nonlinearity(weak_comp)

print(f"AES nonlinearity: {aes_nl}")  # 112
print(f"Weak nonlinearity: {weak_nl}")  # Much lower

# Linearity test
weak_tester = PropertyTester(weak_comp)
print(f"Weak S-box passes linearity: {weak_tester.blr_linearity_test()}")
```

## Full Analysis Report

```python
def analyze_sbox(sbox: list, name: str = "S-box"):
    """Generate comprehensive S-box analysis report."""
    print(f"=== {name} Analysis ===\n")
    
    components = [get_sbox_component(sbox, i) for i in range(8)]
    
    # Per-component analysis
    for i, comp in enumerate(components):
        nl = compute_nonlinearity(comp)
        deg = comp.degree()
        bal = PropertyTester(comp).balanced_test()
        inf = comp.total_influence()
        
        print(f"Bit {i}: NL={nl}, deg={deg}, balanced={bal}, I[f]={inf:.2f}")
    
    # Summary
    nonlinearities = [compute_nonlinearity(c) for c in components]
    print(f"\nMin nonlinearity: {min(nonlinearities)}")
    print(f"Max nonlinearity: {max(nonlinearities)}")
    print(f"All balanced: {all(PropertyTester(c).balanced_test() for c in components)}")

analyze_sbox(AES_SBOX, "AES")
```

## Key Takeaways

1. **High nonlinearity** (112 for AES) provides resistance to linear cryptanalysis
2. **Balancedness** ensures no output bias
3. **High algebraic degree** resists algebraic attacks
4. **BoolFunc makes it easy** to compute all these properties with a unified API

## References

- Nyberg, K. (1991). "Perfect nonlinear S-boxes"
- Carlet, C. (2010). "Boolean Functions for Cryptography"
- O'Donnell, R. (2014). "Analysis of Boolean Functions" - Chapter on noise stability
