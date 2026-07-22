# Function Families and Growth Analysis

This guide covers how to work with function families - parameterized collections of Boolean functions that grow with n.

## What is a Function Family?

A **function family** is a sequence of Boolean functions indexed by the number of variables n:

- **Majority_n**: Returns 1 if more than half of inputs are 1
- **Parity_n**: XOR of all inputs
- **Tribes_{k,n}**: OR of k groups, each an AND of n/k variables
- **AND_n / OR_n**: All inputs must be 1 / at least one input must be 1

Studying how properties change as n grows reveals asymptotic behavior and theoretical bounds.

## Built-in Families

```python
from boofun.families import (
    MajorityFamily,
    ParityFamily,
    TribesFamily,
    ANDFamily,
    ORFamily,
    DictatorFamily,
    ThresholdFamily,
    LTFFamily,
    RecursiveMajority3Family,
)

# Generate a specific instance
maj = MajorityFamily()
f_5 = maj.generate(5)  # Majority on 5 variables
f_7 = maj(7)           # Shorthand

# Generate multiple
functions = maj.generate_range([3, 5, 7, 9, 11])
```

### Family Metadata

Each family has metadata about its properties:

```python
maj = MajorityFamily()
meta = maj.metadata

print(meta.name)           # "Majority"
print(meta.description)    # "MAJ_n(x) = 1 if Σx_i > n/2"
print(meta.asymptotics)    # Known theoretical formulas
print(meta.universal_properties)  # ["monotone", "symmetric", "balanced"]
```

### Theoretical Values

Get theoretical predictions for properties:

```python
# Total influence of Majority_n ≈ √(2/π) · √n
theory = maj.theoretical_value('total_influence', n=15)
print(f"I[MAJ_15] ≈ {theory:.4f}")  # ≈ 3.09
```

## Tracking Growth

The `GrowthTracker` observes properties as n increases:

```python
from boofun.families import GrowthTracker, MajorityFamily

# Create tracker
maj = MajorityFamily()
tracker = GrowthTracker(maj)

# Mark properties to track
tracker.mark('total_influence')
tracker.mark('noise_stability', rho=0.9)
tracker.mark('expectation')
tracker.mark('variance')

# Observe over range of n
results = tracker.observe(n_values=[3, 5, 7, 9, 11, 13, 15])

# Get summary
print(tracker.summary())
```

### Available Markers

```python
from boofun.families import PropertyMarker

# Built-in markers
tracker.mark('total_influence')
tracker.mark('influences')              # All variable influences
tracker.mark('influence_0')             # Specific variable
tracker.mark('noise_stability', rho=0.5)
tracker.mark('fourier_degree')
tracker.mark('spectral_concentration', k=2)
tracker.mark('expectation')
tracker.mark('variance')
tracker.mark('sensitivity')
tracker.mark('block_sensitivity')
tracker.mark('is_monotone')             # Boolean properties

# Custom marker
tracker.mark('custom',
    compute_fn=lambda f: f.sparsity(),
    description="Fourier sparsity")
```

## Visualization

### Single Family Growth

```python
from boofun.visualization.growth_plots import GrowthVisualizer

viz = GrowthVisualizer()

# Plot with theoretical comparison
fig, ax = viz.plot_growth(
    tracker,
    'total_influence',
    show_theory=True,
    log_y=False
)
```

### Family Comparison

Compare multiple families on the same property:

```python
from boofun.families import MajorityFamily, ParityFamily, ANDFamily

# Track each family
families = {
    'Majority': MajorityFamily(),
    'Parity': ParityFamily(),
    'AND': ANDFamily(),
}

trackers = {}
for name, family in families.items():
    tracker = GrowthTracker(family)
    tracker.mark('total_influence')
    tracker.observe(n_values=list(range(3, 12)))
    trackers[name] = tracker

# Compare
viz = GrowthVisualizer()
fig = viz.plot_family_comparison(
    trackers,
    'total_influence',
    log_y=True,
    title="Total Influence: Parity vs Majority vs AND"
)
```

### Convergence Rate

Analyze how a property converges to its asymptotic value:

```python
fig = viz.plot_convergence_rate(
    tracker,
    'total_influence',
    reference='sqrt_n'  # Divide by √n to see constant factor
)
```

## Custom Families

### From a Generator Function

```python
from boofun.families import FunctionFamily, FamilyMetadata
import boofun as bf

class MyCustomFamily(FunctionFamily):
    @property
    def metadata(self):
        return FamilyMetadata(
            name="MyFunction",
            description="Custom function family",
            asymptotics={
                'total_influence': lambda n: n * 0.5,
            },
            universal_properties=['balanced'],
        )

    def generate(self, n, **kwargs):
        # Your custom logic
        return bf.majority(n) ^ bf.parity(n)
```

### Inductive Families

Define families recursively:

```python
from boofun.families import InductiveFamily

class RecursiveFamily(InductiveFamily):
    def __init__(self):
        super().__init__(
            name="Recursive",
            base_cases={1: bf.dictator(1, 0)},
            step_size=1
        )

    def step(self, f_prev, n, n_prev):
        # Extend from n-1 to n variables
        return f_prev.extend(n, fill=0)
```

### Weight Pattern LTFs

LTF families with custom weight patterns:

```python
from boofun.families import LTFFamily

# Geometric weights: w_i = 0.5^i
geometric = LTFFamily.geometric(ratio=0.5)

# Harmonic weights: w_i = 1/(i+1)
harmonic = LTFFamily.harmonic()

# Power-law weights: w_i = (n-i)^2
power = LTFFamily.power_law(power=2.0)

# Custom pattern
custom = LTFFamily(
    weight_pattern=lambda i, n: 1.0 / (i + 1)**0.5,
    name="SqrtHarmonic"
)
```

## Theoretical Bounds

The library includes known asymptotic formulas:

| Family | Total Influence | Max Influence | Noise Stability |
|--------|-----------------|---------------|-----------------|
| Majority_n | √(2/π) · √n | √(2/(πn)) | (1/2) + (1/π)arcsin(ρ) |
| Parity_n | n | 1 | ρ^n |
| AND_n | n · 2^{-(n-1)} | 2^{-(n-1)} | - |
| Tribes | O(log n) | O(log n / n) | - |
| Dictator | 1 | 1 | ρ |

Access these via:

```python
maj = MajorityFamily()
print(maj.metadata.asymptotics)
```

## Quick Growth Plot

For quick exploration:

```python
from boofun.visualization.growth_plots import quick_growth_plot

# One-liner visualization
fig = quick_growth_plot(
    'majority',
    properties=['total_influence', 'expectation'],
    n_values=[3, 5, 7, 9, 11, 13]
)
```

## See Also

- [Spectral Analysis Guide](spectral_analysis.md) - Fourier analysis fundamentals
- [Query Complexity Guide](query_complexity.md) - Complexity measures
- [API Reference](../api/boofun.families.rst) - Full API documentation
