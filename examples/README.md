# BoolFunc Tutorials & Examples

This folder contains tutorials organized by skill level.

## Tutorial Index

### Beginner
1. **[01_getting_started.py](01_getting_started.py)** - Installation, basic usage, your first Boolean function
2. **[02_fourier_basics.py](02_fourier_basics.py)** - Walsh-Hadamard transform, coefficients, Parseval's identity
3. **[03_common_families.py](03_common_families.py)** - AND, OR, majority, parity, tribes

### Intermediate
4. **[04_property_testing.py](04_property_testing.py)** - BLR linearity, junta, monotonicity testing
5. **[05_query_complexity.py](05_query_complexity.py)** - Sensitivity, block sensitivity, decision tree depth
6. **[06_noise_stability.py](06_noise_stability.py)** - Noise stability, influences, FKN theorem

### Advanced
7. **[07_quantum_applications.py](07_quantum_applications.py)** - Grover, quantum walks, quantum advantage
8. **[08_cryptographic_analysis.py](08_cryptographic_analysis.py)** - S-box analysis, nonlinearity
9. **[09_research_applications.py](09_research_applications.py)** - Communication complexity, advanced topics

## Quick Start

```python
import boolfunc as bf

# Create a function
f = bf.majority(5)

# Analyze it
print(f"Total influence: {f.total_influence():.3f}")
print(f"Fourier degree: {f.degree()}")
print(f"Noise stability: {f.noise_stability(0.9):.3f}")
```

## Running Examples

```bash
# Run any tutorial
python examples/01_getting_started.py

# Run all tutorials
for f in examples/0*.py; do python "$f"; done
```
