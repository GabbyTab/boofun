# BoolFunc Tutorials

A comprehensive tutorial series for learning Boolean function analysis with BoolFunc.

## Quick Start

```bash
# Install BoolFunc
pip install -e ".[full]"

# Run a tutorial
python examples/01_getting_started.py
```

## Tutorial Index

### Beginner Tutorials

| # | Tutorial | Topics | Run Command |
|---|----------|--------|-------------|
| 1 | [Getting Started](01_getting_started.py) | Installation, basic usage, first function, evaluation, truth tables | `python examples/01_getting_started.py` |
| 2 | [Fourier Basics](02_fourier_basics.py) | WHT, Fourier coefficients, Parseval's identity, spectral weight | `python examples/02_fourier_basics.py` |
| 3 | [Common Families](03_common_families.py) | AND, OR, Majority, Parity, Tribes, Threshold, Dictator | `python examples/03_common_families.py` |

### Intermediate Tutorials

| # | Tutorial | Topics | Run Command |
|---|----------|--------|-------------|
| 4 | [Property Testing](04_property_testing.py) | BLR linearity, junta, monotonicity, symmetry, dictator tests | `python examples/04_property_testing.py` |
| 5 | [Query Complexity](05_query_complexity.py) | Sensitivity, block sensitivity, certificate complexity, D(f) | `python examples/05_query_complexity.py` |
| 6 | [Noise & Stability](06_noise_stability.py) | Noise stability, influences, total influence, voting applications | `python examples/06_noise_stability.py` |

### Advanced Tutorials

| # | Tutorial | Topics | Run Command |
|---|----------|--------|-------------|
| 7 | [Quantum Applications](07_quantum_applications.py) | Grover speedup, quantum walks, element distinctness, Q₂(f) | `python examples/07_quantum_applications.py` |

### Reference Examples

These files contain additional examples for specific use cases:

| File | Description |
|------|-------------|
| [educational_examples.py](educational_examples.py) | Examples for teaching Boolean logic |
| [representations_demo.py](representations_demo.py) | Circuit, BDD, and other representations |
| [advanced_features_demo.py](advanced_features_demo.py) | ANF, batch processing, GPU acceleration |

## Learning Path

### For Students

1. Start with tutorials 1-3 (Beginner)
2. Move to tutorials 4-5 (Testing & Complexity)
3. Explore tutorial 6 (Noise) for voting applications
4. Finish with tutorial 7 (Quantum) for quantum computing

### For Researchers

1. Skim tutorials 1-3 for API familiarity
2. Focus on tutorials 5-7 for complexity measures
3. Check `representations_demo.py` for advanced representations
4. See `advanced_features_demo.py` for performance optimization

### For Practitioners

1. Tutorial 1 for basic usage
2. Tutorial 4 for property testing
3. Check Jupyter notebooks in `notebooks/` for real-world examples

## Running All Tutorials

```bash
# Run all beginner tutorials
for i in 01 02 03; do python examples/${i}_*.py; done

# Run all tutorials
for f in examples/0*.py; do echo "=== $f ==="; python $f; done
```

## What You'll Learn

After completing these tutorials, you'll be able to:

- Create and manipulate Boolean functions
- Compute Fourier transforms and analyze spectra
- Test functions for properties (linearity, monotonicity, etc.)
- Measure query complexity (sensitivity, certificate complexity)
- Analyze noise stability and influences
- Estimate quantum speedups for Boolean functions
- Work with various representations (circuit, BDD, ANF)

## See Also

- **Jupyter Notebooks**: `notebooks/` - Interactive educational content
- **Documentation**: `docs/` - API reference and guides
- **Real-world Examples**: `docs/examples/` - S-box analysis, voting, ML
