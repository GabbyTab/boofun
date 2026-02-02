# Contributing to BooFun

Welcome! This guide helps you set up your development environment and navigate the codebase.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/GabbyTab/boofun.git
cd boofun

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with all dependencies
pip install -e ".[dev,visualization]"

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/ -v --tb=short
```

Or use Docker:
```bash
docker-compose up dev      # Interactive dev shell
docker-compose run test    # Run tests
docker-compose up notebook # Jupyter at localhost:8888
```

## Project Structure

```
boofun/
├── src/boofun/           # Main library code
│   ├── __init__.py       # Public API exports
│   ├── api.py            # High-level create() function
│   ├── core/             # Core Boolean function implementation
│   │   ├── base.py       # BooleanFunction class
│   │   ├── builtins.py   # Built-in functions (AND, OR, majority, etc.)
│   │   ├── optimizations.py  # Numba-accelerated algorithms
│   │   ├── representations/  # Different function representations
│   │   └── conversion_graph.py  # Automatic conversions
│   ├── analysis/         # Analysis modules
│   │   ├── fourier.py    # Fourier/spectral analysis
│   │   ├── sensitivity.py    # Sensitivity measures
│   │   ├── query_complexity.py  # BFW-style complexity
│   │   └── ...
│   ├── testing/          # Property testing (BLR, monotonicity, etc.)
│   ├── families/         # Function families with growth analysis
│   ├── quantum/          # Quantum computing extensions
│   └── visualization/    # Plotting and visualization
├── tests/                # Test suite
│   ├── core/             # Core functionality tests
│   ├── analysis/         # Analysis module tests
│   ├── correctness/      # Mathematical correctness tests
│   └── golden/           # Golden test data
├── notebooks/            # Educational Jupyter notebooks
├── examples/             # Example scripts
└── docs/                 # Documentation
```

## Key Concepts

### Bit-Ordering Convention

BooFun uses **LSB = x₀** (least significant bit is variable 0):

```python
# For n=3 variables, input index 5 = 0b101 means:
# x₀ = 1, x₁ = 0, x₂ = 1

def index_to_bits(idx, n):
    """Convert integer index to variable assignment."""
    return [(idx >> i) & 1 for i in range(n)]

# Example: index 5 with n=3 → [1, 0, 1] → x₀=1, x₁=0, x₂=1
```

### Fourier Convention

BooFun follows the **O'Donnell convention** (Analysis of Boolean Functions):

- Boolean domain: `{0, 1}` maps to `{+1, -1}` via `(-1)^x`
- Fourier basis: `χ_S(x) = ∏_{i∈S} (-1)^{x_i}`
- Parseval: `∑_S f̂(S)² = 1` for balanced functions

```python
# Truth table to ±1 conversion
pm_values = 1 - 2 * truth_table  # 0→+1, 1→-1

# Fourier coefficients satisfy Parseval's identity
assert abs(sum(coeff**2 for coeff in fourier_coeffs) - 1.0) < 1e-10
```

### Representations

Boolean functions can have multiple representations:

| Representation | Description | Use Case |
|---------------|-------------|----------|
| `truth_table` | Array of 2^n outputs | Small n, exact operations |
| `anf` | Algebraic Normal Form | GF(2) analysis |
| `fourier` | Fourier coefficients | Spectral analysis |
| `function` | Python callable | Large n, lazy evaluation |
| `circuit` | Gate-based circuit | Complexity analysis |
| `bdd` | Binary Decision Diagram | SAT, optimization |

## Development Workflow

### Before Making Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Ensure pre-commit hooks are installed: `pre-commit install`

### Making Changes

1. Write tests first (TDD recommended)
2. Implement your changes
3. Run pre-commit: `pre-commit run --all-files`
4. Run tests: `pytest tests/ -v`
5. Check types: `mypy src/boofun --ignore-missing-imports`

### Code Style

- **Formatting**: Black (line length 100)
- **Imports**: isort (black profile)
- **Linting**: flake8
- **Types**: mypy (strict mode goal)

### Writing Tests

#### Test Correctness, Not Just "No Exception"

```python
# BAD: Only checks no exception
def test_majority_runs():
    f = bf.majority(5)
    f.fourier()  # No assertion!

# GOOD: Verifies mathematical correctness
def test_majority_fourier_correctness():
    f = bf.majority(5)
    coeffs = f.fourier()

    # Verify Parseval's identity
    assert abs(sum(c**2 for c in coeffs) - 1.0) < 1e-10

    # Verify known property: majority has specific weight distribution
    total_influence = sum(i * c**2 for i, c in enumerate(coeffs) if bin(i).count('1') == 1)
    assert 0 < total_influence < f.n_vars
```

#### Golden Tests for Bit-Ordering

```python
def test_bit_ordering_golden():
    """Verify bit-ordering matches golden data."""
    import json
    with open('tests/golden/bit_ordering.json') as f:
        golden = json.load(f)

    for case in golden['cases']:
        f = bf.create(n=case['n'], truth_table=case['truth_table'])
        assert f.evaluate(case['input']) == case['expected_output']
```

### Commit Messages

Follow conventional commits:

```
feat: add noise stability computation
fix: correct bit-ordering in ANF conversion
docs: update Fourier convention documentation
test: add golden tests for majority function
refactor: extract WHT to separate module
perf: optimize influence computation with Numba
```

## Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/unit/`): Test individual functions
2. **Integration Tests** (`tests/integration/`): Test module interactions
3. **Correctness Tests** (`tests/correctness/`): Mathematical verification
4. **Golden Tests** (`tests/golden/`): Regression with known-good data
5. **Property Tests** (`tests/property/`): Hypothesis-based fuzzing
6. **Adversarial Tests** (`tests/adversarial/`): Edge cases and stress tests

### Running Specific Tests

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/correctness/

# Run tests matching pattern
pytest tests/ -k "fourier"

# Run with coverage
pytest tests/ --cov=boofun --cov-report=html

# Run mutation testing (slow)
mutmut run --paths-to-mutate=src/boofun/core/optimizations.py
```

## Common Tasks

### Adding a New Analysis Function

1. Add implementation to appropriate module in `src/boofun/analysis/`
2. Export in `src/boofun/analysis/__init__.py`
3. Add method to `SpectralAnalyzer` if appropriate
4. Write tests in `tests/analysis/`
5. Add docstring with mathematical definition and example
6. Update documentation if user-facing

### Adding a New Representation

1. Create file in `src/boofun/core/representations/`
2. Implement `BooleanFunctionRepresentation` protocol
3. Register with `@register_strategy` decorator
4. Add conversion functions to/from other representations
5. Update conversion graph in `conversion_graph.py`
6. Write comprehensive tests including round-trip conversions

### Debugging Numba Issues

```python
# Disable Numba for debugging
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

# Check if Numba is being used
from boofun.core.optimizations import HAS_NUMBA
print(f"Numba available: {HAS_NUMBA}")
```

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: All PRs require review before merging

## Mathematical References

- O'Donnell, R. (2014). *Analysis of Boolean Functions*. Cambridge University Press.
- Buhrman, H., & de Wolf, R. (2002). *Complexity measures and decision tree complexity*.
- Blais, E. (2009). *Testing juntas nearly optimally*.
