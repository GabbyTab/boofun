# BoolFunc: Boolean Function Analysis Library

A Python library for creating, analyzing, and visualizing Boolean functions using multiple representations and advanced mathematical techniques.



## Features

- **Multiple Representations**: Truth tables, polynomials (ANF), circuits, BDDs, and more
- **Spectral Analysis**: Fourier analysis, influence computation, noise stability  
- **Property Testing**: Linearity, constancy, balance testing, and cryptographic properties
- **Built-in Functions**: Majority, parity, dictator, constant, and tribes functions
- **Visualization**: Interactive plots, dashboards, and function comparisons
- **Educational Tools**: Examples suitable for teaching theoretical computer science

## Built-in Functions

- `majority(n)` - Returns 1 if more than half of n inputs are 1 (n must be odd)
- `parity(n)` - Returns 1 if odd number of inputs are 1 (XOR function)  
- `dictator(i, n)` - Returns the value of the i-th input variable
- `constant(value, n)` - Always returns the same value
- `tribes(k, n)` - Disjunction of k-wise conjunctions


## Installation

### From Source (Development)
```bash
git clone https://github.com/GabbyTab/boolfunc.git
cd boolfunc
pip install -e .
```

### With Optional Dependencies
```bash
# For visualization features
pip install -e ".[visualization]"

# For development (testing, linting)
pip install -e ".[dev]"
```
## Quick Start

```python
import boolfunc as bf

# Create Boolean functions
xor = bf.create([0, 1, 1, 0])  # XOR function from truth table
majority = bf.BooleanFunctionBuiltins.majority(3)  # Built-in majority function

# Evaluate functions
print(f"XOR(1,0) = {xor.evaluate([1, 0])}")  # True
print(f"Majority(1,1,0) = {majority.evaluate([1, 1, 0])}")  # True

# Spectral analysis
analyzer = bf.SpectralAnalyzer(xor)
influences = analyzer.influences()
print(f"Variable influences: {influences}")  # [1.0, 1.0] for XOR

# Property testing
tester = bf.PropertyTester(xor)
is_balanced = tester.balanced_test()
print(f"XOR is balanced: {is_balanced}")  # True
```

See the `examples/` directory for comprehensive tutorials covering all features.


## Testing

Run the test suite to verify installation:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=boolfunc

# Run specific test file
pytest tests/integration/test_basic_functionality.py
```

## Examples

The `examples/` directory contains comprehensive usage examples:

```bash
# Basic library usage and core functionality
python examples/usage.py

# Educational examples for teaching and learning
python examples/educational_examples.py

# Advanced analysis and research workflows  
python examples/advanced_analysis.py

# Representation demos (circuits, BDDs, etc.)
python examples/representations_demo.py

# Visualization examples (requires matplotlib/plotly)
python examples/visualization_examples.py
```

### Example Categories

- **`usage.py`**: Core functionality, perfect for getting started
- **`educational_examples.py`**: Boolean logic fundamentals, suitable for teaching
- **`advanced_analysis.py`**: Research workflows and mathematical property verification
- **`representations_demo.py`**: Multiple representations (circuits, BDDs, conversions)
- **`visualization_examples.py`**: Plotting and interactive analysis dashboards
- **`complete_demo.py`**: Comprehensive showcase of all library features

### Quick Demo
For a complete overview of all capabilities:
```bash
python examples/complete_demo.py
```

## Development

For development setup:
```bash
# Clone and install in development mode
git clone https://github.com/GabbyTab/boolfunc.git
cd boolfunc
pip install -e ".[dev]"

# Run tests and linting
pytest
black src/
flake8 src/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use BoolFunc in your research:

```bibtex
@software{boolfunc2024,
  title={BoolFunc: A Python Library for Boolean Function Analysis},
  author={Gabriel Taboada},
  year={2024},
  url={https://github.com/GabbyTab/boolfunc},
  version={0.2.0}
}
```
