<p align="center">
  <img src="logos/boo_horizontal.png" alt="BoolFunc Logo" width="600"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/boolfunc/"><img src="https://img.shields.io/pypi/v/boolfunc.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/boolfunc/"><img src="https://img.shields.io/pypi/dm/boolfunc.svg?label=PyPI%20downloads" alt="PyPI downloads"></a>
  <a href="https://github.com/GabbyTab/boolfunc/actions"><img src="https://github.com/GabbyTab/boolfunc/workflows/CI/badge.svg" alt="Build Status"></a>
  <a href="https://github.com/GabbyTab/boolfunc/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/GabbyTab/boolfunc/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/GabbyTab/boolfunc/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/typing-checked-blue.svg" alt="Typing"></a>
</p>

# 🚀 Features

## Core Capabilities

- **Multiple Representations**  
  Seamless conversion between truth tables, polynomials (ANF), circuits, BDDs, and spectral forms.

- **Spectral Analysis**  
  Complete Fourier analysis toolkit with influence computation and noise stability.

- **Property Testing**  
  Classical property testing algorithms including linearity, balance, and cryptographic properties.

- **Advanced Visualization**  
  Interactive plots for spectral properties, influences, and function behavior.

- **High Performance**  
  Optimized implementations using NumPy and SciPy with optional acceleration.

- **Educational Tools**  
  Comprehensive examples suitable for teaching theoretical computer science.

---

## Built-in Function Library

- **Tribes functions** - Disjunction of k-wise conjunctions  
- **Majority functions** - Returns 1 if more than half inputs are 1
- **Parity functions** - XOR operations and linear functions
- **Dictator functions** - Single variable dependencies
- **Constant functions** - Always return the same value
- **Random Boolean functions** - For testing and analysis

# 📦 Installation

## Install from PyPI (Recommended)

```bash
pip install boolfunc
```

## From Source (Development)

```bash
git clone https://github.com/GabbyTab/boolfunc.git
cd boolfunc
pip install -e .
```

## With Optional Dependencies

```bash
# For visualization features
pip install boolfunc[visualization]

# For development (testing, linting)
pip install boolfunc[dev]

# All features
pip install boolfunc[visualization,dev]
```

# 🏃‍♀️ Quick Start

```python
import boolfunc as bf
import numpy as np

# Create Boolean functions from truth table
xor = bf.create([0, 1, 1, 0])  # XOR function
majority = bf.BooleanFunctionBuiltins.majority(3)  # Built-in majority function
tribes = bf.BooleanFunctionBuiltins.tribes(k=2, n=6)  # Tribes function

# Evaluate functions
print(f"XOR(1,0) = {xor.evaluate([1, 0])}")  # True
print(f"Majority(1,1,0) = {majority.evaluate([1, 1, 0])}")  # True

# Seamless representation conversion
polynomial = xor.to_polynomial()
circuit = xor.to_circuit()

# Spectral analysis
analyzer = bf.SpectralAnalyzer(xor)
influences = analyzer.influences()
total_influence = analyzer.total_influence()
noise_stability = analyzer.noise_stability(rho=0.9)

# Fourier analysis
fourier_coeffs = analyzer.fourier_expansion()

# Property testing
tester = bf.PropertyTester(xor)
is_balanced = tester.balanced_test()
is_linear = tester.blr_linearity_test(num_queries=3)

# Visualization (requires matplotlib/plotly)
viz = bf.Visualizer(xor)
viz.plot_influences()
viz.plot_fourier_spectrum()
```

---

# 📚 Documentation

Comprehensive documentation and examples are available in the repository:

- **Getting Started Guide:** Basic concepts and first steps in the examples directory
- **API Reference:** Complete function and class documentation via docstrings
- **Mathematical Background:** Theory and algorithms explained in source comments
- **Advanced Examples:** Research-oriented tutorials and use cases

**Quick Links:**
- [Examples Directory](examples/) - Comprehensive usage examples
- [API Documentation](src/boolfunc/) - Source code with detailed docstrings
- [Contributing Guide](CONTRIBUTING.md) - Development and contribution guidelines

---

# 🧪 Core Modules

## `boolfunc.core`
- Multiple Boolean function representations with automatic conversion:
  - Truth table representation
  - Polynomial (ANF) representation  
  - Circuit representation
  - BDD representation
  - Spectral representation

## `boolfunc.analysis`
- Comprehensive spectral analysis tools:
  - Fourier expansion and Walsh-Hadamard transforms
  - Variable influences and sensitivity analysis
  - Noise stability and hypercontractivity
  - Spectral concentration measures

## `boolfunc.testing`
- Property testing algorithms:
  - BLR linearity testing
  - Constant function testing
  - Balance testing
  - Custom testing framework

## `boolfunc.visualization`
- Advanced plotting capabilities:
  - Influence distribution plots
  - Fourier spectrum visualization
  - Noise stability curves
  - Interactive dashboards

---

# 🔬 Research Applications

BoolFunc is designed for researchers in:

- **Theoretical Computer Science:** Analysis of computational complexity and Boolean circuits
- **Cryptography:** Boolean function cryptanalysis and S-box design
- **Property Testing:** Development of efficient testing algorithms
- **Machine Learning:** Boolean function learning and feature analysis
- **Combinatorics:** Extremal problems and probabilistic methods

---

# 📈 Performance

BoolFunc is optimized for both small research examples and large-scale computations:

- **Vectorized Operations:** Efficient NumPy-based implementations
- **Memory Efficiency:** Sparse representations for large functions
- **Scalable Design:** Handles functions with up to 20+ variables efficiently

**Performance Example:**
```python
# Analyze 10-variable Boolean function
import boolfunc as bf
import time

f = bf.BooleanFunctionBuiltins.tribes(k=2, n=10)
start = time.time()

analyzer = bf.SpectralAnalyzer(f)
influences = analyzer.influences()
fourier_coeffs = analyzer.fourier_expansion()

print(f"Analysis completed in {time.time() - start:.3f} seconds")
```

---

# 🧪 Testing

Run the test suite to verify installation:
```bash
# Run all tests
pytest

# Run with coverage  
pytest --cov=boolfunc

# Run specific test file
pytest tests/integration/test_basic_functionality.py
```

# 📋 Examples

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

# Complete showcase of all library features
python examples/complete_demo.py
```

### Example Categories

- **`usage.py`**: Core functionality, perfect for getting started
- **`educational_examples.py`**: Boolean logic fundamentals, suitable for teaching
- **`advanced_analysis.py`**: Research workflows and mathematical property verification
- **`representations_demo.py`**: Multiple representations (circuits, BDDs, conversions)
- **`visualization_examples.py`**: Plotting and interactive analysis dashboards
- **`complete_demo.py`**: Comprehensive showcase of all library features

---

# 🤝 Contributing

We welcome contributions from the research community!  
Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Development Setup

```bash
# Clone repository
git clone https://github.com/GabbyTab/boolfunc.git
cd boolfunc

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests and linting
pytest
black src/
flake8 src/
mypy src/
```

## How to Contribute

- **Bug Reports:** Use GitHub issues with detailed reproduction steps
- **Feature Requests:** Discuss proposals in GitHub discussions
- **Code Contributions:** Submit pull requests with tests and documentation
- **Documentation:** Help improve docs and add examples
- **Research Integration:** Share your research applications and use cases

---

# 📞 Support and Community

- **Issues:** [GitHub Issues](https://github.com/GabbyTab/boolfunc/issues)
- **Discussions:** [GitHub Discussions](https://github.com/GabbyTab/boolfunc/discussions)
- **Repository:** [GitHub Repository](https://github.com/GabbyTab/boolfunc)

---

# 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

# 🙏 Acknowledgments

BoolFunc builds upon decades of research in Boolean function analysis.  
We acknowledge the foundational work of researchers in:

- Harmonic analysis on Boolean cubes
- Property testing theory
- Computational complexity theory
- Modern cryptanalysis techniques

---

# 📖 Citation

If you use BoolFunc in your research, please cite:

```bibtex
@software{boolfunc2024,
  title={BoolFunc: A Python Library for Boolean Function Analysis},
  author={Gabriel Taboada},
  year={2024},
  url={https://github.com/GabbyTab/boolfunc},
  version={0.2.0}
}
```

---

Happy Boolean function analyzing! 🎯

<p align="center">
  <img src="logos/boo_alt.png" alt="BoolFunc Logo" width="200"/>
</p>
