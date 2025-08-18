<p align="center">
  <img src="logos/boo_horizontal.png" alt="BoolFunc Logo" width="1000"/>
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

## 🎯 Core Capabilities

- **Multiple Representations** 🔄  
  Seamless conversion between 12+ representations including truth tables, polynomials (ANF), circuits, BDDs, DNF/CNF forms, and spectral expansions with intelligent conversion graph system.

- **Spectral Analysis** 📊  
  Complete Fourier analysis toolkit with influence computation, noise stability, and spectral concentration with Numba JIT acceleration.

- **Property Testing** 🔬  
  Classical and quantum property testing algorithms including linearity, monotonicity, balance, and cryptographic properties.

- **Advanced Visualization** 📈  
  Interactive plots for spectral properties, influences, function behavior, and quantum analysis with matplotlib/plotly backends.

- **High Performance** ⚡  
  Optimized implementations with GPU acceleration (CuPy/CUDA), Numba JIT compilation, and intelligent batch processing.

- **Educational Tools** 🎓  
  Comprehensive examples and testing framework suitable for teaching theoretical computer science and quantum computing.

## 🆕 Advanced Features (New!)

- **🔄 Intelligent Conversion Graph**  
  Dijkstra-based optimal path finding between representations with cost analysis and caching.

- **⚡ Batch Processing Engine**  
  Automatic selection of vectorized, parallel, or GPU-accelerated processing based on data size and complexity.

- **🚀 GPU Acceleration**  
  CUDA/OpenCL support with automatic GPU selection and CPU fallback for large-scale computations.

- **🔥 Numba JIT Optimization**  
  Just-in-time compilation of critical operations with parallel execution and automatic warm-up.

- **🧪 Comprehensive Testing Framework**  
  Built-in validation, property testing, and performance profiling with detailed reporting.

- **🔌 Adapter System**  
  Easy integration with legacy functions, SymPy expressions, and external Boolean function libraries.

- **🎯 Advanced Error Models**  
  PAC learning bounds, noise models, and uncertainty propagation for robust analysis.

- **⚛️ Quantum Extensions**  
  Quantum Boolean function analysis, quantum property testing, and resource estimation.

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

# Evaluate functions (with automatic batch optimization)
print(f"XOR(1,0) = {xor.evaluate([1, 0])}")  # True
print(f"Majority(1,1,0) = {majority.evaluate([1, 1, 0])}")  # True

# Large batch evaluation (uses GPU/Numba if available)
large_batch = np.random.randint(0, 4, 10000)
results = xor.evaluate(large_batch)  # Optimized automatically

# Intelligent representation conversion
anf_data = xor.get_representation('anf')  # Algebraic Normal Form
fourier_coeffs = xor.get_representation('fourier_expansion')  # Uses conversion graph

# Advanced analysis with optimizations
analyzer = bf.SpectralAnalyzer(xor)
influences = analyzer.influences()  # Numba-accelerated
noise_stability = analyzer.noise_stability(0.9)

# Validate functions
is_valid = bf.quick_validate(xor)
print(f"Function validation: {is_valid}")

# Adapt external functions
python_func = lambda x: x[0] ^ x[1]
adapted = bf.adapt_callable(python_func, n_vars=2)

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

# 🔄 Supported Representations

BoolFunc supports **12+ different representations** for Boolean functions, each optimized for specific use cases:

## **Truth Table Representations**
- **Standard Truth Table** - Complete function mapping using NumPy arrays
  - Fast evaluation and conversion
  - Memory-intensive for large functions (2^n storage)
- **Sparse Truth Table** - Memory-efficient for functions with few true outputs
  - Compact storage for sparse functions
  - Efficient for functions with low density

## **Algebraic Representations**
- **Polynomial (ANF)** - Algebraic Normal Form over GF(2)
  - Canonical representation: f(x) = ⊕ᵢ aᵢ ∏ⱼ∈Sᵢ xⱼ
  - Essential for cryptographic analysis
  - Efficient for degree-limited functions
- **DNF Form** - Disjunctive Normal Form (OR of AND terms)
  - Natural representation for many functions
  - Useful for satisfiability problems
- **CNF Form** - Conjunctive Normal Form (AND of OR terms)
  - Standard form for SAT solvers
  - Important for constraint satisfaction

## **Structural Representations**
- **Boolean Circuits** - Gate-level implementations
  - Supports AND, OR, NOT, XOR, NAND, NOR gates
  - Hardware-friendly representation
  - Circuit optimization and analysis
- **Binary Decision Diagrams (BDD)** - Reduced Ordered BDDs
  - Canonical representation for many functions
  - Efficient manipulation and analysis
  - Memory-efficient for structured functions
- **Linear Threshold Functions (LTF)** - Weighted threshold-based functions
  - Neural network-inspired representation
  - Useful for learning and approximation

## **Spectral Representations**
- **Fourier Expansion** - Walsh-Hadamard transform coefficients
  - Essential for spectral analysis
  - Influence and noise stability computation
  - Property testing algorithms
- **Distribution** - Probability distribution over function outputs
  - Statistical analysis and sampling
  - Random function generation

## **Symbolic Representations**
- **Symbolic** - Abstract symbolic manipulation
  - Formal verification and analysis
  - Mathematical theorem proving

**Automatic Conversion:** All representations support seamless conversion between each other, enabling flexible analysis workflows.

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

### **Truth Table Representations**
- **Standard Truth Table** - Complete function mapping using NumPy arrays
- **Sparse Truth Table** - Memory-efficient for functions with few true outputs

### **Algebraic Representations**
- **Polynomial (ANF)** - Algebraic Normal Form over GF(2)
- **DNF Form** - Disjunctive Normal Form (OR of AND terms)
- **CNF Form** - Conjunctive Normal Form (AND of OR terms)

### **Structural Representations**
- **Boolean Circuits** - Gate-level implementations (AND, OR, NOT, XOR, NAND, NOR)
- **Binary Decision Diagrams (BDD)** - Reduced Ordered BDDs for efficient manipulation
- **Linear Threshold Functions (LTF)** - Weighted threshold-based functions

### **Spectral Representations**
- **Fourier Expansion** - Walsh-Hadamard transform coefficients
- **Distribution** - Probability distribution over function outputs

### **Symbolic Representations**
- **Symbolic** - Abstract symbolic manipulation and analysis

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

# 🆕 NEW: Advanced features demo (ANF, conversion graph, batch processing, GPU)
python examples/advanced_features_demo.py

# 🆕 NEW: Quantum Boolean function analysis and algorithms
python examples/quantum_analysis_demo.py

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

# 🧹 Cleanup old visualization files (if any)
python cleanup_old_files.py
```

### Example Categories

- **`usage.py`**: Core functionality, perfect for getting started
- **🆕 `advanced_features_demo.py`**: Complete showcase of all advanced features including ANF representation, conversion graph, batch processing, GPU acceleration, Numba JIT, testing framework, and adapters
- **🆕 `quantum_analysis_demo.py`**: Quantum Boolean function analysis including quantum property testing, resource estimation, and quantum vs classical comparisons
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
