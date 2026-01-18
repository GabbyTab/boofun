<p align="center">
  <img src="logos/boo_horizontal.png" alt="BoolFunc Logo" width="800"/>
</p>

<p align="center">
  <strong>A Comprehensive Python Library for Boolean Function Analysis and Computation</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/boolfunc/"><img src="https://img.shields.io/pypi/v/boolfunc.svg" alt="PyPI version"></a>
  <a href="https://github.com/GabbyTab/boolfunc/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
  <a href="https://github.com/GabbyTab/boolfunc/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://gabbytab.github.io/boolfunc/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg" alt="Documentation"></a>
</p>

## Overview

BoolFunc is a comprehensive Python library for the analysis and manipulation of Boolean functions, designed for researchers and practitioners in theoretical computer science, computational complexity, and quantum computing. The library provides a unified framework for working with Boolean functions across multiple mathematical representations, enabling efficient computation of spectral properties, influence measures, and complexity-theoretic characteristics.

## Key Features

### Multiple Representations
- **Truth Tables**: Standard binary representation for direct evaluation
- **Algebraic Normal Form (ANF)**: Polynomial representation over GF(2) 
- **Fourier Expansion**: Spectral representation 
- **Binary Decision Diagrams (BDDs)**: Compressed graph-based representation
- **Circuit Representation**: Gate-based computational model
- **CNF/DNF Forms**: Conjunctive and disjunctive normal forms

### Spectral Analysis
- **Fourier Coefficients**: Complete Walsh-Hadamard transform computation
- **Influence Measures**: Variable influence and total influence calculation  
- **Noise Stability**: Correlation under random noise for fixed correlation parameters
- **Spectral Concentration**: Degree-based spectral weight distribution

### Property Testing
- **Linearity Testing**: BLR-style linearity detection algorithms
- **Monotonicity Testing**: Efficient monotonicity verification
- **Balance Testing**: Statistical balance and bias measurement
- **Junta Testing**: k-junta identification and variable relevance

### Built-in Function Classes
- **Tribes Functions**: Disjunction of k-wise conjunctions for complexity analysis
- **Majority Functions**: Threshold functions for voting theory applications  
- **Parity Functions**: Linear functions over GF(2) for coding theory
- **Dictator Functions**: Single-variable dependencies for social choice theory

## Architecture Overview

The library is organized into several interconnected modules:

![Module Architecture](docs/architecture_diagram.png)

*The diagram above illustrates the modular architecture of BoolFunc, showing how different components interact to provide a comprehensive Boolean function analysis framework.*

## Quick Start

### Installation

```bash
pip install boolfunc
```

### Basic Usage

```python
import boolfunc as bf

# Create Boolean functions (multiple ways)
xor = bf.create([0, 1, 1, 0])       # From truth table
maj = bf.majority(5)                 # Built-in majority
parity = bf.parity(4)                # Built-in parity
dictator = bf.dictator(3, i=0)       # Dictator on first variable
tribes = bf.tribes(2, 6)             # Tribes function

# Evaluate
print(f"MAJ(1,1,0,0,1) = {maj.evaluate([1,1,0,0,1])}")  # 1

# Direct analysis methods (NEW simplified API!)
print(f"Fourier degree: {maj.degree()}")
print(f"Total influence: {maj.total_influence():.4f}")
print(f"Max influence: {maj.max_influence():.4f}")
print(f"Variance: {maj.variance():.4f}")
print(f"Noise stability (ρ=0.9): {maj.noise_stability(0.9):.4f}")

# Quick summary of all metrics
print(maj.analyze())
# {'n_vars': 5, 'is_balanced': True, 'variance': 1.0, 'degree': 5, ...}

# Fourier analysis
coeffs = maj.fourier()                    # All Fourier coefficients
heavy = maj.heavy_coefficients(tau=0.3)   # Coefficients above threshold
weights = maj.spectral_weight_by_degree() # Weight at each degree

# Property testing
print(f"Is linear: {maj.is_linear()}")
print(f"Is monotone: {maj.is_monotone()}")
print(f"Is balanced: {maj.is_balanced()}")
print(f"Is 2-junta: {maj.is_junta(2)}")
```

### Advanced Features

```python
# Linear Threshold Functions (LTFs)
ltf = bf.weighted_majority(5, weights=[3, 2, 2, 1, 1])
from boolfunc.analysis.ltf_analysis import analyze_ltf, is_ltf
print(f"Is LTF: {is_ltf(maj)}")
analysis = analyze_ltf(ltf)

# Function Growth Tracking
from boolfunc.families import MajorityFamily, GrowthTracker
tracker = GrowthTracker(MajorityFamily())
tracker.mark("total_influence")
tracker.mark("max_influence")
tracker.observe([3, 5, 7, 9, 11, 13])
tracker.plot("total_influence", show_theory=True)

# Global Hypercontractivity (p-biased analysis)
from boolfunc.analysis.global_hypercontractivity import GlobalHypercontractivityAnalyzer
analyzer = GlobalHypercontractivityAnalyzer(maj, p=0.3)
print(analyzer.summary())
```

## Mathematical Foundation

BoolFunc operates on Boolean functions f: {0,1}ⁿ → {0,1}, providing tools for:

- **Fourier Analysis**: Walsh-Hadamard transform and spectral properties
- **Influence Theory**: Variable influence I_i(f) = Pr[f(x) ≠ f(x ⊕ eᵢ)]  
- **Noise Stability**: NS_ρ(f) = E[f(x)f(N_ρ(x))] for noise operator N_ρ
- **Complexity Measures**: Certificate complexity, sensitivity, block sensitivity
- **Learning Theory**: PAC learning with membership and equivalence queries
- **p-Biased Analysis**: Generalized Fourier analysis for biased input distributions
- **Global Hypercontractivity**: Inverse Cauchy-Schwarz with noise

## Applications

- **Computational Complexity**: Analysis of Boolean function complexity classes
- **Social Choice Theory**: Voting systems and preference aggregation
- **Cryptography**: Security analysis of Boolean functions in stream ciphers  
- **Quantum Computing**: Boolean function analysis in quantum algorithms
- **Machine Learning**: Feature selection and Boolean concept learning

## Documentation

### Generating Documentation

BoolFunc uses Sphinx for comprehensive documentation generation with automatic API reference creation:

#### Quick Documentation Setup

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Generate HTML documentation
cd docs/
make html

# Open in browser
open _build/html/index.html  # macOS
firefox _build/html/index.html  # Linux
```

#### Documentation Components

The documentation system includes:

- **API Reference**: Auto-generated from docstrings using Sphinx autodoc
- **Mathematical Notation**: LaTeX math rendering with MathJax
- **Code Examples**: Syntax-highlighted Python code blocks
- **Architecture Diagrams**: Visual module interaction diagrams
- **Theory Sections**: Mathematical foundations and algorithmic details

#### Available Documentation Formats

```bash
# HTML documentation (recommended)
make html

# PDF documentation (requires LaTeX)
make latexpdf

# EPUB documentation
make epub

# Clean build files
make clean
```

#### Customizing Documentation

The documentation configuration is in `docs/conf.py`. Key settings:

- **Theme**: `sphinx_rtd_theme` for professional appearance
- **Extensions**: Autodoc, Napoleon, MathJax for comprehensive documentation
- **API Generation**: Automatic module documentation with autosummary

### API Reference

The complete API documentation includes:

- **Core Classes**: BooleanFunction, SpectralAnalyzer, PropertyTester
- **Representations**: All 12+ representation types with mathematical descriptions
- **Analysis Tools**: Fourier analysis, influence computation, property testing
- **Advanced Features**: Conversion graph, batch processing, GPU acceleration
- **Examples**: Comprehensive usage examples with mathematical context

#### Documentation Structure

```
docs/
├── _build/html/          # Generated HTML documentation
├── conf.py               # Sphinx configuration
├── index.rst            # Main documentation index
├── quickstart.rst       # Quick start guide
├── architecture_diagram.png  # Module interaction diagram
└── Makefile             # Documentation build commands
```

#### Regenerating Documentation

After making changes to docstrings or adding new modules:

```bash
# Clean previous build
cd docs && make clean

# Regenerate with latest changes
make html

# View updated documentation
open _build/html/index.html
```

Online documentation: [https://boolfunc.readthedocs.io](https://boolfunc.readthedocs.io)

---

## 📓 Educational Notebooks

The `notebooks/` directory contains 16 Jupyter notebooks aligned with CS 294-92: Analysis of Boolean Functions (O'Donnell book):

### Homework Notebooks
| Notebook | Topics |
|----------|--------|
| `hw1_fourier_expansion.ipynb` | Fourier basics, Parseval, Plancherel |
| `hw2_ltf_decision_trees.ipynb` | LTFs, decision trees, Banzhaf power |
| `hw3_dnf_restrictions.ipynb` | DNFs, random restrictions, Switching Lemma |
| `hw4_hypercontractivity.ipynb` | KKL Theorem, Bonami's Lemma, noise stability |

### Lecture Notebooks (Complete!)
| Lecture | Topic |
|---------|-------|
| 1 | Fourier Expansion, Orthogonality |
| 2 | BLR Linearity Testing, Convolution |
| 3 | Social Choice, Influences |
| 4 | Poincaré Inequality, KKL preview |
| 5 | Noise Stability, Arrow's Theorem |
| 6 | Spectral Concentration, Learning |
| 7 | Goldreich-Levin Algorithm |
| 8 | Learning Juntas |
| 9 | DNFs, Random Restrictions |
| 10 | Fourier Concentration of DNFs |
| 11 | Invariance Principle, Gaussian Analysis |

### Research Paper Notebooks
| Paper | Topic |
|-------|-------|
| `global_hypercontractivity.ipynb` | Keevash et al. p-biased hypercontractivity |
| `asymptotic_visualization.ipynb` | Growth tracking & visualization |

```bash
# Launch Jupyter
cd notebooks
jupyter notebook
```

---

## Examples

The `examples/` directory contains comprehensive usage examples:

```bash
# Basic usage and core functionality
python examples/usage.py

# Advanced features (ANF, conversion graph, batch processing, GPU)
python examples/advanced_features_demo.py

# Quantum Boolean function analysis
python examples/quantum_analysis_demo.py

# Educational examples for teaching
python examples/educational_examples.py

# Visualization examples
python examples/visualization_examples.py
```

## Installation

### Install from PyPI (Recommended)

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

## Testing

Run the test suite to verify installation:

```bash
pytest tests/
```

For coverage analysis:
```bash
pytest --cov=boolfunc tests/
```

## Supported Representations

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

## Advanced Implementation Details

### Performance Optimizations

- **🔄 Intelligent Conversion Graph**: Dijkstra-based optimal path finding between representations with cost analysis and caching
- **⚡ Batch Processing Engine**: Automatic selection of vectorized, parallel, or GPU-accelerated processing based on data size and complexity  
- **🚀 GPU Acceleration**: CUDA/OpenCL support with automatic GPU selection and CPU fallback for large-scale computations
- **🔥 Numba JIT Optimization**: Just-in-time compilation of critical operations with parallel execution and automatic warm-up

### Integration Features

- **🔌 Adapter System**: Easy integration with legacy functions, SymPy expressions, and external Boolean function libraries
- **🎯 Advanced Error Models**: PAC learning bounds, noise models, and uncertainty propagation for robust analysis
- **🧪 Comprehensive Testing Framework**: Built-in validation, property testing, and performance profiling with detailed reporting
- **⚛️ Quantum Extensions**: Quantum Boolean function analysis, quantum property testing, and resource estimation

### Technical Architecture

The library employs several design patterns for extensibility and performance:

- **Strategy Pattern**: Pluggable representation implementations with unified interface
- **Factory Pattern**: Automatic type detection and object creation from diverse input formats
- **Observer Pattern**: Event-driven computation caching and invalidation
- **Registry Pattern**: Dynamic discovery and registration of representation strategies

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

See our [ROADMAP.md](ROADMAP.md) for planned features and areas where help is needed!

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

We would like to acknowledge the material learned in CS 294-92: Analysis of Boolean Functions (Spring 2025), which provided the theoretical foundation for this project.

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
