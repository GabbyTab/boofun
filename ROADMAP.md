# Roadmap

**Version:** 1.3.0 (planned)
**Updated:** February 2026

## Vision

BooFun is the computational companion to O'Donnell's *Analysis of Boolean Functions*. The library makes every theorem runnable: create a function, compute its Fourier coefficients, verify KKL, visualize thresholds. 23 interactive notebooks cover Chapters 1-11.

**Primary audience**: TCS graduate students and researchers.
**Secondary**: Cryptographers analyzing S-boxes. Complexity theorists exploring conjectures.

---

## Current State (v1.2.0)

72% test coverage, 3200+ tests, 13 examples, 23 notebooks. Published on PyPI. Cross-validated against SageMath, thomasarmel/boolean_function, and BoolForge. See [CHANGELOG.md](CHANGELOG.md) for what shipped.

### What shipped in v1.2.0

- Numba moved to optional `[performance]` extra (ADR-004)
- Standard `__init__` pattern replaces `__new__`/`_init` (ADR-009)
- `f.is_global(alpha)` method on BooleanFunction
- `Measure` class in `core/spaces.py` for p-biased analysis
- `bf.f2_polynomial(n, monomials)` for F2-polynomial creation
- Adaptive sampling with `target_error` in `estimate_fourier_coefficient`
- Representation round-trip tests (truth_table <-> fourier, truth_table <-> ANF)
- Per-module mypy enforcement (20+ modules)
- `ComputeCache` rewritten with `OrderedDict` for O(1) eviction
- conda-forge recipe submitted (PR #31964)

---

## v1.3.0 Goals

### Cleanup & Architecture

| Item | Why |
|------|-----|
| Consolidate GPU modules | Merge `gpu_acceleration.py` and `gpu.py` into one. Remove unused stubs, keep CuPy WHT. |
| Rename `quantum/` → `quantum_complexity/` | Renamed module to honestly reflect what it does: classical computation of quantum complexity bounds. Deleted classical algorithm duplicates (Fourier, influences, property testing) that were dressed up with `quantum_` prefixes — use `SpectralAnalyzer` and `PropertyTester` instead. Backwards-compat aliases provided. Full quantum simulation deferred to v2.0.0. |
| Delete `setup.cfg` | Consolidate into `pyproject.toml`. |
| Simplify conversion graph | Replace Dijkstra with two-level dispatch (source -> truth_table -> target). |
| Lazy imports in `__init__.py` | Only if import time exceeds 5 seconds. |
| Upgrade mypy to latest | Bump `mypy~=1.13.0` pin, fix ~59 `var-annotated` errors from numpy 2.x stubs, remove `ignore_errors` overrides for 20 modules in `pyproject.toml`. |

---

## v1.4.0 Goals

### Pseudorandomness (CHLT ITCS 2019)
- Fractional PRG utilities: truncated Gaussian sampler, polarizing walk
- Epsilon-biased distribution construction
- Fooling verification

### Performance (for research at larger n)
- GPU-accelerated WHT via CuPy for n > 20
- Parallel batch processing for exhaustive search
- Lazy evaluation for oracle-access functions
- Rust FFI for performance-critical paths (thomasarmel integration)

### Infrastructure
- ~~conda-forge~~ (PR #31964 submitted)
- Interactive Jupyter widgets
- Manim animations for educational content

### Function Families
- N-value generators (`n_values.odd()`, `n_values.powers()`)
- Parameter sweep visualization
- BoolForge-inspired constrained random generation

---

## v2.0.0 Goals

### Quantum Simulation (extending `quantum_complexity`)

The `boofun.quantum_complexity` module (renamed from `quantum` in v1.3.0) currently provides honest classical computation of quantum complexity bounds — Grover iteration formulas, quantum walk hitting times, element distinctness bounds. v2.0.0 will add actual quantum simulation alongside these analytical results.

| Item | Why |
|------|-----|
| Statevector simulation backend | Implement a lightweight statevector simulator (no Qiskit dependency) for small-n quantum circuits, so users can see simulated Grover and quantum walk results up to ~12 qubits. |
| Qiskit / Cirq integration (optional extra) | `pip install boofun[quantum]` pulls in Qiskit. Oracle construction, Grover circuit execution, and quantum walk simulation run on Qiskit's `AerSimulator`. Cirq backend as alternative. |
| Quantum property testing | Implement genuine quantum testers (BLR, monotonicity, junta) that demonstrate quadratic query savings in simulation, separate from the classical testers in `PropertyTester`. |
| Grover & amplitude amplification | Execute Grover iterations on the simulator and return measured success probabilities alongside the closed-form estimates already provided. |
| Quantum walk simulation | Simulate Szegedy walks on the hypercube and report empirical hitting times alongside the analytical bounds. |
| Documentation & notebooks | Add a dedicated "Quantum Boolean Function Analysis" notebook showing the difference between classical estimates and simulated quantum results. Clearly mark which features require `[quantum]` extra. |

### Other v2.0.0 Candidates

| Item | Why |
|------|-----|
| Symbolic / oracle representations | Move beyond explicit truth tables to support functions on n > 20 variables via BDD, ZDD, or oracle-access representations. |
| Large-scale research mode | Combine symbolic representations with lazy evaluation to enable conjecture-checking workflows at n = 30+. |
| Classroom feedback integration | Incorporate feedback from at least one round of student/instructor testing into notebooks and API. |

---

## Architecture Decisions

See [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) for documented design choices (god object rationale, eager imports, conversion graph costs, etc.).

## Fourier Convention

O'Donnell standard (Chapter 1): Boolean 0 -> +1, Boolean 1 -> -1, f-hat(empty) = E[f].

## Prior Art

- **Scott Aaronson's Boolean Function Wizard** (2000): D(f), R(f), Q(f), sensitivity measures.
- **Avishay Tal's BooleanFunc.py**: Sensitivity, p-biased analysis, decision trees. See [migration guide](docs/guides/migration_from_tal.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). High-value contributions: bug reports with reproducible examples, mathematical error corrections, cross-validation tests, notebook improvements.
