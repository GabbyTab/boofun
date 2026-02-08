# Architecture Decision Records

Intentional design choices in BooFun, documented to prevent re-litigating them.

---

## ADR-001: BooleanFunction is a God Object (Intentional)

**Decision**: `BooleanFunction` has 60+ public methods spanning evaluation, Fourier analysis, sensitivity, restrictions, and composition.

**Why**: The primary users are researchers in a Jupyter notebook. They expect `f.fourier()`, `f.influences()`, `f.is_monotone()` -- not `SpectralAnalyzer(f).fourier_expansion()`. The fluent API is the reason people use this library over raw Tal scripts.

**Trade-off**: Harder to maintain. Mixing concerns. But we keep the class as a thin delegation layer: the real logic lives in `analysis/` modules, and `BooleanFunction` methods are one-line wrappers.

**Future**: If the class exceeds ~2000 lines, refactor into mixins or extract to standalone functions with thin wrappers.

---

## ADR-002: Eager Import of Analysis Modules

**Decision**: `import boofun` eagerly loads all 28 analysis modules.

**Why**: For Jupyter notebook workflows (the primary use case), a 2-second import on first cell is invisible. Lazy imports would add complexity and risk "NameError" surprises when users access `bf.analysis.fourier` in later cells.

**Trade-off**: CLI scripts pay a startup cost. Lambda/serverless environments would suffer.

**Future**: If import time exceeds 5 seconds or a CLI use case emerges, implement `__getattr__`-based lazy loading on `analysis/`.

---

## ADR-003: Conversion Graph Uses Hardcoded Costs

**Decision**: The Dijkstra-based conversion graph uses hardcoded `ConversionCost` values.

**Why**: The graph has only ~14 nodes and ~30 edges. Most conversions go through truth tables. Benchmarking real costs for all edges would be fragile (hardware-dependent, n-dependent). The hardcoded costs are "directionally correct" (truth table lookups are cheap, BDD construction is expensive).

**Trade-off**: The "optimal" path might not be truly optimal. But since most paths are 1-2 hops through truth tables, the cost of a wrong choice is small.

**Future**: If a conversion is measurably slow, adjust its cost. Don't benchmark everything.

---

## ADR-004: Numba as a Hard Dependency

**Decision**: `numba>=0.56.0` is in core `dependencies` in `pyproject.toml`.

**Why (historical)**: Numba was added early for WHT and influence computation speedups.

**Known issue**: Numba adds significant install weight (LLVM), warns on import if unavailable, and is rarely needed for the typical `n <= 14` use case.

**Future**: Move to `[project.optional-dependencies]` as `performance` extra. All Numba code paths already have fallbacks. This is a Phase 2 priority.

---

## ADR-005: GPU and Quantum Modules Are Stubs

**Decision**: `gpu_acceleration.py`, `gpu.py`, and `quantum/` exist but contain minimal working code.

**Why**: They were created aspirationally during initial development. They serve as extension points and are excluded from `__all__` in practice.

**Known issue**: The audit correctly identifies these as dead code. They inflate the surface area.

**Future**: Phase 2 cleanup. Delete the GPU stubs (keep only the CuPy WHT call in `optimizations.py`). Keep `quantum/` as a stub but remove from public API.

---

## ADR-006: Dual Tribes Convention (AND-of-ORs)

**Decision**: `bf.tribes(k, n)` implements AND-of-ORs (dual tribes), not O'Donnell's OR-of-ANDs.

**Why (historical)**: The implementation followed a specific reference that used the dual convention.

**Mitigation**: The docstring explicitly documents the convention, notes the duality with O'Donnell Ch. 4, and provides examples. The migration guide documents this for Tal's BooleanFunc.py users.

---

## ADR-007: mypy Disabled on Most Modules

**Decision**: `pyproject.toml` has 23 `ignore_errors = true` overrides.

**Why**: The codebase was written without type annotations and adding them retroactively to 40k LoC is a multi-week project. The `py.typed` marker was added optimistically.

**Known issue**: This is the single largest technical debt item.

**Future**: Phase 3 priority. Enable mypy module-by-module, starting with `core/base.py` and `core/factory.py`. Consider removing `py.typed` until this is done.

---

## ADR-008: 856-Line ROADMAP

**Decision**: The ROADMAP tracks both completed work and future plans in detail.

**Why**: This is a course companion project. The ROADMAP serves as a development journal for the instructor, showing what was delivered each week, what's planned, and what trade-offs were made. Students and collaborators reference it.

**Trade-off**: Long. Could be split into CHANGELOG (completed) and ROADMAP (planned).

---

## ADR-009: No __init__ Subclassing Pattern

**Status**: FIXED (Feb 2026)

The original `__new__`/`_init` pattern was a historical artifact that prevented normal subclassing. It has been replaced with standard `__init__`.
