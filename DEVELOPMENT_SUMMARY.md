# Development Notes

Internal notes on library development. Not user-facing documentation.

## Structure

```
src/boofun/
├── core/           # BooleanFunction, representations, conversion
├── analysis/       # Fourier, influences, property testing, complexity
├── families/       # Function families with growth tracking
├── visualization/  # Plots (requires matplotlib)
├── quantum/        # Speedup estimation (theoretical)
└── testing/        # Validation utilities
```

## API Surface

Primary: `create()`, `majority()`, `parity()`, `tribes()`, `dictator()`, `AND()`, `OR()`, `random()`

Analysis: Methods on `BooleanFunction` objects (`fourier()`, `influences()`, `is_linear()`, etc.)

## Installation

```bash
pip install -e .              # Basic
pip install -e ".[dev]"       # Development
pip install -e ".[visualization]"  # With matplotlib
```

## Testing

```bash
pytest tests/
pytest --cov=boofun tests/
```

Coverage is incomplete. Core paths are better tested than edge cases.

## Known Issues

- Test coverage low (~38%)
- Some visualization code untested
- Quantum module is theoretical only (no actual quantum execution)
- Edge cases in representations may have bugs

## Dependencies

Required: numpy, scipy

Optional: numba (JIT), cupy (GPU), matplotlib/plotly (visualization), qiskit (quantum oracles)
