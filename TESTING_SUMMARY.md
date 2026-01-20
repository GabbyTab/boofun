# Testing Notes

## Current State

Coverage: ~38% (estimated). Core paths are better tested than edge cases.

## Test Structure

```
tests/
├── unit/           # Individual function tests
├── integration/    # Cross-module tests
├── property/       # Hypothesis-based tests
├── fuzz/           # Edge case fuzzing
├── benchmarks/     # Performance tests
└── test_cross_validation.py  # Against known results
```

## Running Tests

```bash
pytest tests/                    # All
pytest --cov=boofun tests/       # With coverage
pytest tests/unit/               # Unit only
pytest tests/property/           # Property-based
```

## What's Well-Tested

- `BooleanFunction` creation and evaluation
- Basic Fourier transform
- Influences computation
- Built-in functions (majority, parity, etc.)
- Property testing (BLR, junta basics)

## What's Not Well-Tested

- Visualization code (~0-13% coverage)
- Quantum module (~11% coverage)
- Many representation edge cases
- Large n behavior
- GPU acceleration paths

## Cross-Validation

`tests/test_cross_validation.py` checks against known mathematical results:
- Parseval's identity
- Known influence values for standard functions
- Complexity measure relationships

These are the most reliable tests because they verify against theory, not just consistency.

## Known Gaps

If you find a bug, it's probably in a low-coverage area. Bug reports with reproducible cases help.
