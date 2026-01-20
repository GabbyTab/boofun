# Error Handling in BooFun

BooFun provides a structured exception hierarchy for clear, actionable error handling.

## Exception Hierarchy

```
BooleanFunctionError (base)
├── ValidationError          - Invalid user input
│   ├── InvalidInputError    - Bad function arguments
│   ├── InvalidRepresentationError - Unsupported representation
│   └── InvalidTruthTableError - Malformed truth table
├── EvaluationError          - Function evaluation failures
├── ConversionError          - Representation conversion failures
├── ConfigurationError       - Setup/configuration errors
├── ResourceUnavailableError - Optional deps unavailable
└── InvariantViolationError  - Internal library bugs
```

## Usage Examples

### Catching All Library Errors

```python
import boofun as bf

try:
    f = bf.create([0, 1, 1])  # Invalid - not power of 2
except bf.BooleanFunctionError as e:
    print(f"Error: {e}")
    print(f"Suggestion: {e.suggestion}")
```

### Catching Specific Errors

```python
import boofun as bf

try:
    f = bf.create([0, 1, 1])
except bf.InvalidTruthTableError as e:
    print(f"Bad truth table: {e}")
except bf.ValidationError as e:
    print(f"Validation failed: {e}")
```

### Checking Error Context

```python
import boofun as bf

try:
    f = bf.create([0, 1, 1, 0])
    f.fix(0, 5)  # Invalid value
except bf.InvalidInputError as e:
    print(f"Parameter: {e.context.get('parameter')}")
    print(f"Received: {e.context.get('received')}")
    print(f"Expected: {e.context.get('expected')}")
```

## Exception Reference

### ValidationError

Raised when user input fails validation.

**Subclasses:**
- `InvalidInputError` - Invalid function arguments (e.g., `rho > 1`)
- `InvalidRepresentationError` - Unknown representation type
- `InvalidTruthTableError` - Size not power of 2, empty table

### EvaluationError

Raised when function evaluation fails.

```python
# Example: evaluation of underlying callable fails
f = bf.create(lambda x: 1/0, n=2)  # Division by zero
f.get_representation("truth_table")  # Raises EvaluationError
```

### ConversionError

Raised when representation conversion fails.

```python
f = bf.BooleanFunction(n=2)  # No representations
f.get_representation("fourier")  # Raises ConversionError
```

### ResourceUnavailableError

Raised when optional dependencies are unavailable.

```python
# Raised when trying to use GPU without CuPy installed
```

### InvariantViolationError

Indicates a bug in BooFun itself. If you see this, please report it!

## Lenient Mode

Some operations support lenient mode for graceful degradation:

```python
from boofun.core.representations.truth_table import TruthTableRepresentation

# Strict mode (default): raises on any failure
tt.convert_from(source, data, space, n_vars)

# Lenient mode: substitutes False and warns
tt.convert_from(source, data, space, n_vars, lenient=True)
```

## Best Practices

1. **Catch specific exceptions** when you can handle them
2. **Catch `BooleanFunctionError`** as a fallback for unexpected errors
3. **Check `e.suggestion`** for actionable fixes
4. **Check `e.context`** for debugging information
