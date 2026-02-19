# Error Handling in BooFun

BooFun provides a structured exception hierarchy with machine-readable error codes
for clear, actionable error handling.

## Exception Hierarchy

```
BooleanFunctionError (base)
├── ValidationError          - Invalid user input (E1xxx)
│   ├── InvalidInputError    - Bad function arguments (E11xx)
│   ├── InvalidRepresentationError - Unsupported representation (E12xx)
│   └── InvalidTruthTableError - Malformed truth table (E13xx)
├── EvaluationError          - Function evaluation failures (E2xxx)
├── ConversionError          - Representation conversion failures (E3xxx)
├── ConfigurationError       - Setup/configuration errors (E4xxx)
├── ResourceUnavailableError - Optional deps unavailable (E5xxx)
└── InvariantViolationError  - Internal library bugs (E9xxx)
```

## Error Codes

BooFun uses machine-readable error codes for programmatic error handling:

| Range | Category | Description |
|-------|----------|-------------|
| E1000-E1999 | Validation | User input problems |
| E2000-E2999 | Evaluation | Function execution problems |
| E3000-E3999 | Conversion | Representation problems |
| E4000-E4999 | Configuration | Setup problems |
| E5000-E5999 | Resource | Dependency problems |
| E9000-E9999 | Internal | Library bugs |

### Common Error Codes

| Code | Name | Description |
|------|------|-------------|
| E1301 | TRUTH_TABLE_WRONG_SIZE | Truth table size is not a power of 2 |
| E1302 | TRUTH_TABLE_EMPTY | Empty truth table provided |
| E1103 | PARAMETER_OUT_OF_RANGE | Parameter value outside valid range |
| E2001 | EVALUATION_FAILED | Function evaluation failed |
| E3001 | NO_CONVERSION_PATH | No path to convert between representations |
| E3004 | NO_REPRESENTATIONS | Function has no representations |
| E5001 | NUMBA_UNAVAILABLE | Numba not installed |
| E5002 | CUPY_UNAVAILABLE | CuPy not installed for GPU |

## Usage Examples

### Catching All Library Errors

```python
import boofun as bf

try:
    f = bf.create([0, 1, 1])  # Invalid - not power of 2
except bf.BooleanFunctionError as e:
    print(f"Error {e.code.value}: {e.message}")
    if e.suggestion:
        print(f"Suggestion: {e.suggestion}")
```

### Catching Specific Errors

```python
import boofun as bf

try:
    f = bf.create([0, 1, 1])
except bf.InvalidTruthTableError as e:
    print(f"Bad truth table: {e.message}")
except bf.ValidationError as e:
    print(f"Validation failed: {e.message}")
```

### Using Error Codes Programmatically

```python
import boofun as bf
from boofun import ErrorCode

try:
    f = bf.create([])
except bf.BooleanFunctionError as e:
    if e.code == ErrorCode.TRUTH_TABLE_EMPTY:
        # Handle empty truth table specifically
        f = bf.create([0])  # Default to constant 0
    else:
        raise
```

### Checking Error Context

```python
import boofun as bf

try:
    f = bf.create([0, 1, 1, 0])
    f.fix(0, 5)  # Invalid value
except bf.InvalidInputError as e:
    print(f"Error code: {e.code.value}")
    print(f"Parameter: {e.context.get('parameter')}")
    print(f"Received: {e.context.get('received')}")
    print(f"Expected: {e.context.get('expected')}")
```

### Logging Errors (JSON-friendly)

```python
import boofun as bf
import json

try:
    f = bf.create([0, 1, 1])
except bf.BooleanFunctionError as e:
    # Convert to dict for structured logging
    error_dict = e.to_dict()
    print(json.dumps(error_dict, indent=2))
```

Output:
```json
{
  "error_code": "E1301",
  "error_type": "InvalidTruthTableError",
  "message": "Truth table size must be a power of 2, got 3",
  "context": {"size": 3, "expected_size": "2 or 4"},
  "suggestion": "For 1-variable function, use 2 entries..."
}
```

## Exception Reference

### ValidationError (E1xxx)

Raised when user input fails validation.

**Subclasses:**
- `InvalidInputError` (E11xx) - Invalid function arguments
- `InvalidRepresentationError` (E12xx) - Unknown representation type
- `InvalidTruthTableError` (E13xx) - Size not power of 2, empty table

### EvaluationError (E2xxx)

Raised when function evaluation fails.

```python
# Example: evaluation of underlying callable fails
f = bf.create(lambda x: 1/0, n=2)  # Division by zero
f.get_representation("truth_table")  # Raises EvaluationError
```

### ConversionError (E3xxx)

Raised when representation conversion fails.

```python
f = bf.BooleanFunction(n=2)  # No representations
f.get_representation("fourier")  # Raises ConversionError (E3004)
```

### ConfigurationError (E4xxx)

Raised when library configuration is invalid.

### ResourceUnavailableError (E5xxx)

Raised when optional dependencies are unavailable.

```python
# When CuPy is not installed:
f.to_gpu()  # Raises ResourceUnavailableError (E5002)
```

### InvariantViolationError (E9xxx)

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

## Debug Logging

Enable debug logging to see detailed error information:

```python
from boofun.utils.logging import enable_debug_logging

enable_debug_logging()  # Now you'll see debug messages for silent fallbacks
```

Or configure manually:

```python
import logging

logging.getLogger("boofun").setLevel(logging.DEBUG)
logging.getLogger("boofun").addHandler(logging.StreamHandler())
```

## Best Practices

1. **Catch specific exceptions** when you can handle them
2. **Catch `BooleanFunctionError`** as a fallback for unexpected errors
3. **Use error codes** for programmatic handling in larger systems
4. **Check `e.suggestion`** for actionable fixes
5. **Check `e.context`** for debugging information
6. **Use `e.to_dict()`** for structured logging
