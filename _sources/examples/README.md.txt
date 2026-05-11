# Real-World Usage Examples

This directory contains practical examples of using BooFun for real-world applications.

## Examples Overview

1. **[Cryptographic S-box Analysis](sbox_analysis.md)** - Analyze AES and other cipher S-boxes
2. **[Voting System Analysis](voting_analysis.md)** - Study weighted voting and influence in elections
3. **[Machine Learning Feature Selection](feature_selection.md)** - Use Boolean function theory for feature importance

## Quick Start

```python
import boofun as bf

# Example 1: Analyze a cryptographic S-box
# See sbox_analysis.md for full details
sbox = [0x63, 0x7c, 0x77, ...]  # AES S-box
component = bf.BooleanFunction.from_truth_table(
    [(sbox[x] >> 0) & 1 for x in range(256)], n_vars=8
)
print(f"Nonlinearity: {compute_nonlinearity(component)}")

# Example 2: Analyze a voting system
# See voting_analysis.md for full details
weighted_vote = bf.weighted_threshold([3, 2, 2, 1, 1, 1], threshold=5)
print(f"Variable influences: {weighted_vote.influences()}")

# Example 3: Feature importance
# See feature_selection.md for full details
from boofun.analysis import PropertyTester
tester = PropertyTester(classifier)
is_junta = tester.junta_test(k=5)  # Is it essentially k features?
```

## Who Uses These Examples?

| Example | Audience | Concepts Used |
|---------|----------|---------------|
| S-box Analysis | Cryptographers | Fourier analysis, nonlinearity, algebraic degree |
| Voting Analysis | Social choice theorists | Influences, dictator proximity, FKN theorem |
| Feature Selection | ML practitioners | Junta testing, influence, property testing |
