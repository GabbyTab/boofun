# Tutorials

```bash
pip install -e ".[all]"
python examples/01_getting_started.py
```

## Index

| # | File | Topic |
|---|------|-------|
| 1 | `01_getting_started.py` | Basics: create, evaluate, properties |
| 2 | `02_fourier_basics.py` | WHT, Parseval, spectral weight |
| 3 | `03_common_families.py` | Majority, Parity, Tribes, Threshold |
| 4 | `04_property_testing.py` | BLR, junta, monotonicity |
| 5 | `05_query_complexity.py` | Sensitivity, certificates |
| 6 | `06_noise_stability.py` | Influences, voting |
| 7 | `07_quantum_applications.py` | Quantum complexity bounds (Grover, walks — classical computation) |
| 8 | `08_cryptographic_analysis.py` | S-box analysis, LAT/DDT, bent functions |
| 9 | `09_partial_functions.py` | Streaming, hex I/O, storage hints |
| 10 | `10_sensitivity_decision_trees.py` | Sensitivity moments, decision trees |

### Additional Examples

- `educational_examples.py` - Teaching examples
- `representations_demo.py` - Circuits, BDDs
- `advanced_features_demo.py` - ANF, GPU acceleration

## Notebooks

`notebooks/` contains 23 Jupyter notebooks aligned with O'Donnell's course:

### Core Lectures
- `lecture1_fourier_expansion.ipynb` - Fourier basics
- `lecture2_linearity_testing.ipynb` - BLR test
- `lecture3_social_choice_influences.ipynb` - Arrow's theorem
- `lecture4_influences_effects.ipynb` - Influence bounds
- `lecture5_noise_stability.ipynb` - Noise sensitivity
- `lecture6_spectral_concentration.ipynb` - Spectral analysis
- `lecture7_goldreich_levin.ipynb` - Learning Fourier
- `lecture8_learning_juntas.ipynb` - Junta learning
- `lecture9_dnf_restrictions.ipynb` - DNF analysis
- `lecture10_fourier_concentration.ipynb` - Advanced concentration
- `lecture11_invariance_principle.ipynb` - Invariance principle and Gaussian analysis

### Homework Solutions
- `hw1_fourier_expansion.ipynb` - Fourier exercises
- `hw2_ltf_decision_trees.ipynb` - LTF analysis
- `hw3_dnf_restrictions.ipynb` - DNF exercises
- `hw4_hypercontractivity.ipynb` - Hypercontractivity

### Special Topics (v1.1)
- `global_hypercontractivity.ipynb` - Global hypercontractivity (Keevash et al.)
- `boolean_functions_as_random_variables.ipynb` - P-biased measures, threshold phenomena, estimation
- `flexible_inputs_and_oracles.ipynb` - Input formats and oracles
- `asymptotic_visualization.ipynb` - Growth plots
- `real_world_applications.ipynb` - Voting, cryptography, feature selection
- `ltf_visualization.ipynb` - LTF geometry and analysis
- `cryptographic_analysis.ipynb` - Walsh spectrum, nonlinearity, SAC
- `fractional_prg.ipynb` - Fractional PRGs and Fourier tails (CHLT ITCS 2019)
