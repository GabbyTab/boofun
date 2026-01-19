# Machine Learning Feature Selection

This example demonstrates how to use Boolean function theory for understanding and improving machine learning models, particularly for feature importance and model interpretability.

## Background

A binary classifier can be viewed as a Boolean function:
- Inputs: Binary features (or binarized continuous features)
- Output: Class prediction (0 or 1)

Boolean function analysis provides:
- **Influence**: Which features matter most?
- **Junta testing**: Does the model depend on only a few features?
- **Fourier analysis**: What interactions exist between features?

## Setup

```python
import numpy as np
import boofun as bf
from boofun.analysis import PropertyTester
from boofun.analysis.query_complexity import QueryComplexityProfile
```

## Example 1: Analyzing a Decision Tree

Decision trees are naturally Boolean functions:

```python
def decision_tree_to_boofun(tree_predict, n_features: int):
    """
    Convert sklearn decision tree to BooleanFunction.
    
    Args:
        tree_predict: Function that takes binary array and returns 0/1
        n_features: Number of binary features
    """
    # Build truth table by evaluating on all inputs
    truth_table = []
    for x in range(2**n_features):
        features = np.array([(x >> i) & 1 for i in range(n_features)])
        pred = int(tree_predict(features.reshape(1, -1))[0])
        truth_table.append(pred)
    
    return bf.BooleanFunction.from_truth_table(truth_table, n_vars=n_features)

# Example: Simple decision tree
# if x0 AND x1: return 1
# elif x2: return 1
# else: return 0
def simple_tree(X):
    if X[0, 0] and X[0, 1]:
        return [1]
    elif X[0, 2]:
        return [1]
    return [0]

tree_func = decision_tree_to_boofun(simple_tree, n_features=5)

# Analyze
print("Feature influences:")
influences = tree_func.influences()
for i, inf in enumerate(influences):
    print(f"  Feature {i}: {inf:.4f}")

# Which features matter?
important = [i for i, inf in enumerate(influences) if inf > 0.01]
print(f"\nImportant features: {important}")
```

## Example 2: Junta Testing for Sparse Models

Many good ML models are "juntas" - they depend on only k features:

```python
def analyze_model_sparsity(f: bf.BooleanFunction):
    """Check if model is essentially a k-junta."""
    tester = PropertyTester(f)
    
    # Test for different k values
    for k in [1, 2, 3, 5, 10]:
        if k > f.n_vars:
            break
        is_junta = tester.junta_test(k=k, num_queries=2000)
        print(f"Is {k}-junta: {is_junta}")
        if is_junta:
            print(f"  -> Model depends on at most {k} features!")
            break

# Test our simple tree (should be 3-junta: depends on x0, x1, x2)
print("Sparsity analysis of decision tree:")
analyze_model_sparsity(tree_func)
```

## Example 3: Feature Interactions via Fourier Analysis

Fourier coefficients reveal feature interactions:

```python
def analyze_feature_interactions(f: bf.BooleanFunction, top_k: int = 10):
    """
    Find the most important feature interactions.
    
    Fourier coefficient f̂(S) measures the interaction strength
    among the features in set S.
    """
    fourier = f.fourier()
    n = f.n_vars
    
    # Get top coefficients by magnitude
    indexed = [(i, fourier[i]) for i in range(len(fourier))]
    sorted_coeffs = sorted(indexed, key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top {top_k} feature interactions:")
    print("-" * 50)
    
    for rank, (idx, coeff) in enumerate(sorted_coeffs[:top_k], 1):
        # Convert index to feature set
        features = [i for i in range(n) if (idx >> i) & 1]
        
        if len(features) == 0:
            feature_str = "∅ (bias)"
        else:
            feature_str = "{" + ", ".join(f"x{i}" for i in features) + "}"
        
        print(f"{rank:2}. {feature_str:20} | Coefficient: {coeff:+.4f}")

# Analyze our decision tree
analyze_feature_interactions(tree_func)
```

## Example 4: Model Complexity Analysis

Query complexity measures model complexity:

```python
def analyze_model_complexity(f: bf.BooleanFunction):
    """Comprehensive complexity analysis of a model."""
    profile = QueryComplexityProfile(f)
    measures = profile.compute()
    
    print("Model Complexity Measures:")
    print("-" * 40)
    print(f"Decision tree depth (D): {measures.get('D', 'N/A')}")
    print(f"Sensitivity (s): {measures.get('s', 'N/A')}")
    print(f"Block sensitivity (bs): {measures.get('bs', 'N/A')}")
    print(f"Certificate complexity (C): {measures.get('C', 'N/A')}")
    print(f"Approximate degree: {measures.get('approx_deg', 'N/A')}")
    
    # Interpretation
    if measures.get('D', float('inf')) <= 3:
        print("\n✓ Simple model (depth ≤ 3)")
    if measures.get('s', 0) <= f.n_vars // 2:
        print("✓ Low sensitivity (robust to single feature changes)")

analyze_model_complexity(tree_func)
```

## Example 5: Comparing Model Interpretability

Compare different models on the same task:

```python
def compare_models_interpretability(models: dict, n_features: int):
    """
    Compare interpretability of different models.
    
    Args:
        models: Dict of {name: predict_function}
        n_features: Number of features
    """
    print("Model Interpretability Comparison")
    print("=" * 60)
    
    results = []
    
    for name, predict in models.items():
        f = decision_tree_to_boofun(predict, n_features)
        
        influences = f.influences()
        tester = PropertyTester(f)
        
        # Compute metrics
        max_inf = max(influences)
        num_influential = sum(1 for inf in influences if inf > 0.05)
        is_monotone = tester.monotonicity_test()
        total_inf = f.total_influence()
        
        results.append({
            'name': name,
            'max_influence': max_inf,
            'num_features_used': num_influential,
            'is_monotone': is_monotone,
            'total_influence': total_inf,
        })
    
    # Print comparison table
    print(f"{'Model':<15} | {'Max Inf':>8} | {'#Features':>9} | {'Monotone':>8} | {'Total I':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<15} | {r['max_influence']:>8.3f} | {r['num_features_used']:>9} | "
              f"{'Yes' if r['is_monotone'] else 'No':>8} | {r['total_influence']:>8.2f}")

# Example models
def linear_threshold(X):
    """Linear model: 1 if x0 + x1 + x2 >= 2"""
    return [1 if sum(X[0, :3]) >= 2 else 0]

def complex_model(X):
    """Complex model with many interactions"""
    return [1 if (X[0,0] ^ X[0,1]) and (X[0,2] or X[0,3]) else 0]

models = {
    'Simple Tree': simple_tree,
    'Linear': linear_threshold,
    'Complex': complex_model,
}

compare_models_interpretability(models, n_features=5)
```

## Example 6: Feature Selection via Influence

Use influence to select features:

```python
def influence_based_feature_selection(f: bf.BooleanFunction, k: int):
    """
    Select top-k features by influence.
    
    Args:
        f: Model as BooleanFunction
        k: Number of features to select
        
    Returns:
        List of selected feature indices
    """
    influences = f.influences()
    
    # Sort by influence
    indexed = [(i, inf) for i, inf in enumerate(influences)]
    sorted_features = sorted(indexed, key=lambda x: x[1], reverse=True)
    
    selected = [i for i, _ in sorted_features[:k]]
    
    print(f"Selected features (top {k} by influence):")
    for i, inf in sorted_features[:k]:
        print(f"  Feature {i}: influence = {inf:.4f}")
    
    return selected

# Select top 3 features from our tree
selected = influence_based_feature_selection(tree_func, k=3)
print(f"\nSelected: {selected}")  # Should be [0, 1, 2]
```

## Example 7: Noise Robustness Analysis

Check how robust predictions are to noise:

```python
def analyze_noise_robustness(f: bf.BooleanFunction):
    """
    Analyze model robustness to input noise.
    
    Noise stability at ρ = probability that prediction
    stays the same when each input is flipped with prob (1-ρ)/2
    """
    print("Noise Robustness Analysis:")
    print("-" * 40)
    
    rhos = [0.99, 0.95, 0.9, 0.8, 0.7]
    
    for rho in rhos:
        stability = f.noise_stability(rho)
        flip_prob = (1 - rho) / 2
        print(f"ρ={rho:.2f} (flip prob {flip_prob:.1%}): stability = {stability:.3f}")
    
    # Interpretation
    stab_95 = f.noise_stability(0.95)
    if stab_95 > 0.9:
        print("\n✓ Model is robust to small input perturbations")
    elif stab_95 < 0.7:
        print("\n⚠ Model is sensitive to input noise")

analyze_noise_robustness(tree_func)
```

## Practical Workflow

```python
def full_model_analysis(predict_func, n_features: int, model_name: str = "Model"):
    """Complete interpretability analysis pipeline."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model_name}")
    print(f"{'='*60}\n")
    
    # Convert to Boolean function
    f = decision_tree_to_boofun(predict_func, n_features)
    
    # 1. Feature importance
    print("1. FEATURE IMPORTANCE")
    influences = f.influences()
    for i, inf in enumerate(influences):
        bar = "█" * int(inf * 50)
        print(f"   x{i}: {inf:.4f} {bar}")
    
    # 2. Model properties
    print("\n2. MODEL PROPERTIES")
    tester = PropertyTester(f)
    print(f"   Monotone: {tester.monotonicity_test()}")
    print(f"   Symmetric: {tester.symmetry_test()}")
    print(f"   Linear: {tester.blr_linearity_test()}")
    
    # 3. Sparsity
    print("\n3. SPARSITY (JUNTA TEST)")
    for k in [1, 2, 3, 5]:
        if k > n_features:
            break
        if tester.junta_test(k=k):
            print(f"   Model is a {k}-junta")
            break
    
    # 4. Complexity
    print("\n4. COMPLEXITY")
    profile = QueryComplexityProfile(f)
    measures = profile.compute()
    print(f"   Decision tree depth: {measures.get('D', 'N/A')}")
    print(f"   Total influence: {f.total_influence():.2f}")
    
    # 5. Noise robustness
    print("\n5. NOISE ROBUSTNESS")
    print(f"   Stab(0.95): {f.noise_stability(0.95):.3f}")
    print(f"   Stab(0.90): {f.noise_stability(0.90):.3f}")
    
    return f

# Run full analysis
f = full_model_analysis(simple_tree, n_features=5, model_name="Simple Decision Tree")
```

## Key Takeaways

1. **Influence = Feature Importance**: Boolean influence directly measures feature importance
2. **Junta Testing**: Identifies if model uses only a subset of features
3. **Fourier Coefficients**: Reveal feature interactions of all orders
4. **Noise Stability**: Measures robustness to input perturbations
5. **Query Complexity**: Provides bounds on model complexity

## Applications

- **Model Debugging**: Why is this feature important?
- **Feature Selection**: Which features actually matter?
- **Model Comparison**: Which model is more interpretable?
- **Robustness Analysis**: How sensitive is the model to noise?

## References

- Blum, A. et al. (2003). "Learning juntas" - k-junta learning
- O'Donnell, R. (2014). "Analysis of Boolean Functions" - Chapter on learning
- Kalai, A. T., Samorodnitsky, A. (2006). "Agnostic learning of juntas"
