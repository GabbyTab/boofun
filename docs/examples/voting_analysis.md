# Voting System Analysis

This example demonstrates how to use BooFun to analyze voting systems using the theory of Boolean functions and social choice.

## Background

Voting rules can be modeled as Boolean functions where:
- Inputs: Individual votes (0 = against, 1 = for)
- Output: Election result (0 = fails, 1 = passes)

Key concepts from Boolean function analysis:
- **Influence**: How much power does each voter have?
- **Dictator proximity**: Is one voter essentially deciding everything?
- **FKN Theorem**: If total influence is low, the system is close to a dictatorship

## Setup

```python
import numpy as np
import boofun as bf
from boofun.analysis import PropertyTester
from boofun.analysis.fkn import (
    distance_to_dictator, closest_dictator, 
    is_close_to_dictator, analyze_dictator_proximity
)
```

## Example 1: Simple Majority

```python
# 5-person committee with simple majority
majority_5 = bf.majority(5)

# Check influences - should be equal
influences = majority_5.influences()
print("Influences:", influences)
# [0.375, 0.375, 0.375, 0.375, 0.375]

# Total influence
total_inf = majority_5.total_influence()
print(f"Total influence: {total_inf:.3f}")
# 1.875

# Check if it's close to a dictatorship (should be NO)
print(f"Close to dictator: {is_close_to_dictator(majority_5)}")
# False
```

## Example 2: Weighted Voting (UN Security Council)

The UN Security Council has 5 permanent members (veto power) and 10 non-permanent members:

```python
def un_security_council():
    """
    Model UN Security Council voting.
    - 5 permanent members (P5): each has veto power
    - 10 non-permanent: need 9 total yes votes (including all P5)
    
    Simplified model: f(x) = 1 iff all P5 vote yes AND ≥4 non-permanent vote yes
    """
    n = 15  # 5 permanent + 10 non-permanent
    
    def evaluate(x):
        # First 5 bits are P5
        p5_votes = [int(x >> i) & 1 for i in range(5)]
        non_perm_votes = [int(x >> i) & 1 for i in range(5, 15)]
        
        # All P5 must vote yes (veto power)
        if not all(p5_votes):
            return 0
        
        # Need at least 4 more yes votes from non-permanent
        if sum(non_perm_votes) < 4:
            return 0
        
        return 1
    
    truth_table = [evaluate(x) for x in range(2**n)]
    return bf.BooleanFunction.from_truth_table(truth_table, n_vars=n)

unsc = un_security_council()

# Analyze influences
influences = unsc.influences()
p5_influences = influences[:5]
non_perm_influences = influences[5:]

print("P5 member influences:", [f"{i:.4f}" for i in p5_influences])
print("Non-permanent avg influence:", f"{np.mean(non_perm_influences):.4f}")
print(f"P5 / non-perm ratio: {np.mean(p5_influences) / np.mean(non_perm_influences):.1f}x")
```

## Example 3: Weighted Threshold Voting

Many real voting systems use weighted votes:

```python
# Example: Company board with different share weights
# Weights: CEO (30), CFO (20), 3 board members (10 each), 2 advisors (5 each)
weights = [30, 20, 10, 10, 10, 5, 5]
threshold = 51  # Need majority of 100 total

# Create weighted threshold function
# f(x) = 1 iff sum(w_i * x_i) >= threshold
def weighted_threshold(weights, threshold):
    n = len(weights)
    def evaluate(x):
        total = sum(w * ((x >> i) & 1) for i, w in enumerate(weights))
        return 1 if total >= threshold else 0
    
    truth_table = [evaluate(x) for x in range(2**n)]
    return bf.BooleanFunction.from_truth_table(truth_table, n_vars=n)

board_vote = weighted_threshold(weights, threshold)

# Analyze power distribution
influences = board_vote.influences()
roles = ["CEO", "CFO", "Board1", "Board2", "Board3", "Adv1", "Adv2"]

print("Voting Power Analysis:")
print("-" * 40)
for role, weight, inf in zip(roles, weights, influences):
    # Banzhaf power index is related to influence
    print(f"{role:8} | Weight: {weight:2} | Influence: {inf:.4f}")

print("-" * 40)
total = sum(influences)
print(f"Total influence: {total:.3f}")
```

## Example 4: Dictator Detection

Use the FKN theorem to detect near-dictatorial systems:

```python
# Obvious dictator: f(x) = x_0 (first person decides)
dictator = bf.dictator(5, 0)

analysis = analyze_dictator_proximity(dictator)
print("Dictator analysis:")
print(f"  Distance to dictator: {analysis['distance']:.4f}")
print(f"  Closest dictator: variable {analysis['closest_dictator']}")
print(f"  Is close: {analysis['is_close']}")

# Hidden near-dictator: weighted system where one person dominates
skewed_weights = [100, 1, 1, 1, 1]  # First person has overwhelming weight
skewed_vote = weighted_threshold(skewed_weights, 51)

analysis = analyze_dictator_proximity(skewed_vote)
print("\nSkewed voting analysis:")
print(f"  Distance to dictator: {analysis['distance']:.4f}")
print(f"  Closest dictator: variable {analysis['closest_dictator']}")
print(f"  Is close: {analysis['is_close']}")  # Likely True!
```

## Example 5: Electoral College Simulation

Simplified model of US Electoral College:

```python
def electoral_college_simplified():
    """
    Simplified electoral college with 5 states.
    State electoral votes: [55, 38, 29, 20, 11] (like CA, TX, FL, NY, smaller)
    Total: 153, need 77 to win
    """
    state_votes = [55, 38, 29, 20, 11]
    threshold = 77
    
    return weighted_threshold(state_votes, threshold)

ec = electoral_college_simplified()
influences = ec.influences()

states = ["CA", "TX", "FL", "NY", "Small"]
votes = [55, 38, 29, 20, 11]

print("Electoral College Power Analysis:")
print("-" * 50)
for state, vote, inf in zip(states, votes, influences):
    # Compare nominal voting power to actual influence
    nominal = vote / sum(votes)
    print(f"{state:6} | Votes: {vote:2} ({nominal:.1%}) | Influence: {inf:.4f}")
```

## Noise Stability and Chaotic Elections

How sensitive is the outcome to random vote changes?

```python
# Compare noise stability of different systems
majority_9 = bf.majority(9)
dictator_9 = bf.dictator(9, 0)

# Weighted system that's somewhat balanced
balanced_weights = [15, 15, 15, 15, 10, 10, 10, 5, 5]
balanced_vote = weighted_threshold(balanced_weights, 51)

# Noise stability: probability outcome stays same if each vote flips with prob (1-ρ)/2
rhos = [0.99, 0.95, 0.9, 0.8]

print("Noise Stability (higher = more stable):")
print("-" * 50)
for rho in rhos:
    maj_stab = majority_9.noise_stability(rho)
    dict_stab = dictator_9.noise_stability(rho)
    bal_stab = balanced_vote.noise_stability(rho)
    print(f"ρ={rho:.2f} | Majority: {maj_stab:.3f} | Dictator: {dict_stab:.3f} | Balanced: {bal_stab:.3f}")
```

## Property Testing for Voting Fairness

```python
def analyze_voting_fairness(f: bf.BooleanFunction, name: str):
    """Comprehensive fairness analysis of a voting system."""
    print(f"\n=== {name} Fairness Analysis ===")
    
    tester = PropertyTester(f)
    
    # Is it dictatorial?
    print(f"Is dictator: {tester.dictator_test()}")
    
    # Is it symmetric (all voters equal)?
    print(f"Is symmetric: {tester.symmetry_test()}")
    
    # Is it monotone (more yes votes can't hurt)?
    print(f"Is monotone: {tester.monotonicity_test()}")
    
    # Influence analysis
    influences = f.influences()
    max_inf = max(influences)
    min_inf = min(influences)
    
    print(f"Max influence: {max_inf:.4f} (voter {influences.tolist().index(max_inf)})")
    print(f"Min influence: {min_inf:.4f}")
    print(f"Influence ratio: {max_inf/min_inf:.2f}x")
    
    # FKN analysis
    analysis = analyze_dictator_proximity(f)
    print(f"Close to dictator: {analysis['is_close']}")

# Test different systems
analyze_voting_fairness(majority_5, "Simple Majority")
analyze_voting_fairness(board_vote, "Weighted Board")
```

## Key Takeaways

1. **Influence ≠ Weight**: Nominal voting weight doesn't equal actual power
2. **FKN Theorem**: Systems with low total influence are near-dictatorships
3. **Noise Stability**: Measures how robust outcomes are to random perturbations
4. **Property Testing**: Can detect dictators, asymmetry, non-monotonicity

## References

- Banzhaf, J. (1965). "Weighted voting doesn't work"
- O'Donnell, R. (2014). "Analysis of Boolean Functions" - Chapter on social choice
- Friedgut, E., Kalai, G., Naor, A. (2002). "Boolean functions whose Fourier transform is concentrated"
