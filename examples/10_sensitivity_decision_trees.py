"""
Advanced Sensitivity and Decision Tree Analysis Demo

This example demonstrates the enhanced sensitivity analysis and
decision tree algorithms added in v1.1.

Features covered:
- Sensitivity moments and histograms
- P-biased sensitivity measures
- Decision tree complexity
- Tree enumeration and DP algorithms
- Randomized decision tree complexity
"""

import numpy as np

import boofun as bf
from boofun.analysis.complexity import decision_tree_depth, max_certificate_complexity
from boofun.analysis.decision_trees import (
    compute_randomized_complexity,
    count_decision_trees,
    decision_tree_depth_dp,
)
from boofun.analysis.sensitivity import (
    arg_max_sensitivity,
    arg_min_sensitivity,
    average_sensitivity,
    average_sensitivity_moment,
    max_sensitivity,
    min_sensitivity,
    sensitive_coordinates,
    sensitivity_at,
    sensitivity_histogram,
)


def demo_pointwise_sensitivity():
    """Demonstrate pointwise sensitivity analysis."""
    print("=" * 60)
    print("Pointwise Sensitivity Analysis")
    print("=" * 60)

    f = bf.majority(5)

    print("\n1. Sensitivity at specific inputs")
    print("-" * 50)
    # Test at different Hamming weights (0, 2, 3, 5 ones)
    # Majority(5) outputs 1 if ≥3 ones
    for x in [0, 3, 7, 31]:  # 0 ones, 2 ones, 3 ones, 5 ones
        s = sensitivity_at(f, x)
        coords = sensitive_coordinates(f, x)
        ones = bin(x).count("1")
        print(f"   x={x:05b} ({ones} ones): s={s}, sensitive coords: {coords}")

    print("\n2. Find extremal inputs")
    print("-" * 50)
    x_max, s_max = arg_max_sensitivity(f)
    x_min, s_min = arg_min_sensitivity(f)
    print(f"   Max sensitivity input: {x_max:05b} (s = {s_max})")
    print(f"   Min sensitivity input: {x_min:05b} (s = {s_min})")


def demo_aggregate_sensitivity():
    """Demonstrate aggregate sensitivity measures."""
    print("\n" + "=" * 60)
    print("Aggregate Sensitivity Measures")
    print("=" * 60)

    functions = {
        "Parity(4)": bf.parity(4),
        "Majority(5)": bf.majority(5),
        "AND(4)": bf.AND(4),
        "OR(4)": bf.OR(4),
    }

    print("\n1. Basic sensitivity measures")
    print("-" * 50)
    print(f"   {'Function':<15} {'s(f)':<8} {'as(f)':<10}")
    print(f"   {'-'*15} {'-'*8} {'-'*10}")
    for name, f in functions.items():
        s = max_sensitivity(f)
        avg_s = average_sensitivity(f)
        print(f"   {name:<15} {s:<8} {avg_s:<10.4f}")

    print("\n2. Sensitivity moments (t-th moment of sensitivity)")
    print("-" * 50)
    f = bf.majority(5)
    for t in [1, 2, 3, 4]:
        moment = average_sensitivity_moment(f, t)
        print(f"   E[s(f,x)^{t}] = {moment:.4f}")


def demo_sensitivity_histogram():
    """Demonstrate sensitivity distribution analysis."""
    print("\n" + "=" * 60)
    print("Sensitivity Histogram")
    print("=" * 60)

    f = bf.majority(5)
    hist = sensitivity_histogram(f)

    print("\n   Sensitivity value distribution for Majority(5):")
    print("-" * 50)
    # hist is an array where hist[s] = count of inputs with sensitivity s
    for sens_val, count in enumerate(hist):
        if count > 0:
            count_int = int(count)
            bar = "█" * (count_int // 2)
            print(f"   s = {sens_val}: {count_int:4d} inputs {bar}")


def demo_decision_trees():
    """Demonstrate decision tree analysis."""
    print("\n" + "=" * 60)
    print("Decision Tree Analysis")
    print("=" * 60)

    functions = {
        "Parity(4)": bf.parity(4),
        "Majority(3)": bf.majority(3),
        "AND(4)": bf.AND(4),
        "OR(4)": bf.OR(4),
    }

    print("\n1. Decision tree depth D(f)")
    print("-" * 50)
    print(f"   {'Function':<15} {'D(f)':<8} {'C₀(f)':<8} {'C₁(f)':<8}")
    print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    for name, f in functions.items():
        d = decision_tree_depth(f)
        c0 = max_certificate_complexity(f, value=0)
        c1 = max_certificate_complexity(f, value=1)
        print(f"   {name:<15} {d:<8} {c0:<8} {c1:<8}")

    print("\n2. Randomized decision tree depth R(f)")
    print("-" * 50)
    for name, f in functions.items():
        r = compute_randomized_complexity(f)
        print(f"   R({name}) ≈ {r:.2f}")


def demo_tree_enumeration():
    """Demonstrate decision tree enumeration."""
    print("\n" + "=" * 60)
    print("Decision Tree Enumeration")
    print("=" * 60)

    print("\n1. Counting decision trees")
    print("-" * 50)
    for n in [2, 3]:
        f = bf.parity(n)
        count = count_decision_trees(f)
        print(f"   Parity({n}): {count} optimal decision trees")


def demo_complexity_relationships():
    """Demonstrate relationships between complexity measures."""
    print("\n" + "=" * 60)
    print("Complexity Measure Relationships")
    print("=" * 60)

    f = bf.majority(5)

    print("\n   For Majority(5):")
    print("-" * 50)

    s = max_sensitivity(f)
    d = decision_tree_depth(f)
    deg = f.degree()

    print(f"   Sensitivity s(f):       {s}")
    print(f"   Decision tree D(f):     {d}")
    print(f"   Fourier degree deg(f):  {deg}")

    print("\n   Key relationships (from Huang's theorem):")
    print(f"   - s(f) ≥ √deg(f): {s} ≥ √{deg} = {np.sqrt(deg):.2f} ✓")
    print(f"   - D(f) ≥ s(f):    {d} ≥ {s} ✓")


def main():
    """Run all sensitivity and decision tree demos."""
    print("\n" + "=" * 60)
    print("BooFun Sensitivity & Decision Tree Analysis Demo")
    print("=" * 60)

    demo_pointwise_sensitivity()
    demo_aggregate_sensitivity()
    demo_sensitivity_histogram()
    demo_decision_trees()
    demo_tree_enumeration()
    demo_complexity_relationships()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
