"""
Advanced Features Demo for BooFun Library

This example demonstrates the advanced features implemented in BooFun:
- ANF (Algebraic Normal Form) representation
- Conversion graph system
- Batch processing
- GPU acceleration (if available)
- Numba JIT optimization
- Testing framework
- Adapter system
- Error models
"""

import os

# Ensure we can import boofun
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boofun as bf


def demo_anf_representation():
    """Demonstrate ANF (Algebraic Normal Form) representation."""
    print("[ANF] ANF Representation Demo")
    print("=" * 50)

    # Create different types of functions
    functions = {
        "XOR": bf.create([False, True, True, False]),
        "AND": bf.create([False, False, False, True]),
        "Majority": bf.create([False, False, False, True, False, True, True, True]),
    }

    for name, func in functions.items():
        print(f"\n{name} Function:")

        # Get ANF representation
        anf_data = func.get_representation("anf")
        print(f"  ANF terms: {len([m for m, c in anf_data.items() if c != 0])}")

        # Analyze ANF properties
        from boofun.core.representations.anf_form import ANFRepresentation, anf_to_string

        anf_repr = ANFRepresentation()

        degree = anf_repr._get_degree(anf_data)
        is_linear = anf_repr.is_linear(anf_data)
        is_quadratic = anf_repr.is_quadratic(anf_data)

        print(f"  Degree: {degree}")
        print(f"  Linear: {is_linear}")
        print(f"  Quadratic: {is_quadratic}")

        # Convert to human-readable string
        anf_string = anf_to_string(anf_data)
        print(f"  ANF: {anf_string}")


def demo_conversion_graph():
    """Demonstrate conversion graph system."""
    print("\n[CONV] Conversion Graph Demo")
    print("=" * 50)

    # Create a function
    majority = bf.create([False, False, False, True, False, True, True, True])

    # Show available conversions
    options = majority.get_conversion_options(max_cost=1000)
    print(f"Available conversions from current representations:")

    for target, info in options.items():
        cost = info["cost"]
        exact = info["exact"]
        print(f"  -> {target}: cost={cost.total_cost:.2f}, exact={exact}")

    # Demonstrate intelligent conversion
    print(f"\nConversion cost estimates:")
    targets = ["anf", "fourier_expansion", "polynomial"]

    for target in targets:
        cost = majority.estimate_conversion_cost(target)
        if cost:
            print(f"  To {target}: {cost.total_cost:.2f}")
        else:
            print(f"  To {target}: Already available")

    # Show conversion graph statistics
    from boofun.core.conversion_graph import get_conversion_graph

    graph = get_conversion_graph()
    stats = graph.get_graph_stats()

    print(f"\nConversion Graph Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Connectivity: {stats['connectivity']:.2%}")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n[BATCH] Batch Processing Demo")
    print("=" * 50)

    # Create a function for testing
    xor = bf.create([False, True, True, False])

    # Test different batch sizes
    batch_sizes = [10, 100, 1000, 10000]

    print("Batch processing performance:")
    for size in batch_sizes:
        # Generate random inputs
        inputs = np.random.randint(0, 4, size)

        # Time the evaluation
        start_time = time.time()
        results = xor.evaluate(inputs)
        end_time = time.time()

        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        throughput = size / (duration / 1000) if duration > 0 else float("inf")

        print(f"  Size {size:5d}: {duration:6.2f}ms, {throughput:8.0f} eval/sec")

    # Show batch processor statistics
    from boofun.core.batch_processing import get_batch_processor_stats

    stats = get_batch_processor_stats()

    print(f"\nBatch Processor Info:")
    print(f"  Available processors: {stats['available_processors']}")
    print(f"  Numba available: {stats['numba_available']}")
    print(f"  CPU count: {stats['cpu_count']}")


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration (if available)."""
    print("\n[PERF] GPU Acceleration Demo")
    print("=" * 50)

    from boofun.core.gpu_acceleration import get_gpu_info, is_gpu_available

    # Check GPU availability
    gpu_info = get_gpu_info()
    print(f"GPU Available: {gpu_info['gpu_available']}")
    print(f"Active Backend: {gpu_info['active_backend']}")
    print(f"Available Backends: {gpu_info['available_backends']}")

    if gpu_info["devices"]:
        print(f"GPU Devices:")
        for device in gpu_info["devices"]:
            print(f"  {device}")
    else:
        print("No GPU devices detected")

    if is_gpu_available():
        print("\n[NOTE] GPU acceleration is available and will be used automatically")
        print("   for large batch operations when beneficial.")
    else:
        print("\n[TIP] GPU acceleration not available - using CPU optimizations")


def demo_numba_optimization():
    """Demonstrate Numba JIT optimization."""
    print("\n[JIT] Numba JIT Optimization Demo")
    print("=" * 50)

    from boofun.core.numba_optimizations import get_numba_stats, is_numba_available

    stats = get_numba_stats()
    print(f"Numba Available: {stats['numba_available']}")

    if stats["numba_available"]:
        print(f"Compiled Functions: {stats['compiled_functions']}")
        print("[NOTE] Numba optimizations are active for:")
        print("  - Batch evaluations")
        print("  - Influence computations")
        print("  - Noise stability calculations")
        print("  - Walsh-Hadamard transforms")

        # Demonstrate optimized analysis
        majority = bf.create([False, False, False, True, False, True, True, True])
        analyzer = bf.SpectralAnalyzer(majority)

        print(f"\nOptimized Analysis Results:")
        start_time = time.time()
        influences = analyzer.influences()
        influence_time = (time.time() - start_time) * 1000

        start_time = time.time()
        stability = analyzer.noise_stability(0.9)
        stability_time = (time.time() - start_time) * 1000

        print(f"  Influences: {influences} ({influence_time:.2f}ms)")
        print(f"  Noise Stability: {stability:.4f} ({stability_time:.2f}ms)")
    else:
        print("[TIP] Numba not available - using pure Python/NumPy implementations")


def demo_testing_framework():
    """Demonstrate testing and validation framework."""
    print("\n[TEST] Testing Framework Demo")
    print("=" * 50)

    # Create a function to test
    xor = bf.create([False, True, True, False])

    # Quick validation
    print("Quick Validation:")
    is_valid = bf.quick_validate(xor, verbose=False)
    print(f"  Valid: {is_valid}")

    # Detailed validation
    print("\nDetailed Validation:")
    validator = bf.BooleanFunctionValidator(xor)
    results = validator.validate_all()

    for category, result in results.items():
        if category != "overall_status" and isinstance(result, dict):
            status = "[PASS] PASS" if result.get("passed", False) else "[FAIL] FAIL"
            print(f"  {category.replace('_', ' ').title()}: {status}")

    # Test representation
    print("\nRepresentation Testing:")
    from boofun.core.representations.truth_table import TruthTableRepresentation

    # Use validate_representation for representation testing
    f = bf.AND(3)
    repr_results = bf.validate_representation(f, "truth_table")
    overall_passed = repr_results.get("valid", False)
    status = "PASS" if overall_passed else "FAIL"
    print(f"  Truth Table Representation: {status}")


def demo_adapter_system():
    """Demonstrate adapter system for external functions."""
    print("\n[ADAPTER] Adapter System Demo")
    print("=" * 50)

    # Example 1: Adapt a Python lambda
    print("1. Python Lambda Adapter:")
    xor_lambda = lambda x: x[0] ^ x[1] if len(x) >= 2 else False
    adapted_xor = bf.adapt_callable(xor_lambda, n_vars=2)

    test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    print("  XOR function results:")
    for inputs in test_inputs:
        result = adapted_xor.evaluate(inputs)
        print(f"    XOR{inputs} = {result}")

    # Example 2: Adapt a legacy function
    print("\n2. Legacy Function Adapter:")

    class LegacyBooleanFunction:
        def legacy_evaluate(self, inputs):
            # Legacy AND function
            return all(inputs)

    legacy_func = LegacyBooleanFunction()
    adapter = bf.LegacyAdapter(evaluation_method="legacy_evaluate")
    adapted_and = adapter.adapt(legacy_func)

    print("  AND function results:")
    for inputs in test_inputs:
        result = adapted_and.evaluate(inputs)
        print(f"    AND{inputs} = {result}")

    # Example 3: NumPy function adapter
    print("\n3. NumPy Function Adapter:")

    def numpy_majority(x):
        """Vectorized majority function."""
        if hasattr(x, "__len__") and len(x) >= 3:
            return np.sum(x) > len(x) // 2
        return False

    adapted_maj = bf.adapt_numpy_function(numpy_majority, n_vars=3, vectorized=False)

    test_inputs_3 = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]
    print("  Majority function results:")
    for inputs in test_inputs_3:
        result = adapted_maj.evaluate(inputs)
        print(f"    MAJ{inputs} = {result}")


def demo_error_models():
    """Demonstrate error models."""
    print("\n[NOTE] Error Models Demo")
    print("=" * 50)

    # Create functions with different error models
    print("1. Exact Error Model (default):")
    exact_func = bf.create([False, True, True, False], error_model=bf.ExactErrorModel())
    result = exact_func.evaluate(1)
    confidence = exact_func.error_model.get_confidence(result)
    print(f"  XOR(1) = {result}, confidence = {confidence}")

    print("\n2. PAC Error Model:")
    pac_func = bf.create([False, True, True, False], error_model=bf.PACErrorModel(0.1, 0.1))
    result = pac_func.evaluate(1)
    confidence = pac_func.error_model.get_confidence(result)
    print(f"  XOR(1) = {result}, confidence = {confidence}")

    print("\n3. Noise Error Model:")
    noise_func = bf.create([False, True, True, False], error_model=bf.NoiseErrorModel(0.05))

    # Multiple evaluations to show noise effects
    results = []
    for _ in range(10):
        result = noise_func.evaluate(1)
        # Extract boolean value from error model result
        if isinstance(result, dict) and "value" in result:
            boolean_result = bool(result["value"])
        else:
            boolean_result = bool(result)
        results.append(boolean_result)

    true_count = sum(results)
    print(f"  XOR(1) over 10 evaluations: {true_count}/10 True")
    print(f"  Expected ~9/10 due to 5% noise rate")


def demo_quantum_complexity():
    """Demonstrate quantum complexity bound computation (experimental)."""
    print("\n[COMPLEXITY] Quantum Complexity Bounds Demo (experimental)")
    print("=" * 50)
    print("This is a playground — all computations are classical (closed-form formulas).")
    print("We're still figuring out what to build here. See ROADMAP.md for v2.0.0 plans.\n")

    from boofun.quantum_complexity import (
        QuantumComplexityAnalyzer,
        element_distinctness_analysis,
        grover_speedup,
        quantum_walk_bounds,
    )

    # Grover's algorithm bounds
    print("1. Grover's Algorithm Complexity Bounds:")
    for name, f in [("AND_4", bf.AND(4)), ("OR_4", bf.OR(4))]:
        result = grover_speedup(f)
        print(f"  {name}: {result['speedup']:.2f}x speedup "
              f"({result['num_solutions']} solutions, "
              f"{result['grover_queries']:.1f} Grover queries vs "
              f"{result['classical_queries']:.0f} classical)")

    # Quantum walk bounds
    print("\n2. Quantum Walk Complexity Bounds (Szegedy 2004):")
    f = bf.AND(4)
    walk = quantum_walk_bounds(f)
    print(f"  AND_4 on hypercube:")
    print(f"    Classical hitting time: {walk['classical_hitting_time']:.1f}")
    print(f"    Quantum hitting time:  {walk['quantum_hitting_time']:.1f}")
    print(f"    Speedup: {walk['speedup_over_classical']:.2f}x")

    # Element distinctness
    print("\n3. Element Distinctness Analysis (Ambainis 2007):")
    result = element_distinctness_analysis(bf.parity(4))
    print(f"  Collisions found: {result['has_collision']}")
    print(f"  Classical complexity: O({result['classical_complexity']})")
    print(f"  Quantum complexity: O({result['quantum_complexity']:.1f})")
    print(f"  Speedup: {result['speedup']:.2f}x")

    # Grover amplitude evolution
    print("\n4. Grover Amplitude Evolution (closed-form):")
    analyzer = QuantumComplexityAnalyzer(bf.AND(4))
    amp = analyzer.grover_amplitude_analysis()
    print(f"  Optimal iterations: {amp['optimal_iterations']}")
    for step in amp["evolution"][:5]:
        print(f"    k={step['iteration']}: "
              f"P(success) = {step['success_probability']:.4f}")


def demo_comprehensive_analysis():
    """Demonstrate comprehensive Boolean function analysis."""
    print("\n[ANALYSIS] Comprehensive Analysis Demo")
    print("=" * 50)

    # Create a more interesting function
    tribes = bf.create(
        [
            # 4-variable tribes function: (x0 ∧ x1) ∨ (x2 ∧ x3)
            False,
            False,
            False,
            True,  # 00xy -> x2 ∧ x3
            False,
            False,
            False,
            True,  # 01xy -> x2 ∧ x3
            False,
            False,
            False,
            True,  # 10xy -> x2 ∧ x3
            True,
            True,
            True,
            True,  # 11xy -> True
        ]
    )

    print("Analyzing Tribes(2,2) function: (x0 ∧ x1) ∨ (x2 ∧ x3)")

    # Multiple representations
    print("\n1. Representations:")
    representations = ["truth_table", "anf", "fourier_expansion"]

    for rep in representations:
        try:
            data = tribes.get_representation(rep)
            if isinstance(data, dict):
                print(f"  {rep}: {len(data)} terms")
            else:
                print(f"  {rep}: {len(data)} elements")
        except Exception as e:
            print(f"  {rep}: Error - {e}")

    # Spectral analysis
    print("\n2. Spectral Analysis:")
    analyzer = bf.SpectralAnalyzer(tribes)

    influences = analyzer.influences()
    total_influence = analyzer.total_influence()
    noise_stability = analyzer.noise_stability(0.9)

    print(f"  Variable influences: {influences}")
    print(f"  Total influence: {total_influence:.4f}")
    print(f"  Noise stability (ρ=0.9): {noise_stability:.4f}")

    # Property testing
    print("\n3. Property Testing:")
    tester = bf.PropertyTester(tribes)
    properties = tester.run_all_tests()

    key_properties = ["constant", "balanced", "monotone", "symmetric"]
    for prop in key_properties:
        if prop in properties:
            result = properties[prop]
            if isinstance(result, bool):
                print(f"  {prop.capitalize()}: {result}")
            else:
                print(f"  {prop.capitalize()}: {result}")

    # Performance analysis
    print("\n4. Performance Analysis:")

    # Batch evaluation performance
    large_batch = np.random.randint(0, 16, 1000)
    start_time = time.time()
    batch_results = tribes.evaluate(large_batch)
    batch_time = (time.time() - start_time) * 1000

    print(f"  Batch evaluation (1000 inputs): {batch_time:.2f}ms")
    print(f"  Throughput: {1000 / (batch_time / 1000):.0f} evaluations/sec")


def create_visualization_examples():
    """Create visualization examples (if matplotlib available)."""
    print("\n[VIZ] Creating Visualization Examples")
    print("=" * 50)

    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "generated")
        os.makedirs(output_dir, exist_ok=True)

        # Example 1: Influence comparison
        functions = {
            "XOR": bf.create([False, True, True, False]),
            "AND": bf.create([False, False, False, True]),
            "OR": bf.create([False, True, True, True]),
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, (name, func) in enumerate(functions.items()):
            analyzer = bf.SpectralAnalyzer(func)
            influences = analyzer.influences()

            axes[i].bar(range(len(influences)), influences)
            axes[i].set_title(f"{name} Function Influences")
            axes[i].set_xlabel("Variable")
            axes[i].set_ylabel("Influence")
            axes[i].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "influence_comparison.png"), dpi=300)
        plt.close()

        print(f"  [PASS] Influence comparison saved to generated/influence_comparison.png")

        # Example 2: Noise stability curves
        majority = bf.create([False, False, False, True, False, True, True, True])
        analyzer = bf.SpectralAnalyzer(majority)

        rho_values = np.linspace(-1, 1, 21)
        stability_values = [analyzer.noise_stability(rho) for rho in rho_values]

        plt.figure(figsize=(10, 6))
        plt.plot(rho_values, stability_values, "b-", linewidth=2, label="Majority Function")
        plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.7, label="Random Function")
        plt.xlabel("Correlation (ρ)")
        plt.ylabel("Noise Stability")
        plt.title("Noise Stability vs Correlation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "noise_stability.png"), dpi=300)
        plt.close()

        print(f"  [PASS] Noise stability curve saved to generated/noise_stability.png")

    except Exception as e:
        print(f"  [WARN]  Visualization examples require matplotlib: {e}")


def main():
    """Run all demos."""
    print("[PERF] BooFun Advanced Features Demo")
    print("=" * 60)
    print("This demo showcases the advanced features of the BooFun library.")
    print("Note: Some features may not be available if dependencies are missing.\n")

    # Run all demos
    demo_anf_representation()
    demo_conversion_graph()
    demo_batch_processing()
    demo_gpu_acceleration()
    demo_numba_optimization()
    demo_testing_framework()
    demo_adapter_system()
    demo_error_models()
    demo_quantum_complexity()
    demo_comprehensive_analysis()
    create_visualization_examples()

    print("\n[DONE] Demo completed! Check the generated/ directory for visualization outputs.")
    print("\n[TIP] Next steps:")
    print("   - Explore the examples/ directory for more specific use cases")
    print("   - Check the documentation for detailed API reference")
    print("   - Run the test suite with: python -m pytest tests/")


if __name__ == "__main__":
    main()
