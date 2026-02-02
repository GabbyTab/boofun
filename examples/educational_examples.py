#!/usr/bin/env python3
"""
BooFun Educational Examples

This file provides examples suitable for teaching Boolean function analysis,
theoretical computer science, and discrete mathematics courses.
"""

import numpy as np

import boofun as bf


def boolean_logic_basics():
    """Demonstrate basic Boolean logic operations."""
    print("=== Boolean Logic Fundamentals ===")

    print("1. Basic Boolean operations (2 variables):")

    # Create fundamental Boolean functions
    operations = {
        "AND": [0, 0, 0, 1],
        "OR": [0, 1, 1, 1],
        "XOR": [0, 1, 1, 0],
        "NAND": [1, 1, 1, 0],
        "NOR": [1, 0, 0, 0],
    }

    print("   Input | AND | OR  | XOR | NAND| NOR")
    print("   " + "-" * 35)

    for i in range(4):
        bits = f"{i:02b}"
        inputs = [int(b) for b in bits]

        results = []
        for op_name, truth_table in operations.items():
            func = bf.create(truth_table)
            result = int(func.evaluate(inputs))
            results.append(result)

        result_str = " | ".join(f"{r:3}" for r in results)
        print(f"   {bits}   | {result_str}")

    print("\n2. De Morgan's Laws verification:")

    # NOT(A AND B) = (NOT A) OR (NOT B)
    and_func = bf.create([0, 0, 0, 1])
    nand_func = bf.create([1, 1, 1, 0])

    print("   Verifying: NOT(A AND B) = NAND(A, B)")
    for i in range(4):
        bits = [(i >> j) & 1 for j in range(1, -1, -1)]
        and_result = and_func.evaluate(bits)
        nand_result = nand_func.evaluate(bits)

        # NAND should be NOT AND
        verification = (not and_result) == nand_result
        status = "[ok]" if verification else "[fail]"
        print(f"   {bits}: NOT({and_result}) = {nand_result} {status}")


def function_complexity_analysis():
    """Demonstrate function complexity concepts."""
    print("\n=== Function Complexity Analysis ===")

    print("1. Influence and sensitivity analysis:")

    functions = [
        ("Constant", bf.BooleanFunctionBuiltins.constant(True, 3)),
        ("Dictator", bf.BooleanFunctionBuiltins.dictator(3, 1)),
        ("Parity", bf.BooleanFunctionBuiltins.parity(3)),
        ("Majority", bf.BooleanFunctionBuiltins.majority(3)),
    ]

    print("   Function  | Variables | Total Influence | Max Influence")
    print("   " + "-" * 55)

    for name, func in functions:
        analyzer = bf.SpectralAnalyzer(func)
        influences = analyzer.influences()
        total_inf = analyzer.total_influence()
        max_inf = np.max(influences)

        print(f"   {name:9} | {func.n_vars:9} | {total_inf:15.3f} | {max_inf:13.3f}")

    print("\n   Interpretation:")
    print("   - Total influence measures how much the function depends on its inputs")
    print("   - Higher influence = more sensitive to input changes")
    print("   - Parity has maximum influence (all variables matter equally)")
    print("   - Constants have zero influence (no variables matter)")

    print("\n2. Function balance analysis:")

    print("   Function  | True outputs | False outputs | Balanced?")
    print("   " + "-" * 50)

    for name, func in functions:
        truth_table = func.get_representation("truth_table")
        true_count = sum(truth_table)
        false_count = len(truth_table) - true_count

        tester = bf.PropertyTester(func)
        is_balanced = tester.balanced_test()

        print(f"   {name:9} | {true_count:12} | {false_count:13} | {is_balanced}")

    print("\n   Interpretation:")
    print("   - Balanced functions have equal numbers of True and False outputs")
    print("   - Balance is important for cryptographic applications")


def fourier_analysis_education():
    """Educational demonstration of Fourier analysis on Boolean functions."""
    print("\n=== Fourier Analysis on Boolean Functions ===")

    print("1. Fourier expansion concepts:")
    print("   Every Boolean function f: {0,1}ⁿ → {0,1} can be written as:")
    print("   f(x) = Σ f̂(S) χₛ(x), where χₛ(x) = ∏ᵢ∈ₛ (-1)ˣⁱ")
    print()

    # Demonstrate with XOR function
    xor = bf.create([0, 1, 1, 0])
    analyzer = bf.SpectralAnalyzer(xor)

    print("2. XOR function Fourier analysis:")
    fourier_coeffs = analyzer.fourier_expansion()

    print(f"   Fourier coefficients: {fourier_coeffs}")
    print("   Interpretation:")
    print("   - Coefficient 0 (empty set): constant term")
    print("   - Coefficient 1 (variable 0): linear term in x₀")
    print("   - Coefficient 2 (variable 1): linear term in x₁")
    print("   - Coefficient 3 (both vars): interaction term x₀x₁")

    # Verify Parseval's theorem
    print("\n3. Parseval's theorem verification:")
    fourier_norm = np.sum(fourier_coeffs**2)
    print(f"   ||f̂||² = {fourier_norm:.6f}")
    print("   (Should equal 1.0 for Boolean functions in ±1 representation)")

    print("\n4. Influence via Fourier coefficients:")
    influences = analyzer.influences()
    print(f"   Computed influences: {influences}")
    print("   Formula: Inf_i(f) = Σₛ:ᵢ∈ₛ f̂(S)²")

    # Manual verification for educational purposes
    manual_inf_0 = fourier_coeffs[1] ** 2 + fourier_coeffs[3] ** 2  # Terms containing variable 0
    manual_inf_1 = fourier_coeffs[2] ** 2 + fourier_coeffs[3] ** 2  # Terms containing variable 1

    print(f"   Manual calculation: Inf₀ = {manual_inf_0:.6f}, Inf₁ = {manual_inf_1:.6f}")


def cryptographic_properties():
    """Demonstrate properties relevant to cryptography."""
    print("\n=== Cryptographic Properties ===")

    print("1. Analyzing cryptographically relevant functions:")

    # Functions of interest in cryptography
    crypto_functions = [
        ("XOR", bf.create([0, 1, 1, 0])),
        ("Majority", bf.BooleanFunctionBuiltins.majority(3)),
        ("Parity", bf.BooleanFunctionBuiltins.parity(3)),
    ]

    print("   Function  | Balanced | Nonlinearity* | Correlation Immunity*")
    print("   " + "-" * 60)

    for name, func in crypto_functions:
        tester = bf.PropertyTester(func)
        analyzer = bf.SpectralAnalyzer(func)

        is_balanced = tester.balanced_test()

        # Nonlinearity approximation (distance to nearest linear function)
        influences = analyzer.influences()
        total_inf = analyzer.total_influence()

        # Simple nonlinearity indicator
        if name == "Parity":
            nonlinearity = "Linear"
        elif total_inf == 0:
            nonlinearity = "Constant"
        else:
            nonlinearity = f"~{2**(func.n_vars-1) - int(total_inf * 2**(func.n_vars-2))}"

        # Correlation immunity (simplified)
        max_inf = np.max(influences)
        if max_inf < 0.1:
            corr_immunity = "High"
        elif max_inf < 0.5:
            corr_immunity = "Medium"
        else:
            corr_immunity = "Low"

        print(f"   {name:9} | {str(is_balanced):8} | {nonlinearity:13} | {corr_immunity}")

    print("\n   *Simplified approximations for educational purposes")

    print("\n2. Avalanche effect analysis:")
    print("   (How much the output changes when one input bit is flipped)")

    for name, func in crypto_functions:
        if func.n_vars <= 3:  # Only for small functions
            total_changes = 0
            total_tests = 0

            for i in range(2**func.n_vars):
                bits = [(i >> j) & 1 for j in range(func.n_vars - 1, -1, -1)]
                original_output = func.evaluate(bits)

                # Flip each bit and count changes
                for bit_pos in range(func.n_vars):
                    flipped_bits = bits.copy()
                    flipped_bits[bit_pos] = 1 - flipped_bits[bit_pos]

                    flipped_output = func.evaluate(flipped_bits)
                    if original_output != flipped_output:
                        total_changes += 1
                    total_tests += 1

            avalanche_ratio = total_changes / total_tests
            print(f"   {name}: {avalanche_ratio:.3f} (ideal = 0.5)")


def main():
    """Run all educational examples."""
    print("BooFun Library - Educational Examples")
    print("=" * 45)
    print("Examples for teaching Boolean function analysis and theoretical computer science.")
    print()

    try:
        boolean_logic_basics()
        function_complexity_analysis()
        fourier_analysis_education()
        cryptographic_properties()

        print("\n[PASS] All educational examples completed!")
        print("\n📚 Educational Topics Covered:")
        print("  - Boolean logic fundamentals")
        print("  - Function complexity and influence")
        print("  - Fourier analysis on Boolean functions")
        print("  - Cryptographic properties")
        print("  - Mathematical property verification")

        print("\n🎓 Suitable for courses in:")
        print("  - Theoretical Computer Science")
        print("  - Discrete Mathematics")
        print("  - Cryptography")
        print("  - Boolean Function Analysis")

    except Exception as e:
        print(f"[FAIL] Error in educational examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
