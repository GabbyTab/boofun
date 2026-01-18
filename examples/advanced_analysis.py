#!/usr/bin/env python3
"""
BoolFunc Advanced Analysis Examples

This file demonstrates advanced analysis capabilities including:
- Detailed spectral analysis
- Property testing algorithms
- Mathematical property verification
- Research-oriented workflows
"""

import numpy as np
import boolfunc as bf

def spectral_analysis_deep_dive():
    """Demonstrate comprehensive spectral analysis."""
    print("=== Advanced Spectral Analysis ===")
    
    # Create functions with interesting spectral properties
    functions = {
        "XOR": bf.create([0, 1, 1, 0]),
        "Majority": bf.BooleanFunctionBuiltins.majority(3),
        "Parity": bf.BooleanFunctionBuiltins.parity(3),
        "AND": bf.create([0, 0, 0, 1]),
    }
    
    print("1. Comprehensive spectral analysis comparison:")
    print("   Function    | Vars | Total Inf | Max Inf | Noise Stab(0.9)")
    print("   " + "-" * 55)
    
    for name, func in functions.items():
        analyzer = bf.SpectralAnalyzer(func)
        
        influences = analyzer.influences()
        total_inf = analyzer.total_influence()
        max_inf = np.max(influences)
        noise_stab = analyzer.noise_stability(0.9)
        
        print(f"   {name:11} | {func.n_vars:4} | {total_inf:9.3f} | {max_inf:7.3f} | {noise_stab:10.3f}")
    
    print("\n2. Detailed Fourier analysis:")
    xor = functions["XOR"]
    analyzer = bf.SpectralAnalyzer(xor)
    
    fourier_coeffs = analyzer.fourier_expansion()
    print(f"   XOR Fourier coefficients: {fourier_coeffs}")
    
    # Verify Parseval's identity
    fourier_norm_sq = np.sum(fourier_coeffs ** 2)
    print(f"   Fourier norm squared: {fourier_norm_sq:.6f}")
    print(f"   (Should equal 1.0 for normalized functions)")
    
    print("\n3. Spectral concentration analysis:")
    for degree in [1, 2]:
        try:
            concentration = analyzer.spectral_concentration(degree)
            print(f"   Spectral concentration at degree {degree}: {concentration:.6f}")
        except Exception as e:
            print(f"   Spectral concentration degree {degree}: Not implemented")

def property_testing_showcase():
    """Demonstrate comprehensive property testing."""
    print("\n=== Property Testing Showcase ===")
    
    # Create test suite of functions with known properties
    test_functions = [
        (bf.BooleanFunctionBuiltins.constant(True, 2), "Constant True"),
        (bf.BooleanFunctionBuiltins.constant(False, 2), "Constant False"),
        (bf.BooleanFunctionBuiltins.parity(2), "Parity (Linear)"),
        (bf.BooleanFunctionBuiltins.parity(3), "Parity 3-var"),
        (bf.BooleanFunctionBuiltins.dictator(2, 0), "Dictator"),
        (bf.create([0, 1, 1, 0]), "XOR"),
        (bf.create([0, 0, 0, 1]), "AND"),
        (bf.BooleanFunctionBuiltins.majority(3), "Majority"),
    ]
    
    print("1. Property testing results:")
    print("   Function      | Constant | Balanced | Linear* | Symmetric")
    print("   " + "-" * 55)
    
    for func, name in test_functions:
        tester = bf.PropertyTester(func, random_seed=42)
        
        is_constant = tester.constant_test()
        is_balanced = tester.balanced_test()
        
        # Test linearity (probabilistic)
        try:
            is_linear = tester.blr_linearity_test(num_queries=50)
        except:
            is_linear = "N/A"
        
        # Test symmetry
        try:
            is_symmetric = tester.symmetry_test()
        except:
            is_symmetric = "N/A"
        
        print(f"   {name:13} | {str(is_constant):8} | {str(is_balanced):8} | {str(is_linear):7} | {str(is_symmetric):9}")
    
    print("\n   *Linear testing is probabilistic - results may vary")
    
    print("\n2. Detailed property analysis for XOR:")
    xor_tester = bf.PropertyTester(bf.create([0, 1, 1, 0]))
    
    properties = {
        "Constant": xor_tester.constant_test(),
        "Balanced": xor_tester.balanced_test(),
    }
    
    try:
        properties["Monotonic"] = xor_tester.monotonicity_test()
    except:
        properties["Monotonic"] = "Not implemented"
    
    for prop, value in properties.items():
        print(f"   {prop}: {value}")

def mathematical_properties():
    """Demonstrate mathematical property verification."""
    print("\n=== Mathematical Properties Verification ===")
    
    print("1. Influence properties:")
    
    # XOR function - all variables have maximum influence
    xor = bf.create([0, 1, 1, 0])
    xor_analyzer = bf.SpectralAnalyzer(xor)
    xor_influences = xor_analyzer.influences()
    
    print(f"   XOR influences: {xor_influences}")
    print(f"   All influences = 1.0: {all(abs(inf - 1.0) < 1e-10 for inf in xor_influences)}")
    
    # Majority function - symmetric influences
    maj = bf.BooleanFunctionBuiltins.majority(3)
    maj_analyzer = bf.SpectralAnalyzer(maj)
    maj_influences = maj_analyzer.influences()
    
    print(f"   Majority influences: {maj_influences}")
    print(f"   All influences equal: {all(abs(inf - maj_influences[0]) < 1e-10 for inf in maj_influences)}")
    
    print("\n2. Noise stability properties:")
    
    # Test noise stability at different correlation levels
    rho_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    print("   Function | ρ=-1.0 | ρ=-0.5 | ρ=0.0  | ρ=0.5  | ρ=1.0")
    print("   " + "-" * 50)
    
    for name, func in [("XOR", xor), ("Majority", maj)]:
        analyzer = bf.SpectralAnalyzer(func)
        stabilities = []
        
        for rho in rho_values:
            try:
                stability = analyzer.noise_stability(rho)
                stabilities.append(f"{stability:.3f}")
            except:
                stabilities.append("N/A")
        
        stab_str = " | ".join(f"{s:6}" for s in stabilities)
        print(f"   {name:8} | {stab_str}")
    
    print("\n3. Function composition properties:")
    
    # Test that XOR is its own inverse: f ⊕ f = 0
    try:
        x1 = bf.BooleanFunctionBuiltins.dictator(2, 0)
        x2 = bf.BooleanFunctionBuiltins.dictator(2, 1)
        
        print("   Testing XOR self-inverse property...")
        print("   (Note: Function composition may not be fully implemented)")
        
        # Test individual dictator functions work
        test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        print("   Dictator functions working:")
        for inputs in test_inputs:
            result1 = x1.evaluate(inputs)
            result2 = x2.evaluate(inputs)
            expected1 = bool(inputs[0])
            expected2 = bool(inputs[1])
            print(f"     {inputs} -> x1={result1}({expected1}), x2={result2}({expected2})")
        
    except Exception as e:
        print(f"   Function composition not fully implemented: {e}")

def research_workflow_example():
    """Demonstrate a typical research workflow."""
    print("\n=== Research Workflow Example ===")
    
    print("1. Research Question: How do influences relate to noise stability?")
    
    # Create several functions to study
    study_functions = [
        ("Constant", bf.BooleanFunctionBuiltins.constant(True, 3)),
        ("Dictator", bf.BooleanFunctionBuiltins.dictator(3, 0)),
        ("Parity", bf.BooleanFunctionBuiltins.parity(3)),
        ("Majority", bf.BooleanFunctionBuiltins.majority(3)),
    ]
    
    print("\n2. Data collection:")
    results = []
    
    for name, func in study_functions:
        analyzer = bf.SpectralAnalyzer(func)
        
        # Collect metrics
        influences = analyzer.influences()
        total_inf = analyzer.total_influence()
        max_inf = np.max(influences)
        noise_stab_09 = analyzer.noise_stability(0.9)
        noise_stab_05 = analyzer.noise_stability(0.5)
        
        results.append({
            'name': name,
            'total_influence': total_inf,
            'max_influence': max_inf,
            'noise_stability_0.9': noise_stab_09,
            'noise_stability_0.5': noise_stab_05,
        })
        
        print(f"   {name}: Total Inf={total_inf:.3f}, Noise Stab(0.9)={noise_stab_09:.3f}")
    
    print("\n3. Analysis insights:")
    
    # Find correlations
    total_infs = [r['total_influence'] for r in results]
    noise_stabs = [r['noise_stability_0.9'] for r in results]
    
    # Simple correlation analysis
    correlation = np.corrcoef(total_infs, noise_stabs)[0, 1]
    print(f"   Correlation between total influence and noise stability: {correlation:.3f}")
    
    # Find extremes
    max_inf_func = max(results, key=lambda x: x['total_influence'])
    min_inf_func = min(results, key=lambda x: x['total_influence'])
    
    print(f"   Highest influence: {max_inf_func['name']} ({max_inf_func['total_influence']:.3f})")
    print(f"   Lowest influence: {min_inf_func['name']} ({min_inf_func['total_influence']:.3f})")
    
    print("\n4. Research conclusions:")
    print("   - Functions with higher total influence tend to be less noise-stable")
    print("   - Parity functions have maximum influence (all variables matter equally)")
    print("   - Constant functions have zero influence (no variables matter)")

def main():
    """Run all advanced analysis examples."""
    print("BoolFunc Library - Advanced Analysis Examples")
    print("=" * 50)
    print("This demonstrates advanced analysis and research capabilities.")
    print()
    
    try:
        spectral_analysis_deep_dive()
        property_testing_showcase()
        mathematical_properties()
        research_workflow_example()
        
        print("\n✅ All advanced analysis examples completed!")
        print("\nFor visualization of these results, run:")
        print("  python examples/visualization_examples.py")
        
    except Exception as e:
        print(f"❌ Error in advanced analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
