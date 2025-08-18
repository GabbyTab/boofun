#!/usr/bin/env python3
"""
BoolFunc Library Usage Examples

This file demonstrates the main functionality of the BoolFunc library
with clear, practical examples suitable for academic and research use.
"""

import numpy as np
import boolfunc as bf

def basic_usage():
    """Demonstrate basic Boolean function creation and evaluation."""
    print("=== Basic Usage ===")
    
    # Create Boolean functions from truth tables
    print("1. Creating Boolean functions from truth tables:")
    
    # XOR function (2 variables)
    xor = bf.create([0, 1, 1, 0])
    print(f"   XOR function: f(00)={xor.evaluate([0,0])}, f(01)={xor.evaluate([0,1])}")
    print(f"                 f(10)={xor.evaluate([1,0])}, f(11)={xor.evaluate([1,1])}")
    
    # AND function (2 variables)  
    and_func = bf.create([0, 0, 0, 1])
    print(f"   AND function: {[and_func.evaluate([i//2, i%2]) for i in range(4)]}")
    
    print()

def builtin_functions():
    """Demonstrate built-in Boolean function generators."""
    print("=== Built-in Functions ===")
    
    # Majority function
    maj3 = bf.BooleanFunctionBuiltins.majority(3)
    print("2. Majority function (3 variables):")
    for i in range(8):
        bits = [(i>>j)&1 for j in range(2, -1, -1)]  # Convert to binary
        result = maj3.evaluate(bits)
        print(f"   maj({bits}) = {result}")
    
    # Parity function
    par3 = bf.BooleanFunctionBuiltins.parity(3)
    print(f"\n   Parity function evaluations: {[par3.evaluate([(i>>j)&1 for j in range(2,-1,-1)]) for i in range(8)]}")
    
    # Dictator function
    dict_func = bf.BooleanFunctionBuiltins.dictator(1, 3)  # Second variable
    print(f"   Dictator(x1) evaluations: {[dict_func.evaluate([(i>>j)&1 for j in range(2,-1,-1)]) for i in range(8)]}")
    
    print()

def spectral_analysis():
    """Demonstrate spectral analysis capabilities."""
    print("=== Spectral Analysis ===")
    
    # Analyze XOR function
    xor = bf.create([0, 1, 1, 0])
    analyzer = bf.SpectralAnalyzer(xor)
    
    print("3. Analyzing XOR function:")
    print(f"   Variable influences: {analyzer.influences()}")
    print(f"   Total influence: {analyzer.total_influence():.3f}")
    print(f"   Noise stability (ρ=0.9): {analyzer.noise_stability(0.9):.3f}")
    
    # Compare with majority function
    maj3 = bf.BooleanFunctionBuiltins.majority(3)
    maj_analyzer = bf.SpectralAnalyzer(maj3)
    
    print("\n   Comparing with Majority function:")
    print(f"   Majority influences: {maj_analyzer.influences()}")
    print(f"   Majority total influence: {maj_analyzer.total_influence():.3f}")
    
    print()

def property_testing():
    """Demonstrate property testing algorithms."""
    print("=== Property Testing ===")
    
    # Test different functions
    functions = {
        "XOR": bf.create([0, 1, 1, 0]),
        "Constant": bf.BooleanFunctionBuiltins.constant(True, 2),
        "Majority": bf.BooleanFunctionBuiltins.majority(3)
    }
    
    print("4. Property testing results:")
    for name, func in functions.items():
        tester = bf.PropertyTester(func)
        
        is_constant = tester.constant_test()
        is_balanced = tester.balanced_test()
        
        print(f"   {name:8} - Constant: {is_constant:5}, Balanced: {is_balanced}")
    
    print()

def advanced_usage():
    """Demonstrate advanced features."""
    print("=== Advanced Usage ===")
    
    print("5. Function composition:")
    # Create composite functions
    x1 = bf.BooleanFunctionBuiltins.dictator(0, 2)  # First variable
    x2 = bf.BooleanFunctionBuiltins.dictator(1, 2)  # Second variable
    
    # XOR as composition: x1 ⊕ x2
    xor_composed = x1 + x2  # Addition in GF(2) = XOR
    print(f"   XOR via composition: {[xor_composed.evaluate([i//2, i%2]) for i in range(4)]}")
    
    print("\n6. Multiple representations:")
    func = bf.create([0, 0, 1, 1])  # OR function
    print(f"   Available representations: {list(func.representations.keys())}")
    
    # Show function properties
    analyzer = bf.SpectralAnalyzer(func)
    summary = analyzer.summary()
    print(f"   Function summary: {len(summary)} spectral properties computed")
    
    print()

def main():
    """Run all examples."""
    print("BoolFunc Library - Usage Examples")
    print("=" * 40)
    print("This demonstrates the core functionality of the Boolean function library.")
    print()
    
    try:
        basic_usage()
        builtin_functions()
        spectral_analysis()
        property_testing()
        advanced_usage()
        
        print("✅ All examples completed successfully!")
        print("\nFor more advanced features, see:")
        print("  - Visualization: bf.BooleanFunctionVisualizer")
        print("  - Circuit representations: bf.BooleanCircuit")
        print("  - Performance benchmarking: bf.PerformanceBenchmark")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
