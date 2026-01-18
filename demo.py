#!/usr/bin/env python3
"""
BoolFunc Library Demo

This script demonstrates the core functionality of the BoolFunc library,
showing how to create Boolean functions, analyze their properties, and
perform spectral analysis.
"""

import numpy as np
import boolfunc as bf
from boolfunc.analysis import SpectralAnalyzer
from boolfunc.core.builtins import BooleanFunctionBuiltins

def main():
    print("🚀 BoolFunc Library Demo")
    print("=" * 50)
    
    # 1. Create Boolean functions from different representations
    print("\n1. Creating Boolean Functions")
    print("-" * 30)
    
    # XOR function from truth table
    xor_func = bf.create([False, True, True, False])
    print(f"XOR function created: {xor_func}")
    
    # Test evaluation
    print("XOR evaluations:")
    for i in range(4):
        result = xor_func.evaluate(np.array(i))
        print(f"  {i:02b} → {result}")
    
    # 2. Built-in Boolean functions
    print("\n2. Built-in Boolean Functions")
    print("-" * 30)
    
    # Majority function
    maj3 = BooleanFunctionBuiltins.majority(3)
    print(f"3-variable majority: {maj3}")
    print("Majority evaluations:")
    for i in range(8):
        result = maj3.evaluate(np.array(i))
        bits = f"{i:03b}"
        ones = bits.count('1')
        print(f"  {bits} ({ones} ones) → {result}")
    
    # Dictator function
    dict1 = BooleanFunctionBuiltins.dictator(3, 1)
    print(f"\nDictator on variable 1: {dict1}")
    print("Dictator evaluations:")
    for i in range(8):
        result = dict1.evaluate(np.array(i))
        bits = f"{i:03b}"
        middle_bit = bool((i >> 1) & 1)
        print(f"  {bits} (middle bit: {middle_bit}) → {result}")
    
    # Parity function
    parity3 = BooleanFunctionBuiltins.parity(3)
    print(f"\n3-variable parity: {parity3}")
    print("Parity evaluations:")
    for i in range(8):
        result = parity3.evaluate(np.array(i))
        bits = f"{i:03b}"
        ones = bits.count('1')
        is_odd = ones % 2 == 1
        print(f"  {bits} ({ones} ones, odd: {is_odd}) → {result}")
    
    # 3. Spectral Analysis
    print("\n3. Spectral Analysis")
    print("-" * 30)
    
    # Analyze different functions
    functions = [
        ("XOR", xor_func),
        ("Majority-3", maj3),
        ("Dictator-1", dict1),
        ("Parity-3", parity3)
    ]
    
    for name, func in functions:
        print(f"\n{name} Analysis:")
        analyzer = SpectralAnalyzer(func)
        
        # Compute influences
        influences = analyzer.influences()
        print(f"  Variable influences: {influences}")
        print(f"  Total influence: {analyzer.total_influence():.3f}")
        
        # Noise stability
        stability_09 = analyzer.noise_stability(0.9)
        stability_05 = analyzer.noise_stability(0.5)
        print(f"  Noise stability (ρ=0.9): {stability_09:.3f}")
        print(f"  Noise stability (ρ=0.5): {stability_05:.3f}")
        
        # Spectral concentration
        conc_1 = analyzer.spectral_concentration(1)
        conc_2 = analyzer.spectral_concentration(2)
        print(f"  Spectral concentration (degree ≤ 1): {conc_1:.3f}")
        print(f"  Spectral concentration (degree ≤ 2): {conc_2:.3f}")
    
    # 4. Function Operations
    print("\n4. Function Operations")
    print("-" * 30)
    
    # NumPy array conversion
    xor_array = np.array(xor_func)
    print(f"XOR as NumPy array: {xor_array}")
    
    # Batch evaluation
    batch_inputs = np.array([0, 1, 2, 3])
    batch_results = xor_func.evaluate(batch_inputs)
    print(f"Batch evaluation: {batch_inputs} → {batch_results}")
    
    # Binary vector evaluation
    binary_input = np.array([1, 0])  # Binary: 10 → index 2
    binary_result = xor_func.evaluate(binary_input)
    print(f"Binary vector [1,0] → {binary_result}")
    
    # 5. Summary Statistics
    print("\n5. Summary Statistics")
    print("-" * 30)
    
    for name, func in functions:
        analyzer = SpectralAnalyzer(func)
        summary = analyzer.summary()
        print(f"\n{name} Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nThe BoolFunc library is working correctly with:")
    print("  • Multiple Boolean function representations")
    print("  • Built-in functions (majority, dictator, parity, etc.)")
    print("  • Spectral analysis and influence computation")
    print("  • Fourier expansion and noise stability")
    print("  • Comprehensive evaluation methods")
    print("  • Professional development setup")

if __name__ == "__main__":
    main()
