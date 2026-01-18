#!/usr/bin/env python3
"""
BoolFunc Complete Demo

A comprehensive demonstration of all BoolFunc library capabilities.
This script showcases the full range of features in a single demonstration.
"""

import numpy as np
import boolfunc as bf

def main():
    """Complete demonstration of BoolFunc capabilities."""
    print("🚀 BoolFunc Library - Complete Demonstration")
    print("=" * 55)
    print("Comprehensive showcase of Boolean function analysis capabilities")
    print()
    
    # === 1. FUNCTION CREATION ===
    print("📊 1. FUNCTION CREATION")
    print("-" * 25)
    
    # Create functions in different ways
    xor = bf.create([0, 1, 1, 0])
    majority = bf.BooleanFunctionBuiltins.majority(3)
    parity = bf.BooleanFunctionBuiltins.parity(3)
    dictator = bf.BooleanFunctionBuiltins.dictator(3, 1)
    
    functions = {
        "XOR": xor,
        "Majority": majority,
        "Parity": parity,
        "Dictator": dictator
    }
    
    for name, func in functions.items():
        print(f"✓ {name}: {func.n_vars} variables, {2**func.n_vars} possible inputs")
    
    # === 2. FUNCTION EVALUATION ===
    print(f"\n🔍 2. FUNCTION EVALUATION")
    print("-" * 25)
    
    print("Testing XOR function:")
    for i in range(4):
        bits = [(i >> j) & 1 for j in range(1, -1, -1)]
        result = xor.evaluate(bits)
        print(f"   XOR{bits} = {result}")
    
    # === 3. SPECTRAL ANALYSIS ===
    print(f"\n📈 3. SPECTRAL ANALYSIS")
    print("-" * 25)
    
    print("Function    | Total Influence | Noise Stability(0.9)")
    print("-" * 45)
    
    for name, func in functions.items():
        analyzer = bf.SpectralAnalyzer(func)
        total_inf = analyzer.total_influence()
        noise_stab = analyzer.noise_stability(0.9)
        print(f"{name:11} | {total_inf:15.3f} | {noise_stab:16.3f}")
    
    # === 4. PROPERTY TESTING ===
    print(f"\n🧪 4. PROPERTY TESTING")
    print("-" * 25)
    
    print("Function    | Constant | Balanced | Symmetric")
    print("-" * 40)
    
    for name, func in functions.items():
        tester = bf.PropertyTester(func)
        is_constant = tester.constant_test()
        is_balanced = tester.balanced_test()
        
        try:
            is_symmetric = tester.symmetry_test()
        except:
            is_symmetric = "N/A"
        
        print(f"{name:11} | {str(is_constant):8} | {str(is_balanced):8} | {str(is_symmetric):9}")
    
    # === 5. MATHEMATICAL PROPERTIES ===
    print(f"\n🔬 5. MATHEMATICAL PROPERTIES")
    print("-" * 30)
    
    print("Influence analysis for XOR function:")
    xor_analyzer = bf.SpectralAnalyzer(xor)
    influences = xor_analyzer.influences()
    fourier_coeffs = xor_analyzer.fourier_expansion()
    
    print(f"   Variable influences: {influences}")
    print(f"   Fourier coefficients: {fourier_coeffs}")
    print(f"   Parseval identity: ||f̂||² = {np.sum(fourier_coeffs**2):.6f}")
    
    # === 6. ADVANCED FEATURES ===
    print(f"\n⚙️  6. ADVANCED FEATURES")
    print("-" * 25)
    
    print("Available advanced capabilities:")
    
    # Test circuit representation
    try:
        from boolfunc.core.representations.circuit import BooleanCircuit, GateType
        circuit = BooleanCircuit(2)
        and_gate = circuit.add_gate(GateType.AND, circuit.input_gates)
        circuit.set_output(and_gate)
        print("✓ Circuit representation: Boolean circuits with multiple gate types")
    except:
        print("⚠️  Circuit representation: Limited availability")
    
    # Test BDD representation
    try:
        from boolfunc.core.representations.bdd import BDD
        bdd = BDD(2)
        bdd.root = bdd.create_node(0, bdd.terminal_false, bdd.terminal_true)
        print("✓ BDD representation: Binary Decision Diagrams")
    except:
        print("⚠️  BDD representation: Limited availability")
    
    # Test visualization
    try:
        from boolfunc.visualization import BooleanFunctionVisualizer
        viz = BooleanFunctionVisualizer(xor, backend="matplotlib")
        print("✓ Visualization: Interactive plots and dashboards")
    except:
        print("⚠️  Visualization: Install with pip install -e \".[visualization]\"")
    
    # === 7. RESEARCH APPLICATIONS ===
    print(f"\n🎓 7. RESEARCH APPLICATIONS")
    print("-" * 30)
    
    print("Example research questions this library can help answer:")
    print()
    print("• How does function complexity relate to noise sensitivity?")
    print("• Which Boolean functions are most suitable for cryptographic applications?")
    print("• How do spectral properties change under function composition?")
    print("• What is the influence distribution for different function classes?")
    print("• How does the Fourier spectrum relate to circuit complexity?")
    
    print(f"\n📊 8. SUMMARY STATISTICS")
    print("-" * 25)
    
    # Compute some interesting statistics
    total_functions_analyzed = len(functions)
    total_evaluations = sum(2**func.n_vars for func in functions.values())
    avg_influence = np.mean([bf.SpectralAnalyzer(func).total_influence() 
                            for func in functions.values()])
    
    print(f"Functions analyzed: {total_functions_analyzed}")
    print(f"Total truth table entries: {total_evaluations}")
    print(f"Average total influence: {avg_influence:.3f}")
    print(f"Library version: {bf.__version__}")
    
    # === 9. NEXT STEPS ===
    print(f"\n🎯 9. NEXT STEPS")
    print("-" * 20)
    
    print("Explore more features:")
    print("• Run individual example files for detailed demonstrations")
    print("• Try visualization features with matplotlib/plotly")
    print("• Experiment with your own Boolean functions")
    print("• Use the library for research or educational projects")
    print()
    print("Example files to try:")
    print("  python examples/educational_examples.py    # For learning")
    print("  python examples/advanced_analysis.py       # For research")
    print("  python examples/visualization_examples.py  # For plotting")
    
    print(f"\n✅ Complete demonstration finished!")
    print("🎉 BoolFunc library is ready for Boolean function analysis!")

if __name__ == "__main__":
    main()
