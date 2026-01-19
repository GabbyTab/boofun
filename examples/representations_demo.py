#!/usr/bin/env python3
"""
BooFun Representations Demo

This file demonstrates the multiple representation capabilities of the BooFun library,
including circuits, BDDs, and other Boolean function representations.
"""

import numpy as np
import boofun as bf

def basic_representations():
    """Demonstrate basic representation functionality."""
    print("=== Boolean Function Representations ===")
    
    # Create a simple function
    and_func = bf.create([0, 0, 0, 1])  # 2-variable AND
    
    print("1. Basic representation information:")
    print(f"   Function: 2-variable AND")
    print(f"   Truth table: {list(and_func.get_representation('truth_table'))}")
    print(f"   Number of variables: {and_func.n_vars}")
    print(f"   Available representations: {list(and_func.representations.keys())}")
    
    print("\n2. Function evaluation examples:")
    test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
    expected = [False, False, False, True]
    
    for inputs, exp in zip(test_cases, expected):
        result = and_func.evaluate(inputs)
        status = "✓" if result == exp else "✗"
        print(f"   AND{inputs} = {result} {status}")

def circuit_representations():
    """Demonstrate circuit representation capabilities."""
    print("\n=== Circuit Representations ===")
    
    try:
        from boofun.core.representations.circuit import BooleanCircuit, GateType
        
        print("1. Building Boolean circuits manually:")
        
        # Create a 2-input XOR circuit
        circuit = BooleanCircuit(2)
        
        # XOR = (A ∧ ¬B) ∨ (¬A ∧ B)
        not_a = circuit.add_gate(GateType.NOT, [circuit.input_gates[0]])
        not_b = circuit.add_gate(GateType.NOT, [circuit.input_gates[1]])
        
        a_and_not_b = circuit.add_gate(GateType.AND, [circuit.input_gates[0], not_b])
        not_a_and_b = circuit.add_gate(GateType.AND, [not_a, circuit.input_gates[1]])
        
        xor_gate = circuit.add_gate(GateType.OR, [a_and_not_b, not_a_and_b])
        circuit.set_output(xor_gate)
        
        print(f"   XOR circuit created with {circuit.get_size()} gates")
        print(f"   Circuit depth: {circuit.get_depth()}")
        
        # Test the circuit
        print("\n2. Testing XOR circuit:")
        test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected = [False, True, True, False]
        
        for inputs, exp in zip(test_cases, expected):
            result = circuit.evaluate(inputs)
            status = "✓" if result == exp else "✗"
            print(f"   XOR_circuit{inputs} = {result} {status}")
        
        print("\n3. Circuit information:")
        circuit_dict = circuit.to_dict()
        print(f"   Total gates: {len(circuit_dict['gates'])}")
        print(f"   Input gates: {circuit_dict['input_gates']}")
        print(f"   Output gate: {circuit_dict['output_gate']}")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Circuit representation not available: {e}")
        return False

def bdd_representations():
    """Demonstrate Binary Decision Diagram capabilities."""
    print("\n=== Binary Decision Diagrams (BDDs) ===")
    
    try:
        from boofun.core.representations.bdd import BDD
        
        print("1. Creating BDD for simple functions:")
        
        # Create BDD for first variable (x0)
        bdd = BDD(2)
        bdd.root = bdd.create_node(0, bdd.terminal_false, bdd.terminal_true)
        
        print("   BDD for x0 created")
        print(f"   Node count: {bdd.get_node_count()}")
        
        # Test the BDD
        print("\n2. Testing BDD evaluation:")
        test_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected = [False, False, True, True]  # x0 function
        
        for inputs, exp in zip(test_cases, expected):
            result = bdd.evaluate(inputs)
            status = "✓" if result == exp else "✗"
            print(f"   BDD(x0){inputs} = {result} {status}")
        
        print("\n3. BDD properties:")
        print(f"   Number of variables: {bdd.n_vars}")
        print(f"   Has root node: {bdd.root is not None}")
        print(f"   Terminal nodes: True and False")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  BDD representation not available: {e}")
        return False

def representation_conversions():
    """Demonstrate conversion between representations."""
    print("\n=== Representation Conversions ===")
    
    print("1. Starting with truth table representation:")
    original_func = bf.create([0, 1, 1, 0])  # XOR
    print(f"   Original function: XOR")
    print(f"   Truth table: {list(original_func.get_representation('truth_table'))}")
    
    print("\n2. Available representation types:")
    
    # Try to add different representations
    representation_types = [
        ("polynomial", "Polynomial (ANF)"),
        ("fourier_expansion", "Fourier Expansion"), 
        ("symbolic", "Symbolic Expression"),
    ]
    
    successful_conversions = []
    
    for rep_type, rep_name in representation_types:
        try:
            # Try to add this representation
            original_func.add_representation(
                original_func.get_representation('truth_table'), 
                rep_type
            )
            successful_conversions.append(rep_name)
            print(f"   ✓ {rep_name} conversion successful")
        except Exception as e:
            print(f"   ⚠️  {rep_name} conversion: {str(e)[:50]}...")
    
    print(f"\n3. Successfully converted to {len(successful_conversions)} additional representations")
    print(f"   Total representations: {len(original_func.representations)}")
    
    # Test that all representations give consistent results
    print("\n4. Verifying conversion consistency:")
    test_inputs = [[0, 1], [1, 0]]
    
    for inputs in test_inputs:
        results = []
        for rep_name in original_func.representations:
            try:
                result = original_func.evaluate(inputs, rep_type=rep_name)
                results.append(bool(result))
            except:
                pass
        
        if len(set(results)) <= 1:  # All results are the same
            print(f"   {inputs}: All representations agree ✓")
        else:
            print(f"   {inputs}: Inconsistent results: {results}")

def advanced_function_analysis():
    """Demonstrate advanced function analysis techniques."""
    print("\n=== Advanced Function Analysis ===")
    
    print("1. Comparing function complexity:")
    
    functions = [
        ("Identity", bf.BooleanFunctionBuiltins.dictator(1, 0)),
        ("XOR", bf.create([0, 1, 1, 0])),
        ("Majority", bf.BooleanFunctionBuiltins.majority(3)),
        ("Parity3", bf.BooleanFunctionBuiltins.parity(3)),
    ]
    
    print("   Function  | Vars | Total Inf | Entropy | Complexity")
    print("   " + "-" * 50)
    
    for name, func in functions:
        analyzer = bf.SpectralAnalyzer(func)
        
        total_inf = analyzer.total_influence()
        
        # Compute function entropy (balance measure)
        truth_table = func.get_representation('truth_table')
        ones = sum(truth_table)
        zeros = len(truth_table) - ones
        total = len(truth_table)
        
        if ones == 0 or zeros == 0:
            entropy = 0.0
        else:
            p1 = ones / total
            p0 = zeros / total
            entropy = -(p1 * np.log2(p1) + p0 * np.log2(p0))
        
        # Simple complexity measure: number of True entries
        complexity = ones / total
        
        print(f"   {name:9} | {func.n_vars:4} | {total_inf:9.3f} | {entropy:7.3f} | {complexity:10.3f}")
    
    print("\n2. Function symmetry analysis:")
    
    # Test if functions are symmetric under variable permutation
    symmetric_functions = []
    
    for name, func in functions:
        if func.n_vars <= 3:  # Only test small functions
            tester = bf.PropertyTester(func)
            try:
                is_symmetric = tester.symmetry_test()
                symmetric_functions.append((name, is_symmetric))
                print(f"   {name}: Symmetric = {is_symmetric}")
            except:
                print(f"   {name}: Symmetry test not implemented")
    
    print("\n3. Influence distribution analysis:")
    
    for name, func in functions:
        if func.n_vars > 1:  # Skip single-variable functions
            analyzer = bf.SpectralAnalyzer(func)
            influences = analyzer.influences()
            
            # Compute influence statistics
            mean_inf = np.mean(influences)
            std_inf = np.std(influences)
            min_inf = np.min(influences)
            max_inf = np.max(influences)
            
            print(f"   {name}: mean={mean_inf:.3f}, std={std_inf:.3f}, range=[{min_inf:.3f}, {max_inf:.3f}]")

def main():
    """Run all representation and analysis examples."""
    print("BooFun Library - Advanced Representations & Analysis")
    print("=" * 60)
    print("This demonstrates advanced features and analysis capabilities.")
    print()
    
    try:
        basic_representations()
        circuit_representations()
        bdd_representations()
        representation_conversions()
        advanced_function_analysis()
        
        print("\n✅ All representation examples completed!")
        print("\nNext steps:")
        print("  - Try visualization: python examples/visualization_examples.py")
        print("  - Explore the library: python examples/usage.py")
        print("  - Run tests: pytest tests/integration/")
        
    except Exception as e:
        print(f"❌ Error in representation examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
