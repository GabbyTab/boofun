"""
Quantum Boolean Function Analysis Demo

This example demonstrates quantum-specific features of the BoolFunc library,
including quantum property testing, quantum Fourier analysis, and quantum
resource estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import boolfunc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import boolfunc as bf


def demo_quantum_oracle_creation():
    """Demonstrate quantum oracle creation."""
    print("⚛️  Quantum Oracle Creation")
    print("=" * 40)
    
    # Create various Boolean functions
    functions = {
        'XOR': bf.create([False, True, True, False]),
        'AND': bf.create([False, False, False, True]),
        'Majority': bf.create([False, False, False, True, False, True, True, True])
    }
    
    for name, func in functions.items():
        quantum_func = bf.create_quantum_boolean_function(func)
        
        print(f"\n{name} Function:")
        print(f"  Variables: {quantum_func.n_vars}")
        
        # Try to create quantum oracle (if Qiskit available)
        try:
            oracle = quantum_func.create_quantum_oracle()
            if oracle:
                print(f"  Quantum Oracle: ✅ Created")
                print(f"  Oracle Depth: {quantum_func._estimate_oracle_depth()}")
            else:
                print(f"  Quantum Oracle: ⚠️  Requires Qiskit")
        except Exception as e:
            print(f"  Quantum Oracle: ⚠️  {e}")


def demo_quantum_fourier_analysis():
    """Demonstrate quantum Fourier analysis."""
    print("\n🌊 Quantum Fourier Analysis")
    print("=" * 40)
    
    # Test different function types
    functions = {
        'Linear (XOR)': bf.create([False, True, True, False]),
        'Quadratic (AND)': bf.create([False, False, False, True]),
        'Balanced (XOR3)': bf.create([False, True, True, False, True, False, False, True]),
        'Unbalanced (Majority)': bf.create([False, False, False, True, False, True, True, True])
    }
    
    for name, func in functions.items():
        quantum_func = bf.create_quantum_boolean_function(func)
        
        print(f"\n{name}:")
        
        # Quantum Fourier analysis
        fourier_results = quantum_func.quantum_fourier_analysis()
        
        print(f"  Method: {fourier_results['method']}")
        print(f"  Quantum Advantage: {fourier_results.get('quantum_advantage', 'N/A')}")
        
        if 'fourier_coefficients' in fourier_results:
            coeffs = fourier_results['fourier_coefficients']
            non_zero_coeffs = np.sum(np.abs(coeffs) > 1e-10)
            print(f"  Non-zero Fourier coefficients: {non_zero_coeffs}/{len(coeffs)}")
            
            # Show spectral concentration
            total_weight = np.sum(coeffs**2)
            degree_1_weight = coeffs[0]**2 + np.sum([coeffs[1 << i]**2 for i in range(func.n_vars)])
            if total_weight > 0:
                concentration = degree_1_weight / total_weight
                print(f"  Degree-1 spectral concentration: {concentration:.3f}")


def demo_quantum_property_testing():
    """Demonstrate quantum property testing algorithms."""
    print("\n🔬 Quantum Property Testing")
    print("=" * 40)
    
    # Create test functions with known properties
    test_functions = {
        'Linear (Parity)': {
            'function': bf.create([False, True, True, False]),
            'expected': {'linearity': True, 'monotonicity': False}
        },
        'Monotone (AND)': {
            'function': bf.create([False, False, False, True]),
            'expected': {'linearity': False, 'monotonicity': True}
        },
        'Majority': {
            'function': bf.create([False, False, False, True, False, True, True, True]),
            'expected': {'linearity': False, 'monotonicity': True}
        },
        'Random-like': {
            'function': bf.create([True, False, True, True, False, True, False, False]),
            'expected': {'linearity': False, 'monotonicity': False}
        }
    }
    
    properties_to_test = ['linearity', 'monotonicity', 'junta']
    
    for func_name, func_data in test_functions.items():
        quantum_func = bf.create_quantum_boolean_function(func_data['function'])
        
        print(f"\n{func_name}:")
        
        for prop in properties_to_test:
            try:
                if prop == 'junta':
                    # Test if function is a k-junta for k=1,2
                    for k in [1, 2]:
                        result = quantum_func.quantum_property_testing(prop, k=k)
                        print(f"  {k}-junta: {result.get('is_k_junta', 'N/A')}")
                else:
                    result = quantum_func.quantum_property_testing(prop)
                    detected = result.get(f'is_{prop}', result.get(f'is_{prop[:-3]}', 'N/A'))
                    expected = func_data['expected'].get(prop, 'N/A')
                    
                    status = "✅" if detected == expected else "⚠️"
                    print(f"  {prop.capitalize()}: {detected} {status}")
                    
                    if 'num_queries' in result:
                        print(f"    Queries used: {result['num_queries']}")
                    if 'quantum_speedup' in result:
                        print(f"    Quantum speedup: {result['quantum_speedup']}")
                        
            except Exception as e:
                print(f"  {prop.capitalize()}: Error - {e}")


def demo_quantum_resource_estimation():
    """Demonstrate quantum resource estimation."""
    print("\n💻 Quantum Resource Estimation")
    print("=" * 40)
    
    # Test functions of different sizes
    function_sizes = [
        (2, "XOR", [False, True, True, False]),
        (3, "Majority", [False, False, False, True, False, True, True, True]),
        (4, "Parity", [i % 2 == 1 for i in range(16) if bin(i).count('1') % 2 == 1] + 
                     [i % 2 == 0 for i in range(16) if bin(i).count('1') % 2 == 0])
    ]
    
    print("Resource requirements by function size:")
    print("Size | Qubits | Depth | Gates | Coherence | Volume")
    print("-" * 55)
    
    for n_vars, name, truth_table in function_sizes:
        func = bf.create(truth_table[:2**n_vars])  # Ensure correct size
        quantum_func = bf.create_quantum_boolean_function(func)
        
        resources = quantum_func.get_quantum_resources()
        
        print(f"{n_vars:4d} | {resources['qubits_required']:6d} | "
              f"{resources['circuit_depth']:5d} | {resources['gate_count']:5d} | "
              f"{resources['coherence_time_needed']:9s} | {resources['quantum_volume_required']:6d}")


def demo_quantum_vs_classical_comparison():
    """Demonstrate quantum vs classical algorithm comparison."""
    print("\n⚖️  Quantum vs Classical Comparison")
    print("=" * 40)
    
    # Test different function sizes
    sizes = [2, 3, 4, 5, 6]
    
    print("Function size analysis:")
    print("Variables | Function Size | Quantum Advantages | Recommendations")
    print("-" * 70)
    
    for n_vars in sizes:
        # Create a representative function
        truth_table = [i % 2 for i in range(2**n_vars)]  # Simple pattern
        func = bf.create(truth_table)
        quantum_func = bf.create_quantum_boolean_function(func)
        
        comparison = quantum_func.quantum_algorithm_comparison()
        
        advantages = ", ".join(comparison.get('quantum_advantages', ['None']))[:20]
        recommendations = comparison.get('recommendations', ['N/A'])[0][:25]
        
        print(f"{n_vars:9d} | {comparison['function_size']:13d} | "
              f"{advantages:18s} | {recommendations}")


def demo_quantum_influence_estimation():
    """Demonstrate quantum influence estimation."""
    print("\n📊 Quantum Influence Estimation")
    print("=" * 40)
    
    # Create functions with different influence patterns
    functions = {
        'Equal Influences (XOR)': bf.create([False, True, True, False]),
        'Unequal Influences (x0)': bf.create([False, False, True, True]),
        'Complex (Majority)': bf.create([False, False, False, True, False, True, True, True])
    }
    
    for name, func in functions.items():
        quantum_func = bf.create_quantum_boolean_function(func)
        
        print(f"\n{name}:")
        
        # Classical influences for comparison
        classical_analyzer = bf.SpectralAnalyzer(func)
        classical_influences = classical_analyzer.influences()
        
        print(f"  Classical influences: {classical_influences}")
        
        # Quantum influence estimation for each variable
        quantum_influences = []
        for var_idx in range(func.n_vars):
            result = quantum_func.quantum_influence_estimation(var_idx, num_queries=100)
            quantum_influences.append(result['influence'])
            
            speedup = "✅" if result.get('quantum_speedup', False) else "⚠️"
            print(f"    Variable {var_idx}: {result['influence']:.3f} {speedup}")
        
        # Compare accuracy
        if len(classical_influences) == len(quantum_influences):
            max_diff = np.max(np.abs(np.array(classical_influences) - np.array(quantum_influences)))
            print(f"  Max difference from classical: {max_diff:.4f}")


def create_quantum_visualizations():
    """Create quantum analysis visualizations."""
    print("\n📈 Creating Quantum Visualizations")
    print("=" * 40)
    
    try:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'generated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualization 1: Quantum resource scaling
        sizes = range(2, 8)
        qubits = []
        depths = []
        volumes = []
        
        for n_vars in sizes:
            # Create representative function
            truth_table = [i % 2 for i in range(2**n_vars)]
            func = bf.create(truth_table)
            quantum_func = bf.create_quantum_boolean_function(func)
            
            resources = quantum_func.get_quantum_resources()
            qubits.append(resources['qubits_required'])
            depths.append(resources['circuit_depth'])
            volumes.append(resources['quantum_volume_required'])
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Qubits scaling
        axes[0].plot(sizes, qubits, 'bo-', linewidth=2)
        axes[0].set_xlabel('Number of Variables')
        axes[0].set_ylabel('Qubits Required')
        axes[0].set_title('Quantum Resource: Qubits')
        axes[0].grid(True, alpha=0.3)
        
        # Circuit depth scaling
        axes[1].plot(sizes, depths, 'ro-', linewidth=2)
        axes[1].set_xlabel('Number of Variables')
        axes[1].set_ylabel('Circuit Depth')
        axes[1].set_title('Quantum Resource: Circuit Depth')
        axes[1].grid(True, alpha=0.3)
        
        # Quantum volume scaling
        axes[2].semilogy(sizes, volumes, 'go-', linewidth=2)
        axes[2].set_xlabel('Number of Variables')
        axes[2].set_ylabel('Quantum Volume (log scale)')
        axes[2].set_title('Quantum Resource: Volume')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quantum_resource_scaling.png'), dpi=300)
        plt.close()
        
        print(f"  ✅ Quantum resource scaling saved to generated/quantum_resource_scaling.png")
        
        # Visualization 2: Quantum vs Classical comparison
        sizes = range(2, 10)
        classical_complexity = [2**n for n in sizes]
        quantum_fourier_complexity = [n * 2**n for n in sizes]
        quantum_search_complexity = [2**(n/2) for n in sizes]
        
        plt.figure(figsize=(12, 8))
        plt.semilogy(sizes, classical_complexity, 'b-', linewidth=2, label='Classical (2^n)')
        plt.semilogy(sizes, quantum_fourier_complexity, 'r--', linewidth=2, label='Quantum Fourier (n·2^n)')
        plt.semilogy(sizes, quantum_search_complexity, 'g:', linewidth=2, label='Quantum Search (2^(n/2))')
        
        plt.xlabel('Number of Variables')
        plt.ylabel('Computational Complexity (log scale)')
        plt.title('Quantum vs Classical Complexity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'quantum_vs_classical_complexity.png'), dpi=300)
        plt.close()
        
        print(f"  ✅ Complexity comparison saved to generated/quantum_vs_classical_complexity.png")
        
    except Exception as e:
        print(f"  ⚠️  Visualization requires matplotlib: {e}")


def demo_quantum_advantage_analysis():
    """Analyze when quantum algorithms provide advantages."""
    print("\n🎯 Quantum Advantage Analysis")
    print("=" * 40)
    
    # Analyze different types of functions
    function_types = {
        'Linear': [False, True, True, False],  # XOR
        'Sparse': [False, False, False, True, False, False, False, False],  # Single 1
        'Dense': [True, True, True, False, True, True, True, True],  # Mostly 1s
        'Symmetric': [False, True, True, True, True, True, True, False],  # Symmetric
        'Random': [bool(np.random.randint(2)) for _ in range(8)]  # Random
    }
    
    print("Quantum advantage analysis by function type:")
    print("Type      | Size | Fourier Adv | Search Adv | Property Adv | Overall")
    print("-" * 70)
    
    for func_type, truth_table in function_types.items():
        func = bf.create(truth_table)
        quantum_func = bf.create_quantum_boolean_function(func)
        
        # Estimate advantages for different analysis types
        fourier_adv = bf.estimate_quantum_advantage(func.n_vars, 'fourier')
        search_adv = bf.estimate_quantum_advantage(func.n_vars, 'search')
        property_adv = bf.estimate_quantum_advantage(func.n_vars, 'property_testing')
        
        fourier_worthwhile = "✅" if fourier_adv['worthwhile'] else "❌"
        search_worthwhile = "✅" if search_adv['worthwhile'] else "❌"
        property_worthwhile = "✅" if property_adv['worthwhile'] else "❌"
        
        # Overall recommendation
        comparison = quantum_func.quantum_algorithm_comparison()
        overall = "✅" if len(comparison['quantum_advantages']) > 0 else "❌"
        
        print(f"{func_type:9s} | {2**func.n_vars:4d} | {fourier_worthwhile:11s} | "
              f"{search_worthwhile:10s} | {property_worthwhile:12s} | {overall:7s}")


def main():
    """Run all quantum demos."""
    print("⚛️  BoolFunc Quantum Analysis Demo")
    print("=" * 50)
    print("This demo showcases quantum-specific features of the BoolFunc library.")
    print("Note: Full quantum features require Qiskit or other quantum libraries.\n")
    
    # Run all quantum demos
    demo_quantum_oracle_creation()
    demo_quantum_fourier_analysis()
    demo_quantum_property_testing()
    demo_quantum_resource_estimation()
    demo_quantum_vs_classical_comparison()
    demo_quantum_influence_estimation()
    demo_quantum_advantage_analysis()
    create_quantum_visualizations()
    
    print("\n🎉 Quantum demo completed!")
    print("\n💡 Next steps:")
    print("   - Install Qiskit for full quantum oracle support: pip install qiskit")
    print("   - Explore quantum algorithms for larger Boolean functions")
    print("   - Check generated/ directory for quantum analysis visualizations")


if __name__ == "__main__":
    main()
