#!/usr/bin/env python3
"""
BoolFunc Visualization Examples

This file demonstrates the visualization capabilities of the BoolFunc library,
including influence plots, Fourier spectrum analysis, and interactive dashboards.

Requirements: pip install -e ".[visualization]"
"""

import numpy as np
import boolfunc as bf

def basic_visualization():
    """Demonstrate basic visualization functionality."""
    print("=== Basic Visualization Examples ===")
    
    # Create some interesting Boolean functions
    xor = bf.create([0, 1, 1, 0])
    majority = bf.BooleanFunctionBuiltins.majority(3)
    parity = bf.BooleanFunctionBuiltins.parity(3)
    
    print("1. Creating visualizations for different functions:")
    print(f"   - XOR function ({xor.n_vars} variables)")
    print(f"   - Majority function ({majority.n_vars} variables)")
    print(f"   - Parity function ({parity.n_vars} variables)")
    
    try:
        from boolfunc.visualization import BooleanFunctionVisualizer
        import os
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'generated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizers
        xor_viz = BooleanFunctionVisualizer(xor, backend="matplotlib")
        maj_viz = BooleanFunctionVisualizer(majority, backend="matplotlib")
        
        print("\n2. Generating influence plots...")
        # Note: In real usage, remove show=False to display plots
        xor_inf_path = os.path.join(output_dir, "xor_influences.png")
        maj_inf_path = os.path.join(output_dir, "majority_influences.png")
        xor_viz.plot_influences(save_path=xor_inf_path, show=False)
        maj_viz.plot_influences(save_path=maj_inf_path, show=False)
        print("   ✓ Influence plots saved to PNG files")
        
        print("\n3. Generating Fourier spectrum plots...")
        xor_fourier_path = os.path.join(output_dir, "xor_fourier.png")
        maj_fourier_path = os.path.join(output_dir, "majority_fourier.png")
        xor_viz.plot_fourier_spectrum(save_path=xor_fourier_path, show=False)
        maj_viz.plot_fourier_spectrum(save_path=maj_fourier_path, show=False)
        print("   ✓ Fourier spectrum plots saved")
        
        print("\n4. Creating comprehensive dashboards...")
        xor_dash_path = os.path.join(output_dir, "xor_dashboard.png")
        maj_dash_path = os.path.join(output_dir, "majority_dashboard.png")
        xor_viz.create_dashboard(save_path=xor_dash_path, show=False)
        maj_viz.create_dashboard(save_path=maj_dash_path, show=False)
        print("   ✓ Analysis dashboards created")
        
        print(f"   ✓ All files saved to: {output_dir}/")
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Visualization not available: {e}")
        print("   Install with: pip install -e \".[visualization]\"")
        return False

def interactive_visualization():
    """Demonstrate interactive Plotly visualizations."""
    print("\n=== Interactive Visualization (Plotly) ===")
    
    try:
        from boolfunc.visualization import BooleanFunctionVisualizer
        import os
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'generated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create function for analysis
        parity = bf.BooleanFunctionBuiltins.parity(3)
        
        # Create Plotly visualizer
        viz = BooleanFunctionVisualizer(parity, backend="plotly")
        
        print("1. Creating interactive Plotly visualizations...")
        
        # Generate interactive plots (saved as HTML in generated/ directory)
        influences_path = os.path.join(output_dir, "parity_influences_interactive.html")
        fourier_path = os.path.join(output_dir, "parity_fourier_interactive.html")
        noise_path = os.path.join(output_dir, "parity_noise_stability.html")
        dashboard_path = os.path.join(output_dir, "parity_dashboard_interactive.html")
        
        viz.plot_influences(save_path=influences_path, show=False)
        viz.plot_fourier_spectrum(save_path=fourier_path, show=False)
        viz.plot_noise_stability_curve(save_path=noise_path, show=False)
        
        print("   ✓ Interactive HTML plots created")
        print(f"   ✓ Files saved to: {output_dir}/")
        print("   ✓ Open the .html files in a web browser for interactive exploration")
        
        # Create interactive dashboard
        viz.create_dashboard(save_path=dashboard_path, show=False)
        print("   ✓ Interactive dashboard created")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Interactive visualization not available: {e}")
        return False

def function_comparison():
    """Demonstrate function comparison visualizations."""
    print("\n=== Function Comparison Visualization ===")
    
    try:
        from boolfunc.visualization import plot_function_comparison
        
        # Create functions to compare (same number of variables)
        functions = {
            "XOR": bf.create([0, 1, 1, 0]),
            "AND": bf.create([0, 0, 0, 1]),
            "OR": bf.create([0, 1, 1, 1]),
            "NAND": bf.create([1, 1, 1, 0])
        }
        
        print("1. Comparing influences across different 2-variable functions:")
        for name, func in functions.items():
            analyzer = bf.SpectralAnalyzer(func)
            influences = analyzer.influences()
            total_inf = analyzer.total_influence()
            print(f"   {name:4}: influences={influences}, total={total_inf:.1f}")
        
        print("\n2. Creating comparison plot...")
        # Note: This creates a matplotlib plot
        fig = plot_function_comparison(
            functions, 
            metric="influences",
            figsize=(10, 6)
        )
        print("   ✓ Function comparison plot displayed")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Function comparison not available: {e}")
        return False

def advanced_visualization():
    """Demonstrate advanced visualization features."""
    print("\n=== Advanced Visualization Features ===")
    
    try:
        from boolfunc.visualization import BooleanFunctionVisualizer
        import os
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), 'generated')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create more complex function for analysis
        # Tribes function (if available) or majority
        try:
            func = bf.BooleanFunctionBuiltins.tribes(2, 4)
            func_name = "Tribes(2,4)"
        except:
            func = bf.BooleanFunctionBuiltins.majority(3)
            func_name = "Majority(3)"
        
        viz = BooleanFunctionVisualizer(func, backend="matplotlib")
        
        print(f"1. Advanced analysis of {func_name}:")
        
        # Analyze spectral properties
        analyzer = bf.SpectralAnalyzer(func)
        influences = analyzer.influences()
        total_influence = analyzer.total_influence()
        
        print(f"   Variables: {func.n_vars}")
        print(f"   Influences: {influences}")
        print(f"   Total influence: {total_influence:.3f}")
        
        # Create noise stability curve
        print("\n2. Generating noise stability analysis...")
        rho_range = np.linspace(-1, 1, 21)
        noise_path = os.path.join(output_dir, f"{func_name.lower()}_noise_stability.png")
        viz.plot_noise_stability_curve(
            rho_range=rho_range,
            save_path=noise_path,
            show=False
        )
        print("   ✓ Noise stability curve generated")
        
        # Create truth table visualization (for small functions)
        if func.n_vars <= 4:
            print("\n3. Creating truth table heatmap...")
            truth_table_path = os.path.join(output_dir, f"{func_name.lower()}_truth_table.png")
            viz.plot_truth_table(save_path=truth_table_path, show=False)
            print("   ✓ Truth table heatmap created")
        
        print(f"   ✓ Files saved to: {output_dir}/")
        return True
        
    except Exception as e:
        print(f"   ⚠️  Advanced visualization failed: {e}")
        return False

def main():
    """Run all visualization examples."""
    import os
    
    print("BoolFunc Library - Visualization Examples")
    print("=" * 45)
    print("This demonstrates the visualization capabilities of the library.")
    print()
    
    success_count = 0
    
    if basic_visualization():
        success_count += 1
    
    if interactive_visualization():
        success_count += 1
        
    if function_comparison():
        success_count += 1
        
    if advanced_visualization():
        success_count += 1
    
    print(f"\n✅ Completed {success_count}/4 visualization example categories")
    
    if success_count >= 2:
        print("\n🎨 Visualization Examples Summary:")
        print("  - Static plots (matplotlib): PNG files for publications")
        print("  - Interactive plots (plotly): HTML files for exploration")
        print("  - Function comparisons: Side-by-side analysis")
        print("  - Comprehensive dashboards: Multi-panel analysis")
        print("\n📁 Generated files location:")
        print("  - All output files saved to: examples/generated/")
        print("  - PNG files: Static plots for publications")
        print("  - HTML files: Interactive plots for web browsers")
        print("\nTo view the generated plots:")
        print("  - Open .png files in any image viewer")
        print("  - Open .html files in a web browser for interactivity")
        print(f"  - Files are in: {os.path.join(os.path.dirname(__file__), 'generated')}/")
    else:
        print("\n📝 To enable visualization features:")
        print("  pip install -e \".[visualization]\"")
        print("  This will install matplotlib and plotly dependencies")

if __name__ == "__main__":
    main()
