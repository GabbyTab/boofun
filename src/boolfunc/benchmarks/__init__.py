"""
Performance benchmarking module for BoolFunc library.

This module provides comprehensive benchmarking tools to measure and compare
the performance of different Boolean function representations and algorithms.
"""

import time
import gc
import numpy as np
import psutil
from typing import Dict, List, Callable, Any, Optional, Tuple
from contextlib import contextmanager
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available - plotting disabled")

from ..core.base import BooleanFunction
from ..core.builtins import BooleanFunctionBuiltins
from ..analysis import SpectralAnalyzer


class PerformanceBenchmark:
    """
    Comprehensive benchmarking suite for Boolean function operations.
    
    Measures timing, memory usage, and scalability across different
    representations and algorithms.
    """
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 10):
        """
        Initialize benchmark suite.
        
        Args:
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations for averaging
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = {}
        
    @contextmanager
    def timer(self):
        """Context manager for high-precision timing."""
        gc.collect()  # Clean up before timing
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        yield
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        self.last_timing = {
            'duration': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'peak_memory': max(start_memory, end_memory)
        }
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """
        Benchmark a single function with multiple runs.
        
        Args:
            func: Function to benchmark
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Dictionary with timing statistics
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Benchmark runs
        durations = []
        memory_deltas = []
        
        for _ in range(self.benchmark_runs):
            with self.timer():
                result = func(*args, **kwargs)
            
            durations.append(self.last_timing['duration'])
            memory_deltas.append(self.last_timing['memory_delta'])
        
        return {
            'mean_time': np.mean(durations),
            'std_time': np.std(durations),
            'min_time': np.min(durations),
            'max_time': np.max(durations),
            'mean_memory': np.mean(memory_deltas),
            'std_memory': np.std(memory_deltas),
            'result': result if 'result' in locals() else None
        }
    
    def benchmark_creation(self, n_vars_range: List[int]) -> Dict[str, Any]:
        """
        Benchmark Boolean function creation across different sizes.
        
        Args:
            n_vars_range: List of variable counts to test
            
        Returns:
            Benchmark results for each method
        """
        results = {
            'truth_table': {'n_vars': [], 'times': [], 'memory': []},
            'majority': {'n_vars': [], 'times': [], 'memory': []},
            'parity': {'n_vars': [], 'times': [], 'memory': []},
        }
        
        for n_vars in n_vars_range:
            print(f"Benchmarking creation for {n_vars} variables...")
            
            # Truth table creation
            if n_vars <= 16:  # Avoid memory explosion
                truth_table = [False] * (2**n_vars)
                stats = self.benchmark_function(BooleanFunction, truth_table=truth_table, n=n_vars)
                results['truth_table']['n_vars'].append(n_vars)
                results['truth_table']['times'].append(stats['mean_time'])
                results['truth_table']['memory'].append(stats['mean_memory'])
            
            # Built-in functions
            stats = self.benchmark_function(BooleanFunctionBuiltins.majority, n_vars)
            results['majority']['n_vars'].append(n_vars)
            results['majority']['times'].append(stats['mean_time'])
            results['majority']['memory'].append(stats['mean_memory'])
            
            stats = self.benchmark_function(BooleanFunctionBuiltins.parity, n_vars)
            results['parity']['n_vars'].append(n_vars)
            results['parity']['times'].append(stats['mean_time'])
            results['parity']['memory'].append(stats['mean_memory'])
        
        return results
    
    def benchmark_evaluation(self, n_vars_range: List[int]) -> Dict[str, Any]:
        """
        Benchmark function evaluation performance.
        
        Args:
            n_vars_range: List of variable counts to test
            
        Returns:
            Evaluation benchmark results
        """
        results = {
            'single_eval': {'n_vars': [], 'times': []},
            'batch_eval': {'n_vars': [], 'times': []},
            'binary_eval': {'n_vars': [], 'times': []}
        }
        
        for n_vars in n_vars_range:
            if n_vars > 10:  # Skip large functions for evaluation benchmarks
                continue
                
            print(f"Benchmarking evaluation for {n_vars} variables...")
            
            # Create test function
            func = BooleanFunctionBuiltins.parity(n_vars)
            
            # Single evaluation
            test_input = np.array(0)
            stats = self.benchmark_function(func.evaluate, test_input)
            results['single_eval']['n_vars'].append(n_vars)
            results['single_eval']['times'].append(stats['mean_time'])
            
            # Batch evaluation
            batch_size = min(100, 2**n_vars)
            batch_inputs = np.random.randint(0, 2**n_vars, batch_size)
            stats = self.benchmark_function(func.evaluate, batch_inputs)
            results['batch_eval']['n_vars'].append(n_vars)
            results['batch_eval']['times'].append(stats['mean_time'])
            
            # Binary vector evaluation
            binary_input = np.random.randint(0, 2, n_vars)
            stats = self.benchmark_function(func.evaluate, binary_input)
            results['binary_eval']['n_vars'].append(n_vars)
            results['binary_eval']['times'].append(stats['mean_time'])
        
        return results
    
    def benchmark_spectral_analysis(self, n_vars_range: List[int]) -> Dict[str, Any]:
        """
        Benchmark spectral analysis algorithms.
        
        Args:
            n_vars_range: List of variable counts to test
            
        Returns:
            Spectral analysis benchmark results
        """
        results = {
            'influences': {'n_vars': [], 'times': []},
            'fourier': {'n_vars': [], 'times': []},
            'noise_stability': {'n_vars': [], 'times': []}
        }
        
        for n_vars in n_vars_range:
            if n_vars > 8:  # Spectral analysis is expensive
                continue
                
            print(f"Benchmarking spectral analysis for {n_vars} variables...")
            
            # Create test function
            func = BooleanFunctionBuiltins.majority(n_vars)
            analyzer = SpectralAnalyzer(func)
            
            # Influences computation
            stats = self.benchmark_function(analyzer.influences)
            results['influences']['n_vars'].append(n_vars)
            results['influences']['times'].append(stats['mean_time'])
            
            # Fourier expansion
            stats = self.benchmark_function(analyzer.fourier_expansion)
            results['fourier']['n_vars'].append(n_vars)
            results['fourier']['times'].append(stats['mean_time'])
            
            # Noise stability
            stats = self.benchmark_function(analyzer.noise_stability, 0.9)
            results['noise_stability']['n_vars'].append(n_vars)
            results['noise_stability']['times'].append(stats['mean_time'])
        
        return results
    
    def benchmark_representations(self, n_vars: int = 6) -> Dict[str, Any]:
        """
        Compare different representation formats.
        
        Args:
            n_vars: Number of variables for comparison
            
        Returns:
            Representation comparison results
        """
        print(f"Benchmarking representations for {n_vars} variables...")
        
        # Create test function
        base_func = BooleanFunctionBuiltins.majority(n_vars)
        
        results = {
            'creation': {},
            'evaluation': {},
            'memory': {}
        }
        
        # Test different representations
        representations = ['truth_table']  # Add more as they become available
        
        for rep_type in representations:
            # Creation time
            stats = self.benchmark_function(base_func.get_representation, rep_type)
            results['creation'][rep_type] = stats['mean_time']
            
            # Evaluation time
            test_input = np.array(0)
            stats = self.benchmark_function(base_func.evaluate, test_input, rep_type=rep_type)
            results['evaluation'][rep_type] = stats['mean_time']
            
            # Memory usage (approximate)
            import sys
            rep_data = base_func.get_representation(rep_type)
            results['memory'][rep_type] = sys.getsizeof(rep_data)
        
        return results
    
    def run_comprehensive_benchmark(self, max_vars: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            max_vars: Maximum number of variables to test
            
        Returns:
            Complete benchmark results
        """
        print("🚀 Running comprehensive BoolFunc benchmark suite...")
        print("=" * 60)
        
        n_vars_range = list(range(2, max_vars + 1))
        
        results = {
            'metadata': {
                'timestamp': time.time(),
                'max_vars': max_vars,
                'warmup_runs': self.warmup_runs,
                'benchmark_runs': self.benchmark_runs,
                'system_info': self._get_system_info()
            }
        }
        
        # Run individual benchmarks
        print("📊 Benchmarking function creation...")
        results['creation'] = self.benchmark_creation(n_vars_range)
        
        print("⚡ Benchmarking function evaluation...")
        results['evaluation'] = self.benchmark_evaluation(n_vars_range)
        
        print("🔬 Benchmarking spectral analysis...")
        results['spectral'] = self.benchmark_spectral_analysis(n_vars_range)
        
        print("🔄 Benchmarking representations...")
        results['representations'] = self.benchmark_representations()
        
        print("✅ Benchmark suite completed!")
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        import platform
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'numpy_version': np.__version__
        }
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted benchmark results."""
        print("\n📈 Benchmark Results Summary")
        print("=" * 50)
        
        # System info
        sys_info = results['metadata']['system_info']
        print(f"System: {sys_info['platform']}")
        print(f"CPU: {sys_info['processor']} ({sys_info['cpu_count']} cores)")
        print(f"Memory: {sys_info['memory_gb']:.1f} GB")
        print(f"Python: {sys_info['python_version']}")
        print()
        
        # Function creation results
        if 'creation' in results:
            print("🔨 Function Creation Performance:")
            creation = results['creation']
            for method in ['majority', 'parity']:
                if method in creation and creation[method]['times']:
                    avg_time = np.mean(creation[method]['times'])
                    print(f"  {method.capitalize()}: {avg_time*1000:.2f}ms average")
        
        # Evaluation results
        if 'evaluation' in results:
            print("\n⚡ Evaluation Performance:")
            evaluation = results['evaluation']
            for eval_type in ['single_eval', 'batch_eval']:
                if eval_type in evaluation and evaluation[eval_type]['times']:
                    avg_time = np.mean(evaluation[eval_type]['times'])
                    print(f"  {eval_type.replace('_', ' ').title()}: {avg_time*1000:.2f}ms average")
        
        # Spectral analysis results
        if 'spectral' in results:
            print("\n🔬 Spectral Analysis Performance:")
            spectral = results['spectral']
            for analysis_type in ['influences', 'fourier', 'noise_stability']:
                if analysis_type in spectral and spectral[analysis_type]['times']:
                    avg_time = np.mean(spectral[analysis_type]['times'])
                    print(f"  {analysis_type.replace('_', ' ').title()}: {avg_time*1000:.2f}ms average")
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot benchmark results if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            print("📊 Matplotlib not available - skipping plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('BoolFunc Performance Benchmarks', fontsize=16)
        
        # Creation performance
        if 'creation' in results:
            ax = axes[0, 0]
            creation = results['creation']
            for method in ['majority', 'parity']:
                if method in creation and creation[method]['n_vars']:
                    ax.plot(creation[method]['n_vars'], 
                           [t*1000 for t in creation[method]['times']], 
                           'o-', label=method.capitalize())
            ax.set_xlabel('Number of Variables')
            ax.set_ylabel('Creation Time (ms)')
            ax.set_title('Function Creation Performance')
            ax.legend()
            ax.grid(True)
        
        # Evaluation performance
        if 'evaluation' in results:
            ax = axes[0, 1]
            evaluation = results['evaluation']
            for eval_type in ['single_eval', 'batch_eval']:
                if eval_type in evaluation and evaluation[eval_type]['n_vars']:
                    ax.plot(evaluation[eval_type]['n_vars'], 
                           [t*1000 for t in evaluation[eval_type]['times']], 
                           'o-', label=eval_type.replace('_', ' ').title())
            ax.set_xlabel('Number of Variables')
            ax.set_ylabel('Evaluation Time (ms)')
            ax.set_title('Evaluation Performance')
            ax.legend()
            ax.grid(True)
        
        # Spectral analysis performance
        if 'spectral' in results:
            ax = axes[1, 0]
            spectral = results['spectral']
            for analysis_type in ['influences', 'fourier']:
                if analysis_type in spectral and spectral[analysis_type]['n_vars']:
                    ax.plot(spectral[analysis_type]['n_vars'], 
                           [t*1000 for t in spectral[analysis_type]['times']], 
                           'o-', label=analysis_type.capitalize())
            ax.set_xlabel('Number of Variables')
            ax.set_ylabel('Analysis Time (ms)')
            ax.set_title('Spectral Analysis Performance')
            ax.legend()
            ax.grid(True)
        
        # Memory usage (if available)
        if 'creation' in results:
            ax = axes[1, 1]
            creation = results['creation']
            for method in ['majority', 'parity']:
                if method in creation and creation[method]['n_vars']:
                    memory_mb = [m / (1024*1024) for m in creation[method]['memory']]
                    ax.plot(creation[method]['n_vars'], memory_mb, 
                           'o-', label=method.capitalize())
            ax.set_xlabel('Number of Variables')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to {save_path}")
        else:
            plt.show()


def run_quick_benchmark() -> None:
    """Run a quick benchmark for development testing."""
    benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=3)
    results = benchmark.run_comprehensive_benchmark(max_vars=6)
    benchmark.print_results(results)
    return results


if __name__ == "__main__":
    run_quick_benchmark()
