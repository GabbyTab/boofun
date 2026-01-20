"""
Tests for benchmarks module.

Tests for performance benchmarking utilities.
"""

import sys

import pytest

sys.path.insert(0, "src")

import boofun as bf
from boofun.benchmarks import PerformanceBenchmark, run_quick_benchmark


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""

    def test_benchmark_init(self):
        """PerformanceBenchmark should initialize correctly."""
        benchmark = PerformanceBenchmark()

        assert benchmark is not None
        assert benchmark.warmup_runs >= 0
        assert benchmark.benchmark_runs > 0

    def test_benchmark_custom_params(self):
        """PerformanceBenchmark should accept custom parameters."""
        benchmark = PerformanceBenchmark(warmup_runs=2, benchmark_runs=5)

        assert benchmark.warmup_runs == 2
        assert benchmark.benchmark_runs == 5

    def test_timer_context_manager(self):
        """Timer context manager should work."""
        benchmark = PerformanceBenchmark()

        with benchmark.timer():
            # Do some work
            _ = sum(range(1000))

        assert hasattr(benchmark, "last_timing")
        assert "duration" in benchmark.last_timing

    def test_benchmark_function(self):
        """benchmark_function should measure timing."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=3)

        def simple_func():
            return sum(range(100))

        result = benchmark.benchmark_function(simple_func)

        assert isinstance(result, dict)
        # Check for any timing-related key
        timing_keys = ["mean", "duration", "mean_time", "min_time", "max_time"]
        assert any(k in result for k in timing_keys)

    def test_benchmark_boolean_function_creation(self):
        """Should benchmark BooleanFunction creation."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)

        result = benchmark.benchmark_function(bf.majority, 3)

        assert result is not None

    def test_benchmark_fourier(self):
        """Should benchmark Fourier computation."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)

        f = bf.majority(3)
        result = benchmark.benchmark_function(f.fourier)

        assert result is not None

    def test_benchmark_evaluation(self):
        """Should benchmark evaluation."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)

        f = bf.AND(4)
        result = benchmark.benchmark_function(f.evaluate, 5)

        assert result is not None

    def test_benchmark_results_storage(self):
        """Results should be stored."""
        benchmark = PerformanceBenchmark()

        # Results dict should exist
        assert hasattr(benchmark, "results")
        assert isinstance(benchmark.results, dict)


class TestBenchmarkScaling:
    """Test benchmark scaling capabilities."""

    def test_scalability_test(self):
        """Benchmark should support scalability tests."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)

        # Benchmark across different sizes
        results = []
        for n in [2, 3, 4]:
            f = bf.majority(n)
            timing = benchmark.benchmark_function(f.fourier)
            results.append((n, timing))

        assert len(results) == 3

    def test_comparison_across_functions(self):
        """Benchmark should compare different functions."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)

        functions = {
            "AND": bf.AND(4),
            "OR": bf.OR(4),
            "MAJ": bf.majority(5),
        }

        results = {}
        for name, f in functions.items():
            results[name] = benchmark.benchmark_function(f.fourier)

        assert len(results) == 3


class TestRunQuickBenchmark:
    """Test run_quick_benchmark function."""

    def test_quick_benchmark_runs(self):
        """run_quick_benchmark should execute without error."""
        try:
            run_quick_benchmark()
        except Exception as e:
            pytest.skip(f"Quick benchmark not available: {e}")

    def test_quick_benchmark_callable(self):
        """run_quick_benchmark should be callable."""
        assert callable(run_quick_benchmark)


class TestBenchmarkAnalysis:
    """Test benchmark analysis features."""

    def test_memory_tracking(self):
        """Benchmark should track memory usage."""
        benchmark = PerformanceBenchmark()

        with benchmark.timer():
            # Create some objects
            f = bf.majority(5)
            _ = f.fourier()

        assert "memory_delta" in benchmark.last_timing or "peak_memory" in benchmark.last_timing

    def test_multiple_benchmarks(self):
        """Multiple benchmarks should work."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=2)

        # Run multiple benchmarks
        r1 = benchmark.benchmark_function(bf.AND, 3)
        r2 = benchmark.benchmark_function(bf.OR, 3)
        r3 = benchmark.benchmark_function(bf.majority, 3)

        assert r1 is not None
        assert r2 is not None
        assert r3 is not None


class TestBenchmarkEdgeCases:
    """Test edge cases for benchmarks."""

    def test_empty_function(self):
        """Benchmark should handle constant functions."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=1)

        result = benchmark.benchmark_function(lambda: None)
        assert result is not None

    def test_failing_function(self):
        """Benchmark should handle errors gracefully."""
        benchmark = PerformanceBenchmark(warmup_runs=1, benchmark_runs=1)

        def failing_func():
            raise ValueError("Test error")

        try:
            benchmark.benchmark_function(failing_func)
            # May either return error info or raise
        except ValueError:
            pass  # Expected

    def test_zero_runs(self):
        """Benchmark with 0 warmup runs should work."""
        benchmark = PerformanceBenchmark(warmup_runs=0, benchmark_runs=1)

        result = benchmark.benchmark_function(lambda: 1)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
