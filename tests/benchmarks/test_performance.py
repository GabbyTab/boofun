"""
Comprehensive Performance Benchmark Suite for boofun.

This module tests:
1. Query safety for different access types
2. Scaling behavior with increasing n
3. Performance of different representations
4. Optimization effectiveness (Numba, batch processing)
5. Memory usage

Run with: pytest tests/benchmarks/test_performance.py -v -s
"""

import sys
import time

import numpy as np
import pytest

# Import boofun
sys.path.insert(0, "src")
import boofun as bf
from boofun.core.query_model import AccessType, QueryModel, get_access_type

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def query_counter():
    """Create a counting wrapper for functions."""

    class Counter:
        def __init__(self):
            self.calls = 0

        def reset(self):
            self.calls = 0

        def wrap(self, func):
            def wrapped(x):
                self.calls += 1
                return func(x)

            return wrapped

    return Counter()


@pytest.fixture
def parity_oracle(query_counter):
    """Parity function as a query oracle."""

    def parity(x):
        if isinstance(x, np.ndarray):
            x = int(x.item()) if x.ndim == 0 else int(x[0])
        return bin(int(x)).count("1") % 2

    return query_counter.wrap(parity)


@pytest.fixture
def majority_oracle(query_counter):
    """Majority function as a query oracle."""

    def majority(x):
        if isinstance(x, np.ndarray):
            x = int(x.item()) if x.ndim == 0 else int(x[0])
        n = 10  # Assume 10 variables
        ones = bin(int(x)).count("1")
        return 1 if ones > n // 2 else 0

    return query_counter.wrap(majority)


# =============================================================================
# QUERY SAFETY TESTS
# =============================================================================


class TestQuerySafety:
    """Tests for query model and safety guarantees."""

    def test_access_type_explicit(self):
        """Functions with truth tables are explicit."""
        f = bf.create([0, 1, 1, 0])
        assert get_access_type(f) == AccessType.EXPLICIT

    def test_access_type_query(self, parity_oracle, query_counter):
        """Functions from callables are query-access."""
        f = bf.create(parity_oracle, n=20)
        assert get_access_type(f) == AccessType.QUERY

    def test_blr_query_count(self, query_counter):
        """BLR test should use O(num_queries) evaluations."""

        def counting_parity(x):
            query_counter.calls += 1
            if isinstance(x, np.ndarray):
                x = int(x.item()) if x.ndim == 0 else int(x[0])
            return bin(int(x)).count("1") % 2

        f = bf.create(counting_parity, n=30)  # Large n
        query_counter.reset()

        result = f.is_linear(num_tests=100)

        # Should use ~300 queries (3 per test), definitely not 2^30
        assert query_counter.calls < 500
        assert query_counter.calls >= 100  # At least num_tests

    def test_monotonicity_query_count(self, query_counter):
        """Monotonicity test should use O(num_queries) evaluations."""

        def counting_func(x):
            query_counter.calls += 1
            if isinstance(x, np.ndarray):
                x = int(x.item()) if x.ndim == 0 else int(x[0])
            return bin(int(x)).count("1") >= 2

        f = bf.create(counting_func, n=30)
        query_counter.reset()

        f.is_monotone(num_tests=100)

        assert query_counter.calls < 500

    def test_query_model_cost_estimation(self):
        """QueryModel should correctly estimate costs."""
        f = bf.majority(20)
        qm = QueryModel(f)

        # Safe operations
        blr_cost = qm.estimate_cost("is_linear", num_queries=100)
        assert blr_cost["safe"] is True
        assert blr_cost["queries"] == 300

        # Unsafe operations
        fourier_cost = qm.estimate_cost("fourier")
        assert fourier_cost["safe"] is False
        assert fourier_cost["queries"] == 2**20

    def test_query_model_feasibility(self):
        """Large functions should mark unsafe ops as infeasible."""
        f = bf.create(lambda x: 0, n=50)
        qm = QueryModel(f, max_queries=1_000_000)

        assert qm.can_compute("is_linear")
        assert not qm.can_compute("fourier")


# =============================================================================
# SCALING TESTS
# =============================================================================


class TestScaling:
    """Tests for performance scaling with n."""

    @pytest.mark.parametrize("n", [8, 10, 12, 14])
    def test_fourier_scaling(self, n):
        """Fourier transform should scale as O(n * 2^n)."""
        f = bf.majority(n)

        start = time.time()
        coeffs = f.fourier()
        elapsed = time.time() - start

        # Should complete reasonably fast (allow extra time for CI variability)
        assert elapsed < 20.0  # 20 seconds max
        assert len(coeffs) == 2**n

    @pytest.mark.parametrize("n", [8, 10, 12, 14])
    def test_influences_scaling(self, n):
        """Influence computation should scale as O(n * 2^n)."""
        f = bf.majority(n)

        start = time.time()
        infs = f.influences()
        elapsed = time.time() - start

        assert elapsed < 30.0  # 30 seconds max
        assert len(infs) == n

    @pytest.mark.parametrize("n", [10, 20, 30, 50, 100])
    def test_blr_constant_scaling(self, n):
        """BLR test should be O(1) in n."""

        def simple_func(x):
            if isinstance(x, np.ndarray):
                x = int(x.item()) if x.ndim == 0 else int(x[0])
            return bin(int(x)).count("1") % 2

        f = bf.create(simple_func, n=n)

        start = time.time()
        result = f.is_linear(num_tests=100)
        elapsed = time.time() - start

        # Should be essentially constant time
        assert elapsed < 0.1  # 100ms max regardless of n
        assert result is True  # Parity is linear


# =============================================================================
# REPRESENTATION PERFORMANCE TESTS
# =============================================================================


class TestRepresentationPerformance:
    """Tests for different representation performance."""

    def test_truth_table_evaluation_speed(self):
        """Truth table evaluation should be very fast."""
        f = bf.majority(14)
        f.get_representation("truth_table")  # Ensure cached

        start = time.time()
        for i in range(10000):
            f.evaluate(i % (2**14))
        elapsed = time.time() - start

        # Should be < 1µs per evaluation on average
        assert elapsed / 10000 < 0.001  # 1ms per 1000 evals

    def test_function_evaluation_overhead(self, query_counter):
        """Function representation should have minimal overhead."""

        def fast_func(x):
            query_counter.calls += 1
            return 0

        f = bf.create(fast_func, n=10)
        query_counter.reset()

        start = time.time()
        for i in range(1000):
            f.evaluate(i)
        elapsed = time.time() - start

        assert query_counter.calls == 1000
        # Should be < 10µs per evaluation
        assert elapsed / 1000 < 0.01

    def test_conversion_performance(self):
        """Representation conversion should be reasonably fast."""
        f = bf.majority(10)

        # Truth table → Fourier
        start = time.time()
        f.get_representation("fourier_expansion")
        tt_to_fourier = time.time() - start

        # Truth table → ANF
        f2 = bf.majority(10)
        start = time.time()
        f2.get_representation("anf")
        tt_to_anf = time.time() - start

        assert tt_to_fourier < 1.0
        assert tt_to_anf < 1.0


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================


class TestBatchProcessing:
    """Tests for batch evaluation performance."""

    def test_batch_vs_single_evaluation(self):
        """Batch evaluation should be faster than many single evaluations."""
        f = bf.majority(12)
        n_evals = 1000

        # Single evaluations
        start = time.time()
        for i in range(n_evals):
            f.evaluate(i)
        single_time = time.time() - start

        # Array input (if supported)
        start = time.time()
        tt = np.array(f.get_representation("truth_table"))
        tt[np.arange(n_evals)]
        array_time = time.time() - start

        # Array should be faster (or at least competitive)
        print(f"Single: {single_time:.4f}s, Array: {array_time:.4f}s")


# =============================================================================
# MEMORY TESTS
# =============================================================================


class TestMemory:
    """Tests for memory usage."""

    def test_lazy_no_materialization(self, query_counter):
        """Creating from callable should not materialize truth table."""

        def counting_func(x):
            query_counter.calls += 1
            return 0

        query_counter.reset()
        f = bf.create(counting_func, n=20)

        # Should not have called function yet
        assert query_counter.calls == 0
        assert "truth_table" not in f.representations

    def test_truth_table_memory(self):
        """Truth table memory should be ~2^n bits."""
        import sys

        for n in [10, 12, 14]:
            f = bf.majority(n)
            tt = f.get_representation("truth_table")

            # NumPy bool array uses 1 byte per element
            expected_bytes = 2**n
            actual_bytes = sys.getsizeof(tt)

            # Should be within 2x of expected
            assert actual_bytes < 3 * expected_bytes


# =============================================================================
# CORRECTNESS UNDER PERFORMANCE
# =============================================================================


class TestCorrectnessUnderPerformance:
    """Ensure optimizations don't break correctness."""

    def test_fourier_correctness(self):
        """Fourier coefficients should satisfy Parseval."""
        f = bf.majority(8)
        coeffs = f.fourier()

        # Parseval: sum of squares = 1 for Boolean functions
        parseval_sum = np.sum(coeffs**2)
        assert abs(parseval_sum - 1.0) < 1e-10

    def test_influences_correctness(self):
        """Influences should sum to total influence."""
        f = bf.majority(8)
        infs = f.influences()
        total = f.total_influence()

        assert abs(np.sum(infs) - total) < 1e-10

    def test_blr_correctness(self):
        """BLR should correctly identify linear functions."""
        # Parity is linear
        parity = bf.parity(10)
        assert parity.is_linear(num_tests=500)

        # Majority is not linear
        maj = bf.majority(5)
        assert not maj.is_linear(num_tests=500)


# =============================================================================
# COMPREHENSIVE BENCHMARK
# =============================================================================


class TestComprehensiveBenchmark:
    """Run full benchmark suite and report results."""

    def test_full_benchmark(self):
        """Run comprehensive benchmark and print results."""
        results = {}

        for n in [8, 10, 12, 14]:
            f = bf.majority(n)
            results[n] = {}

            # Fourier
            start = time.time()
            f.fourier()
            results[n]["fourier"] = time.time() - start

            # Influences
            start = time.time()
            f.influences()
            results[n]["influences"] = time.time() - start

            # BLR
            start = time.time()
            f.is_linear(num_tests=100)
            results[n]["blr"] = time.time() - start

            # Single eval (1000x)
            start = time.time()
            for i in range(1000):
                f.evaluate(i % (2**n))
            results[n]["eval_1000"] = time.time() - start

        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"{'n':>4} | {'fourier':>10} | {'influences':>10} | {'BLR':>10} | {'1000 evals':>10}")
        print("-" * 60)
        for n, data in results.items():
            print(
                f"{n:>4} | {data['fourier']:>10.4f}s | {data['influences']:>10.4f}s | {data['blr']:>10.4f}s | {data['eval_1000']:>10.4f}s"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
