"""High-yield behavioral tests for core dispatch and fallback workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from boofun.core import batch_processing, gpu
from boofun.core.batch_processing import (
    BatchProcessorManager,
    OptimizedANFProcessor,
    OptimizedTruthTableProcessor,
    ParallelBatchProcessor,
    VectorizedBatchProcessor,
)
from boofun.core.conversion_graph import (
    ConversionCost,
    ConversionEdge,
    ConversionGraph,
    ConversionPath,
    set_large_n_policy,
)
from boofun.core.spaces import Space


def test_vectorized_chunking_and_invalid_lookup() -> None:
    processor = VectorizedBatchProcessor(chunk_size=2)
    truth_table = np.array([False, True, True, False])

    result = processor.process_batch(np.array([0, 1, 2, 3, 99]), truth_table, Space.BOOLEAN_CUBE, 2)

    assert np.array_equal(result, [False, True, True, False, False])


def test_parallel_worker_failure_falls_back_to_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = ParallelBatchProcessor(n_workers=2, use_processes=False)
    truth_table = np.array([False, True, True, False])
    inputs = np.tile(np.arange(4), 6)

    def fail_chunk(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("worker failed")

    monkeypatch.setattr(processor, "_process_chunk", fail_chunk)
    with pytest.warns(UserWarning, match="Parallel processing failed"):
        result = processor.process_batch(inputs, truth_table, Space.BOOLEAN_CUBE, 2)

    assert np.array_equal(result, truth_table[inputs])


def test_truth_table_gpu_failure_uses_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = OptimizedTruthTableProcessor()
    truth_table = np.array([False, True, True, False])

    monkeypatch.setattr(batch_processing, "should_use_gpu", lambda *args: True)

    def fail_gpu(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("gpu failed")

    monkeypatch.setattr(batch_processing, "gpu_accelerate", fail_gpu)
    with pytest.warns(UserWarning, match="GPU acceleration failed"):
        result = processor.process_batch(np.arange(4), truth_table, Space.BOOLEAN_CUBE, n_vars=2)

    assert np.array_equal(result, truth_table)


def test_batch_manager_safe_strategy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = BatchProcessorManager()

    def fail_batch(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("processor failed")

    monkeypatch.setattr(manager.processors["truth_table"], "process_batch", fail_batch)

    class Strategy:
        def evaluate(self, value: int, *args: Any) -> bool:
            if int(value) == 2:
                raise ValueError("bad input")
            return bool(int(value) % 2)

    from boofun.core.representations import registry

    monkeypatch.setattr(registry, "get_strategy", lambda representation: Strategy())
    with pytest.warns(UserWarning, match="Batch processing failed"):
        result = manager.process_batch(
            np.arange(4), np.zeros(4), "truth_table", Space.BOOLEAN_CUBE, 2
        )

    assert np.array_equal(result, [False, True, False, True])


def test_anf_integer_and_vector_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = OptimizedANFProcessor()
    monkeypatch.setattr(batch_processing, "HAS_NUMBA", False)
    anf = {frozenset({0}): 1, frozenset({1}): 1}

    integers = processor.process_batch(np.arange(4), anf, Space.BOOLEAN_CUBE, 2)
    vectors = processor.process_batch(
        np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), anf, Space.BOOLEAN_CUBE, 2
    )

    assert np.array_equal(integers, [False, True, True, False])
    assert np.array_equal(vectors, [False, True, True, False])


def test_conversion_path_large_n_policies_and_converter() -> None:
    calls: list[tuple[Any, Space, int]] = []

    def converter(data: Any, space: Space, n_vars: int) -> list[Any]:
        calls.append((data, space, n_vars))
        return [data, "converted"]

    path = ConversionPath(
        [
            ConversionEdge(
                "truth_table",
                "custom",
                ConversionCost(1, 1),
                converter,
            )
        ]
    )

    try:
        set_large_n_policy("warn", threshold=2)
        with pytest.warns(UserWarning, match="materialise"):
            assert path.execute("data", Space.BOOLEAN_CUBE, 3) == ["data", "converted"]

        set_large_n_policy("raise", threshold=2)
        with pytest.raises(ValueError, match="materialise"):
            path.execute("data", Space.BOOLEAN_CUBE, 3)

        set_large_n_policy("off")
        assert path.execute("again", Space.BOOLEAN_CUBE, 3) == ["again", "converted"]
    finally:
        set_large_n_policy("warn", threshold=25)

    assert len(calls) == 2
    with pytest.raises(ValueError, match="policy must be"):
        set_large_n_policy("invalid")


def test_conversion_graph_cache_stats_and_rendering() -> None:
    graph = ConversionGraph()
    first = graph.find_optimal_path("anf", "fourier_expansion")
    assert first is not None
    assert len(first) == 2
    assert graph.find_optimal_path("anf", "anf") is None
    assert graph.find_optimal_path("missing", "also_missing") is None

    graph.add_edge("anf", "fourier_expansion", ConversionCost(1, 1))
    direct = graph.find_optimal_path("anf", "fourier_expansion")
    assert direct is not None
    assert len(direct) == 1

    stats = graph.get_graph_stats()
    assert stats["num_nodes"] > 1
    assert "Conversion Graph" in graph.visualize_graph()
    assert graph.visualize_graph("dot").startswith("digraph")
    assert graph.get_conversion_options("anf", max_cost=10)
    graph.clear_cache()
    assert graph.path_cache == {}


def test_gpu_heuristics_dispatch_and_auto_acceleration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu, "is_gpu_enabled", lambda: True)
    assert gpu.should_use_gpu("truth_table", 10_001, 2)
    assert gpu.should_use_gpu("fourier", 1, 10)
    assert gpu.should_use_gpu("wht", 1, 11)
    assert gpu.should_use_gpu("custom", 10_001, 1)
    assert not gpu.should_use_gpu("truth_table", 10_000, 2)

    transferred: list[str] = []
    monkeypatch.setattr(gpu, "to_gpu", lambda array: transferred.append("gpu") or array)
    monkeypatch.setattr(gpu, "to_cpu", lambda array: transferred.append("cpu") or array)

    @gpu.auto_accelerate
    def double(array: np.ndarray) -> np.ndarray:
        return array * 2

    result = double(np.ones(2**14))
    assert np.array_equal(result, np.full(2**14, 2.0))
    assert transferred == ["gpu", "cpu"]

    monkeypatch.setattr(gpu, "is_gpu_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        gpu.gpu_accelerate("truth_table_batch", np.arange(2), np.arange(2))
