"""Behavioral coverage for query complexity and PAC helper APIs."""

from __future__ import annotations

import numpy as np
import pytest

import boofun as bf
from boofun.analysis.certificates import min_certificate_size
from boofun.analysis.pac_learning import PACLearner
from boofun.analysis.query_complexity import (
    QueryComplexityProfile,
    average_deterministic_complexity,
    average_everywhere_sensitivity,
    bounded_error_randomized_complexity,
    nondeterministic_complexity,
    one_sided_randomized_complexity,
    zero_error_randomized_complexity,
)


def test_minimum_certificate_sizes_by_output() -> None:
    assert min_certificate_size(bf.AND(3), 0) == 1
    assert min_certificate_size(bf.AND(3), 1) == 3
    assert min_certificate_size(bf.OR(3), 0) == 3
    assert min_certificate_size(bf.OR(3), 1) == 1
    assert min_certificate_size(bf.parity(3), 0) == 3
    assert min_certificate_size(bf.parity(3), 1) == 3


def test_query_complexity_scalar_measures() -> None:
    function = bf.AND(2)

    assert average_deterministic_complexity(function) == pytest.approx(1.25)
    assert nondeterministic_complexity(function, 1) == 2
    assert nondeterministic_complexity(function, 0) == 1
    assert average_everywhere_sensitivity(function, 1) == pytest.approx(2.0)
    assert average_everywhere_sensitivity(function, 0) == pytest.approx(2 / 3)

    deterministic = function.n_vars
    assert deterministic is not None
    assert 0 <= bounded_error_randomized_complexity(function) <= deterministic
    assert 0 <= one_sided_randomized_complexity(function) <= deterministic
    assert 0 <= zero_error_randomized_complexity(function) <= deterministic


def test_query_complexity_profile_caches_and_summarizes() -> None:
    profile = QueryComplexityProfile(bf.parity(2))

    measures = profile.compute()
    assert profile.compute() is measures
    assert measures["D"] == 2
    assert measures["C"] == 2
    assert measures["deg"] == 2
    assert "Query Complexity Profile" in profile.summary()
    assert all(profile.check_known_relations().values())


def test_pac_learner_sampling_hypothesis_and_adaptation() -> None:
    function = bf.dictator(3, 1)
    learner = PACLearner(
        function,
        epsilon=0.2,
        delta=0.1,
        rng=np.random.default_rng(0),
    )

    samples = learner.sample(3)
    assert len(samples) == 3
    assert learner.sample_count == 3
    assert "ε=0.2" in learner.summary()

    coefficients = {2: 1.0}
    for value in range(8):
        assert learner.evaluate_hypothesis(coefficients, value) == function.evaluate(value)
    assert learner.test_accuracy(coefficients, num_tests=20) == 1.0

    adaptive = learner.learn_adaptive()
    assert adaptive["algorithm"] == "junta"
