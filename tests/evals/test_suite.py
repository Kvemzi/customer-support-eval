"""
LLM-as-a-judge test suite for the ShopCo support agent.

Run with the DeepEval CLI (recommended; produces the DeepEval run report):

    deepeval test run tests/evals/test_suite.py

or with plain pytest (same assertions, no DeepEval run report):

    pytest tests/evals/test_suite.py -v

Dataset mode is controlled by EVAL_DATASET_MODE (static | live). Every declared
case/metric pair is collected automatically from the dataset specifications.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest
from deepeval import assert_test

from agent_eval.metrics import build_metric
from agent_eval.test_dataset import (
    ALL_CASE_SPECS,
    get_all_test_cases,
    get_metadata,
)

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY is not set; the LLM judge cannot run.",
    ),
]


@pytest.fixture(scope="module")
def cases() -> Dict[str, object]:
    """Build each case once so all metric checks judge the same agent output."""
    return {get_metadata(case)["id"]: case for case in get_all_test_cases()}


_EVALUATIONS = [
    (spec.id, metric, spec.expected_verdict)
    for spec in ALL_CASE_SPECS
    for metric in spec.target_metrics
]


@pytest.mark.parametrize(
    ("case_id", "metric_key", "expected_verdict"),
    _EVALUATIONS,
    ids=[f"{case_id}-{metric}" for case_id, metric, _ in _EVALUATIONS],
)
def test_case_metric_matches_expected_verdict(cases, case_id, metric_key, expected_verdict):
    """Known-bad controls must fail; all other case/metric pairs must pass."""
    test_case = cases[case_id]
    metric = build_metric(metric_key)
    if expected_verdict == "fail":
        with pytest.raises(AssertionError):
            assert_test(test_case, [metric])
    else:
        assert_test(test_case, [metric])
