"""
tests/evals/test_suite.py
=========================
LLM-as-a-judge test suite for the ShopCo support agent.

Run with the DeepEval CLI (recommended; produces the DeepEval run report):

    deepeval test run tests/evals/test_suite.py

or with plain pytest (same assertions, no DeepEval run report):

    pytest tests/evals/test_suite.py -v

Dataset mode is controlled by EVAL_DATASET_MODE (static | live).

Conventions
-----------
* One test function per (test case, metric) pair for granular reporting.
* Metrics and thresholds come from agent_eval.metrics (single source of truth).
* assert_test(test_case, [metric]) raises AssertionError if the metric fails.
* TC-02 is a NEGATIVE CONTROL: the hallucination metric must FAIL on it,
  so the assertion is inverted with pytest.raises.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest
from deepeval import assert_test

from agent_eval.metrics import build_metric
from agent_eval.test_dataset import (
    M_ANSWER_RELEVANCY,
    M_BIAS,
    M_CONVERSATION_COMPLETENESS,
    M_HALLUCINATION,
    M_TOXICITY,
    M_TURN_RELEVANCY,
    get_all_test_cases,
    get_metadata,
)

# Applied to every test in this module.
pytestmark = [
    pytest.mark.llm,  # lets you deselect paid tests: pytest -m "not llm"
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY is not set; the LLM judge cannot run.",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cases() -> Dict[str, object]:
    """
    All test cases keyed by ID (TC-01 ... TC-04).

    Built once per module. In live mode this is where the SUT is called,
    so every metric test judges the same agent output.
    """
    return {get_metadata(c)["id"]: c for c in get_all_test_cases()}


# =============================================================================
# TC-01: Happy Path
# =============================================================================

def test_tc01_answer_relevancy(cases):
    """The answer about the return policy must be relevant and direct."""
    assert_test(cases["TC-01"], [build_metric(M_ANSWER_RELEVANCY)])


def test_tc01_hallucination(cases):
    """The answer must not contradict the 30-day policy in the context."""
    assert_test(cases["TC-01"], [build_metric(M_HALLUCINATION)])


# =============================================================================
# TC-02: Hallucination (negative control)
# =============================================================================

def test_tc02_hallucination_is_detected(cases):
    """
    The recorded answer claims a 60-day return window that contradicts the
    30-day policy. The HallucinationMetric MUST fail here.

    If assert_test does NOT raise, the judge missed an obvious hallucination
    and the whole evaluation setup is untrustworthy.
    """
    with pytest.raises(AssertionError):
        assert_test(cases["TC-02"], [build_metric(M_HALLUCINATION)])


# =============================================================================
# TC-03: Safety (hostile, bias-baiting input)
# =============================================================================

def test_tc03_toxicity(cases):
    """The agent must not mirror the user's insults."""
    assert_test(cases["TC-03"], [build_metric(M_TOXICITY)])


def test_tc03_bias(cases):
    """The agent must not endorse the age stereotype."""
    assert_test(cases["TC-03"], [build_metric(M_BIAS)])


def test_tc03_answer_relevancy(cases):
    """Despite the abuse, the agent must still address the refund request."""
    assert_test(cases["TC-03"], [build_metric(M_ANSWER_RELEVANCY)])


# =============================================================================
# TC-04: Multi-turn conversation
# =============================================================================

def test_tc04_turn_relevancy(cases):
    """Each assistant turn must be relevant given the preceding dialogue."""
    assert_test(cases["TC-04"], [build_metric(M_TURN_RELEVANCY)])


def test_tc04_conversation_completeness(cases):
    """By the end, the user's intentions (delay, return, refund) must be satisfied."""
    assert_test(cases["TC-04"], [build_metric(M_CONVERSATION_COMPLETENESS)])