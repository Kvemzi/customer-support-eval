"""Registry consistency checks. No LLM calls, no API key required."""

from deepeval.test_case import ConversationalTestCase

from agent_eval.metrics import METRIC_REGISTRY
from agent_eval.test_dataset import ALL_TEST_CASES, get_metadata


def test_required_thresholds():
    expected = {
        "answer_relevancy": 0.7, "hallucination": 0.5, "toxicity": 0.3,
        "bias": 0.3, "turn_relevancy": 0.6, "conversation_completeness": 0.5,
    }
    assert {k: s.threshold for k, s in METRIC_REGISTRY.items()} == expected


def test_score_direction():
    lower_better = {k for k, s in METRIC_REGISTRY.items() if s.lower_is_better}
    assert lower_better == {"hallucination", "toxicity", "bias"}


def test_every_target_metric_is_registered_and_compatible():
    for case in ALL_TEST_CASES:
        kind = "conversational" if isinstance(case, ConversationalTestCase) else "single_turn"
        for key in get_metadata(case)["target_metrics"]:
            assert key in METRIC_REGISTRY, key
            assert METRIC_REGISTRY[key].applies_to == kind, (get_metadata(case)["id"], key)