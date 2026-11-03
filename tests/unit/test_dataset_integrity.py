"""Structural checks on the evaluation dataset. No LLM calls are made."""

from deepeval.test_case import ConversationalTestCase, LLMTestCase

from agent_eval.test_dataset import (
    ALL_TEST_CASES, TC_01, TC_02, TC_03, TC_04, get_metadata,
)


def test_ids_are_unique():
    ids = [get_metadata(c)["id"] for c in ALL_TEST_CASES]
    assert len(ids) == len(set(ids)) == 4


def test_case_types():
    assert all(isinstance(c, LLMTestCase) for c in (TC_01, TC_02, TC_03))
    assert isinstance(TC_04, ConversationalTestCase)


def test_single_turn_cases_have_ground_truth_context():
    for case in (TC_01, TC_02, TC_03):
        assert case.input and case.actual_output and case.context


def test_tc02_contradicts_context():
    # The negative control is only valid if output and context truly disagree.
    assert "60-day" in TC_02.actual_output
    assert any("30 days" in c for c in TC_02.context)
    assert not any("60" in c for c in TC_02.context)
    assert get_metadata(TC_02)["expected_verdict"] == "fail"


def test_tc04_is_well_formed_dialogue():
    roles = [t.role for t in TC_04.turns]
    assert roles == ["user", "assistant"] * 4
    assert TC_04.scenario and TC_04.expected_outcome


def test_tc04_assistant_remembers_order_number():
    assistant_msgs = [t.content for t in TC_04.turns if t.role == "assistant"]
    assert sum("784512" in m for m in assistant_msgs) >= 2