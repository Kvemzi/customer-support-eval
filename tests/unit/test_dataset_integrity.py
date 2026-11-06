"""Structural checks on the evaluation dataset. No LLM calls are made."""

from deepeval.test_case import ConversationalTestCase, LLMTestCase

from agent_eval.test_dataset import (
    ALL_CASE_SPECS,
    ALL_TEST_CASES,
    CONVERSATIONAL_TEST_CASES,
    SINGLE_TURN_TEST_CASES,
    CONVERSATIONAL_SPECS,
    SINGLE_TURN_SPECS,
    TC_01,
    TC_02,
    TC_03,
    TC_04,
    get_all_test_cases,
    get_metadata,
)
from support_agent.config import Settings
from support_agent.knowledge_base import KnowledgeBase


def test_ids_are_unique():
    ids = [get_metadata(c)["id"] for c in ALL_TEST_CASES]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 20


def test_combined_case_order_places_tc04_after_tc03():
    expected_ids = [spec.id for spec in ALL_CASE_SPECS]
    assert expected_ids[:5] == ["TC-01", "TC-02", "TC-03", "TC-04", "TC-05"]
    assert [get_metadata(case)["id"] for case in ALL_TEST_CASES] == expected_ids
    assert [get_metadata(case)["id"] for case in get_all_test_cases()] == expected_ids


def test_case_types():
    assert all(isinstance(c, LLMTestCase) for c in SINGLE_TURN_TEST_CASES)
    assert all(isinstance(c, ConversationalTestCase) for c in CONVERSATIONAL_TEST_CASES)
    assert all(isinstance(c, LLMTestCase) for c in (TC_01, TC_02, TC_03))
    assert isinstance(TC_04, ConversationalTestCase)


def test_single_turn_cases_have_ground_truth_context():
    for case in SINGLE_TURN_TEST_CASES:
        assert case.input and case.actual_output and case.context
        assert get_metadata(case)["target_metrics"]


def test_tc09_checks_accidental_damage_with_grounding_metric():
    case = next(case for case in SINGLE_TURN_TEST_CASES if get_metadata(case)["id"] == "TC-09")
    assert get_metadata(case)["target_metrics"] == ["hallucination"]
    assert "accidental damage" in case.input.lower()
    assert "not cover accidental damage" in case.actual_output.lower()


def test_tc16_uses_grounding_metric_for_untrusted_conflicting_instruction():
    case = next(case for case in SINGLE_TURN_TEST_CASES if get_metadata(case)["id"] == "TC-16")
    assert get_metadata(case)["target_metrics"] == ["hallucination"]


def test_tc20_directly_answers_independence_question():
    case = next(case for case in SINGLE_TURN_TEST_CASES if get_metadata(case)["id"] == "TC-20")
    answer = case.actual_output.lower()
    assert get_metadata(case)["target_metrics"] == [
        "bias", "answer_relevancy", "hallucination"
    ]
    assert "do not need someone younger" in answer
    assert "handle the return yourself" in answer
    assert "contact shopco support" in answer
    assert "30 days" not in answer


def test_tc02_contradicts_context():
    # The negative control is only valid if output and context truly disagree.
    assert "60-day" in TC_02.actual_output
    assert any("30 days" in c for c in TC_02.context)
    assert not any("60" in c for c in TC_02.context)
    assert get_metadata(TC_02)["expected_verdict"] == "fail"


def test_negative_controls_are_fixed_and_marked_expected_fail():
    cases_by_id = {get_metadata(case)["id"]: case for case in ALL_TEST_CASES}
    for case_id in ("TC-02", "TC-13", "TC-14", "TC-15"):
        case = cases_by_id[case_id]
        assert get_metadata(case)["expected_verdict"] == "fail"
        assert get_metadata(case)["output_source"] == "static"


def test_live_inputs_retrieve_their_declared_policy_topics():
    kb = KnowledgeBase.from_json(Settings.from_env().kb_path)
    for spec in SINGLE_TURN_SPECS:
        if spec.live_eligible:
            retrieved = kb.retrieve(spec.input)
            assert all(kb.get(topic) in retrieved for topic in spec.context_topics), spec.id

    for spec in CONVERSATIONAL_SPECS:
        for index, (user_message, _, topics) in enumerate(spec.static_turns, 1):
            retrieved = kb.retrieve(user_message)
            assert all(kb.get(topic) in retrieved for topic in topics), (
                spec.id,
                index,
            )


def test_tc04_is_well_formed_dialogue():
    roles = [t.role for t in TC_04.turns]
    assert roles == ["user", "assistant"] * 4
    assert TC_04.scenario and TC_04.expected_outcome


def test_tc04_assistant_remembers_order_number():
    assistant_msgs = [t.content for t in TC_04.turns if t.role == "assistant"]
    assert sum("784512" in m for m in assistant_msgs) >= 2