import pytest

from agent_eval import test_dataset
from run_evaluation import load_cases
from support_agent.schemas import AgentResponse


class FakeAgent:
    def __init__(self):
        self.messages = []

    def reset(self):
        pass

    def chat(self, message):
        self.messages.append(message)
        return AgentResponse(user_input=message, answer="recorded answer")


def test_selected_live_case_generates_only_selected_output(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(test_dataset, "_make_agent", lambda mode: agent)
    cases = load_cases("live", ["TC-01"])
    assert [test_dataset.get_metadata(case)["id"] for case in cases] == ["TC-01"]
    assert len(agent.messages) == 1


def test_static_negative_control_does_not_create_live_agent(monkeypatch):
    def fail_if_called(mode):
        pytest.fail("A static negative control must not create an agent")

    monkeypatch.setattr(test_dataset, "_make_agent", fail_if_called)
    case = test_dataset.get_case_by_id("TC-02", mode="live")
    assert test_dataset.get_metadata(case)["output_source"] == "static"


def test_unknown_case_is_rejected_before_agent_creation(monkeypatch):
    def fail_if_called(mode):
        pytest.fail("Unknown case must not create an agent")

    monkeypatch.setattr(test_dataset, "_make_agent", fail_if_called)
    with pytest.raises(KeyError, match="Unknown case"):
        test_dataset.get_all_test_cases("live", case_ids=["not-a-case"])


def test_selected_cases_keep_dataset_order():
    cases = test_dataset.get_all_test_cases("static", case_ids=["TC-04", "TC-01"])
    assert [test_dataset.get_metadata(case)["id"] for case in cases] == ["TC-01", "TC-04"]
