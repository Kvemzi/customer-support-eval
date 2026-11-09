from unittest.mock import Mock

from support_agent.agent import SupportAgent
from support_agent.memory import ConversationMemory
from support_agent.prompts import PROMPT_VERSION, build_system_prompt


def test_prompt_requests_focused_but_qualified_answers():
    prompt = build_system_prompt(["A policy fact."])

    assert PROMPT_VERSION == "v1.3"
    assert "Answer each part of the customer's question directly" in prompt
    assert "mention that other conditions also apply" in prompt
    assert "eligibility or coverage depends on unknown" in prompt
    assert "- A policy fact." in prompt


def test_agent_passes_focused_prompt_and_retrieved_context_to_backend(kb):
    backend = Mock()
    backend.name = "test"
    backend.generate.return_value = "A grounded response."
    agent = SupportAgent(backend=backend, knowledge_base=kb, memory=ConversationMemory())

    response = agent.chat("My headphones have a defect. Are they under warranty?")

    system_prompt, history, question, context = backend.generate.call_args.args
    assert "Answer each part of the customer's question directly" in system_prompt
    assert kb.get("warranty") in context
    assert kb.get("warranty") in system_prompt
    assert history == []
    assert question == response.user_input
