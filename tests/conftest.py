"""Shared fixtures: the agent always uses the offline mock backend here."""

import pytest

from support_agent.agent import SupportAgent
from support_agent.backends.mock_backend import MockBackend
from support_agent.knowledge_base import KnowledgeBase
from support_agent.memory import ConversationMemory


@pytest.fixture
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_json()


@pytest.fixture
def mock_agent(kb: KnowledgeBase) -> SupportAgent:
    return SupportAgent(backend=MockBackend(), knowledge_base=kb,
                        memory=ConversationMemory(max_turns=5))