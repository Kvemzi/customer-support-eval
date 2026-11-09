from unittest.mock import Mock, patch

import pytest

from support_agent.agent import FALLBACK_ANSWER, SupportAgent
from support_agent.backends.http_backend import HttpBackend
from support_agent.memory import ConversationMemory


@pytest.mark.parametrize("body", [{}, {"answer": None}, {"answer": ""}, {"answer": 123}, []])
def test_http_backend_rejects_invalid_answers(body):
    response = Mock()
    response.json.return_value = body
    with patch("support_agent.backends.http_backend.requests.post", return_value=response):
        with pytest.raises(ValueError, match="HTTP backend response"):
            HttpBackend("http://example.invalid").generate("", [], "question", [])


def test_http_backend_returns_valid_answer():
    response = Mock()
    response.json.return_value = {"answer": "  A valid answer  "}
    with patch("support_agent.backends.http_backend.requests.post", return_value=response):
        assert HttpBackend("http://example.invalid").generate("", [], "question", []) == (
            "A valid answer"
        )


def test_invalid_http_answer_uses_logged_agent_fallback(kb, caplog):
    response = Mock()
    response.json.return_value = {}
    agent = SupportAgent(HttpBackend("http://example.invalid"), kb, ConversationMemory())
    with patch("support_agent.backends.http_backend.requests.post", return_value=response):
        assert agent.chat("What is the return policy?").answer == FALLBACK_ANSWER
    assert "Backend 'http' failed" in caplog.text
