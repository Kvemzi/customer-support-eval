from unittest.mock import Mock, patch

from support_agent.backends.openai_backend import OpenAIBackend


def test_openai_backend_preserves_history_roles_and_message_order():
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="  answer  "))
    ]
    with patch("support_agent.backends.openai_backend.OpenAI", return_value=client):
        backend = OpenAIBackend(model="test-model", temperature=0.2)
        answer = backend.generate(
            "system prompt",
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "system", "content": "extra instruction"},
            ],
            "next question",
            [],
        )
    assert answer == "answer"
    client.chat.completions.create.assert_called_once_with(
        model="test-model", temperature=0.2,
        messages=[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "system", "content": "extra instruction"},
            {"role": "user", "content": "next question"},
        ],
    )
