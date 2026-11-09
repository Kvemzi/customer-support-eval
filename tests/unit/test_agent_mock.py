def test_agent_remembers_order_number_across_turns(mock_agent):
    mock_agent.chat("My order 123456 has not arrived yet.")
    response = mock_agent.chat("Can I return it?")
    assert "#123456" in response.answer
    assert "30 days" in response.answer


def test_agent_refuses_when_no_context(mock_agent):
    response = mock_agent.chat("Do you sell cars on Mars?")
    assert response.retrieval_context == []
    assert "don't have verified information" in response.answer


def test_reset_clears_memory(mock_agent):
    mock_agent.chat("Where is my package?")
    mock_agent.reset()
    assert len(mock_agent.memory) == 0


def test_referential_follow_up_reuses_previous_context(mock_agent):
    first = mock_agent.chat("How long do refunds take?")
    follow_up = mock_agent.chat("And how long is that?")
    assert first.retrieval_context
    assert follow_up.retrieval_context == first.retrieval_context
    assert "5-7 business days" in follow_up.answer


def test_unrelated_turn_and_reset_clear_follow_up_context(mock_agent):
    mock_agent.chat("How long do refunds take?")
    assert mock_agent.chat("Do you sell this car on Mars?").retrieval_context == []
    assert mock_agent.chat("How long is that?").retrieval_context == []

    mock_agent.chat("How long do refunds take?")
    mock_agent.reset()
    assert mock_agent.chat("How long is that?").retrieval_context == []