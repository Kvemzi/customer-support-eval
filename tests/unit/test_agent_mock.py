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