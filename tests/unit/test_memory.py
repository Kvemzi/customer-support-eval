from support_agent.memory import ConversationMemory


def test_memory_trims_to_max_turns():
    memory = ConversationMemory(max_turns=2)
    for i in range(5):
        memory.add_user_message(f"q{i}")
        memory.add_assistant_message(f"a{i}")
    assert len(memory) == 4
    assert memory.get_messages()[0]["content"] == "q3"


def test_get_messages_returns_copy():
    memory = ConversationMemory()
    memory.add_user_message("hello")
    memory.get_messages().clear()
    assert len(memory) == 1