"""Sliding-window dialogue memory."""

from __future__ import annotations

from typing import List

from support_agent.schemas import Message


class ConversationMemory:
    """
    Keeps the last `max_turns` user/assistant pairs.

    The bounded window keeps prompt size, and therefore token cost, under
    control while preserving enough history for multi-turn coherence.
    """

    def __init__(self, max_turns: int = 10) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        self._messages: List[Message] = []

    def add_user_message(self, content: str) -> None:
        self._append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self._append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Message]:
        """Return a copy so callers cannot mutate the internal state."""
        return [message.copy() for message in self._messages]

    def reset(self) -> None:
        self._messages.clear()

    def _append(self, message: Message) -> None:
        self._messages.append(message)
        max_messages = self.max_turns * 2  # one turn = user + assistant
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def __len__(self) -> int:
        return len(self._messages)