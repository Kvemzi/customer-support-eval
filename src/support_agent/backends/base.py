"""Abstract interface for every LLM backend (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from support_agent.schemas import Message


class LLMBackend(ABC):
    """A backend turns (system prompt, history, user message, context) into a reply."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        history: List[Message],
        user_message: str,
        context: List[str],
    ) -> str:
        """Return the assistant's reply text."""