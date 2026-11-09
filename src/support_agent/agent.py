"""SupportAgent: orchestrates retrieval, memory, prompting and generation."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from support_agent.backends import LLMBackend, build_backend
from support_agent.config import Settings
from support_agent.knowledge_base import KnowledgeBase
from support_agent.memory import ConversationMemory
from support_agent.prompts import build_system_prompt
from support_agent.schemas import AgentResponse

logger = logging.getLogger(__name__)

FOLLOW_UP_REFERENCE = re.compile(
    r"^\s*(?:(?:and|but|so)\s+)?"
    r"(?:how long (?:is|does|will) (?:it|that|this)(?: take| last)?"
    r"|what about (?:it|that|this|them|those|these))\s*[?.!]*\s*$",
    re.IGNORECASE,
)

FALLBACK_ANSWER = (
    "I'm sorry, I'm having technical difficulties right now. "
    "Please contact support@shopco.example."
)


class SupportAgent:
    """
    Customer-support agent (the System Under Test).

    Per-turn pipeline:
        1. retrieve  -> facts from the knowledge base
        2. prompt    -> inject facts into the system prompt
        3. generate  -> call the configured LLM backend with history
        4. remember  -> store the exchange; return a structured AgentResponse

    All collaborators are injectable, so tests can swap any component.
    """

    def __init__(
        self,
        backend: LLMBackend,
        knowledge_base: KnowledgeBase,
        memory: ConversationMemory,
        top_k: int = 2,
    ) -> None:
        self.backend = backend
        self.knowledge_base = knowledge_base
        self.memory = memory
        self.top_k = top_k
        self._turn_counter = 0
        self._last_context: List[str] = []

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "SupportAgent":
        """Convenience constructor wiring all components from configuration."""
        settings = settings or Settings.from_env()
        agent = cls(
            backend=build_backend(settings),
            knowledge_base=KnowledgeBase.from_json(settings.kb_path),
            memory=ConversationMemory(max_turns=settings.memory_max_turns),
            top_k=settings.retrieval_top_k,
        )
        logger.info("SupportAgent ready (backend=%s, kb=%s)",
                    agent.backend.name, agent.knowledge_base.version)
        return agent

    def chat(self, user_message: str) -> AgentResponse:
        """Process one user message and return the grounded reply."""
        if not user_message or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")

        context = self.knowledge_base.retrieve(user_message, top_k=self.top_k)
        if not context and FOLLOW_UP_REFERENCE.search(user_message):
            context = list(self._last_context)
        system_prompt = build_system_prompt(context)
        history = self.memory.get_messages()  # excludes the current message

        try:
            answer = self.backend.generate(system_prompt, history, user_message, context)
        except Exception:  # network or API errors must not crash the session
            logger.exception("Backend '%s' failed", self.backend.name)
            answer = FALLBACK_ANSWER

        self.memory.add_user_message(user_message)
        self.memory.add_assistant_message(answer)
        self._last_context = list(context)
        self._turn_counter += 1

        return AgentResponse(
            user_input=user_message,
            answer=answer,
            retrieval_context=context,
            backend=self.backend.name,
            turn_index=self._turn_counter,
        )

    def reset(self) -> None:
        """Start a new, independent conversation."""
        self.memory.reset()
        self._last_context.clear()
        self._turn_counter = 0