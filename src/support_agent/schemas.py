"""Data contracts shared between the SUT and the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, TypedDict


class Message(TypedDict):
    """A single chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class AgentResponse:
    """
    Structured output of one agent turn.

    Field names map directly onto DeepEval test-case fields:
        user_input        -> LLMTestCase.input
        answer            -> LLMTestCase.actual_output
        retrieval_context -> LLMTestCase.context / retrieval_context
    """

    user_input: str
    answer: str
    retrieval_context: List[str] = field(default_factory=list)
    backend: str = ""
    turn_index: int = 0