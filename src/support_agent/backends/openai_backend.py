"""Real LLM backend via the OpenAI Chat Completions API (SDK v1+)."""

from __future__ import annotations

from typing import List

from openai import OpenAI

from support_agent.backends.base import LLMBackend
from support_agent.schemas import Message


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3) -> None:
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model
        self.temperature = temperature

    def generate(self, system_prompt, history: List[Message], user_message, context) -> str:
        messages = [{"role": "system", "content": system_prompt}, *history,
                    {"role": "user", "content": user_message}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return (completion.choices[0].message.content or "").strip()