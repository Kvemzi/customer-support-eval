"""Real LLM backend via the OpenAI Chat Completions API (SDK v1+)."""

from __future__ import annotations

from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from support_agent.backends.base import LLMBackend
from support_agent.schemas import Message


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3) -> None:
        self.client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.model = model
        self.temperature = temperature

    def generate(self, system_prompt, history: List[Message], user_message, context) -> str:
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        for message in history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["content"]})
            elif message["role"] == "assistant":
                messages.append({"role": "assistant", "content": message["content"]})
            elif message["role"] == "system":
                messages.append({"role": "system", "content": message["content"]})
            else:
                raise ValueError(f"Unsupported message role: {message['role']}")
        messages.append({"role": "user", "content": user_message})
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return (completion.choices[0].message.content or "").strip()