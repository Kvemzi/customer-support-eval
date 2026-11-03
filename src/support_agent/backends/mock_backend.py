"""
Deterministic offline backend.

Used for unit tests, CI without API keys, and as a stable baseline to
contrast against the stochastic OpenAI backend.
"""

from __future__ import annotations

import re
from typing import List, Optional

from support_agent.backends.base import LLMBackend
from support_agent.schemas import Message

HOSTILE_PATTERN = re.compile(
    r"\b(stupid|idiot|useless|hate|garbage|scam|worst|dumb)\b", re.IGNORECASE
)
ORDER_NUMBER_PATTERN = re.compile(r"(\d{5,})")


class MockBackend(LLMBackend):
    name = "mock"

    def generate(self, system_prompt, history: List[Message], user_message, context) -> str:
        parts: List[str] = []

        # 1) De-escalation: never mirror a hostile tone.
        if HOSTILE_PATTERN.search(user_message):
            parts.append(
                "I'm sorry you're having a frustrating experience. "
                "I'm here to help and will do my best to resolve this for you."
            )

        # 2) Memory: reuse an order number mentioned anywhere in the dialogue.
        all_msgs: List[Message] = [*history, {"role": "user", "content": user_message}]
        order_number = self._find_order_number(all_msgs)
        if order_number:
            parts.append(f"Regarding your order #{order_number}:")

        # 3) Grounding: answer strictly from the retrieved context, or refuse.
        if context:
            parts.append(" ".join(context))
        else:
            parts.append(
                "I don't have verified information about that. I can connect you with "
                "a human agent (Mon-Fri, 09:00-17:00 CET) via support@shopco.example."
            )
        return " ".join(parts)

    @staticmethod
    def _find_order_number(messages: List[Message]) -> Optional[str]:
        for msg in reversed(messages):
            if msg["role"] == "user":
                match = ORDER_NUMBER_PATTERN.search(msg["content"])
                if match:
                    return match.group(1)
        return None