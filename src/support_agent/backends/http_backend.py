"""
Backend for an agent deployed behind a REST API.

Contract:
    POST {endpoint}
    body:     {"system": str, "messages": [...], "context": [...]}
    response: {"answer": str}
"""

from __future__ import annotations

from typing import List

import requests

from support_agent.backends.base import LLMBackend
from support_agent.schemas import Message


class HttpBackend(LLMBackend):
    name = "http"

    def __init__(self, endpoint: str, timeout: int = 30) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, system_prompt, history: List[Message], user_message, context) -> str:
        payload = {
            "system": system_prompt,
            "messages": [*history, {"role": "user", "content": user_message}],
            "context": context,
        }
        resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json().get("answer", "")).strip()