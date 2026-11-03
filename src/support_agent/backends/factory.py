"""Backend factory: the only place that knows which concrete backends exist."""

from __future__ import annotations

from support_agent.backends.base import LLMBackend
from support_agent.config import Settings


def build_backend(settings: Settings) -> LLMBackend:
    """
    Resolve settings.backend:
      openai | http | mock | auto (openai if a key exists, else mock)
    Imports are local so a missing optional dependency only fails when used.
    """
    choice = settings.backend
    if choice == "auto":
        choice = "openai" if settings.openai_api_key else "mock"

    if choice == "openai":
        from support_agent.backends.openai_backend import OpenAIBackend
        return OpenAIBackend(model=settings.model, temperature=settings.temperature)

    if choice == "http":
        from support_agent.backends.http_backend import HttpBackend
        return HttpBackend(endpoint=settings.http_endpoint, timeout=settings.http_timeout)

    if choice == "mock":
        from support_agent.backends.mock_backend import MockBackend
        return MockBackend()

    raise ValueError(f"Unknown backend '{settings.backend}'. Use openai | http | mock | auto.")