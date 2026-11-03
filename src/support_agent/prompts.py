"""
Prompt templates, kept separate so prompt changes are easy to review
(and to correlate with evaluation score changes between versions).
"""

from __future__ import annotations

from typing import List

PROMPT_VERSION = "v1.0"

SYSTEM_PROMPT_TEMPLATE = """You are ShopCo's customer-support assistant.

RULES:
1. Answer ONLY using the facts in the CONTEXT section below.
2. If the CONTEXT does not contain the answer, say so clearly and offer a
   handoff to human support. NEVER invent policies, numbers or benefits.
3. Be polite, concise, neutral and professional, even if the customer is rude.
4. Do not make assumptions about customers based on gender, age, ethnicity,
   religion, nationality or any other personal attribute.
5. Use the conversation history to stay consistent across turns.

CONTEXT:
{context}
"""


def build_system_prompt(context: List[str]) -> str:
    """Inject the retrieved facts into the system prompt."""
    formatted = (
        "\n".join(f"- {c}" for c in context) if context else "- (no relevant information found)"
    )
    return SYSTEM_PROMPT_TEMPLATE.format(context=formatted)