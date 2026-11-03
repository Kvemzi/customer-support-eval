"""Keyword-based retriever over the JSON knowledge base in data/knowledge_base/."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from support_agent.config import DEFAULT_KB_PATH


@dataclass(frozen=True)
class Document:
    topic: str
    text: str
    keywords: List[str]


class KnowledgeBase:
    """
    Loads policy documents from JSON and retrieves the most relevant ones.

    Keyword matching keeps retrieval deterministic, which matters for
    reproducible evaluation. The `retrieve()` interface is stable, so it can
    later be swapped for an embedding or vector-store retriever without
    touching the agent.
    """

    def __init__(self, documents: Dict[str, Document], version: str = "unknown") -> None:
        self.documents = documents
        self.version = version

    # ---- Construction --------------------------------------------------------
    @classmethod
    def from_json(cls, path: Optional[Path] = None) -> "KnowledgeBase":
        path = Path(path or DEFAULT_KB_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        documents = {
            topic: Document(
                topic=topic,
                text=doc["text"],
                keywords=[k.lower() for k in doc.get("keywords", [])],
            )
            for topic, doc in raw["documents"].items()
        }
        return cls(documents=documents, version=raw.get("version", "unknown"))

    # ---- Access --------------------------------------------------------------
    def get(self, topic: str) -> str:
        """Return the text of one topic (used by test_dataset.py for `context`)."""
        try:
            return self.documents[topic].text
        except KeyError as exc:
            raise KeyError(f"Unknown topic '{topic}'. Available: {list(self.documents)}") from exc

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """
        Return up to `top_k` passages ranked by keyword overlap.
        An empty list means "no grounded answer available", and the agent
        must then refuse rather than invent a policy.
        """
        query_lower = query.lower()
        scores = {
            topic: sum(1 for kw in doc.keywords if kw in query_lower)
            for topic, doc in self.documents.items()
        }
        ranked = [t for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True) if s > 0]
        return [self.documents[t].text for t in ranked[:top_k]]