from __future__ import annotations

# Deprecated: branch facts are produced by the LLM from textbook excerpts, not injected here.
CANONICAL_TOPIC_FACTS: dict[str, list[str]] = {}


def canonical_facts_for_topic(topic: str) -> list[str]:
    """Return canonical branch facts for a topic (empty — LLM generates all content)."""
    return list(CANONICAL_TOPIC_FACTS.get(topic, ()))
