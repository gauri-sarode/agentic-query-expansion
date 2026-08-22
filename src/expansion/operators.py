"""generate_expansion(action, query, evidence) -> rewritten query. The
single LLM tool the agent can invoke; one call per action, hard maximum
two calls per query episode (src/config.py:max_llm_calls_per_query).
"""
from __future__ import annotations

from src.agent.actions import EXPANSION_ACTIONS, Action
from src.expansion.prompts import (
    AMBIGUITY_PROMPT,
    CONTEXT_PROMPT,
    ENTITY_NORMALIZE_PROMPT,
    VOCABULARY_PROMPT,
)
from src.llm_client import chat

_PROMPTS = {
    Action.VOCABULARY: VOCABULARY_PROMPT,
    Action.CONTEXT: CONTEXT_PROMPT,
    Action.AMBIGUITY: AMBIGUITY_PROMPT,
    Action.ENTITY_NORMALIZE: ENTITY_NORMALIZE_PROMPT,
}


def generate_expansion(action: Action, query: str, evidence: list[str]) -> str:
    if action not in EXPANSION_ACTIONS:
        raise ValueError(f"{action} is not an expansion action; expected one of {EXPANSION_ACTIONS}")
    prompt = _PROMPTS[action].format(query=query, evidence="\n---\n".join(evidence))
    raw = chat(messages=[{"role": "user", "content": prompt}])
    rewritten = raw.strip().strip('"').strip()
    return rewritten or query
