"""Provider-agnostic LLM client. Default provider is a local Ollama model
(Qwen3.5-2B, 4-bit) so the main experiment has no per-call API cost;
huggingface/anthropic are available for the small robustness subset.
Callers use chat() with a fixed signature regardless of backend.
"""
from __future__ import annotations

import logging

import requests

from src.config import CONFIG

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when a live LLM call can't be made; callers decide fallback behavior."""


def _chat_ollama(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    response = requests.post(
        f"{CONFIG.ollama_base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["message"]["content"] or ""


def _chat_huggingface(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    if not CONFIG.hf_api_token:
        raise LLMUnavailableError("no HF_API_TOKEN configured")
    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {CONFIG.hf_api_token}"},
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"] or ""


def _chat_anthropic(model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    if not CONFIG.anthropic_api_key:
        raise LLMUnavailableError("no ANTHROPIC_API_KEY configured")
    import anthropic

    client = anthropic.Anthropic(api_key=CONFIG.anthropic_api_key)
    response = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature, messages=messages
    )
    return response.content[0].text if response.content else ""


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    model = model or CONFIG.expansion_model
    max_tokens = max_tokens or CONFIG.llm_output_tokens
    try:
        if CONFIG.llm_provider == "ollama":
            return _chat_ollama(model, messages, temperature, max_tokens)
        if CONFIG.llm_provider == "huggingface":
            return _chat_huggingface(model, messages, temperature, max_tokens)
        if CONFIG.llm_provider == "anthropic":
            return _chat_anthropic(model, messages, temperature, max_tokens)
        raise LLMUnavailableError(f"unknown provider {CONFIG.llm_provider!r}")
    except requests.RequestException as e:
        raise LLMUnavailableError(str(e)) from e
