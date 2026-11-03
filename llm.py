"""Thin LLM abstraction so the agents don't care which provider is behind them.

Providers (LLM_PROVIDER): gemini (default), anthropic, openai, mock.
Override the model with LLM_MODEL.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}
KEY_VARS = {
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class LLM(Protocol):
    name: str

    def structured(self, system: str, user: str, schema: type[T]) -> T: ...


class LLMConfigError(RuntimeError):
    pass


class LangChainLLM:
    """Structured-output calls through LangChain chat models."""

    def __init__(self, provider: str, model: str | None = None):
        if provider not in DEFAULT_MODELS:
            raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider}'")
        key_var = KEY_VARS[provider]
        if not os.getenv(key_var):
            raise LLMConfigError(f"{key_var} is not set (LLM_PROVIDER={provider})")
        model = model or DEFAULT_MODELS[provider]
        self.name = f"{provider}:{model}"

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._chat = ChatGoogleGenerativeAI(model=model, temperature=0)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self._chat = ChatAnthropic(model=model, temperature=0, max_tokens=4096)
        else:
            from langchain_openai import ChatOpenAI

            self._chat = ChatOpenAI(model=model, temperature=0)

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        runnable = self._chat.with_structured_output(schema)
        messages = [("system", system), ("human", user)]
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                out = runnable.invoke(messages)
                if isinstance(out, schema):
                    return out
                return schema.model_validate(out)
            except Exception as exc:  # noqa: BLE001 - retry once on any parse/transport error
                last_exc = exc
                log.warning("LLM call failed (attempt %s): %s", attempt, exc)
        raise RuntimeError(f"LLM call failed after retry: {last_exc}") from last_exc


def get_llm() -> LLM:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "mock":
        from .mock_llm import MockLLM

        return MockLLM()
    return LangChainLLM(provider, os.getenv("LLM_MODEL") or None)
