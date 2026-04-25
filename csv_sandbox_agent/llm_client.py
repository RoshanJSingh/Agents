from __future__ import annotations

import os
from typing import Protocol

from .schemas import LLMResponseError


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        ...


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat-completions client.

    The OpenAI SDK is imported lazily so tests and fake-mode usage do not need
    network credentials or the SDK installed.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("CSV_SANDBOX_AGENT_MODEL", "gpt-4.1-mini")
        self.base_url = base_url or os.getenv("CSV_SANDBOX_AGENT_BASE_URL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMResponseError("OPENAI_API_KEY is not set")

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on optional SDK
            raise LLMResponseError(
                "The OpenAI SDK is not installed. Install package dependencies or use fake mode."
            ) from exc

        client_kwargs: dict[str, str] = {"api_key": self.api_key or ""}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMResponseError("LLM returned an empty response")
        return content


class FakeLLMClient:
    """Deterministic LLM client for tests and dry-run style local checks."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, object]] = []

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        if not self.responses:
            raise LLMResponseError("FakeLLMClient has no response queued")
        return self.responses.pop(0)
