"""LLM providers behind one small interface.

Generation only needs "give me an instance of this pydantic schema". Anything
that implements `StructuredLLM` works, so swapping providers touches only this
file. GUESS: Anthropic (Claude) is the default until the provider choice is
confirmed.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"


class LLMRefusal(RuntimeError):
    """The model declined to answer."""


class LLMOutputError(RuntimeError):
    """The model returned no usable structured output."""


class StructuredLLM(Protocol):
    model: str

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T: ...


class AnthropicLLM:
    """Claude via the Messages API with structured outputs.

    Uses adaptive thinking (the default on this model) and the server-side
    refusal fallback, so a refused request is retried on a fallback model
    inside the same call.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        effort: str = "high",
        max_tokens: int = 16000,
        client: object | None = None,
    ) -> None:
        self.model = model or os.environ.get("GENERATION_MODEL", DEFAULT_MODEL)
        self.effort = effort
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal(f"model declined: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMOutputError("output hit max_tokens before finishing")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMOutputError("no parsed output in response")
        return parsed


class FakeLLM:
    """Deterministic stand-in for tests and offline runs.

    `responder` gets (system, prompt, schema) and returns a schema instance.
    Calls are recorded in `calls` so tests can inspect the prompts.
    """

    model = "fake"

    def __init__(self, responder: Callable[[str, str, type], BaseModel]) -> None:
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        self.calls.append((system, prompt))
        result = self._responder(system, prompt, schema)
        if not isinstance(result, schema):
            result = schema.model_validate(result)
        return result
