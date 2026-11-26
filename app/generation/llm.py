"""LLM providers behind one small interface.

Generation only needs "give me an instance of this pydantic schema". Anything
that implements `StructuredLLM` works, so swapping providers touches only this
file. Gemini is the default (chosen 2026-11-26); Claude stays available.
Pick with LLM_PROVIDER=gemini|anthropic, or construct a class directly.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"
# Most capable Gemini model with a free API tier as of 2026-11-26.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


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


class GeminiLLM:
    """Gemini via the google-genai SDK with JSON structured output.

    Needs a Gemini API key from Google AI Studio in GEMINI_API_KEY (or
    GOOGLE_API_KEY). A consumer Gemini app subscription does not include
    API access. On the free tier Google may use prompts to improve its
    products, so do not send confidential documents on a free key.
    """

    _BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}

    def __init__(
        self,
        model: str | None = None,
        *,
        max_output_tokens: int = 16000,
        client: object | None = None,
    ) -> None:
        self.model = model or os.environ.get("GENERATION_MODEL", DEFAULT_GEMINI_MODEL)
        self.max_output_tokens = max_output_tokens
        if client is None:
            from google import genai

            client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self._client = client

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMRefusal(f"prompt blocked: {feedback.block_reason}")
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        reason_name = getattr(reason, "name", str(reason) if reason else "")
        if reason_name in self._BLOCKED:
            raise LLMRefusal(f"model declined: {reason_name}")
        if reason_name == "MAX_TOKENS":
            raise LLMOutputError("output hit max_output_tokens before finishing")
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise LLMOutputError("no structured output in response")
        try:
            return schema.model_validate_json(text)
        except ValueError as exc:
            raise LLMOutputError(f"output did not match schema: {exc}") from exc


def get_llm(provider: str | None = None) -> StructuredLLM:
    provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}")


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
