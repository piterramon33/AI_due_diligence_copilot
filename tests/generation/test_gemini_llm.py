from types import SimpleNamespace

import pytest

from app.generation import DraftReport, GeminiLLM, LLMOutputError, LLMRefusal, get_llm


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def client_for(response):
    return SimpleNamespace(models=FakeModels(response))


def resp(finish="STOP", parsed=None, text=None, block=None):
    return SimpleNamespace(
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        parsed=parsed, text=text,
    )


def test_request_shape_and_parsed_output():
    parsed = DraftReport(claims=[], insufficient_evidence=True)
    client = client_for(resp(parsed=parsed))
    out = GeminiLLM(client=client).generate(system="sys", prompt="p", schema=DraftReport)
    assert out is parsed
    kw = client.models.kwargs
    assert kw["model"] == "gemini-3.8-flash"
    assert kw["contents"] == "p"
    assert kw["config"].system_instruction == "sys"
    assert kw["config"].response_mime_type == "application/json"


def test_falls_back_to_json_text():
    text = '{"claims": [], "insufficient_evidence": true}'
    out = GeminiLLM(client=client_for(resp(text=text))).generate(
        system="s", prompt="p", schema=DraftReport)
    assert out.insufficient_evidence


@pytest.mark.parametrize("r, exc", [
    (resp(finish="SAFETY"), LLMRefusal),
    (resp(block="SAFETY"), LLMRefusal),
    (resp(finish="MAX_TOKENS"), LLMOutputError),
    (resp(text="not json"), LLMOutputError),
])
def test_errors(r, exc):
    with pytest.raises(exc):
        GeminiLLM(client=client_for(r)).generate(system="s", prompt="p", schema=DraftReport)


def test_factory_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(get_llm(), GeminiLLM)
