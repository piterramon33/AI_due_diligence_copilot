from types import SimpleNamespace

import pytest

from app.generation import AnthropicLLM, DraftReport, LLMRefusal


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_request_shape_and_parsed_output():
    parsed = DraftReport(claims=[], insufficient_evidence=True)
    client = FakeClient(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed))
    llm = AnthropicLLM(client=client)
    assert llm.generate(system="s", prompt="p", schema=DraftReport) is parsed
    kw = client.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_format"] is DraftReport
    assert kw["output_config"] == {"effort": "high"}
    assert kw["fallbacks"] == "default"
    assert "thinking" not in kw  # adaptive thinking is the model default


def test_refusal_raises():
    client = FakeClient(SimpleNamespace(stop_reason="refusal", parsed_output=None, stop_details=None))
    with pytest.raises(LLMRefusal):
        AnthropicLLM(client=client).generate(system="s", prompt="p", schema=DraftReport)
