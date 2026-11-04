"""LLM layer: client retries/parsing, narration extractor validation, explanations with fallback."""

import json

import httpx
import pandas as pd
import pytest
import respx

from reconlens.config import Settings
from reconlens.llm.client import LLMError, OpenAICompatClient, build_client, parse_json_object
from reconlens.llm.explain import explain_clusters
from reconlens.llm.narration import LLMNarrationExtractor

BASE = "https://llm.test/v1"


def chat(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_parse_json_object_tolerates_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_object("no json here")
    with pytest.raises(LLMError):
        parse_json_object("[1, 2]")


@respx.mock
def test_client_retries_on_429_then_succeeds():
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[httpx.Response(429), httpx.Response(503), chat('{"ok": true}')])
    c = OpenAICompatClient(BASE, "m", "k", max_retries=3)
    assert c.complete_json("s", "u") == {"ok": True}
    assert route.call_count == 3
    sent = json.loads(route.calls[-1].request.content)
    assert sent["response_format"] == {"type": "json_object"} and sent["temperature"] == 0
    assert route.calls[-1].request.headers["authorization"] == "Bearer k"


@respx.mock
def test_client_gives_up_and_does_not_retry_400():
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(400))
    with pytest.raises(LLMError):
        OpenAICompatClient(BASE, "m", "k", max_retries=3).complete_json("s", "u")
    assert route.call_count == 1


@respx.mock
def test_azure_uses_api_key_header_and_version():
    route = respx.post(url__startswith=f"{BASE}/chat/completions").mock(return_value=chat("{}"))
    OpenAICompatClient(BASE, "m", "secret", provider="azure", api_version="2024-10-21").complete_json("s", "u")
    req = route.calls[0].request
    assert req.headers["api-key"] == "secret" and "api-version=2024-10-21" in str(req.url)


def test_build_client_requires_provider_and_key():
    assert build_client(Settings(llm_provider="none", llm_api_key="x", _env_file=None)) is None
    assert build_client(Settings(llm_provider="groq", llm_api_key=None, _env_file=None)) is None
    c = build_client(Settings(llm_provider="groq", llm_api_key="x", _env_file=None))
    assert c is not None and c.model == "llama-3.3-70b-versatile"


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete_json(self, system, user, *, max_tokens=1500):
        self.prompts.append(user)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_narration_extractor_validates_and_aligns():
    llm = Scripted({"items": [
        {"index": 1, "utr": "utib n26034 000001", "confidence": 0.9},
        {"index": 0, "utr": "x!", "confidence": 0.9},          # invalid shape -> dropped
        {"index": 2, "utr": "UTIBN26034000002", "confidence": 7},  # confidence out of range -> dropped
        {"index": 9, "utr": "UTIBN26034000003", "confidence": 0.9},  # index out of range -> ignored
    ]})
    out = LLMNarrationExtractor(llm, batch_size=10).extract_many(["a", "b", "c"])
    assert out[0] is None and out[2] is None
    assert out[1].utr == "UTIBN26034000001"


def test_narration_extractor_batches_and_survives_failed_batch():
    llm = Scripted(LLMError("down"), {"items": [{"index": 0, "utr": "UTIBN26034000009", "confidence": 0.8}]})
    out = LLMNarrationExtractor(llm, batch_size=2).extract_many(["a", "b", "c"])
    assert out[:2] == [None, None] and out[2].utr == "UTIBN26034000009"
    assert len(llm.prompts) == 2


def _exc_frame():
    ev = [{"source": "gateway", "id": "pay_1", "fee_charged": 23.6, "fee_expected": 19.5}]
    return pd.DataFrame([{"type": "mdr_overcharge", "title": "MDR charged above contracted rate", "impact_paise": 486,
                          "recoverable": True, "suggested_action": "Email gateway", "evidence": json.dumps(ev)}])


def test_explanations_without_llm_use_template(cfg):
    out = explain_clusters(_exc_frame(), cfg, None)
    assert out[0]["source"] == "template" and "PayFlow support" in out[0]["draft_email"]
    assert out[0]["cited_evidence_ids"] == ["pay_1"]


def test_explanations_drop_hallucinated_citations(cfg):
    llm = Scripted({"summary": "One card payment was charged 2.36% instead of 1.95%.",
                    "likely_cause": "Wrong pricing plan", "next_step": "Send the email",
                    "draft_email": "Hello PayFlow...", "cited_evidence_ids": ["pay_1", "pay_999"]})
    out = explain_clusters(_exc_frame(), cfg, llm)[0]
    assert out["source"] == "llm" and out["cited_evidence_ids"] == ["pay_1"]
    assert out["dropped_citations"] == ["pay_999"]


def test_explanations_fall_back_on_invalid_llm_output(cfg):
    out = explain_clusters(_exc_frame(), cfg, Scripted({"summary": "x"}))[0]
    assert out["source"] == "template" and "llm_error" in out
