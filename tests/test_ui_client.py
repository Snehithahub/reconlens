"""Tests for the dashboard's API client: error mapping, pagination, fallback and an in-process round trip."""

from __future__ import annotations

import io
import json

import httpx
import pandas as pd
import pytest

from reconlens.config import Settings
from reconlens.ui.client import PAGE_SIZE, ApiError, ReconClient, Upload, connect, local_client


def mock_client(handler) -> ReconClient:
    http = httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(handler))
    return ReconClient(http, "api", "http://api.test")


def test_error_envelope_becomes_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {"code": "invalid_input", "message": "bank statement could not "
                                                   "be read", "details": ["missing required columns: ['credit']"]}})

    with pytest.raises(ApiError) as e:
        mock_client(handler).get_run("run_x")
    assert e.value.status == 422
    assert e.value.message == "bank statement could not be read"
    assert e.value.details == ["missing required columns: ['credit']"]


def test_non_json_error_body():
    with pytest.raises(ApiError) as e:
        mock_client(lambda r: httpx.Response(502, text="Bad Gateway")).health()
    assert e.value.status == 502 and "Bad Gateway" in e.value.message


def test_transport_failure_is_reported_as_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ApiError) as e:
        mock_client(handler).list_runs()
    assert e.value.status == 0 and "unreachable" in e.value.message


def test_exceptions_follow_pagination():
    total = PAGE_SIZE + 5
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        seen.append(offset)
        n = min(PAGE_SIZE, total - offset)
        return httpx.Response(200, json={"total": total, "items": [{"exception_id": f"e{offset + i}"}
                                                                   for i in range(n)]})

    items = mock_client(handler).exceptions("run_1")
    assert len(items) == total and seen == [0, PAGE_SIZE]
    assert items[-1]["exception_id"] == f"e{total - 1}"


def test_upload_sends_three_files_and_flags():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["force"] = request.url.params["force"]
        captured["body"] = request.content
        return httpx.Response(201, json={"run_id": "run_1", "reused": False})

    c = mock_client(handler)
    c.run_upload(Upload("g.csv", b"gw"), Upload("b.xlsx", b"bk"), Upload("l.csv", b"ld"), force=True)
    body = captured["body"].decode(errors="replace")
    assert captured["force"] == "true"
    for field, name in (("gateway_file", "g.csv"), ("bank_file", "b.xlsx"), ("ledger_file", "l.csv")):
        assert f'name="{field}"; filename="{name}"' in body


def test_resolve_sends_nulls_for_blank_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json.loads(request.content))

    out = mock_client(handler).resolve("exc_1", "resolved", note="", resolved_by="")
    assert out == {"status": "resolved", "note": None, "resolved_by": None}


def test_connect_falls_back_to_in_process_when_api_is_down():
    c = connect("http://127.0.0.1:9", "auto")
    assert c.mode == "local"
    assert c.health()["status"] == "ok"


def test_connect_api_mode_requires_server():
    with pytest.raises(ApiError):
        connect("http://127.0.0.1:9", "api")


@pytest.fixture(scope="module")
def local():
    return local_client(":memory:", Settings(_env_file=None, llm_provider="none"))


def test_in_process_round_trip(local):
    run = local.run_demo(400, 7)
    run_id = run["run_id"]
    assert run["summary"]["payments"] == 400 and not run["reused"]
    assert local.run_demo(400, 7)["reused"] is True  # same inputs and config are not reconciled twice

    items = local.exceptions(run_id)
    assert len(items) == run["summary"]["exceptions"] > 0
    first = items[0]
    assert {"exception_id", "type", "impact_inr", "age_days", "evidence", "status"} <= set(first)

    updated = local.resolve(first["exception_id"], "resolved", "ticket PF-1", "asha")
    assert (updated["status"], updated["note"], updated["resolved_by"]) == ("resolved", "ticket PF-1", "asha")
    assert next(i for i in local.exceptions(run_id) if i["exception_id"] == first["exception_id"])["status"] \
        == "resolved"

    csv = pd.read_csv(io.BytesIO(local.report(run_id, "csv")))
    assert len(csv) == len(items) and "resolved" in set(csv["status"])
    xlsx = pd.ExcelFile(io.BytesIO(local.report(run_id, "xlsx")))
    assert {"exceptions", "settlements"} <= set(xlsx.sheet_names)

    assert local.explanations(run_id) == []
    expl = local.explain(run_id, use_llm=False)
    assert not expl["llm_used"] and {e["source"] for e in expl["items"]} == {"template"}
    assert len(local.explanations(run_id)) == len(expl["items"])
    assert any(r["run_id"] == run_id for r in local.list_runs())


def test_in_process_bad_upload_is_a_clear_error(local):
    bad = Upload("bank.csv", b"date,narration\n01-05-2026,NEFT\n")
    good = Upload("x.csv", b"order_id,amount,status\nA,1,paid\n")
    with pytest.raises(ApiError) as e:
        local.run_upload(good, bad, good)
    assert e.value.status == 422


def test_in_process_llm_requested_without_llm(local):
    with pytest.raises(ApiError) as e:
        local.run_demo(100, 1, use_llm=True)
    assert e.value.status == 503
