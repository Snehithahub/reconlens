import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from reconlens.api.app import create_app
from reconlens.config import Settings
from reconlens.store import Store


@pytest.fixture(scope="module")
def client():
    app = create_app(Settings(llm_provider="none", demo_payments=1000, _env_file=None), store=Store(":memory:"))
    return TestClient(app)


@pytest.fixture(scope="module")
def run_id(client):
    r = client.post("/runs", data={"demo": "true"})
    assert r.status_code == 201
    return r.json()["run_id"]


def test_health_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json() == {"status": "ready", "llm_enabled": False, "gateway": "PayFlow"}


def test_run_summary_and_reuse(client, run_id):
    run = client.get(f"/runs/{run_id}").json()
    assert 0 < run["summary"]["auto_match_rate"] < 1 and run["summary"]["exceptions"] > 0
    again = client.post("/runs", data={"demo": "true"})
    assert again.status_code == 200 and again.json()["reused"] is True
    assert any(r["run_id"] == run_id for r in client.get("/runs").json())


def test_exception_filters_and_resolve(client, run_id):
    body = client.get(f"/runs/{run_id}/exceptions", params={"type": "mdr_overcharge", "limit": 2}).json()
    assert body["total"] >= 2 and len(body["items"]) == 2
    item = body["items"][0]
    assert item["type"] == "mdr_overcharge" and item["evidence"][0]["source"] == "gateway"
    assert item["impact_inr"] == item["impact_paise"] / 100
    high = client.get(f"/runs/{run_id}/exceptions", params={"severity": "high"}).json()
    assert all(i["severity"] == "high" for i in high["items"])
    r = client.post(f"/exceptions/{item['exception_id']}/resolve", json={"status": "resolved", "note": "credit note"})
    assert r.json()["status"] == "resolved" and r.json()["note"] == "credit note"
    resolved = client.get(f"/runs/{run_id}/exceptions", params={"status": "resolved"}).json()
    assert [i["exception_id"] for i in resolved["items"]] == [item["exception_id"]]


def test_resolve_validation_and_404(client):
    assert client.post("/exceptions/nope/resolve", json={"status": "resolved"}).status_code == 404
    r = client.post("/exceptions/nope/resolve", json={"status": "maybe"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_export_csv_and_xlsx(client, run_id):
    csv = client.get(f"/runs/{run_id}/report", params={"format": "csv"})
    df = pd.read_csv(io.StringIO(csv.text))
    assert {"exception_id", "type", "impact_inr", "suggested_action"} <= set(df.columns) and len(df) > 0
    xlsx = client.get(f"/runs/{run_id}/report", params={"format": "xlsx"})
    sheets = pd.read_excel(io.BytesIO(xlsx.content), sheet_name=None)
    assert set(sheets) == {"exceptions", "settlements"}


def test_upload_files_roundtrip(client, cfg):
    from reconlens.config import Settings as S
    from reconlens.service import demo_inputs

    gw, bank, led = demo_inputs(S(_env_file=None), cfg, n=300, seed=9)
    files = {"gateway_file": (gw.name, gw.content), "bank_file": (bank.name, bank.content),
             "ledger_file": (led.name, led.content)}
    r = client.post("/runs", files=files)
    assert r.status_code == 201 and r.json()["source"] == "upload"


def test_bad_upload_is_422_with_details(client):
    files = {"gateway_file": ("g.csv", b"foo,bar\n1,2\n"), "bank_file": ("b.csv", b"x\n1\n"),
             "ledger_file": ("l.csv", b"y\n1\n")}
    r = client.post("/runs", files=files)
    assert r.status_code == 422 and "missing required columns" in r.json()["error"]["details"][0]


def test_missing_inputs_and_llm_not_configured(client):
    assert client.post("/runs").status_code == 400
    assert client.post("/runs", data={"demo": "true", "use_llm": "true"}).status_code == 503


def test_explain_without_llm_uses_templates(client, run_id):
    r = client.post(f"/runs/{run_id}/explain").json()
    assert r["llm_used"] is False and all(i["source"] == "template" for i in r["items"])
    assert len(client.get(f"/runs/{run_id}/explanations").json()["items"]) == len(r["items"])


def test_unknown_run_and_type(client, run_id):
    assert client.get("/runs/run_missing").status_code == 404
    assert client.get(f"/runs/{run_id}/exceptions", params={"type": "bogus"}).status_code == 422
