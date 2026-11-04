"""Headless Streamlit test: run the dashboard script, reconcile demo data, open and resolve an exception."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import reconlens.ui

APP = str(Path(reconlens.ui.__file__).with_name("app.py"))


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setenv("RECONLENS_UI_MODE", "local")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_landing_page_without_a_run(app):
    assert not app.exception
    assert app.title[0].value == "Settlement reconciliation"
    assert "in-process" in " ".join(c.value for c in app.sidebar.caption)
    assert len(app.tabs) == 0


def test_demo_run_renders_kpis_queue_and_resolve(app):
    app.sidebar.number_input[0].set_value(300)
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Overview", "Exception queue", "Download", "Business impact"]

    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Auto-match rate"].endswith("%")
    assert metrics["Recoverable"].startswith("₹")
    assert "Estimated hours saved per run" in metrics

    run = app.session_state["run"]
    assert run["summary"]["payments"] == 300
    items = app.session_state["exc_items"]
    assert len(items) == run["summary"]["exceptions"]
    assert app.session_state["explanations"], "rule-based explanations load automatically without an LLM"
    assert any("Rule-based explanation" in m.value for m in app.markdown)

    # Resolve the exception shown in the detail view.
    form_status = next(s for s in app.selectbox if s.label == "Status")
    form_status.set_value("resolved")
    next(t for t in app.text_area if t.label == "Note").input("raised with PayFlow")
    app.button(key="FormSubmitter:resolve_" + items[0]["exception_id"] + "-Save").click().run()
    assert not app.exception
    updated = next(i for i in app.session_state["exc_items"] if i["exception_id"] == items[0]["exception_id"])
    assert updated["status"] == "resolved" and updated["note"] == "raised with PayFlow"


def test_impact_assumption_is_editable(app):
    app.sidebar.button[0].click().run()  # default demo size
    per_match = next(n for n in app.number_input if n.label == "Minutes per manual match")
    per_match.set_value(6.0).run()
    s = app.session_state["run"]["summary"]
    expected = round((s["auto_matched_order_lines"] + s["auto_matched_settlements"]) * 6 / 60, 1)
    assert {m.label: m.value for m in app.metric}["Estimated hours saved per run"] == f"{expected:,.1f}"
