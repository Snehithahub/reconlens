"""Unit tests for the dashboard's pure data helpers."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from reconlens.ui import data


def item(exc_id: str, type_: str = "mdr_overcharge", *, severity: str = "high", impact: float = 10.0,
         age: int | None = 5, status: str = "open", recoverable: bool = True, evidence=None, detail: str = "x"
         ) -> dict:
    return {
        "exception_id": exc_id, "run_id": "run_1", "type": type_, "title": type_.replace("_", " ").title(),
        "severity": severity, "key": f"key_{exc_id}", "impact_paise": int(impact * 100), "impact_inr": impact,
        "recoverable": recoverable, "needs_review": False, "event_date": "2026-05-01", "age_days": age,
        "detail": detail, "suggested_action": "do something",
        "evidence": evidence if evidence is not None else [{"source": "gateway", "id": f"pay_{exc_id}"}],
        "status": status, "note": None, "resolved_by": None,
    }


@pytest.fixture
def frame() -> pd.DataFrame:
    return data.exceptions_frame([
        item("a", "mdr_overcharge", impact=12.5, age=3),
        item("b", "mdr_overcharge", impact=7.5, age=10, status="resolved"),
        item("c", "missing_settlement", impact=50_000, age=45, severity="high",
             evidence=[{"source": "settlement", "id": "setl_9", "utr": "UTIBN123"}]),
        item("d", "split_credit", severity="low", impact=0, age=None, recoverable=False,
             evidence=[{"source": "bank", "id": "b1"}, {"source": "settlement", "id": "setl_7"},
                       {"source": "bank", "id": "b2"}]),
        item("e", "amount_mismatch", severity="medium", impact=-200, age=100, recoverable=False,
             detail="ledger says 999, gateway 799"),
    ])


@pytest.mark.parametrize(("age", "bucket"), [
    (0, "0-7 days"), (7, "0-7 days"), (8, "8-15 days"), (30, "16-30 days"), (31, "31-60 days"),
    (90, "61-90 days"), (91, "90+ days"), (5000, "90+ days"), (-2, "0-7 days"), (None, "Unknown"),
    (float("nan"), "Unknown"), (pd.NA, "Unknown"),
])
def test_ageing_bucket(age, bucket):
    assert data.ageing_bucket(age) == bucket


def test_inr_compact_uses_lakh_and_crore():
    assert data.inr_compact(94_180_070) == "₹9.42 L"
    assert data.inr_compact(2_083_027_035) == "₹2.08 Cr"
    assert data.inr_compact(1_234_550) == "₹12,345.50"
    assert data.inr_compact(-50_000_000) == "-₹5.00 L"


def test_kpis_from_summary():
    summary = {
        "payments": 5000, "auto_match_rate": 0.9614, "reconciled_paise": 100, "unreconciled_paise": 50,
        "recoverable_paise": 900, "exceptions": 81, "settlements": 282, "settlement_auto_match_rate": 0.95,
        "as_of": "2026-06-04",
        "exceptions_by_type": {"mdr_overcharge": {"impact_paise": 300}, "gst_mismatch": {"impact_paise": 40},
                               "missing_settlement": {"impact_paise": 560}},
    }
    k = data.kpis(summary)
    assert k.fee_gst_recoverable_paise == 340
    assert k.auto_match_rate == pytest.approx(0.9614)
    assert (k.payments, k.exceptions, k.as_of) == (5000, 81, "2026-06-04")


def test_kpis_tolerates_missing_fields():
    k = data.kpis({})
    assert k.payments == 0 and k.fee_gst_recoverable_paise == 0 and k.as_of is None


def test_exceptions_frame_adds_ageing_and_handles_empty(frame):
    assert list(frame["ageing"]) == ["0-7 days", "8-15 days", "31-60 days", "Unknown", "90+ days"]
    empty = data.exceptions_frame([])
    assert empty.empty and {"impact_inr", "evidence", "ageing"} <= set(empty.columns)
    assert data.filter_exceptions(empty, search="x").empty


def test_filters_combine(frame):
    assert set(data.filter_exceptions(frame)["exception_id"]) == set("abcde")
    assert set(data.filter_exceptions(frame, types=["mdr_overcharge"])["exception_id"]) == {"a", "b"}
    assert set(data.filter_exceptions(frame, types=["mdr_overcharge"], statuses=["open"])["exception_id"]) == {"a"}
    assert set(data.filter_exceptions(frame, severities=["low", "medium"])["exception_id"]) == {"d", "e"}
    assert set(data.filter_exceptions(frame, ageing=["Unknown", "90+ days"])["exception_id"]) == {"d", "e"}
    assert set(data.filter_exceptions(frame, recoverable_only=True)["exception_id"]) == {"a", "b", "c"}


def test_min_impact_uses_absolute_value(frame):
    assert set(data.filter_exceptions(frame, min_impact_inr=100)["exception_id"]) == {"c", "e"}


def test_search_matches_ids_detail_and_evidence(frame):
    assert set(data.filter_exceptions(frame, search="utibn123")["exception_id"]) == set()  # utr is not an id
    assert set(data.filter_exceptions(frame, search="setl_9")["exception_id"]) == {"c"}
    assert set(data.filter_exceptions(frame, search="  LEDGER says ")["exception_id"]) == {"e"}
    assert set(data.filter_exceptions(frame, search="key_b")["exception_id"]) == {"b"}
    assert set(data.filter_exceptions(frame, search="(")["exception_id"]) == set()  # not treated as regex


def test_by_type_counts_and_sums(frame):
    bt = data.by_type(frame)
    assert bt.iloc[0]["type"] == "mdr_overcharge"
    row = bt.set_index("type").loc["mdr_overcharge"]
    assert row["count"] == 2 and row["impact_inr"] == pytest.approx(20.0)
    assert data.by_type(frame.iloc[0:0]).empty


def test_by_ageing_keeps_bucket_order_and_zero_buckets(frame):
    ba = data.by_ageing(frame)
    assert list(ba["ageing"]) == data.AGEING_ORDER
    counts = dict(zip(ba["ageing"], ba["count"], strict=True))
    assert counts["0-7 days"] == 1 and counts["16-30 days"] == 0 and counts["Unknown"] == 1
    no_unknown = data.by_ageing(frame[frame["ageing"] != "Unknown"])
    assert "Unknown" not in set(no_unknown["ageing"])
    empty = data.by_ageing(frame.iloc[0:0])
    assert empty["count"].sum() == 0 and len(empty) == len(data.AGEING_BUCKETS)


def test_evidence_grouped_by_source_in_reading_order():
    groups = data.evidence_by_source([
        {"source": "ledger", "id": "o1", "amount": 10},
        {"source": "bank", "id": "b1", "narration": "NEFT PAYFLOW"},
        {"source": "gateway", "id": "p1"},
        {"source": "bank", "id": "b2", "narration": "NEFT"},
        {"id": "x"},
    ])
    assert list(groups) == ["Gateway report", "Bank statement", "Order ledger", "Other"]
    assert list(groups["Bank statement"]["id"]) == ["b1", "b2"]
    assert "source" not in groups["Bank statement"].columns
    assert data.evidence_by_source([]) == {}


def test_explanation_lookup_and_labels():
    expl = [{"cluster": "mdr_overcharge", "source": "llm"}, {"cluster": "gst_mismatch", "source": "template"},
            {"cluster": "short_credit", "source": "template", "llm_error": "timeout"}]
    assert data.explanation_for("mdr_overcharge", expl)["source"] == "llm"
    assert data.explanation_for("duplicate_refund", expl) is None
    assert data.explanation_label(expl[0]) == "AI explanation"
    assert data.explanation_label(expl[1]) == "Rule-based explanation"
    assert "fell back" in data.explanation_label(expl[2])


def test_impact_estimate():
    summary = {"auto_matched_order_lines": 4807, "auto_matched_settlements": 258, "exceptions": 81}
    est = data.impact_estimate(summary, minutes_per_match=2, minutes_per_exception=15)
    assert est.auto_matched_units == 5065
    assert est.hours_saved == pytest.approx(168.8)
    assert est.review_hours == pytest.approx(20.2)
    assert data.impact_estimate(summary, 0).hours_saved == 0
    assert data.impact_estimate({}, 3).auto_matched_units == 0
    with pytest.raises(ValueError):
        data.impact_estimate(summary, -1)


def test_queue_export_csv_and_xlsx(frame):
    csv = data.queue_export(frame, "csv").decode()
    out = pd.read_csv(io.StringIO(csv))
    assert len(out) == 5 and "evidence_ids" in out.columns
    assert out.set_index("exception_id").loc["d", "evidence_ids"] == "b1, setl_7, b2"
    xlsx = pd.read_excel(io.BytesIO(data.queue_export(frame, "xlsx")), sheet_name="exceptions")
    assert list(xlsx["exception_id"]) == list("abcde")
    with pytest.raises(ValueError):
        data.queue_export(frame, "pdf")
