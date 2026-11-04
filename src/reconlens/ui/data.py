"""Pure data helpers for the dashboard: KPIs, filters, ageing, evidence, impact estimate.

Nothing here imports Streamlit, so it is unit-tested directly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import pandas as pd

from reconlens.money import fmt_inr

AGEING_BUCKETS: list[tuple[str, int, int]] = [
    ("0-7 days", 0, 7), ("8-15 days", 8, 15), ("16-30 days", 16, 30), ("31-60 days", 31, 60),
    ("61-90 days", 61, 90), ("90+ days", 91, 10**9),
]
AGEING_ORDER = [b[0] for b in AGEING_BUCKETS] + ["Unknown"]
SEVERITY_ORDER = ["high", "medium", "low"]
STATUSES = ["open", "in_progress", "resolved", "ignored"]
FEE_TYPES = ("mdr_overcharge", "gst_mismatch")
SOURCE_LABELS = {"gateway": "Gateway report", "settlement": "Gateway settlement", "bank": "Bank statement",
                 "ledger": "Order ledger"}
SOURCE_ORDER = list(SOURCE_LABELS)

QUEUE_COLUMNS = ["exception_id", "severity", "title", "key", "impact_inr", "age_days", "ageing", "status",
                 "recoverable", "needs_review", "event_date", "type", "detail", "note"]


def inr_compact(paise: int) -> str:
    """Rupees in lakh/crore for KPI tiles: 94180070 paise -> '₹9.42 L'."""
    rupees = paise / 100
    sign = "-" if rupees < 0 else ""
    r = abs(rupees)
    if r >= 1e7:
        return f"{sign}₹{r / 1e7:.2f} Cr"
    if r >= 1e5:
        return f"{sign}₹{r / 1e5:.2f} L"
    return fmt_inr(int(paise))


def ageing_bucket(age_days: Any) -> str:
    if age_days is None or pd.isna(age_days):
        return "Unknown"
    days = max(int(age_days), 0)
    for label, lo, hi in AGEING_BUCKETS:
        if lo <= days <= hi:
            return label
    return "Unknown"


@dataclass(frozen=True)
class Kpis:
    payments: int
    auto_match_rate: float
    reconciled_paise: int
    unreconciled_paise: int
    recoverable_paise: int
    fee_gst_recoverable_paise: int
    exceptions: int
    settlements: int
    settlement_auto_match_rate: float
    as_of: str | None


def kpis(summary: dict[str, Any]) -> Kpis:
    by_type = summary.get("exceptions_by_type", {})
    fee_gst = sum(max(int(by_type.get(t, {}).get("impact_paise", 0)), 0) for t in FEE_TYPES)
    return Kpis(
        payments=int(summary.get("payments", 0)),
        auto_match_rate=float(summary.get("auto_match_rate", 0.0)),
        reconciled_paise=int(summary.get("reconciled_paise", 0)),
        unreconciled_paise=int(summary.get("unreconciled_paise", 0)),
        recoverable_paise=int(summary.get("recoverable_paise", 0)),
        fee_gst_recoverable_paise=fee_gst,
        exceptions=int(summary.get("exceptions", 0)),
        settlements=int(summary.get("settlements", 0)),
        settlement_auto_match_rate=float(summary.get("settlement_auto_match_rate", 0.0)),
        as_of=summary.get("as_of"),
    )


def exceptions_frame(items: list[dict[str, Any]]) -> pd.DataFrame:
    """API exception items -> queue frame (keeps the raw evidence list in an extra column)."""
    if not items:
        return pd.DataFrame(columns=[*QUEUE_COLUMNS, "impact_paise", "suggested_action", "evidence",
                                     "resolved_by"])
    df = pd.DataFrame(items)
    df["age_days"] = pd.to_numeric(df["age_days"], errors="coerce").astype("Int64")
    df["ageing"] = df["age_days"].map(ageing_bucket)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    for col in ("note", "resolved_by"):
        if col not in df.columns:
            df[col] = None
    return df


def filter_exceptions(df: pd.DataFrame, *, types: list[str] | None = None, severities: list[str] | None = None,
                      statuses: list[str] | None = None, ageing: list[str] | None = None,
                      min_impact_inr: float = 0.0, recoverable_only: bool = False, search: str = ""
                      ) -> pd.DataFrame:
    """Every filter is optional; an empty list means 'no filter on this field'."""
    mask = pd.Series(True, index=df.index)
    if types:
        mask &= df["type"].isin(types)
    if severities:
        mask &= df["severity"].isin(severities)
    if statuses:
        mask &= df["status"].isin(statuses)
    if ageing:
        mask &= df["ageing"].isin(ageing)
    if min_impact_inr > 0:
        mask &= df["impact_inr"].abs() >= min_impact_inr
    if recoverable_only:
        mask &= df["recoverable"].astype(bool)
    term = search.strip().lower()
    if term:
        hay = (df["exception_id"].astype(str) + " " + df["key"].astype(str) + " " + df["detail"].astype(str) + " "
               + df["evidence"].map(lambda ev: " ".join(str(e.get("id", "")) for e in ev or [])))
        mask &= hay.str.lower().str.contains(term, regex=False)
    return df[mask]


def by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Count and ₹ impact per exception type, largest count first."""
    if df.empty:
        return pd.DataFrame(columns=["type", "title", "count", "impact_inr"])
    g = (df.groupby(["type", "title"], as_index=False)
         .agg(count=("exception_id", "size"), impact_inr=("impact_inr", "sum")))
    g["impact_inr"] = g["impact_inr"].round(2)
    return g.sort_values(["count", "impact_inr"], ascending=False).reset_index(drop=True)


def by_ageing(df: pd.DataFrame) -> pd.DataFrame:
    """Count and ₹ impact per ageing bucket, in bucket order (empty buckets included)."""
    if df.empty:
        g = pd.DataFrame(columns=["ageing", "count", "impact_inr"])
    else:
        g = df.groupby("ageing", as_index=False).agg(count=("exception_id", "size"),
                                                     impact_inr=("impact_inr", "sum"))
    full = pd.DataFrame({"ageing": AGEING_ORDER}).merge(g, on="ageing", how="left")
    if full.loc[full["ageing"] == "Unknown", "count"].isna().all():
        full = full[full["ageing"] != "Unknown"]
    full["count"] = full["count"].fillna(0).astype(int)
    full["impact_inr"] = full["impact_inr"].astype(float).fillna(0.0).round(2)
    return full.reset_index(drop=True)


def evidence_by_source(evidence: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Group evidence rows by source system, in a fixed reading order (gateway, settlement, bank, ledger)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in evidence or []:
        src = str(row.get("source", "other"))
        groups.setdefault(src, []).append({k: v for k, v in row.items() if k != "source"})
    order = sorted(groups, key=lambda s: (SOURCE_ORDER.index(s) if s in SOURCE_ORDER else len(SOURCE_ORDER), s))
    return {SOURCE_LABELS.get(s, s.title()): pd.DataFrame(groups[s]) for s in order}


def explanation_for(exc_type: str, explanations: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((e for e in explanations if e.get("cluster") == exc_type), None)


def explanation_label(expl: dict[str, Any]) -> str:
    if expl.get("source") == "llm":
        return "AI explanation"
    if expl.get("llm_error"):
        return "Rule-based explanation (LLM call failed, fell back to template)"
    return "Rule-based explanation"


@dataclass(frozen=True)
class ImpactEstimate:
    auto_matched_units: int
    minutes_per_match: float
    hours_saved: float
    exceptions_to_review: int
    review_hours: float


def impact_estimate(summary: dict[str, Any], minutes_per_match: float, minutes_per_exception: float = 0.0
                    ) -> ImpactEstimate:
    """Hours a person would spend matching what the engine auto-matched (an estimate, not a measurement)."""
    if minutes_per_match < 0 or minutes_per_exception < 0:
        raise ValueError("minutes must be non-negative")
    units = int(summary.get("auto_matched_order_lines", 0)) + int(summary.get("auto_matched_settlements", 0))
    n_exc = int(summary.get("exceptions", 0))
    return ImpactEstimate(
        auto_matched_units=units, minutes_per_match=minutes_per_match,
        hours_saved=round(units * minutes_per_match / 60, 1), exceptions_to_review=n_exc,
        review_hours=round(n_exc * minutes_per_exception / 60, 1),
    )


def queue_export(df: pd.DataFrame, fmt: str = "csv") -> bytes:
    """The current (filtered) queue as CSV or XLSX, with evidence ids flattened."""
    out = df.reindex(columns=[c for c in QUEUE_COLUMNS if c != "ageing"] + ["ageing", "suggested_action"]).copy()
    out["evidence_ids"] = df["evidence"].map(lambda ev: ", ".join(str(e.get("id")) for e in ev or [])) \
        if "evidence" in df.columns else ""
    if fmt == "csv":
        return out.to_csv(index=False).encode()
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            out.to_excel(xw, sheet_name="exceptions", index=False)
        return buf.getvalue()
    raise ValueError(f"unsupported format: {fmt}")
