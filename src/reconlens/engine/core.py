"""Reconciliation orchestrator: gateway checks, bank matching, order matching, SLA, summary."""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from reconlens.bankcal import BankCalendar
from reconlens.config import ReconConfig
from reconlens.engine.bank_match import BankMatcher, NarrationExtractor
from reconlens.engine.gateway_checks import (
    build_settlements,
    expected_fees,
    fee_exceptions,
    refund_exceptions,
)
from reconlens.engine.orders import order_exceptions
from reconlens.engine.types import TAXONOMY, EngineResult, ExceptionType, ReconException
from reconlens.money import to_rupees

log = logging.getLogger(__name__)

ENGINE_VERSION = "1.0.0"
PAYMENT_LEVEL = {ExceptionType.MDR_OVERCHARGE, ExceptionType.GST_MISMATCH, ExceptionType.PAYMENT_NOT_RECORDED,
                 ExceptionType.PAYMENT_WITHOUT_ORDER, ExceptionType.AMOUNT_MISMATCH, ExceptionType.DUPLICATE_CAPTURE}
SETTLED_OK = {"auto"}


def _sla_exceptions(settlements: pd.DataFrame, credit_dates: dict, cal: BankCalendar, grace: int
                    ) -> list[ReconException]:
    out = []
    for r in settlements.itertuples(index=False):
        if pd.isna(r.expected_by) or r.settlement_id not in credit_dates:
            continue
        expected = r.expected_by
        actual = max(credit_dates[r.settlement_id], r.settled_at)
        late = cal.business_days_between(expected, actual)
        if late > grace:
            out.append(ReconException(
                ExceptionType.SETTLEMENT_DELAYED, r.settlement_id, int(r.net_paise), actual,
                [{"source": "settlement", "id": r.settlement_id, "expected_by": expected.isoformat(),
                  "settled_at": r.settled_at.isoformat(), "bank_credit_date": credit_dates[r.settlement_id].isoformat(),
                  "business_days_late": late, "net_amount": to_rupees(int(r.net_paise))}],
                f"credited {late} business days after contracted date {expected.isoformat()}",
            ))
    return out


def reconcile(
    gateway: pd.DataFrame,
    bank: pd.DataFrame,
    ledger: pd.DataFrame,
    cfg: ReconConfig,
    *,
    extractor: NarrationExtractor | None = None,
    min_confidence: float = 0.7,
) -> EngineResult:
    t0 = time.perf_counter()
    cal = BankCalendar(cfg.bank_holidays)
    gw = gateway.copy()
    pays = expected_fees(gw[gw["entity_type"] == "payment"], cfg)

    exceptions: list[ReconException] = []
    exceptions += fee_exceptions(pays, cfg)
    exceptions += refund_exceptions(gw, set(ledger["order_id"]))
    order_exc, order_ok = order_exceptions(pays, ledger)
    exceptions += order_exc

    settlements = build_settlements(gw, cfg, cal)
    matcher = BankMatcher(settlements, bank, cfg, cal, extractor, min_confidence)
    bm = matcher.run()
    exceptions += bm.exceptions
    exceptions += _sla_exceptions(settlements, bm.settlement_credit_date, cal, cfg.tolerances.sla_grace_business_days)

    settlements["status"] = settlements["settlement_id"].map(bm.settlement_status).fillna("unknown")
    sid_to_match = {s: m for m in bm.matches for s in m.settlement_ids}
    settlements["bank_txn_ids"] = settlements["settlement_id"].map(
        lambda s: ",".join(sid_to_match[s].bank_txn_ids) if s in sid_to_match else "")
    settlements["match_method"] = settlements["settlement_id"].map(
        lambda s: sid_to_match[s].method if s in sid_to_match else None)

    # Per-payment status: reconciled only when the order, fees and the settlement payout all check out.
    flagged = defaultdict(list)
    for e in exceptions:
        if e.type in PAYMENT_LEVEL:
            flagged[e.key].append(str(e.type))
    setl_status = pays["settlement_id"].map(bm.settlement_status).fillna("unsettled")
    pay_status = pd.Series("exception", index=pays.index)
    pay_status[setl_status.isin(["in_transit", "unsettled"])] = "pending"
    pay_status[setl_status == "needs_review"] = "needs_review"
    pay_status[setl_status.isin(SETTLED_OK)] = "reconciled"
    has_exc = pays["payment_id"].isin(flagged.keys()) | ~order_ok
    pay_status[has_exc] = "exception"
    pays = pays.assign(settlement_status=setl_status, recon_status=pay_status,
                       exception_types=pays["payment_id"].map(lambda p: ",".join(flagged.get(p, []))))

    bank_out = bank.merge(bm.credits, on="txn_id", how="left")
    bank_out["gateway_related"] = bank_out["gateway_related"].astype("boolean").fillna(False).astype(bool)
    bank_out["match_status"] = bank_out["match_status"].fillna("debit")

    summary = _summary(pays, settlements, bank_out, exceptions, bm, ledger)
    summary["engine_version"] = ENGINE_VERSION
    summary["runtime_s"] = round(time.perf_counter() - t0, 3)
    log.info("reconciled %d payments, %d settlements, %d exceptions in %.2fs", len(pays), len(settlements),
             len(exceptions), summary["runtime_s"])
    return EngineResult(payments=pays, gateway=gw, settlements=settlements, bank=bank_out, orders=ledger,
                        matches=bm.matches, exceptions=exceptions, summary=summary)


def _summary(pays, settlements, bank, exceptions, bm, ledger) -> dict[str, Any]:
    n = len(pays)
    rec = pays["recon_status"] == "reconciled"
    gross = int(pays["amount_paise"].sum())
    by_type: dict[str, dict[str, Any]] = {}
    counts = Counter(e.type for e in exceptions)
    for t, c in counts.items():
        impact = sum(e.impact_paise for e in exceptions if e.type == t)
        by_type[str(t)] = {"count": c, "impact_paise": impact, "severity": str(TAXONOMY[t].severity),
                           "recoverable": TAXONOMY[t].recoverable, "title": TAXONOMY[t].title}
    recoverable = sum(max(e.impact_paise, 0) for e in exceptions if TAXONOMY[e.type].recoverable)
    st = settlements["status"].value_counts().to_dict()
    payable = settlements[settlements["status"] != "no_payout"]
    s_auto = int((payable["status"] == "auto").sum())
    dates = [d for d in [bank["txn_date"].max(), pd.to_datetime(settlements["settled_at"]).max()] if pd.notna(d)]
    gw_bank = bank[bank["gateway_related"]]
    return {
        "as_of": max(dates).date().isoformat() if dates else None,
        "payments": n,
        "payments_reconciled": int(rec.sum()),
        "auto_match_rate": round(rec.mean(), 4) if n else 0.0,
        "payment_status_counts": pays["recon_status"].value_counts().to_dict(),
        "gross_paise": gross,
        "reconciled_paise": int(pays.loc[rec, "amount_paise"].sum()),
        "unreconciled_paise": gross - int(pays.loc[rec, "amount_paise"].sum()),
        "settlements": len(settlements),
        "settlement_status_counts": {str(k): int(v) for k, v in st.items()},
        "settlement_auto_match_rate": round(s_auto / len(payable), 4) if len(payable) else 0.0,
        "settled_net_paise": int(payable["net_paise"].sum()),
        "bank_matched_net_paise": int(payable.loc[payable["status"].isin(["auto", "needs_review"]), "net_paise"].sum()),
        "bank_credits": int((bank["credit_paise"] > 0).sum()),
        "gateway_bank_credits": len(gw_bank),
        "gateway_credits_matched": int(gw_bank["match_status"].isin(["auto", "needs_review", "short", "excess"]).sum()),
        "ledger_orders": len(ledger),
        "matches": len(bm.matches),
        "matches_needs_review": sum(1 for m in bm.matches if m.status == "needs_review"),
        "exceptions": len(exceptions),
        "exceptions_by_type": by_type,
        "recoverable_paise": recoverable,
        "narration": bm.narration_stats,
        # Units of work a person would otherwise match by hand (used by the business-impact estimate).
        "auto_matched_order_lines": int(rec.sum()),
        "auto_matched_settlements": s_auto,
    }
