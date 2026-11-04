"""Checks that need only the gateway report: fee/GST recomputation, refunds, settlement totals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reconlens.bankcal import BankCalendar
from reconlens.config import ReconConfig
from reconlens.engine.types import ExceptionType, ReconException
from reconlens.money import gst_on, to_rupees


def expected_fees(pays: pd.DataFrame, cfg: ReconConfig) -> pd.DataFrame:
    """Add expected_fee / gst columns from the contracted rate card (integer paise math)."""
    rc = cfg.rate_card
    m = pays["method"]
    known = m.isin(list(rc.methods))
    rate = m.map({k: v.rate_ppm for k, v in rc.methods.items()}).fillna(0).astype("int64")
    flat = m.map({k: v.flat_paise for k, v in rc.methods.items()}).fillna(0).astype("int64")
    mn = m.map({k: v.min_fee_paise for k, v in rc.methods.items()}).fillna(0).astype("int64")
    amount = pays["amount_paise"].to_numpy(dtype="int64")
    exp_fee = np.maximum((amount * rate.to_numpy() + 500_000) // 1_000_000 + flat.to_numpy(), mn.to_numpy())
    out = pays.copy()
    out["rate_known"] = known.to_numpy()
    out["expected_fee_paise"] = exp_fee
    out["gst_on_charged_fee_paise"] = gst_on(out["fee_paise"].to_numpy(), rc.gst_bps)
    out["gst_on_expected_fee_paise"] = gst_on(exp_fee, rc.gst_bps)
    return out


def fee_exceptions(pays: pd.DataFrame, cfg: ReconConfig) -> list[ReconException]:
    tol = cfg.tolerances
    rc = cfg.rate_card
    out: list[ReconException] = []
    fee_diff = pays["fee_paise"] - pays["expected_fee_paise"]
    over = pays[pays["rate_known"] & (fee_diff > tol.fee_paise)]
    for r in over.itertuples(index=False):
        impact = (r.fee_paise + r.gst_on_charged_fee_paise) - (r.expected_fee_paise + r.gst_on_expected_fee_paise)
        contracted = float(rc.methods[r.method].percent)
        effective = round(r.fee_paise / r.amount_paise * 100, 3) if r.amount_paise else 0.0
        out.append(ReconException(
            ExceptionType.MDR_OVERCHARGE, r.payment_id, int(impact), r.created_at.date(),
            [{"source": "gateway", "id": r.payment_id, "method": r.method, "amount": to_rupees(r.amount_paise),
              "fee_charged": to_rupees(r.fee_paise), "fee_expected": to_rupees(r.expected_fee_paise),
              "contracted_rate_pct": contracted, "effective_rate_pct": effective,
              "settlement_id": r.settlement_id}],
            f"{r.method} fee {effective}% vs contracted {contracted}%",
        ))
    gst_diff = pays["tax_paise"] - pays["gst_on_charged_fee_paise"]
    bad = pays[gst_diff.abs() > tol.gst_paise]
    for r, d in zip(bad.itertuples(index=False), gst_diff[gst_diff.abs() > tol.gst_paise], strict=True):
        out.append(ReconException(
            ExceptionType.GST_MISMATCH, r.payment_id, int(d), r.created_at.date(),
            [{"source": "gateway", "id": r.payment_id, "fee_charged": to_rupees(r.fee_paise),
              "gst_charged": to_rupees(r.tax_paise), "gst_expected": to_rupees(r.gst_on_charged_fee_paise),
              "gst_rate_pct": float(rc.gst_rate_percent), "settlement_id": r.settlement_id}],
            f"GST {to_rupees(r.tax_paise)} vs {float(rc.gst_rate_percent)}% of fee = "
            f"{to_rupees(r.gst_on_charged_fee_paise)}",
        ))
    return out


def refund_exceptions(gw: pd.DataFrame, ledger_orders: set[str]) -> list[ReconException]:
    out: list[ReconException] = []
    refs = gw[gw["entity_type"] == "refund"].sort_values(["created_at", "entity_id"])
    first = refs.groupby(["payment_id", "amount_paise"])["entity_id"].transform("first")
    dup = refs[refs.duplicated(["payment_id", "amount_paise"], keep="first")]
    for r in dup.itertuples():
        orig = str(first.loc[r.Index])
        out.append(ReconException(
            ExceptionType.DUPLICATE_REFUND, r.entity_id, int(r.amount_paise), r.created_at.date(),
            [{"source": "gateway", "id": r.entity_id, "payment_id": r.payment_id, "amount": to_rupees(r.amount_paise),
              "created_at": str(r.created_at), "duplicate_of": orig, "settlement_id": r.settlement_id}],
            f"second refund of {to_rupees(r.amount_paise)} on {r.payment_id}",
        ))
    orphan = refs[~refs["order_id"].isin(ledger_orders)]
    for r in orphan.itertuples(index=False):
        out.append(ReconException(
            ExceptionType.REFUND_WITHOUT_ORDER, r.entity_id, int(r.amount_paise), r.created_at.date(),
            [{"source": "gateway", "id": r.entity_id, "payment_id": r.payment_id, "order_id": r.order_id or None,
              "amount": to_rupees(r.amount_paise), "created_at": str(r.created_at)}],
            f"refund {r.entity_id} references order {r.order_id or '(blank)'} not in the ledger",
        ))
    return out


def build_settlements(gw: pd.DataFrame, cfg: ReconConfig, cal: BankCalendar) -> pd.DataFrame:
    """One row per settlement batch with its components, net payout and contracted due date."""
    rc = cfg.rate_card
    g = gw[gw["settlement_id"] != ""].copy()
    is_pay = g["entity_type"] == "payment"
    g["net_paise"] = np.where(is_pay, g["amount_paise"] - g["fee_paise"] - g["tax_paise"], -g["amount_paise"])
    default_t = max((m.settlement_days for m in rc.methods.values()), default=2)
    t_days = g["method"].map({k: v.settlement_days for k, v in rc.methods.items()}).fillna(default_t).astype(int)
    pairs = pd.Series(list(zip(g["created_at"].dt.date, t_days, strict=True)), index=g.index)
    due_map = {p: cal.add_business_days(p[0], int(p[1])) for p in set(pairs)}
    g["due_date"] = pairs.map(due_map).where(is_pay)

    def comp(kind: str, col: str) -> pd.Series:
        return g[g["entity_type"] == kind].groupby("settlement_id")[col].sum()

    grp = g.groupby("settlement_id", sort=True)
    s = pd.DataFrame({
        "account": grp["account"].first(),
        "settled_at": grp["settled_at"].min(),
        "utr": grp["utr"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else ""),
        "n_payments": grp["entity_type"].agg(lambda x: int((x == "payment").sum())),
        "net_paise": grp["net_paise"].sum(),
        "expected_by": grp["due_date"].max(),
    })
    s["gross_paise"] = comp("payment", "amount_paise").reindex(s.index).fillna(0).astype("int64")
    s["fees_paise"] = comp("payment", "fee_paise").reindex(s.index).fillna(0).astype("int64")
    s["tax_paise"] = comp("payment", "tax_paise").reindex(s.index).fillna(0).astype("int64")
    s["refunds_paise"] = comp("refund", "amount_paise").reindex(s.index).fillna(0).astype("int64")
    s["chargebacks_paise"] = comp("chargeback", "amount_paise").reindex(s.index).fillna(0).astype("int64")
    s["net_paise"] = s["net_paise"].astype("int64")
    s["settled_at"] = s["settled_at"].dt.date
    return s.reset_index()
