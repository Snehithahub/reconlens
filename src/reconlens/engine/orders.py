"""Order ledger <-> gateway payment matching by order_id."""

from __future__ import annotations

import pandas as pd

from reconlens.engine.types import ExceptionType, ReconException
from reconlens.money import to_rupees

PAID_STATUSES = {"paid", "refunded", "partially_refunded", "shipped", "delivered", "completed", "fulfilled"}


def order_exceptions(pays: pd.DataFrame, ledger: pd.DataFrame) -> tuple[list[ReconException], pd.Series]:
    """Return exceptions and a per-payment 'order_ok' flag aligned to pays.index."""
    out: list[ReconException] = []
    led = ledger.drop_duplicates("order_id", keep="first").set_index("order_id")
    j = pays.join(led[["status", "amount_paise", "payment_mode"]].rename(
        columns={"amount_paise": "order_amount_paise", "status": "order_status"}), on="order_id")
    order_ok = pd.Series(True, index=pays.index)

    def ev(r) -> list[dict]:
        return [
            {"source": "gateway", "id": r.payment_id, "order_id": r.order_id, "amount": to_rupees(r.amount_paise),
             "created_at": str(r.created_at), "settlement_id": r.settlement_id},
            {"source": "ledger", "id": r.order_id,
             "status": None if pd.isna(r.order_status) else r.order_status,
             "amount": None if pd.isna(r.order_amount_paise) else to_rupees(int(r.order_amount_paise))},
        ]

    dup = j[j.duplicated("order_id", keep="first") & (j["order_id"] != "")]
    for r in dup.itertuples():
        order_ok[r.Index] = False
        out.append(ReconException(ExceptionType.DUPLICATE_CAPTURE, r.payment_id, int(r.amount_paise),
                                  r.created_at.date(), ev(r), f"second captured payment for order {r.order_id}"))
    for r in j[j["order_status"].isna()].itertuples():
        order_ok[r.Index] = False
        out.append(ReconException(ExceptionType.PAYMENT_WITHOUT_ORDER, r.payment_id, int(r.amount_paise),
                                  r.created_at.date(), ev(r), f"order {r.order_id or '(blank)'} not in ledger"))
    present = j[j["order_status"].notna()]
    for r in present[~present["order_status"].isin(PAID_STATUSES)].itertuples():
        order_ok[r.Index] = False
        out.append(ReconException(ExceptionType.PAYMENT_NOT_RECORDED, r.payment_id, int(r.amount_paise),
                                  r.created_at.date(), ev(r),
                                  f"payment captured but order {r.order_id} is '{r.order_status}'"))
    mism = present[present["order_amount_paise"] != present["amount_paise"]]
    for r in mism.itertuples():
        order_ok[r.Index] = False
        out.append(ReconException(ExceptionType.AMOUNT_MISMATCH, r.payment_id,
                                  int(r.amount_paise - r.order_amount_paise), r.created_at.date(), ev(r),
                                  f"paid {to_rupees(r.amount_paise)} vs order {to_rupees(int(r.order_amount_paise))}"))

    # Prepaid orders the store thinks are paid, with no captured payment in the report period.
    paid_orders = set(pays["order_id"])
    lo, hi = pays["created_at"].min().normalize(), pays["created_at"].max().normalize()
    cand = ledger[(ledger["payment_mode"] != "cod") & ledger["status"].isin(PAID_STATUSES)
                  & ~ledger["order_id"].isin(paid_orders)]
    if cand["order_date"].notna().any():
        cand = cand[cand["order_date"].isna() | cand["order_date"].between(lo, hi)]
    for r in cand.itertuples(index=False):
        out.append(ReconException(
            ExceptionType.ORDER_PAID_NO_PAYMENT, r.order_id, int(r.amount_paise),
            r.order_date.date() if pd.notna(r.order_date) else None,
            [{"source": "ledger", "id": r.order_id, "status": r.status, "amount": to_rupees(r.amount_paise),
              "payment_ref": r.payment_ref or None, "order_date": str(r.order_date.date()) if pd.notna(r.order_date)
              else None}],
            f"order {r.order_id} is '{r.status}' but no captured payment exists"
            + (f" (ledger cites {r.payment_ref})" if r.payment_ref else ""),
        ))
    return out, order_ok
