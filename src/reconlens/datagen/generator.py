"""Seeded synthetic data for a D2C merchant on the fictional PayFlow gateway.

Produces the three inputs finance ops reconcile by hand (gateway report, bank
statement, order ledger) plus gold labels for every injected discrepancy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from reconlens.bankcal import BankCalendar
from reconlens.config import ReconConfig
from reconlens.money import gst_on

METHODS = ["upi", "card", "netbanking", "wallet"]
METHOD_P = [0.55, 0.25, 0.10, 0.10]
PRICE_POINTS = np.array([199, 299, 349, 399, 499, 599, 699, 799, 899, 999, 1199, 1299, 1499, 1999, 2499, 2999,
                         3999, 4999])
UTR_BANKS = ["UTIB", "HDFC"]


@dataclass
class GenConfig:
    n_payments: int = 10_000
    start: date = date(2026, 1, 1)
    days: int = 120
    accounts: tuple[str, ...] = ("WEB", "APP", "B2B")
    seed: int = 42
    # Payment-level discrepancy rates (per payment), each with a floor.
    mdr_overcharge_rate: float = 0.002
    gst_mismatch_rate: float = 0.001
    duplicate_refund_rate: float = 0.0005
    refund_without_order_rate: float = 0.0003
    order_paid_no_payment_rate: float = 0.0005
    payment_not_recorded_rate: float = 0.001
    min_per_type: int = 3
    # Settlement-level discrepancy rates (per settlement).
    settlement_type_rate: float = 0.02
    # Narration variants on otherwise clean credits.
    narration_absent_rate: float = 0.03
    narration_hard_rate: float = 0.04
    narration_fragment_rate: float = 0.05
    refund_rate: float = 0.04
    chargeback_rate: float = 0.003


@dataclass
class GeneratedData:
    gateway: pd.DataFrame
    bank: pd.DataFrame
    ledger: pd.DataFrame
    gold: dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir) -> None:
        import json
        from pathlib import Path

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.gateway.to_csv(d / "gateway_report.csv", index=False)
        self.bank.to_csv(d / "bank_statement.csv", index=False)
        self.ledger.to_csv(d / "order_ledger.csv", index=False)
        (d / "gold.json").write_text(json.dumps(self.gold, indent=1, default=str))


def _rupees(paise: int) -> str:
    sign = "-" if paise < 0 else ""
    r, p = divmod(abs(int(paise)), 100)
    return f"{sign}{r}.{p:02d}"


class _Ids:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.seen: set[str] = set()

    def make(self, prefix: str, n: int = 1) -> list[str]:
        out: list[str] = []
        while len(out) < n:
            for v in self.rng.integers(0, 2**56, size=n - len(out), dtype=np.uint64):
                s = f"{prefix}_{int(v):014x}"
                if s not in self.seen:
                    self.seen.add(s)
                    out.append(s)
        return out


def _count(rate: float, n: int, floor: int) -> int:
    return max(floor, round(rate * n))


def generate(cfg: GenConfig, recon: ReconConfig) -> GeneratedData:
    rng = np.random.default_rng(cfg.seed)
    ids = _Ids(rng)
    cal = BankCalendar(recon.bank_holidays)
    rc = recon.rate_card
    labels: list[dict[str, Any]] = []

    def label(kind: str, key: str, impact: int = 0, **detail: Any) -> None:
        labels.append({"type": kind, "key": key, "impact_paise": int(impact), **detail})

    # ---- payments -------------------------------------------------------------------------
    n = cfg.n_payments
    day_off = rng.integers(0, cfg.days, n)
    secs = rng.integers(0, 86_400, n)
    created = pd.to_datetime(cfg.start) + pd.to_timedelta(day_off, unit="D") + pd.to_timedelta(secs, unit="s")
    method = rng.choice(METHODS, size=n, p=METHOD_P)
    acc_w = np.array([0.6, 0.3, 0.1][: len(cfg.accounts)] + [0.1] * max(0, len(cfg.accounts) - 3))
    account = rng.choice(list(cfg.accounts), size=n, p=acc_w / acc_w.sum())
    qty = np.minimum(rng.geometric(0.6, n), 4)
    base = rng.choice(PRICE_POINTS, n) * qty * np.where(account == "B2B", 8, 1)
    shipping = rng.choice([0, 49, 99], n, p=[0.7, 0.2, 0.1])
    disc = rng.choice([0, 5, 10], n, p=[0.7, 0.2, 0.1])
    amount = ((base * 100) * (100 - disc) // 100 + shipping * 100).astype("int64")

    rate_ppm = np.array([rc.methods[m].rate_ppm for m in method])
    flat = np.array([rc.methods[m].flat_paise for m in method])
    min_fee = np.array([rc.methods[m].min_fee_paise for m in method])
    fee = np.maximum((amount * rate_ppm + 500_000) // 1_000_000 + flat, min_fee)
    # The gateway occasionally rounds differently (truncate / ceil): a 1-paise wobble the
    # engine must tolerate rather than flag.
    wobble = rng.choice([0, -1, 1], n, p=[0.97, 0.02, 0.01])
    fee = np.maximum(fee + wobble, 0)
    tax = gst_on(fee, rc.gst_bps)

    pay_ids = ids.make("pay", n)
    order_ids = ids.make("order", n)
    pay = pd.DataFrame({
        "payment_id": pay_ids, "order_id": order_ids, "method": method, "account": account,
        "amount": amount, "fee": fee.astype("int64"), "tax": np.asarray(tax, dtype="int64"), "created_at": created,
    })
    pay["created_date"] = pay["created_at"].dt.date
    pay["ledger_status"] = "paid"

    # ---- refunds and chargebacks -------------------------------------------------------------
    refund_idx = rng.choice(n, size=int(cfg.refund_rate * n), replace=False)
    full = rng.random(len(refund_idx)) < 0.7
    frac = rng.uniform(0.1, 0.9, len(refund_idx))
    ref_amt = np.where(full, amount[refund_idx], (amount[refund_idx] * frac // 100 * 100).astype("int64"))
    ref_amt = np.maximum(ref_amt, 100)
    ref_created = created[refund_idx] + pd.to_timedelta(rng.integers(1, 16, len(refund_idx)), unit="D") \
        + pd.to_timedelta(rng.integers(0, 86_400, len(refund_idx)), unit="s")
    refunds = pd.DataFrame({
        "entity_id": ids.make("rfnd", len(refund_idx)), "entity_type": "refund",
        "payment_id": pay["payment_id"].values[refund_idx], "order_id": pay["order_id"].values[refund_idx],
        "method": method[refund_idx], "account": account[refund_idx], "amount": ref_amt.astype("int64"),
        "created_at": ref_created,
    })
    pay.loc[pay.index[refund_idx[full]], "ledger_status"] = "refunded"

    refunded = set(refund_idx.tolist())
    pool = np.array([i for i in range(n) if i not in refunded])
    cb_idx = rng.choice(pool, size=int(cfg.chargeback_rate * n), replace=False)
    chargebacks = pd.DataFrame({
        "entity_id": ids.make("chbk", len(cb_idx)), "entity_type": "chargeback",
        "payment_id": pay["payment_id"].values[cb_idx], "order_id": pay["order_id"].values[cb_idx],
        "method": method[cb_idx], "account": account[cb_idx], "amount": amount[cb_idx],
        "created_at": created[cb_idx] + pd.to_timedelta(rng.integers(10, 41, len(cb_idx)), unit="D"),
    })

    # ---- payment-level injections (disjoint, clean payments only) ---------------------------
    clean = [i for i in range(n) if i not in refunded and i not in set(cb_idx.tolist())]
    rng.shuffle(clean)
    take_pos = 0

    def take(k: int) -> list[int]:
        nonlocal take_pos
        out = clean[take_pos: take_pos + k]
        take_pos += k
        return out

    for i in take(_count(cfg.mdr_overcharge_rate, n, cfg.min_per_type)):
        m = method[i]
        exp_fee = int(pay.at[i, "fee"]) - int(wobble[i])
        extra_ppm = int(rng.choice([3000, 5000, 6000, 10000]))
        charged = max(int((amount[i] * (rate_ppm[i] + extra_ppm) + 500_000) // 1_000_000) + int(flat[i]),
                      exp_fee + 2)
        pay.at[i, "fee"] = charged
        pay.at[i, "tax"] = int(gst_on(charged, rc.gst_bps))
        impact = (charged + int(gst_on(charged, rc.gst_bps))) - (exp_fee + int(gst_on(exp_fee, rc.gst_bps)))
        label("mdr_overcharge", pay_ids[i], impact, method=m)

    for i in take(_count(cfg.gst_mismatch_rate, n, cfg.min_per_type)):
        f = int(pay.at[i, "fee"])
        right = int(gst_on(f, rc.gst_bps))
        wrong = int(gst_on(f, 2800)) if rng.random() < 0.5 else right + int(rng.integers(100, 2001))
        wrong = max(wrong, right + 2)
        pay.at[i, "tax"] = wrong
        label("gst_mismatch", pay_ids[i], wrong - right)

    for i in take(_count(cfg.payment_not_recorded_rate, n, cfg.min_per_type)):
        pay.at[i, "ledger_status"] = str(rng.choice(["pending", "created"]))
        label("payment_not_recorded", pay_ids[i], int(amount[i]))

    dup_src = refunds.sample(n=min(len(refunds), _count(cfg.duplicate_refund_rate, n, cfg.min_per_type)),
                             random_state=cfg.seed)
    dups = dup_src.copy()
    dups["entity_id"] = ids.make("rfnd", len(dups))
    dups["created_at"] = dups["created_at"] + pd.to_timedelta(rng.integers(600, 2 * 86_400, len(dups)), unit="s")
    for _, r in dups.iterrows():
        label("duplicate_refund", r["entity_id"], int(r["amount"]), original_payment=r["payment_id"])

    k = _count(cfg.refund_without_order_rate, n, cfg.min_per_type)
    orphan = pd.DataFrame({
        "entity_id": ids.make("rfnd", k), "entity_type": "refund", "payment_id": ids.make("pay", k),
        "order_id": ids.make("order", k), "method": "card",
        "account": rng.choice(list(cfg.accounts), k),
        "amount": (rng.choice(PRICE_POINTS, k) * 100).astype("int64"),
        "created_at": pd.to_datetime(cfg.start) + pd.to_timedelta(rng.integers(5, cfg.days, k), unit="D")
        + pd.to_timedelta(rng.integers(0, 86_400, k), unit="s"),
    })
    for _, r in orphan.iterrows():
        label("refund_without_order", r["entity_id"], int(r["amount"]))

    debits = pd.concat([refunds, dups, chargebacks, orphan], ignore_index=True)
    debits["fee"] = 0
    debits["tax"] = 0

    # ---- settlement batches -------------------------------------------------------------
    due = [cal.add_business_days(d, rc.methods[m].settlement_days) for d, m in zip(pay["created_date"], method,
                                                                                     strict=True)]
    pay["settle_date"] = due
    debits["settle_date"] = [cal.add_business_days(d, 1) for d in debits["created_at"].dt.date]

    keys = sorted(set(zip(pay["account"], pay["settle_date"], strict=True))
                  | set(zip(debits["account"], debits["settle_date"], strict=True)), key=lambda t: (t[1], t[0]))
    setl_ids = ids.make("setl", len(keys))
    key_to_sid = dict(zip(keys, setl_ids, strict=True))
    pay["settlement_id"] = [key_to_sid[k] for k in zip(pay["account"], pay["settle_date"], strict=True)]
    debits["settlement_id"] = [key_to_sid[k] for k in zip(debits["account"], debits["settle_date"], strict=True)]

    net: dict[str, int] = defaultdict(int)
    for sid, a, f, t in zip(pay["settlement_id"], pay["amount"], pay["fee"], pay["tax"], strict=True):
        net[sid] += int(a) - int(f) - int(t)
    for sid, a in zip(debits["settlement_id"], debits["amount"], strict=True):
        net[sid] -= int(a)
    settle_date = {key_to_sid[k]: k[1] for k in keys}

    # ---- settlement-level injections ---------------------------------------------------
    last = max(settle_date.values())
    eligible = [s for s in setl_ids if net[s] > 500_000 and (last - settle_date[s]).days > 10]
    rng.shuffle(eligible)
    ns = _count(cfg.settlement_type_rate, len(setl_ids), 2)
    chosen: set[str] = set()

    def pick(k: int) -> list[str]:
        out = [s for s in eligible if s not in chosen][:k]
        chosen.update(out)
        return out

    delayed = pick(ns)
    for s in delayed:
        shift = int(rng.integers(3, 6))
        settle_date[s] = cal.add_business_days(settle_date[s], shift)
        label("settlement_delayed", s, net[s], business_days_late=shift)

    # Merged: two settlements on the same date paid out as one transfer sharing one UTR.
    merged_pairs: list[tuple[str, str]] = []
    by_date: dict[date, list[str]] = defaultdict(list)
    for s in eligible:
        if s not in chosen:
            by_date[settle_date[s]].append(s)
    for _d, group in sorted(by_date.items()):
        if len(merged_pairs) >= ns:
            break
        group = [s for s in group if s not in chosen]
        if len(group) >= 2:
            merged_pairs.append((group[0], group[1]))
            chosen.update(group[:2])
    missing = set(pick(ns))
    short = set(pick(ns))
    rounding = set(pick(round(ns * 1.5)))
    split = set(pick(ns))
    for s in missing:
        label("missing_settlement", s, net[s])

    # ---- UTRs -----------------------------------------------------------------------------
    utr: dict[str, str] = {}
    seq = rng.permutation(900_000)[: len(setl_ids)] + 100_000
    for s, q in zip(setl_ids, seq, strict=True):
        d = settle_date[s]
        utr[s] = f"{rng.choice(UTR_BANKS)}N{d.strftime('%y')}{d.timetuple().tm_yday:03d}{int(q):06d}"
    for a, b in merged_pairs:
        utr[b] = utr[a]

    # ---- bank statement ---------------------------------------------------------------------
    bank_rows: list[dict[str, Any]] = []
    bank_gold: dict[int, list[str]] = {}
    narr_gold: dict[int, dict[str, str]] = {}

    def credit(d: date, narration: str, amt: int, sids: list[str] | None = None,
               variant: str | None = None, true_utr: str | None = None) -> int:
        idx = len(bank_rows)
        bank_rows.append({"date": d, "narration": narration, "credit": amt, "debit": 0})
        if sids:
            bank_gold[idx] = sids
        if variant:
            narr_gold[idx] = {"variant": variant, "utr": true_utr or ""}
        return idx

    def narration(u: str, d: date, variant: str) -> str:
        ddmm = d.strftime("%d%m")
        if variant == "standard":
            return str(rng.choice([
                f"NEFT CR-{u[:4]}0000123-PAYFLOW PAYMENTS PVT LTD-SETTL {ddmm}-{u}",
                f"NEFT-{u}-PAYFLOW PAYMENTS PRIVATE LIMITED-SETTLEMENT",
                f"BY TRANSFER NEFT*{u}*PAYFLOW SETTL",
                f"NEFT CR {u} PAYFLOW SETL {ddmm}",
            ]))
        if variant == "fragment":
            return f"NEFT CR-PAYFLOW SETTL {ddmm} UTR {u[-10:]}"
        if variant == "hard":
            return str(rng.choice([
                f"PAYFLOW SETTLEMENT REF {u[:4]} {u[4:7]} {u[7:10]} {u[10:]} NEFT CR",
                f"NEFT CR PAYFLOW {u[:8]} REF {u[8:]}",
                f"PAYFLOW/U.T.R/{u[:4]}.{u[4:10]}.{u[10:]}/SETTL",
            ]))
        return str(rng.choice([
            "NEFT CR-PAYFLOW PAYMENTS PVT LTD SETTLEMENT",
            f"PAYFLOW SETTL {ddmm} BATCH PAYOUT",
            "NEFT CR PAYFLOW PAYMENTS PRIVATE LTD",
        ]))

    merged_second = {b for _, b in merged_pairs}
    merged_first = {a: b for a, b in merged_pairs}
    for s in setl_ids:
        if s in missing or s in merged_second or net[s] <= 0:
            continue
        d = settle_date[s]
        if rng.random() < 0.1:
            d = cal.add_business_days(d, 1)
        u = utr[s]
        amt = net[s]
        if s in merged_first:
            other = merged_first[s]
            i = credit(d, narration(u, d, "standard"), amt + net[other], [s, other], "standard", u)
            label("merged_credit", f"bank:{i}", 0, settlements=[s, other])
            continue
        if s in short:
            gap = int(max(10_000, min(amt * float(rng.uniform(0.005, 0.05)), 500_000)))
            credit(d, narration(u, d, "standard"), amt - gap, [s], "standard", u)
            label("short_credit", s, gap)
            continue
        if s in rounding:
            diff = int(rng.integers(1, 100)) * int(rng.choice([-1, 1]))
            credit(d, narration(u, d, "standard"), amt + diff, [s], "standard", u)
            label("rounding_difference", s, abs(diff))
            continue
        if s in split:
            part = int(amt * float(rng.uniform(0.55, 0.75)))
            credit(d, narration(u, d, "standard"), part, [s], "standard", u)
            d2 = d if rng.random() < 0.5 else cal.add_business_days(d, 1)
            credit(d2, narration(u, d2, "absent"), amt - part, [s], "absent", "")
            label("split_credit", s, 0)
            continue
        if s in delayed:
            credit(d, narration(u, d, "standard"), amt, [s], "standard", u)
            continue
        r = rng.random()
        if r < cfg.narration_absent_rate:
            variant = "absent"
        elif r < cfg.narration_absent_rate + cfg.narration_hard_rate:
            variant = "hard"
        elif r < cfg.narration_absent_rate + cfg.narration_hard_rate + cfg.narration_fragment_rate:
            variant = "fragment"
        else:
            variant = "standard"
        i = credit(d, narration(u, d, variant), amt, [s], variant, u)
        if variant in ("absent", "hard"):
            label("narration_no_utr", f"bank:{i}", 0, variant=variant)

    # Unrelated credits and debits that a real statement is full of.
    span_end = last + timedelta(days=3)
    cur = cfg.start
    inv = 1000
    while cur <= span_end:
        for _ in range(int(rng.poisson(0.5))):
            inv += 1
            choice = rng.integers(0, 3)
            if choice == 0:
                credit(cur, f"NEFT CR-ACME RETAIL DISTRIBUTORS-INV {inv}", int(rng.integers(5_000, 500_000)) * 100)
            elif choice == 1:
                ref = int(rng.integers(10**11, 10**12))
                credit(cur, f"IMPS/{ref}/CUSTOMER TRANSFER", int(rng.integers(100, 20_000)) * 100)
            else:
                credit(cur, "CASH DEPOSIT BY SELF", int(rng.integers(1_000, 50_000)) * 100)
        for _ in range(int(rng.poisson(1.2))):
            bank_rows.append({"date": cur, "narration": str(rng.choice([
                "NEFT DR-SHIPFAST LOGISTICS-AWB BILL", "ACH DR-OFFICE RENT", "UPI/DR/ADS/MARKETING",
                "NEFT DR-PACKAGING SUPPLIES"])), "credit": 0, "debit": int(rng.integers(1_000, 300_000)) * 100})
        if cur.day == 1 and cur.month in (1, 4, 7, 10) and cur != cfg.start:
            credit(cur, f"INT.PD:{cur.strftime('%d-%m-%Y')}", int(rng.integers(500, 5_000)) * 100)
        cur += timedelta(days=1)

    order = sorted(range(len(bank_rows)), key=lambda i: (bank_rows[i]["date"], i))
    txn_id = {old: f"BNK{pos + 1:07d}" for pos, old in enumerate(order)}
    balance = 2_500_000_000
    bank_out = []
    for old in order:
        r = bank_rows[old]
        balance += r["credit"] - r["debit"]
        bank_out.append({
            "txn_id": txn_id[old], "txn_date": r["date"].strftime("%d/%m/%Y"),
            "value_date": r["date"].strftime("%d/%m/%Y"), "narration": r["narration"],
            "ref_no": "", "debit": _rupees(r["debit"]) if r["debit"] else "",
            "credit": _rupees(r["credit"]) if r["credit"] else "", "balance": _rupees(balance),
        })
    bank_df = pd.DataFrame(bank_out)
    for lab in labels:
        if str(lab["key"]).startswith("bank:"):
            lab["key"] = txn_id[int(lab["key"].split(":")[1])]

    # ---- gateway report ------------------------------------------------------------------
    pay_rows = pd.DataFrame({
        "entity_id": pay["payment_id"], "entity_type": "payment", "payment_id": pay["payment_id"],
        "order_id": pay["order_id"], "method": pay["method"], "merchant_account": pay["account"],
        "amount": pay["amount"], "fee": pay["fee"], "tax": pay["tax"], "created_at": pay["created_at"],
        "settlement_id": pay["settlement_id"],
    })
    deb_rows = pd.DataFrame({
        "entity_id": debits["entity_id"], "entity_type": debits["entity_type"], "payment_id": debits["payment_id"],
        "order_id": debits["order_id"], "method": debits["method"], "merchant_account": debits["account"],
        "amount": debits["amount"], "fee": 0, "tax": 0, "created_at": debits["created_at"],
        "settlement_id": debits["settlement_id"],
    })
    gw = pd.concat([pay_rows, deb_rows], ignore_index=True)
    gw["settled_at"] = gw["settlement_id"].map(lambda s: settle_date[s].isoformat())
    gw["settlement_utr"] = gw["settlement_id"].map(utr)
    gw = gw.sort_values(["settled_at", "created_at", "entity_id"]).reset_index(drop=True)
    gw["created_at"] = gw["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ("amount", "fee", "tax"):
        gw[c] = gw[c].map(_rupees)

    # ---- order ledger -------------------------------------------------------------------
    ledger = pd.DataFrame({
        "order_id": pay["order_id"], "order_date": pay["created_date"].map(date.isoformat),
        "customer_id": [f"cust_{x:06d}" for x in rng.integers(0, 10**6, n)],
        "amount": pay["amount"].map(_rupees), "payment_mode": "prepaid", "status": pay["ledger_status"],
        "payment_ref": np.where(rng.random(n) < 0.9, pay["payment_id"], ""),
    })
    extra: list[dict[str, Any]] = []

    def rand_day() -> str:
        return (cfg.start + timedelta(days=int(rng.integers(0, cfg.days)))).isoformat()

    for oid in ids.make("order", int(0.25 * n)):
        extra.append({"order_id": oid, "order_date": rand_day(), "customer_id": f"cust_{rng.integers(0, 10**6):06d}",
                      "amount": _rupees(int(rng.choice(PRICE_POINTS)) * 100), "payment_mode": "cod",
                      "status": str(rng.choice(["paid", "pending", "cancelled"], p=[0.8, 0.1, 0.1])),
                      "payment_ref": ""})
    for oid in ids.make("order", int(0.05 * n)):
        extra.append({"order_id": oid, "order_date": rand_day(), "customer_id": f"cust_{rng.integers(0, 10**6):06d}",
                      "amount": _rupees(int(rng.choice(PRICE_POINTS)) * 100), "payment_mode": "prepaid",
                      "status": str(rng.choice(["pending", "failed", "cancelled"], p=[0.6, 0.3, 0.1])),
                      "payment_ref": ""})
    for oid in ids.make("order", _count(cfg.order_paid_no_payment_rate, n, cfg.min_per_type)):
        amt = int(rng.choice(PRICE_POINTS)) * 100
        extra.append({"order_id": oid, "order_date": rand_day(), "customer_id": f"cust_{rng.integers(0, 10**6):06d}",
                      "amount": _rupees(amt), "payment_mode": "prepaid", "status": "paid",
                      "payment_ref": ids.make("pay")[0]})
        label("order_paid_no_payment", oid, amt)
    ledger = pd.concat([ledger, pd.DataFrame(extra)], ignore_index=True)
    ledger = ledger.sample(frac=1.0, random_state=cfg.seed).sort_values("order_date", kind="stable")
    ledger.insert(len(ledger.columns), "invoice_no", [f"INV/26/{i + 1:07d}" for i in range(len(ledger))])
    ledger = ledger.reset_index(drop=True)

    gold = {
        "config": {**asdict(cfg), "start": cfg.start.isoformat()},
        "labels": labels,
        "bank_to_settlements": {txn_id[i]: sids for i, sids in bank_gold.items()},
        "narrations": {txn_id[i]: v for i, v in narr_gold.items()},
        "counts": {
            "payments": int(n), "gateway_rows": len(gw), "settlements": len(setl_ids),
            "bank_rows": len(bank_df), "ledger_rows": len(ledger),
        },
    }
    return GeneratedData(gateway=gw, bank=bank_df, ledger=ledger, gold=gold)
