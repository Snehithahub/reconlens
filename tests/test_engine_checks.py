"""Fee/GST recomputation, refunds, order<->payment and SLA checks through the full engine."""

from __future__ import annotations

import io

import pandas as pd

from reconlens.engine import reconcile
from reconlens.engine.types import ExceptionType as T
from reconlens.ingest import RawInput, load_bank, load_gateway, load_ledger

GW_HEAD = ("entity_id,entity_type,payment_id,order_id,method,merchant_account,amount,fee,tax,created_at,"
           "settlement_id,settled_at,settlement_utr\n")
UTR = "UTIBN26034000001"


def gw_row(eid, kind, pid, oid, method, amount, fee, tax, created="2026-02-02 10:00:00",
           sid="setl_1", settled="2026-02-04", utr=UTR):
    return f"{eid},{kind},{pid},{oid},{method},WEB,{amount},{fee},{tax},{created},{sid},{settled},{utr}\n"


def run(gw_rows, ledger_rows, bank_rows=None, cfg=None):
    gw = load_gateway(RawInput("g.csv", (GW_HEAD + "".join(gw_rows)).encode()))
    led = load_ledger(RawInput("l.csv", ("order_id,order_date,amount,payment_mode,status\n"
                                         + "".join(ledger_rows)).encode()))
    if bank_rows is None:  # credit exactly the settlement net so only the checks under test fire
        net = 0
        for r in gw.itertuples():
            net += (r.amount_paise - r.fee_paise - r.tax_paise) if r.entity_type == "payment" else -r.amount_paise
        bank_rows = [f"B1,04/02/2026,NEFT-{UTR}-PAYFLOW,{net / 100:.2f}\n"]
    bank = load_bank(RawInput("b.csv", ("txn_id,txn_date,narration,credit\n" + "".join(bank_rows)).encode()))
    return reconcile(gw, bank, led, cfg)


def keys(res, kind):
    return sorted(e.key for e in res.exceptions if e.type == kind)


# card 1.95%: Rs 1000 -> fee 19.50, GST 3.51
CLEAN = gw_row("pay_a", "payment", "pay_a", "ord_a", "card", "1000.00", "19.50", "3.51")
LEDGER_A = "ord_a,2026-02-02,1000.00,prepaid,paid\n"


def test_clean_payment_reconciles(cfg):
    res = run([CLEAN], [LEDGER_A], cfg=cfg)
    assert res.exceptions == []
    assert res.payments["recon_status"].tolist() == ["reconciled"]
    assert res.summary["auto_match_rate"] == 1.0


def test_mdr_overcharge_detected_with_impact(cfg):
    # charged 2.36% (23.60 + GST 4.25) vs contracted 1.95% (19.50 + 3.51)
    res = run([gw_row("pay_a", "payment", "pay_a", "ord_a", "card", "1000.00", "23.60", "4.25")], [LEDGER_A], cfg=cfg)
    exc = [e for e in res.exceptions if e.type == T.MDR_OVERCHARGE]
    assert [e.key for e in exc] == ["pay_a"]
    assert exc[0].impact_paise == (2360 + 425) - (1950 + 351)
    assert res.payments["recon_status"].tolist() == ["exception"]


def test_fee_one_paise_wobble_tolerated(cfg):
    res = run([gw_row("pay_a", "payment", "pay_a", "ord_a", "card", "1000.00", "19.51", "3.51")], [LEDGER_A], cfg=cfg)
    assert keys(res, T.MDR_OVERCHARGE) == [] and keys(res, T.GST_MISMATCH) == []


def test_min_fee_applies(cfg):
    # netbanking 1.90% of Rs 100 = 1.90, but contracted minimum is Rs 5.00
    res = run([gw_row("pay_a", "payment", "pay_a", "ord_a", "netbanking", "100.00", "5.00", "0.90")],
              ["ord_a,2026-02-02,100.00,prepaid,paid\n"], cfg=cfg)
    assert res.exceptions == []


def test_gst_mismatch(cfg):
    res = run([gw_row("pay_a", "payment", "pay_a", "ord_a", "card", "1000.00", "19.50", "5.46")], [LEDGER_A], cfg=cfg)
    exc = [e for e in res.exceptions if e.type == T.GST_MISMATCH]
    assert [e.key for e in exc] == ["pay_a"] and exc[0].impact_paise == 546 - 351


def test_duplicate_refund_and_refund_without_order(cfg):
    rows = [CLEAN,
            gw_row("rfnd_1", "refund", "pay_a", "ord_a", "card", "200.00", "0", "0", "2026-02-02 11:00:00"),
            gw_row("rfnd_2", "refund", "pay_a", "ord_a", "card", "200.00", "0", "0", "2026-02-02 12:00:00"),
            gw_row("rfnd_3", "refund", "pay_old", "ord_unknown", "card", "50.00", "0", "0", "2026-02-02 13:00:00")]
    res = run(rows, [LEDGER_A], cfg=cfg)
    assert keys(res, T.DUPLICATE_REFUND) == ["rfnd_2"]
    assert keys(res, T.REFUND_WITHOUT_ORDER) == ["rfnd_3"]


def test_order_side_exceptions(cfg):
    rows = [CLEAN,
            gw_row("pay_b", "payment", "pay_b", "ord_b", "card", "1000.00", "19.50", "3.51"),
            gw_row("pay_c", "payment", "pay_c", "ord_c", "card", "1000.00", "19.50", "3.51"),
            gw_row("pay_d", "payment", "pay_d", "ord_d", "card", "1000.00", "19.50", "3.51"),
            gw_row("pay_e", "payment", "pay_e", "ord_a", "card", "1000.00", "19.50", "3.51")]
    ledger = [LEDGER_A,
              "ord_b,2026-02-02,1000.00,prepaid,pending\n",       # captured but not updated
              "ord_d,2026-02-02,900.00,prepaid,paid\n",           # amount mismatch
              "ord_x,2026-02-02,500.00,prepaid,paid\n",           # paid, no payment
              "ord_y,2026-02-02,500.00,cod,paid\n",               # COD: out of scope
              "ord_z,2026-02-02,500.00,prepaid,pending\n"]        # abandoned cart: fine
    res = run(rows, ledger, cfg=cfg)
    assert keys(res, T.PAYMENT_NOT_RECORDED) == ["pay_b"]
    assert keys(res, T.PAYMENT_WITHOUT_ORDER) == ["pay_c"]
    assert keys(res, T.AMOUNT_MISMATCH) == ["pay_d"]
    assert keys(res, T.ORDER_PAID_NO_PAYMENT) == ["ord_x"]
    assert keys(res, T.DUPLICATE_CAPTURE) == ["pay_e"]


def test_settlement_delayed_beyond_sla(cfg):
    # card captured Mon 2 Feb, T+2 = Wed 4 Feb; settled and credited Mon 9 Feb
    # = 4 business days late (Sat 7 Feb is a working 1st Saturday)
    row = gw_row("pay_a", "payment", "pay_a", "ord_a", "card", "1000.00", "19.50", "3.51", settled="2026-02-09")
    res = run([row], [LEDGER_A], [f"B1,09/02/2026,NEFT-{UTR}-PAYFLOW,977.00\n"], cfg=cfg)
    exc = [e for e in res.exceptions if e.type == T.SETTLEMENT_DELAYED]
    assert [e.key for e in exc] == ["setl_1"] and exc[0].evidence[0]["business_days_late"] == 4


def test_one_day_bank_lag_is_within_grace(cfg):
    res = run([CLEAN], [LEDGER_A], [f"B1,05/02/2026,NEFT-{UTR}-PAYFLOW,976.90\n"], cfg=cfg)
    assert keys(res, T.SETTLEMENT_DELAYED) == []
    assert keys(res, T.ROUNDING_DIFFERENCE) == ["setl_1"]


def test_settlement_net_includes_fees_tax_refunds_chargebacks(cfg):
    rows = [CLEAN,
            gw_row("rfnd_1", "refund", "pay_a", "ord_a", "card", "100.00", "0", "0"),
            gw_row("chbk_1", "chargeback", "pay_a", "ord_a", "card", "50.00", "0", "0")]
    res = run(rows, [LEDGER_A], cfg=cfg)
    s = res.settlements.iloc[0]
    assert s["net_paise"] == 100000 - 1950 - 351 - 10000 - 5000
    assert (s["refunds_paise"], s["chargebacks_paise"]) == (10000, 5000)


def test_result_frames_are_json_safe(cfg):
    res = run([CLEAN], [LEDGER_A], cfg=cfg)
    assert isinstance(res.bank, pd.DataFrame) and "match_status" in res.bank
    io.StringIO(res.settlements.to_json())
