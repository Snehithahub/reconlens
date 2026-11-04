import io

import pandas as pd
import pytest

from reconlens.ingest import InputError, RawInput, load_bank, load_gateway, load_ledger
from reconlens.service import demo_inputs, execute_run
from reconlens.store import Store


def test_bank_aliases_dayfirst_and_row_ids():
    raw = RawInput("b.csv", b"Date,Particulars,Withdrawal Amt,Deposit Amt\n04/02/2026,NEFT CR PAYFLOW,,\"1,234.50\"\n"
                            b"05/02/2026,RENT,500.00,\n")
    b = load_bank(raw)
    assert b["txn_date"].iloc[0] == pd.Timestamp(2026, 2, 4)
    assert b["credit_paise"].tolist() == [123450, 0] and b["debit_paise"].tolist() == [0, 50000]
    assert b["txn_id"].tolist() == ["row-2", "row-3"]


def test_xlsx_ledger():
    buf = io.BytesIO()
    pd.DataFrame({"Order ID": ["o1"], "Order Amount": ["499.00"], "Order Status": ["Paid"]}).to_excel(buf, index=False)
    led = load_ledger(RawInput("ledger.xlsx", buf.getvalue()))
    assert led.loc[0, "amount_paise"] == 49900 and led.loc[0, "status"] == "paid"


@pytest.mark.parametrize("content,needle", [
    (b"entity_id,amount\nx,1\n", "missing required columns"),
    (b"entity_id,entity_type,order_id,method,amount,fee,tax,created_at,settlement_id,settled_at,utr\n"
     b"p1,payment,o1,upi,abc,0,0,2026-01-01,s1,2026-01-02,U\n", "unparseable amounts"),
    (b"entity_id,entity_type,order_id,method,amount,fee,tax,created_at,settlement_id,settled_at,utr\n"
     b"p1,transfer,o1,upi,1,0,0,2026-01-01,s1,2026-01-02,U\n", "entity_type"),
    (b"entity_id,entity_type,order_id,method,amount,fee,tax,created_at,settlement_id,settled_at,utr\n"
     b"p1,payment,o1,upi,1,0,0,not-a-date,s1,2026-01-02,U\n", "unparseable dates"),
])
def test_gateway_validation_errors(content, needle):
    with pytest.raises(InputError) as e:
        load_gateway(RawInput("g.csv", content))
    assert needle in str(e.value)


def test_unsupported_file_type():
    with pytest.raises(InputError, match="unsupported"):
        load_ledger(RawInput("ledger.pdf", b"%PDF"))


def test_idempotent_rerun_same_ids_and_resolutions_survive(cfg):
    from reconlens.config import Settings

    store = Store(":memory:")
    gw, bank, led = demo_inputs(Settings(_env_file=None), cfg, n=800, seed=5)
    first = execute_run(store, cfg, gw, bank, led, source="t")
    _, df1 = store.exceptions(first.run_id, limit=10_000)
    store.resolve(df1["exception_id"].iat[0], "resolved", "claimed", "qa")
    again = execute_run(store, cfg, gw, bank, led, source="t")
    assert again.reused and again.run_id == first.run_id
    forced = execute_run(store, cfg, gw, bank, led, source="t", force=True)
    _, df2 = store.exceptions(forced.run_id, limit=10_000)
    assert not forced.reused and forced.run_id == first.run_id
    assert sorted(df1["exception_id"]) == sorted(df2["exception_id"])  # no duplicates, same ids
    assert store.get_exception(df1["exception_id"].iat[0])["status"] == "resolved"
    other = execute_run(store, cfg, gw, bank, RawInput("l.csv", led.content + b"\n"), source="t")
    assert other.run_id != first.run_id  # different input bytes -> different run
