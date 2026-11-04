from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from reconlens.bankcal import BankCalendar
from reconlens.config import load_recon_config


@pytest.fixture(scope="session")
def cfg():
    return load_recon_config()


@pytest.fixture(scope="session")
def cal(cfg):
    return BankCalendar(cfg.bank_holidays)


def settlements_df(rows: list[tuple[str, int, date, str]]) -> pd.DataFrame:
    """rows: (settlement_id, net_paise, settled_at, utr)"""
    return pd.DataFrame([{"settlement_id": s, "net_paise": n, "settled_at": d, "account": "WEB", "utr": u}
                         for s, n, d, u in rows])


def bank_df(rows: list[tuple[str, date, str, int]]) -> pd.DataFrame:
    """rows: (txn_id, date, narration, credit_paise)"""
    return pd.DataFrame([{"txn_id": t, "txn_date": pd.Timestamp(d), "narration": n, "ref_no": "",
                          "credit_paise": c, "debit_paise": 0} for t, d, n, c in rows],
                        columns=["txn_id", "txn_date", "narration", "ref_no", "credit_paise", "debit_paise"])


@pytest.fixture(scope="session")
def small_dataset(cfg):
    from reconlens.datagen import GenConfig, generate

    return generate(GenConfig(n_payments=3000, seed=11), cfg)
