"""Load the three input files (CSV or XLSX) into validated, typed frames.

Normalised frames use integer paise for money and naive datetimes / python dates.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reconlens.money import series_to_paise


class InputError(ValueError):
    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        super().__init__(f"{source}: " + "; ".join(problems))


@dataclass(frozen=True)
class RawInput:
    name: str
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @classmethod
    def from_path(cls, path: Path | str) -> RawInput:
        p = Path(path)
        return cls(p.name, p.read_bytes())


GATEWAY_ALIASES = {
    "type": "entity_type", "settlement_utr": "utr", "utr_number": "utr", "merchant_account": "account",
    "gst": "tax", "gst_on_fee": "tax", "mdr": "fee", "mdr_fee": "fee", "gross_amount": "amount",
    "settlement_date": "settled_at", "captured_at": "created_at",
}
BANK_ALIASES = {
    "date": "txn_date", "transaction_date": "txn_date", "description": "narration", "particulars": "narration",
    "remarks": "narration", "deposit": "credit", "deposit_amt": "credit", "credit_amount": "credit",
    "withdrawal": "debit", "withdrawal_amt": "debit", "debit_amount": "debit", "amount": "credit",
    "chq_ref_no": "ref_no", "reference": "ref_no",
}
LEDGER_ALIASES = {
    "order_amount": "amount", "total": "amount", "order_status": "status", "payment_id": "payment_ref",
    "created_at": "order_date",
}

GATEWAY_REQUIRED = ["entity_id", "entity_type", "order_id", "method", "amount", "fee", "tax",
                    "created_at", "settlement_id", "settled_at", "utr"]
BANK_REQUIRED = ["txn_date", "narration", "credit"]
LEDGER_REQUIRED = ["order_id", "amount", "status"]

ENTITY_TYPES = {"payment", "refund", "chargeback"}


def read_table(raw: RawInput) -> pd.DataFrame:
    name = raw.name.lower()
    try:
        if name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(io.BytesIO(raw.content), dtype=str, engine="openpyxl")
        elif name.endswith((".csv", ".txt")) or "." not in name:
            df = pd.read_csv(io.BytesIO(raw.content), dtype=str, keep_default_na=False)
        else:
            raise InputError(raw.name, ["unsupported file type; upload .csv or .xlsx"])
    except InputError:
        raise
    except Exception as e:  # pandas raises many parser-specific errors
        raise InputError(raw.name, [f"could not parse file: {e}"]) from e
    df.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "").replace("/", "_") for c in df.columns]
    return df


def _normalise(df: pd.DataFrame, aliases: dict[str, str], required: list[str], source: str) -> pd.DataFrame:
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases and aliases[c] not in df.columns})
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputError(source, [f"missing required columns: {missing}"])
    if df.empty:
        raise InputError(source, ["file has no rows"])
    return df


def _text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _money(df: pd.DataFrame, col: str, source: str, problems: list[str]) -> pd.Series:
    try:
        return series_to_paise(df[col].fillna("") if col in df.columns else pd.Series("", index=df.index))
    except ValueError as e:
        problems.append(f"{col}: {e}")
        return pd.Series(0, index=df.index, dtype="int64")


def _dates(s: pd.Series, col: str, problems: list[str], *, required: bool, dayfirst: bool = False) -> pd.Series:
    txt = _text(s)
    parsed = pd.to_datetime(txt.where(txt != ""), errors="coerce", format="mixed", dayfirst=dayfirst)
    bad = parsed.isna() & (txt != "")
    if bad.any():
        problems.append(f"{col}: unparseable dates at rows {list(bad[bad].index[:5])}")
    if required and (txt == "").any():
        rows = list((txt == "")[txt == ""].index[:5])
        problems.append(f"{col}: empty at rows {rows}")
    return parsed


def load_gateway(raw: RawInput) -> pd.DataFrame:
    df = _normalise(read_table(raw), GATEWAY_ALIASES, GATEWAY_REQUIRED, "gateway report")
    problems: list[str] = []
    out = pd.DataFrame(index=df.index)
    out["entity_id"] = _text(df["entity_id"])
    out["entity_type"] = _text(df["entity_type"]).str.lower()
    bad_type = ~out["entity_type"].isin(ENTITY_TYPES)
    if bad_type.any():
        bad_rows = list(bad_type[bad_type].index[:5])
        problems.append(f"entity_type must be one of {sorted(ENTITY_TYPES)}; bad rows {bad_rows}")
    if (out["entity_id"] == "").any():
        problems.append("entity_id is empty on some rows")
    dup = out["entity_id"].duplicated() & (out["entity_id"] != "")
    if dup.any():
        problems.append(f"duplicate entity_id values, e.g. {out.loc[dup, 'entity_id'].head(3).tolist()}")
    payment_id = _text(df["payment_id"]) if "payment_id" in df.columns else pd.Series("", index=df.index)
    out["payment_id"] = payment_id.where(payment_id != "", out["entity_id"].where(out["entity_type"] == "payment", ""))
    out["order_id"] = _text(df["order_id"])
    out["method"] = _text(df["method"]).str.lower()
    out["account"] = _text(df["account"]) if "account" in df.columns else "default"
    out["amount_paise"] = _money(df, "amount", "gateway", problems).abs()
    out["fee_paise"] = _money(df, "fee", "gateway", problems)
    out["tax_paise"] = _money(df, "tax", "gateway", problems)
    out["created_at"] = _dates(df["created_at"], "created_at", problems, required=True)
    out["settlement_id"] = _text(df["settlement_id"])
    out["settled_at"] = _dates(df["settled_at"], "settled_at", problems, required=False).dt.normalize()
    out["utr"] = _text(df["utr"]).str.upper()
    if problems:
        raise InputError("gateway report", problems)
    return out.reset_index(drop=True)


def load_bank(raw: RawInput) -> pd.DataFrame:
    df = _normalise(read_table(raw), BANK_ALIASES, BANK_REQUIRED, "bank statement")
    problems: list[str] = []
    out = pd.DataFrame(index=df.index)
    ids = _text(df["txn_id"]) if "txn_id" in df.columns else pd.Series("", index=df.index)
    # Statements rarely carry a stable id; fall back to the row position in the file.
    out["txn_id"] = ids.where(ids != "", pd.Series([f"row-{i + 2}" for i in df.index], index=df.index))
    out["txn_date"] = _dates(df["txn_date"], "txn_date", problems, required=True, dayfirst=True).dt.normalize()
    out["narration"] = _text(df["narration"])
    out["ref_no"] = _text(df["ref_no"]) if "ref_no" in df.columns else ""
    out["credit_paise"] = _money(df, "credit", "bank", problems)
    out["debit_paise"] = _money(df, "debit", "bank", problems) if "debit" in df.columns else 0
    if out["txn_id"].duplicated().any():
        problems.append("duplicate txn_id values")
    if problems:
        raise InputError("bank statement", problems)
    return out.reset_index(drop=True)


def load_ledger(raw: RawInput) -> pd.DataFrame:
    df = _normalise(read_table(raw), LEDGER_ALIASES, LEDGER_REQUIRED, "order ledger")
    problems: list[str] = []
    out = pd.DataFrame(index=df.index)
    out["order_id"] = _text(df["order_id"])
    if (out["order_id"] == "").any():
        problems.append("order_id is empty on some rows")
    out["order_date"] = (
        _dates(df["order_date"], "order_date", problems, required=False).dt.normalize()
        if "order_date" in df.columns else pd.NaT
    )
    out["amount_paise"] = _money(df, "amount", "ledger", problems)
    out["status"] = _text(df["status"]).str.lower()
    out["payment_mode"] = _text(df["payment_mode"]).str.lower() if "payment_mode" in df.columns else "prepaid"
    out["payment_ref"] = _text(df["payment_ref"]) if "payment_ref" in df.columns else ""
    out["invoice_no"] = _text(df["invoice_no"]) if "invoice_no" in df.columns else ""
    if problems:
        raise InputError("order ledger", problems)
    return out.reset_index(drop=True)
