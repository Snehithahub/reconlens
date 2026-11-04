"""Money helpers. All amounts are integer paise; rupees exist only at the edges."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import numpy as np
import pandas as pd

_CLEAN = re.compile(r"[₹,\s]|INR|Rs\.?", re.IGNORECASE)


def to_paise(value: object) -> int:
    """Parse '₹1,234.50', '1234.5', 1234.5 or '(12.00)' into paise (half-up)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    s = _CLEAN.sub("", str(value))
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if s in ("", "-"):
        return 0
    try:
        d = Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"not an amount: {value!r}") from e
    p = int((d * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return -p if neg else p


def series_to_paise(s: pd.Series) -> pd.Series:
    """Vectorised to_paise. Raises ValueError listing the first bad row positions."""
    if pd.api.types.is_numeric_dtype(s):
        num = s.astype("float64")
    else:
        cleaned = s.astype("string").str.replace(_CLEAN, "", regex=True).str.strip()
        neg = cleaned.str.startswith("(", na=False) & cleaned.str.endswith(")", na=False)
        cleaned = cleaned.str.strip("()").replace({"": None, "-": None})
        num = pd.to_numeric(cleaned, errors="coerce")
        bad = num.isna() & cleaned.notna()
        if bad.any():
            rows = list(bad[bad].index[:5])
            raise ValueError(f"unparseable amounts at rows {rows}")
        num = num.where(~neg, -num)
    return (num.fillna(0.0) * 100).round().astype("int64")


def pct_fee(amount_paise: int | np.ndarray, rate_ppm: int) -> int | np.ndarray:
    """amount * rate, rounded half-up to the paise. rate_ppm is parts per million."""
    return (np.asarray(amount_paise, dtype="int64") * rate_ppm + 500_000) // 1_000_000


def gst_on(fee_paise: int | np.ndarray, gst_bps: int) -> int | np.ndarray:
    """GST on a fee, rounded half-up. gst_bps=1800 for 18%."""
    return (np.asarray(fee_paise, dtype="int64") * gst_bps + 5_000) // 10_000


def fmt_inr(paise: int) -> str:
    """Indian digit grouping: 12345678 paise -> '₹1,23,456.78'."""
    neg = paise < 0
    rupees, p = divmod(abs(int(paise)), 100)
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join([*groups, tail])
    return f"{'-' if neg else ''}₹{s}.{p:02d}"


def to_rupees(paise: int) -> float:
    return round(paise / 100, 2)
