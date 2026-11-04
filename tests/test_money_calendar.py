from datetime import date

import numpy as np
import pandas as pd
import pytest

from reconlens.money import fmt_inr, gst_on, pct_fee, series_to_paise, to_paise


@pytest.mark.parametrize("raw,paise", [
    ("₹1,234.50", 123450), ("1234.5", 123450), (1234.5, 123450), ("(12.00)", -1200), ("", 0),
    ("Rs. 99", 9900), ("0.005", 1), ("INR 10,00,000.00", 100000000),
])
def test_to_paise(raw, paise):
    assert to_paise(raw) == paise


def test_to_paise_rejects_garbage():
    with pytest.raises(ValueError):
        to_paise("12abc")


def test_series_to_paise_vectorised_and_errors():
    assert series_to_paise(pd.Series(["1,000.10", "", "(5.00)"])).tolist() == [100010, 0, -500]
    with pytest.raises(ValueError, match="rows"):
        series_to_paise(pd.Series(["1.00", "oops"]))


def test_fee_half_up_rounding():
    # 1.95% of Rs 999.00 = 19.4805 -> 19.48; of Rs 1000.26 = 19.50507 -> 19.51
    assert pct_fee(99900, 19_500) == 1948
    assert pct_fee(100026, 19_500) == 1951
    # exactly half a paise rounds up
    assert pct_fee(50, 10_000) == 1  # 1% of 50 paise = 0.5 paise


def test_gst_18_percent_half_up():
    assert gst_on(1948, 1800) == 351  # 350.64 -> 351
    assert gst_on(np.array([0, 100, 2778]), 1800).tolist() == [0, 18, 500]  # 500.04 -> 500


def test_fmt_inr_indian_grouping():
    assert fmt_inr(12345678) == "₹1,23,456.78"
    assert fmt_inr(-5) == "-₹0.05"
    assert fmt_inr(100000000000) == "₹1,00,00,00,000.00"


def test_calendar_weekends_and_second_fourth_saturdays(cal):
    assert not cal.is_business_day(date(2026, 1, 4))  # Sunday
    assert cal.is_business_day(date(2026, 1, 3))  # 1st Saturday
    assert not cal.is_business_day(date(2026, 1, 10))  # 2nd Saturday
    assert cal.is_business_day(date(2026, 1, 17))  # 3rd Saturday
    assert not cal.is_business_day(date(2026, 1, 24))  # 4th Saturday
    assert not cal.is_business_day(date(2026, 1, 26))  # Republic Day (config)


def test_t_plus_n(cal):
    assert cal.add_business_days(date(2026, 1, 9), 1) == date(2026, 1, 12)  # Fri -> Mon (2nd Sat closed)
    assert cal.add_business_days(date(2026, 1, 23), 1) == date(2026, 1, 27)  # skips 4th Sat, Sun, holiday
    assert cal.add_business_days(date(2026, 1, 23), 2) == date(2026, 1, 28)
    assert cal.add_business_days(date(2026, 1, 4), 0) == date(2026, 1, 5)
    assert cal.business_days_between(date(2026, 1, 9), date(2026, 1, 13)) == 2
    assert cal.business_days_between(date(2026, 1, 13), date(2026, 1, 9)) == -2
