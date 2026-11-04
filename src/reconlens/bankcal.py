"""Indian bank-working-day calendar: Sundays, 2nd/4th Saturdays and listed holidays are closed."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from functools import lru_cache


class BankCalendar:
    def __init__(self, holidays: Iterable[date] = ()) -> None:
        self.holidays = frozenset(holidays)
        self._is_bd = lru_cache(maxsize=4096)(self._compute_is_business_day)
        self._add = lru_cache(maxsize=16384)(self._compute_add)

    def _compute_is_business_day(self, d: date) -> bool:
        if d in self.holidays or d.weekday() == 6:
            return False
        if d.weekday() == 5:
            nth = (d.day - 1) // 7 + 1
            return nth not in (2, 4)
        return True

    def is_business_day(self, d: date) -> bool:
        return self._is_bd(d)

    def _compute_add(self, d: date, n: int) -> date:
        cur = d
        added = 0
        while added < n:
            cur += timedelta(days=1)
            if self._is_bd(cur):
                added += 1
        if n == 0:
            while not self._is_bd(cur):
                cur += timedelta(days=1)
        return cur

    def add_business_days(self, d: date, n: int) -> date:
        """T+n: the n-th business day after d. n=0 rolls forward to a business day."""
        return self._add(d, n)

    def business_days_between(self, start: date, end: date) -> int:
        """Business days in (start, end]; negative if end < start."""
        if end == start:
            return 0
        sign = 1
        if end < start:
            start, end, sign = end, start, -1
        count = 0
        cur = start
        while cur < end:
            cur += timedelta(days=1)
            if self._is_bd(cur):
                count += 1
        return sign * count
