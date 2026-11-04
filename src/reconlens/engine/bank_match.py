"""Multi-pass matching of gateway settlements to bank credits.

Pass order (each pass only sees what earlier passes left open):
  1. UTR       exact or unique-suffix UTR from the narration; groups handle merged (N:1) and split (1:N)
  2. LLM UTR   optional: model extracts a UTR the regex missed; matches are always "needs_review"
  3. Amount    settlement net vs credit amount within tolerance and a business-day window,
               1:1 with mutual uniqueness, then N:1 and 1:N subset sums
  4. Leftovers missing settlements, unmatched gateway credits
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from typing import Any, Protocol

import pandas as pd

from reconlens.bankcal import BankCalendar
from reconlens.config import ReconConfig
from reconlens.engine.types import ExceptionType, Match, ReconException
from reconlens.llm.schemas import NarrationExtraction
from reconlens.money import to_rupees
from reconlens.narration import UTRIndex, extract_utr, is_gateway_narration

log = logging.getLogger(__name__)
MAX_SUBSET_CANDIDATES = 24


class NarrationExtractor(Protocol):
    def extract_many(self, narrations: Sequence[str]) -> list[NarrationExtraction | None]: ...


@dataclass
class _Setl:
    sid: str
    net: int
    day: date
    account: str
    utr: str


@dataclass
class _Credit:
    tid: str
    amount: int
    day: date
    narration: str
    token: str | None
    rule: str | None
    sids: list[str]
    gateway: bool
    llm_utr: str | None = None
    llm_confidence: float | None = None


@dataclass
class BankMatchOutcome:
    matches: list[Match] = field(default_factory=list)
    exceptions: list[ReconException] = field(default_factory=list)
    settlement_status: dict[str, str] = field(default_factory=dict)
    settlement_credit_date: dict[str, date] = field(default_factory=dict)
    credits: pd.DataFrame = field(default_factory=pd.DataFrame)
    narration_stats: dict[str, int] = field(default_factory=dict)


def unique_subset(target: int, items: list[tuple[str, int]], tol: int, sizes: Sequence[int]) -> list[str] | None:
    """The single combination (smallest size first) whose sum is within tol of target, else None."""
    items = items[:MAX_SUBSET_CANDIDATES]
    for k in sizes:
        found = [c for c in combinations(items, k) if abs(sum(a for _, a in c) - target) <= tol]
        if len(found) == 1:
            return [i for i, _ in found[0]]
        if len(found) > 1:
            return None
    return None


class BankMatcher:
    def __init__(
        self,
        settlements: pd.DataFrame,
        bank: pd.DataFrame,
        cfg: ReconConfig,
        cal: BankCalendar,
        extractor: NarrationExtractor | None = None,
        min_confidence: float = 0.7,
    ) -> None:
        self.cfg = cfg
        self.cal = cal
        self.tol = cfg.tolerances.settlement_paise
        self.window = cfg.tolerances.credit_window_business_days
        self.extractor = extractor
        self.min_confidence = min_confidence
        self.out = BankMatchOutcome()

        self.setl = {
            r.settlement_id: _Setl(r.settlement_id, int(r.net_paise), r.settled_at, r.account, r.utr)
            for r in settlements.itertuples(index=False)
            if pd.notna(r.settled_at)
        }
        utr_map: dict[str, list[str]] = defaultdict(list)
        for s in self.setl.values():
            if s.utr:
                utr_map[s.utr].append(s.sid)
        self.index = UTRIndex(utr_map)

        credits = bank[bank["credit_paise"] > 0]
        self.statement_end: date = bank["txn_date"].max().date() if len(bank) else date.min
        self.credits: dict[str, _Credit] = {}
        for r in credits.itertuples(index=False):
            parsed = extract_utr(r.narration)
            _, sids = self.index.resolve(parsed.utr)
            gw = bool(sids) or is_gateway_narration(r.narration, cfg.gateway_keywords)
            self.credits[r.txn_id] = _Credit(r.txn_id, int(r.credit_paise), r.txn_date.date(), r.narration,
                                             parsed.utr, parsed.rule, sids, gw)
        self.open_s = {sid for sid, s in self.setl.items() if s.net > 0}
        for sid, s in self.setl.items():
            if s.net <= 0:
                self.out.settlement_status[sid] = "no_payout"
        self.open_c = {tid for tid, c in self.credits.items() if c.gateway}
        self.credit_state: dict[str, dict[str, Any]] = {}

    # ---- helpers ---------------------------------------------------------------------------
    def _in_window(self, s_day: date, c_day: date) -> bool:
        gap = (c_day - s_day).days
        if gap < 0 or gap > self.window * 2 + 6:
            return False
        return 0 <= self.cal.business_days_between(s_day, c_day) <= self.window

    def _s_ev(self, sid: str) -> dict[str, Any]:
        s = self.setl[sid]
        return {"source": "settlement", "id": sid, "account": s.account, "settled_at": s.day.isoformat(),
                "utr": s.utr, "net_amount": to_rupees(s.net)}

    def _c_ev(self, tid: str) -> dict[str, Any]:
        c = self.credits[tid]
        ev = {"source": "bank", "id": tid, "date": c.day.isoformat(), "narration": c.narration,
              "amount": to_rupees(c.amount), "utr_found": c.token}
        if c.llm_utr:
            ev.update(llm_utr=c.llm_utr, llm_confidence=c.llm_confidence)
        return ev

    def _record(self, sids: list[str], tids: list[str], method: str, status: str) -> int:
        diff = sum(self.credits[t].amount for t in tids) - sum(self.setl[s].net for s in sids)
        self.out.matches.append(Match("settlement_bank", sorted(sids), sorted(tids), method, status, diff))
        first_day = min(self.credits[t].day for t in tids)
        for s in sids:
            self.open_s.discard(s)
            self.out.settlement_status[s] = status
            self.out.settlement_credit_date[s] = first_day
        for t in tids:
            self.open_c.discard(t)
            self.credit_state[t] = {"match_method": method, "match_status": status,
                                    "matched_settlements": ",".join(sorted(sids))}
        if 0 < abs(diff) <= self.tol:
            self._exc(ExceptionType.ROUNDING_DIFFERENCE, sorted(sids)[0], abs(diff), sids, tids,
                      f"bank credit differs from settlement by {to_rupees(diff)}")
        return diff

    def _exc(self, kind: ExceptionType, key: str, impact: int, sids: list[str], tids: list[str], detail: str,
             needs_review: bool = False) -> None:
        days = [self.setl[s].day for s in sids] + [self.credits[t].day for t in tids]
        ev = [self._s_ev(s) for s in sorted(sids)] + [self._c_ev(t) for t in sorted(tids)]
        self.out.exceptions.append(ReconException(kind, key, int(impact), min(days) if days else None, ev, detail,
                                                   needs_review))

    def _no_utr_credits_near(self, day: date, exclude: set[str]) -> list[tuple[str, int]]:
        return sorted(
            ((t, self.credits[t].amount) for t in self.open_c
             if t not in exclude and not self.credits[t].sids and self._in_window(day, self.credits[t].day)),
            key=lambda x: (abs(self.credits[x[0]].day.toordinal() - day.toordinal()), x[0]),
        )

    def _open_settlements_near(self, day: date, exclude: set[str]) -> list[tuple[str, int]]:
        return sorted(
            ((s, self.setl[s].net) for s in self.open_s
             if s not in exclude and self._in_window(self.setl[s].day, day)),
            key=lambda x: (abs(self.setl[x[0]].day.toordinal() - day.toordinal()), x[0]),
        )

    # ---- passes ----------------------------------------------------------------------------
    def pass_utr(self) -> None:
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for tid in sorted(self.open_c):
            c = self.credits[tid]
            if c.sids:
                groups[tuple(c.sids)].append(tid)
        for key in sorted(groups):
            tids = groups[key]
            sids = [s for s in key if s in self.open_s]
            if not sids:
                continue  # UTR of an already-paid settlement: left for the unmatched-credit pass
            diff = sum(self.credits[t].amount for t in tids) - sum(self.setl[s].net for s in sids)
            if abs(diff) <= self.tol:
                method = "utr_merged" if len(sids) > 1 else "utr_split" if len(tids) > 1 else "utr"
                self._record(sids, tids, method, "auto")
                self._group_exceptions(sids, tids)
                continue
            if diff < 0 and len(sids) == 1:
                extra = unique_subset(-diff, self._no_utr_credits_near(self.setl[sids[0]].day, set(tids)),
                                      self.tol, (1, 2))
                if extra:
                    self._record(sids, tids + extra, "utr_split", "auto")
                    self._group_exceptions(sids, tids + extra)
                    continue
            if diff > 0 and len(tids) == 1:
                day = self.credits[tids[0]].day
                more = unique_subset(diff, self._open_settlements_near(day, set(sids)), self.tol, (1, 2))
                if more:
                    self._record(sids + more, tids, "utr_merged", "auto")
                    self._group_exceptions(sids + more, tids)
                    continue
            status = "short" if diff < 0 else "excess"
            self._record(sids, tids, "utr", status)
            kind = ExceptionType.SHORT_CREDIT if diff < 0 else ExceptionType.EXCESS_CREDIT
            self._exc(kind, sorted(sids)[0], abs(diff), sids, tids,
                      f"bank credited {to_rupees(sum(self.credits[t].amount for t in tids))} against settlement "
                      f"net {to_rupees(sum(self.setl[s].net for s in sids))}")

    def _group_exceptions(self, sids: list[str], tids: list[str]) -> None:
        if len(sids) > 1:
            self._exc(ExceptionType.MERGED_CREDIT, sorted(tids)[0], 0, sids, tids,
                      f"{len(sids)} settlements paid in one credit")
        if len(tids) > 1:
            self._exc(ExceptionType.SPLIT_CREDIT, sorted(sids)[0], 0, sids, tids,
                      f"settlement paid in {len(tids)} credits")

    def pass_llm(self) -> None:
        stats = self.out.narration_stats
        todo = sorted(t for t in self.open_c if not self.credits[t].sids)
        if not self.extractor or not todo:
            return
        stats["llm_attempted"] = len(todo)
        try:
            results = self.extractor.extract_many([self.credits[t].narration for t in todo])
        except Exception as e:  # the engine must finish even if the model is down
            log.warning("llm narration extraction failed: %s", e)
            stats["llm_errors"] = stats.get("llm_errors", 0) + 1
            return
        for tid, res in zip(todo, results, strict=False):
            if res is None or not res.utr:
                continue
            c = self.credits[tid]
            c.llm_utr, c.llm_confidence = res.utr, res.confidence
            stats["llm_utr_found"] = stats.get("llm_utr_found", 0) + 1
            if res.confidence < self.min_confidence:
                continue
            _, sids = self.index.resolve(res.utr)
            sids = [s for s in sids if s in self.open_s]
            if not sids:
                continue
            stats["llm_resolved"] = stats.get("llm_resolved", 0) + 1
            if abs(c.amount - sum(self.setl[s].net for s in sids)) <= self.tol:
                self._record(sids, [tid], "llm_utr", "needs_review")
                stats["llm_matched"] = stats.get("llm_matched", 0) + 1
                self._exc(ExceptionType.NARRATION_NO_UTR, tid, 0, sids, [tid],
                          f"UTR {res.utr} extracted by LLM (confidence {res.confidence:.2f}); review before accepting",
                          needs_review=True)

    def pass_amount(self) -> None:
        # 1:1 with mutual uniqueness so greedy order cannot cross-wire similar amounts.
        changed = True
        while changed:
            changed = False
            cand: dict[str, list[str]] = {}
            rev: dict[str, list[str]] = defaultdict(list)
            for tid in sorted(self.open_c):
                c = self.credits[tid]
                hits = [s for s, net in self._open_settlements_near(c.day, set()) if abs(net - c.amount) <= self.tol]
                cand[tid] = hits
                for s in hits:
                    rev[s].append(tid)
            for tid, hits in cand.items():
                if len(hits) == 1 and len(rev[hits[0]]) == 1 and tid in self.open_c and hits[0] in self.open_s:
                    self._record(hits, [tid], "amount_date", "auto")
                    self._exc(ExceptionType.NARRATION_NO_UTR, tid, 0, hits, [tid],
                              "no usable UTR in narration; matched on amount and date window")
                    changed = True
        for tid in sorted(self.open_c):
            if tid not in self.open_c:
                continue
            c = self.credits[tid]
            combo = unique_subset(c.amount, self._open_settlements_near(c.day, set()), self.tol, (2, 3))
            if combo:
                self._record(combo, [tid], "amount_merged", "needs_review")
                self._group_exceptions(combo, [tid])
        for sid in sorted(self.open_s):
            if sid not in self.open_s:
                continue
            s = self.setl[sid]
            pool = sorted((t, self.credits[t].amount) for t in self.open_c
                          if self._in_window(s.day, self.credits[t].day))
            combo = unique_subset(s.net, pool, self.tol, (2, 3))
            if combo:
                self._record([sid], combo, "amount_split", "needs_review")
                self._group_exceptions([sid], combo)

    def pass_leftovers(self) -> None:
        grace = self.cfg.tolerances.missing_grace_business_days
        for sid in sorted(self.open_s):
            s = self.setl[sid]
            if self.cal.business_days_between(s.day, self.statement_end) <= grace:
                self.out.settlement_status[sid] = "in_transit"
                continue
            self.out.settlement_status[sid] = "missing"
            near = [t for t, _ in self._no_utr_credits_near(s.day, set())][:3]
            self._exc(ExceptionType.MISSING_SETTLEMENT, sid, s.net, [sid], [],
                      f"no bank credit for settlement {sid} (UTR {s.utr}) by statement end {self.statement_end}")
            if near:
                self.out.exceptions[-1].evidence.extend(
                    {**self._c_ev(t), "note": "unmatched credit in window (different amount)"} for t in near)
        for tid in sorted(self.open_c):
            c = self.credits[tid]
            note = "UTR belongs to a settlement already matched (possible duplicate credit)" if c.sids else ""
            self._exc(ExceptionType.UNMATCHED_BANK_CREDIT, tid, c.amount, [], [tid],
                      note or "gateway-looking credit with no matching settlement")

    def run(self) -> BankMatchOutcome:
        st = self.out.narration_stats
        gw = [c for c in self.credits.values() if c.gateway]
        st["bank_credits"] = len(self.credits)
        st["gateway_credits"] = len(gw)
        st["rule_utr_found"] = sum(1 for c in gw if c.token)
        st["rule_utr_resolved"] = sum(1 for c in gw if c.sids)
        self.pass_utr()
        self.pass_llm()
        self.pass_amount()
        self.pass_leftovers()
        rows = []
        for c in self.credits.values():
            state = self.credit_state.get(c.tid, {})
            rows.append({
                "txn_id": c.tid, "gateway_related": c.gateway, "rule_utr": c.token, "rule": c.rule,
                "resolved_settlements": ",".join(c.sids), "llm_utr": c.llm_utr, "llm_confidence": c.llm_confidence,
                "match_method": state.get("match_method"),
                "match_status": state.get("match_status", "unmatched" if c.gateway else "not_gateway"),
                "matched_settlements": state.get("matched_settlements"),
            })
        self.out.credits = pd.DataFrame(rows)
        return self.out
