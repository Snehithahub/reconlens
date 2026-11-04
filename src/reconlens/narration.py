"""Rule-based UTR extraction from bank narrations and UTR resolution against settlements."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

# NEFT/RTGS UTR: 4-letter bank code, N or R, then 11+ digits (e.g. UTIBN26034001234).
_FULL_UTR = re.compile(r"(?<![A-Z0-9])([A-Z]{4}[NR]\d{11,20})(?![A-Z0-9])")
# Labelled reference, possibly a truncated fragment: "UTR N2341..." / "UTR NO: 6034001234".
_LABELLED = re.compile(r"\bUTR(?:\s*NO)?[\s:#./-]*([A-Z0-9]{8,22})\b")
# IMPS reference number: 12 digits after an IMPS marker.
_IMPS = re.compile(r"\bIMPS[\s/:-]+(?:[A-Z]+[\s/:-]+)?(\d{12})\b")

MIN_FRAGMENT = 8


@dataclass(frozen=True)
class ParsedNarration:
    utr: str | None
    rule: str | None  # full | labelled | imps


def extract_utr(narration: str) -> ParsedNarration:
    text = narration.upper()
    if m := _FULL_UTR.search(text):
        return ParsedNarration(m.group(1), "full")
    if m := _LABELLED.search(text):
        token = m.group(1)
        if sum(c.isdigit() for c in token) >= 6:
            return ParsedNarration(token, "labelled")
    if m := _IMPS.search(text):
        return ParsedNarration(m.group(1), "imps")
    return ParsedNarration(None, None)


def is_gateway_narration(narration: str, keywords: Iterable[str]) -> bool:
    text = re.sub(r"\s+", " ", narration.upper())
    return any(k.upper() in text for k in keywords)


class UTRIndex:
    """Resolve a full UTR or a trailing fragment to the settlement ids that carry it."""

    def __init__(self, utr_to_settlements: dict[str, list[str]]) -> None:
        self._exact = {u.upper(): sorted(s) for u, s in utr_to_settlements.items() if u}
        self._suffix: dict[str, set[str]] = defaultdict(set)
        for utr in self._exact:
            for i in range(0, len(utr) - MIN_FRAGMENT + 1):
                self._suffix[utr[i:]].add(utr)

    def resolve(self, token: str | None) -> tuple[str | None, list[str]]:
        """Return (full_utr, settlement_ids). Fragments must identify exactly one UTR."""
        if not token:
            return None, []
        token = token.upper()
        if token in self._exact:
            return token, self._exact[token]
        hits = self._suffix.get(token, set())
        if len(hits) == 1:
            utr = next(iter(hits))
            return utr, self._exact[utr]
        return None, []
