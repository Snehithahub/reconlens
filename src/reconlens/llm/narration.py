"""LLM fallback for bank narrations the UTR regex could not parse.

The model's output is validated by Pydantic; anything malformed is dropped. A match
built on an LLM-extracted UTR is always marked needs_review by the engine.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from pydantic import ValidationError

from reconlens.llm.client import JSONLLM, LLMError
from reconlens.llm.schemas import NarrationExtraction

log = logging.getLogger(__name__)

SYSTEM = (
    "You extract payment references from Indian bank statement narrations. A UTR is a NEFT/RTGS reference: "
    "4 letters (bank code), then N or R, then 11+ digits, e.g. UTIBN26034001234. Narrations may split the UTR "
    "with spaces, dots, dashes or words such as REF. Rejoin the pieces only when you are confident they belong "
    "to one UTR. Never invent characters. Reply with JSON only: "
    '{"items": [{"index": int, "utr": string|null, "batch_ref": string|null, "value_date": "YYYY-MM-DD"|null, '
    '"confidence": number 0-1}]}'
)


class LLMNarrationExtractor:
    def __init__(self, client: JSONLLM, batch_size: int = 20) -> None:
        self.client = client
        self.batch_size = batch_size

    def extract_many(self, narrations: Sequence[str]) -> list[NarrationExtraction | None]:
        out: list[NarrationExtraction | None] = [None] * len(narrations)
        for start in range(0, len(narrations), self.batch_size):
            chunk = narrations[start: start + self.batch_size]
            user = json.dumps([{"index": i, "narration": n} for i, n in enumerate(chunk)])
            try:
                raw = self.client.complete_json(SYSTEM, user, max_tokens=120 * len(chunk) + 200)
            except LLMError as e:
                log.warning("narration batch %d failed: %s", start, e)
                continue
            for item in raw.get("items", []) if isinstance(raw.get("items"), list) else []:
                try:
                    ex = NarrationExtraction.model_validate(item)
                except ValidationError:
                    continue
                if ex.index < len(chunk) and out[start + ex.index] is None:
                    out[start + ex.index] = ex
        return out
