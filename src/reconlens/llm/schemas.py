"""Pydantic contracts for everything an LLM returns. Nothing from a model is used unvalidated."""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, Field, field_validator

_UTR_SHAPE = re.compile(r"^[A-Z0-9]{8,22}$")


class NarrationExtraction(BaseModel):
    index: int = Field(ge=0)
    utr: str | None = None
    batch_ref: str | None = None
    value_date: date | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("utr", mode="before")
    @classmethod
    def _clean_utr(cls, v: object) -> str | None:
        if v in (None, "", "null", "NONE", "none"):
            return None
        s = re.sub(r"[\s.\-/]", "", str(v)).upper()
        if not _UTR_SHAPE.match(s):
            raise ValueError("utr must be 8-22 alphanumeric characters")
        return s


class NarrationBatch(BaseModel):
    items: list[NarrationExtraction]


class ClusterExplanation(BaseModel):
    summary: str = Field(min_length=10, max_length=1500)
    likely_cause: str = Field(min_length=3, max_length=800)
    next_step: str = Field(min_length=3, max_length=800)
    draft_email: str | None = Field(None, max_length=4000)
    cited_evidence_ids: list[str] = Field(default_factory=list)


class SQLAnswer(BaseModel):
    sql: str = Field(min_length=6, max_length=4000)
    explanation: str = Field("", max_length=1000)
