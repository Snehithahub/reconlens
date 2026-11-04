"""Plain-English explanation and next step per exception cluster.

Always produces a rule-based template; when an LLM is configured it is asked for a
richer write-up (e.g. a draft email to gateway support) that must cite only evidence
ids we gave it. Invalid citations are dropped, invalid output falls back to the template.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from pydantic import ValidationError

from reconlens.config import ReconConfig
from reconlens.llm.client import JSONLLM, LLMError
from reconlens.llm.schemas import ClusterExplanation
from reconlens.money import fmt_inr

log = logging.getLogger(__name__)
MAX_EVIDENCE = 12

SYSTEM = (
    "You help the finance-ops team of an Indian D2C merchant reconcile payment-gateway settlements. "
    "Given one cluster of reconciliation exceptions with evidence rows, write a short explanation for a finance "
    "executive. Use only facts in the evidence; never invent amounts or ids. Cite evidence ids you rely on. "
    "If the gateway owes the merchant money, include a polite, specific draft email to the gateway's support "
    "team listing the ids and amounts. Reply with JSON only: {\"summary\": str, \"likely_cause\": str, "
    "\"next_step\": str, \"draft_email\": str|null, \"cited_evidence_ids\": [str]}"
)


def cluster_payloads(exc: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for t, g in exc.groupby("type", sort=True):
        g = g.sort_values("impact_paise", key=abs, ascending=False)
        evidence = []
        for ev in g["evidence"].head(MAX_EVIDENCE):
            evidence.extend(json.loads(ev) if isinstance(ev, str) else ev)
        out.append({
            "cluster": str(t), "title": g["title"].iat[0], "count": len(g),
            "impact_paise": int(g["impact_paise"].sum()), "recoverable": bool(g["recoverable"].iat[0]),
            "suggested_action": g["suggested_action"].iat[0], "evidence": evidence,
        })
    return out


def template_explanation(c: dict[str, Any], cfg: ReconConfig) -> dict[str, Any]:
    ids = sorted({str(e["id"]) for e in c["evidence"]})[:5]
    money = fmt_inr(c["impact_paise"])
    summary = f"{c['count']} exception(s) of type '{c['title']}' with total impact {money}."
    email = None
    if c["recoverable"] and c["impact_paise"] > 0:
        email = (
            f"Subject: Reconciliation query - {c['title']} ({c['count']} items, {money})\n\n"
            f"Hello {cfg.rate_card.gateway} support,\n\nWhile reconciling our settlements we found {c['count']} "
            f"item(s) of type '{c['title']}' totalling {money}. Examples: {', '.join(ids)}. "
            "Please review and confirm the reversal or adjustment, or share the breakup if we have misread "
            "the report.\n\nRegards,\nFinance team"
        )
    return {"cluster": c["cluster"], "source": "template", "summary": summary,
            "likely_cause": "See the exception detail and evidence rows.", "next_step": c["suggested_action"],
            "draft_email": email, "cited_evidence_ids": ids, "count": c["count"], "impact_paise": c["impact_paise"]}


def explain_clusters(exc: pd.DataFrame, cfg: ReconConfig, llm: JSONLLM | None) -> list[dict[str, Any]]:
    results = []
    for c in cluster_payloads(exc):
        base = template_explanation(c, cfg)
        if llm is None:
            results.append(base)
            continue
        user = json.dumps({k: c[k] for k in ("cluster", "title", "count", "recoverable", "suggested_action",
                                              "evidence")} | {"total_impact": fmt_inr(c["impact_paise"])},
                          default=str)
        try:
            parsed = ClusterExplanation.model_validate(llm.complete_json(SYSTEM, user, max_tokens=1200))
        except (LLMError, ValidationError) as e:
            log.warning("explanation for %s fell back to template: %s", c["cluster"], e)
            results.append({**base, "llm_error": str(e)[:200]})
            continue
        known = {str(e["id"]) for e in c["evidence"]}
        cited = [i for i in parsed.cited_evidence_ids if i in known]
        dropped = [i for i in parsed.cited_evidence_ids if i not in known]
        results.append({**base, **parsed.model_dump(), "cited_evidence_ids": cited, "dropped_citations": dropped,
                        "source": "llm"})
    return results
