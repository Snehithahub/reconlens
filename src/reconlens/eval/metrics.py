"""Score an engine result against generator gold labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from reconlens.engine.types import EngineResult
from reconlens.narration import UTRIndex


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 4),
            "recall": None if r is None else round(r, 4)}


def detection_scores(result: EngineResult, gold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gold_sets: dict[str, set[str]] = defaultdict(set)
    for lab in gold["labels"]:
        gold_sets[lab["type"]].add(str(lab["key"]))
    pred: dict[str, set[str]] = defaultdict(set)
    for e in result.exceptions:
        pred[str(e.type)].add(e.key)
    out = {}
    for t in sorted(gold_sets):
        g, p = gold_sets[t], pred.get(t, set())
        out[t] = _prf(len(g & p), len(p - g), len(g - p))
    extra = sorted(set(pred) - set(gold_sets))
    for t in extra:  # types the generator never injects: every hit is a false positive
        out[t] = _prf(0, len(pred[t]), 0)
    return out


def match_quality(result: EngineResult, gold: dict[str, Any]) -> dict[str, Any]:
    truth = {t: set(s) for t, s in gold["bank_to_settlements"].items()}
    edges = [(t, s, m.status) for m in result.matches for t in m.bank_txn_ids for s in m.settlement_ids]
    wrong = [(t, s) for t, s, _ in edges if s not in truth.get(t, set())]
    auto_edges = [(t, s) for t, s, st in edges if st == "auto"]
    auto_wrong = [(t, s) for t, s in auto_edges if s not in truth.get(t, set())]
    gold_edges = {(t, s) for t, ss in truth.items() for s in ss}
    found = {(t, s) for t, s, _ in edges}
    return {
        "match_edges": len(edges),
        "false_match_edges": len(wrong),
        "false_match_rate": round(len(wrong) / len(edges), 4) if edges else 0.0,
        "auto_false_match_rate": round(len(auto_wrong) / len(auto_edges), 4) if auto_edges else 0.0,
        "gold_edges": len(gold_edges),
        "gold_edges_recovered": len(gold_edges & found),
    }


def narration_rates(result: EngineResult, gold: dict[str, Any]) -> dict[str, Any]:
    """Share of gateway credits whose narration carries a UTR where we recover the right settlement(s)."""
    s = result.settlements
    index = UTRIndex({u: list(g["settlement_id"]) for u, g in s.groupby("utr")})
    bank = result.bank.set_index("txn_id")
    by_variant: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "rule": 0, "llm_assisted": 0})
    for tid, info in gold["narrations"].items():
        if info["variant"] == "absent" or tid not in bank.index:
            continue
        truth = set(gold["bank_to_settlements"].get(tid, []))
        row = bank.loc[tid]
        rule_ok = set(filter(None, str(row.get("resolved_settlements") or "").split(","))) == truth
        llm_ok = False
        llm_utr = row.get("llm_utr")
        if isinstance(llm_utr, str) and llm_utr:
            llm_ok = set(index.resolve(llm_utr)[1]) == truth
        v = by_variant[info["variant"]]
        v["n"] += 1
        v["rule"] += int(rule_ok)
        v["llm_assisted"] += int(rule_ok or llm_ok)
    total = {k: sum(v[k] for v in by_variant.values()) for k in ("n", "rule", "llm_assisted")}
    return {
        "by_variant": dict(by_variant),
        "with_utr": total["n"],
        "rule_extraction_rate": round(total["rule"] / total["n"], 4) if total["n"] else None,
        "llm_assisted_extraction_rate": round(total["llm_assisted"] / total["n"], 4) if total["n"] else None,
    }
