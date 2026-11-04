"""python -m reconlens.eval [--sizes 10000,100000] [--llm]

Generates seeded data with gold labels, runs ingest + reconciliation, and reports
per-type precision/recall, auto-match rate, false-match rate, runtime and the
narration UTR extraction rate (rule-only, and LLM-assisted when --llm is given).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from reconlens.config import get_settings, load_recon_config
from reconlens.datagen import GenConfig, generate
from reconlens.engine import reconcile
from reconlens.eval.metrics import detection_scores, match_quality, narration_rates
from reconlens.ingest import RawInput, load_bank, load_gateway, load_ledger


def run_one(n: int, seed: int, use_llm: bool, stress: bool = False) -> dict:
    cfg = load_recon_config()
    t0 = time.perf_counter()
    gen = GenConfig(n_payments=n, seed=seed)
    if stress:  # most narrations lose their UTR, so matching leans on amount + date
        gen = GenConfig(n_payments=n, seed=seed, narration_absent_rate=0.5, narration_hard_rate=0.3,
                        narration_fragment_rate=0.1, accounts=("WEB", "APP", "B2B", "POS", "MKT", "WHS"))
    data = generate(gen, cfg)
    gen_s = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as d:
        data.write(d)
        t1 = time.perf_counter()
        gw = load_gateway(RawInput.from_path(Path(d) / "gateway_report.csv"))
        bank = load_bank(RawInput.from_path(Path(d) / "bank_statement.csv"))
        led = load_ledger(RawInput.from_path(Path(d) / "order_ledger.csv"))
        load_s = time.perf_counter() - t1
    extractor = None
    if use_llm:
        from reconlens.llm.client import build_client
        from reconlens.llm.narration import LLMNarrationExtractor

        settings = get_settings()
        client = build_client(settings)
        if client is None:
            raise SystemExit("--llm needs LLM_PROVIDER and LLM_API_KEY set (see .env.example)")
        extractor = LLMNarrationExtractor(client, batch_size=settings.llm_narration_batch_size)
    t2 = time.perf_counter()
    result = reconcile(gw, bank, led, cfg, extractor=extractor)
    recon_s = time.perf_counter() - t2
    s = result.summary
    return {
        "payments": n, "seed": seed, "llm": use_llm, "stress": stress,
        "rows": data.gold["counts"],
        "runtime_s": {"generate": round(gen_s, 2), "ingest": round(load_s, 2), "reconcile": round(recon_s, 2),
                      "ingest_plus_reconcile": round(load_s + recon_s, 2)},
        "auto_match_rate": s["auto_match_rate"],
        "settlement_auto_match_rate": s["settlement_auto_match_rate"],
        "match_quality": match_quality(result, data.gold),
        "detection": detection_scores(result, data.gold),
        "narration": narration_rates(result, data.gold),
        "narration_stats": s["narration"],
    }


def to_markdown(results: list[dict]) -> str:
    lines = []
    for r in results:
        mode = ", stress: 80% of narrations without a parseable UTR, 6 accounts" if r.get("stress") else ""
        lines.append(f"### {r['payments']:,} payments (seed {r['seed']}, LLM {'on' if r['llm'] else 'off'}{mode})\n")
        rt = r["runtime_s"]
        mq = r["match_quality"]
        lines += [
            f"- rows: {r['rows']}",
            f"- runtime: ingest {rt['ingest']}s + reconcile {rt['reconcile']}s = {rt['ingest_plus_reconcile']}s "
            f"(data generation {rt['generate']}s, not counted)",
            f"- payment auto-match rate: {r['auto_match_rate']:.2%}; settlement auto-match rate: "
            f"{r['settlement_auto_match_rate']:.2%}",
            f"- false-match rate (settlement-to-bank edges): {mq['false_match_rate']:.2%} "
            f"({mq['false_match_edges']}/{mq['match_edges']}); gold edges recovered "
            f"{mq['gold_edges_recovered']}/{mq['gold_edges']}",
            f"- narration UTR extraction (credits whose narration carries a UTR): rule-only "
            f"{r['narration']['rule_extraction_rate']:.2%}"
            + (f", LLM-assisted {r['narration']['llm_assisted_extraction_rate']:.2%}" if r["llm"] else
               ", LLM-assisted: pending a live run"),
            "",
            "| exception type | gold | TP | FP | FN | precision | recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for t, m in r["detection"].items():
            p = "-" if m["precision"] is None else f"{m['precision']:.3f}"
            rc = "-" if m["recall"] is None else f"{m['recall']:.3f}"
            lines.append(f"| {t} | {m['tp'] + m['fn']} | {m['tp']} | {m['fp']} | {m['fn']} | {p} | {rc} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,100000")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--llm", action="store_true", help="use the configured LLM for narration fallback")
    ap.add_argument("--stress", action="store_true", help="also run a hard-narration stress scenario")
    ap.add_argument("--out", default="eval_results")
    args = ap.parse_args()
    results = [run_one(int(n), args.seed, args.llm) for n in args.sizes.split(",")]
    if args.stress:
        results.append(run_one(int(args.sizes.split(",")[0]), args.seed, args.llm, stress=True))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    suffix = "_llm" if args.llm else ""
    (out / f"results{suffix}.json").write_text(json.dumps(results, indent=1))
    md = to_markdown(results)
    (out / f"results{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
