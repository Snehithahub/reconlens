"""End-to-end: generated data with every injected discrepancy type is detected exactly."""

import io

from reconlens.engine import reconcile
from reconlens.eval.metrics import detection_scores, match_quality, narration_rates
from reconlens.ingest import RawInput, load_bank, load_gateway, load_ledger

GOLD_TYPES = {"missing_settlement", "short_credit", "mdr_overcharge", "gst_mismatch", "duplicate_refund",
              "refund_without_order", "order_paid_no_payment", "payment_not_recorded", "settlement_delayed",
              "narration_no_utr", "split_credit", "merged_credit", "rounding_difference"}


def _raw(df, name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return RawInput(name, buf.getvalue().encode())


def _run(data, cfg, extractor=None):
    return reconcile(load_gateway(_raw(data.gateway, "g.csv")), load_bank(_raw(data.bank, "b.csv")),
                     load_ledger(_raw(data.ledger, "l.csv")), cfg, extractor=extractor)


def test_generator_injects_every_type(small_dataset):
    assert {lab["type"] for lab in small_dataset.gold["labels"]} == GOLD_TYPES


def test_generator_is_deterministic(cfg):
    from reconlens.datagen import GenConfig, generate

    a = generate(GenConfig(n_payments=500, seed=3), cfg)
    b = generate(GenConfig(n_payments=500, seed=3), cfg)
    assert a.gateway.equals(b.gateway) and a.bank.equals(b.bank) and a.ledger.equals(b.ledger)
    assert a.gold == b.gold


def test_all_types_detected_with_no_false_matches(small_dataset, cfg):
    res = _run(small_dataset, cfg)
    scores = detection_scores(res, small_dataset.gold)
    for t in GOLD_TYPES:
        assert scores[t]["recall"] == 1.0, (t, scores[t])
        assert scores[t]["precision"] == 1.0, (t, scores[t])
    assert match_quality(res, small_dataset.gold)["false_match_rate"] == 0.0


def test_hard_narrations_defeat_rules_but_fake_llm_recovers_them(small_dataset, cfg):
    from reconlens.llm.schemas import NarrationExtraction

    truth = {tid: v["utr"] for tid, v in small_dataset.gold["narrations"].items() if v["variant"] == "hard"}
    by_narr = dict(zip(small_dataset.bank["narration"], small_dataset.bank["txn_id"], strict=True))

    class Oracle:  # stands in for a perfect model
        def extract_many(self, narrations):
            return [NarrationExtraction(index=i, utr=truth.get(by_narr.get(n)), confidence=0.9)
                    if truth.get(by_narr.get(n)) else None for i, n in enumerate(narrations)]

    rule = narration_rates(_run(small_dataset, cfg), small_dataset.gold)
    res = _run(small_dataset, cfg, Oracle())
    assisted = narration_rates(res, small_dataset.gold)
    assert rule["by_variant"]["hard"]["rule"] == 0
    assert assisted["by_variant"]["hard"]["llm_assisted"] == assisted["by_variant"]["hard"]["n"] > 0
    assert any(m.method == "llm_utr" and m.status == "needs_review" for m in res.matches)
    assert match_quality(res, small_dataset.gold)["false_match_rate"] == 0.0
