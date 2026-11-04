"""Each settlement<->bank matching pass on hand-built inputs."""

from datetime import date

from conftest import bank_df, settlements_df
from reconlens.engine.bank_match import BankMatcher, unique_subset
from reconlens.engine.types import ExceptionType as T
from reconlens.llm.schemas import NarrationExtraction

D = date(2026, 2, 3)  # Tuesday
U1, U2, U3 = "UTIBN26034000001", "UTIBN26034000002", "HDFCN26034000003"


def run(cfg, cal, setl, bank, extractor=None):
    return BankMatcher(settlements_df(setl), bank_df(bank), cfg, cal, extractor).run()


def types(out):
    return sorted((str(e.type), e.key) for e in out.exceptions)


def test_exact_utr_one_to_one(cfg, cal):
    out = run(cfg, cal, [("s1", 1_000_000, D, U1)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 1_000_000)])
    assert [(m.settlement_ids, m.bank_txn_ids, m.method, m.status) for m in out.matches] == [
        (["s1"], ["b1"], "utr", "auto")]
    assert out.exceptions == []


def test_rounding_within_tolerance_is_matched_and_flagged(cfg, cal):
    out = run(cfg, cal, [("s1", 1_000_000, D, U1)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 999_963)])
    assert out.settlement_status["s1"] == "auto"
    assert types(out) == [("rounding_difference", "s1")]
    assert out.exceptions[0].impact_paise == 37


def test_tolerance_boundary(cfg, cal):
    tol = cfg.tolerances.settlement_paise
    at = run(cfg, cal, [("s1", 1_000_000, D, U1)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 1_000_000 - tol)])
    over = run(cfg, cal, [("s1", 1_000_000, D, U1)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 1_000_000 - tol - 1)])
    assert at.settlement_status["s1"] == "auto"
    assert over.settlement_status["s1"] == "short"
    assert types(over) == [("short_credit", "s1")] and over.exceptions[0].impact_paise == tol + 1


def test_excess_credit(cfg, cal):
    out = run(cfg, cal, [("s1", 1_000_000, D, U1)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 1_200_000)])
    assert types(out) == [("excess_credit", "s1")]


def test_fragment_utr_resolves_by_suffix(cfg, cal):
    out = run(cfg, cal, [("s1", 500_000, D, U1)], [("b1", D, f"NEFT CR-PAYFLOW SETTL UTR {U1[-10:]}", 500_000)])
    assert out.matches[0].method == "utr"


def test_merged_credit_shared_utr(cfg, cal):
    out = run(cfg, cal, [("s1", 300_000, D, U1), ("s2", 200_000, D, U1)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 500_000)])
    assert out.matches[0].method == "utr_merged" and out.matches[0].settlement_ids == ["s1", "s2"]
    assert types(out) == [("merged_credit", "b1")]


def test_merged_credit_completed_from_other_settlement(cfg, cal):
    out = run(cfg, cal, [("s1", 300_000, D, U1), ("s2", 200_000, D, U2)], [("b1", D, f"NEFT-{U1}-PAYFLOW", 500_000)])
    assert out.matches[0].settlement_ids == ["s1", "s2"]
    assert ("merged_credit", "b1") in types(out)


def test_split_credit_second_part_without_utr(cfg, cal):
    out = run(cfg, cal, [("s1", 1_000_000, D, U1)], [
        ("b1", D, f"NEFT-{U1}-PAYFLOW", 600_000), ("b2", date(2026, 2, 4), "NEFT CR PAYFLOW BAL", 400_000)])
    assert out.matches[0].bank_txn_ids == ["b1", "b2"] and out.matches[0].method == "utr_split"
    assert types(out) == [("split_credit", "s1")]


def test_amount_date_match_when_no_utr(cfg, cal):
    out = run(cfg, cal, [("s1", 777_700, D, U1)], [("b1", date(2026, 2, 4), "NEFT CR-PAYFLOW PAYMENTS", 777_700)])
    assert out.matches[0].method == "amount_date"
    assert types(out) == [("narration_no_utr", "b1")]


def test_amount_date_respects_window(cfg, cal):
    late = date(2026, 2, 13)  # > 3 business days after D
    out = run(cfg, cal, [("s1", 777_700, D, U1)], [("b1", late, "NEFT CR-PAYFLOW PAYMENTS", 777_700)])
    assert out.matches == []
    assert ("missing_settlement", "s1") in types(out) and ("unmatched_bank_credit", "b1") in types(out)


def test_amount_date_ambiguous_is_not_guessed(cfg, cal):
    out = run(cfg, cal, [("s1", 500_000, D, U1), ("s2", 500_000, D, U2)],
              [("b1", D, "PAYFLOW SETTL", 500_000)])
    assert out.matches == []


def test_mutual_uniqueness_pairs_two_credits(cfg, cal):
    out = run(cfg, cal, [("s1", 500_000, D, U1), ("s2", 800_000, D, U2)],
              [("b1", D, "PAYFLOW SETTL", 800_000), ("b2", D, "PAYFLOW SETTL", 500_000)])
    pairs = sorted((m.settlement_ids[0], m.bank_txn_ids[0]) for m in out.matches)
    assert pairs == [("s1", "b2"), ("s2", "b1")]


def test_missing_vs_in_transit(cfg, cal):
    out = run(cfg, cal, [("old", 500_000, D, U1), ("new", 400_000, date(2026, 2, 19), U2)],
              [("b9", date(2026, 2, 20), "NEFT CR-ACME DISTRIBUTORS", 1_000)])
    assert out.settlement_status == {"old": "missing", "new": "in_transit"}
    assert types(out) == [("missing_settlement", "old")]


def test_non_gateway_credit_ignored(cfg, cal):
    out = run(cfg, cal, [("s1", 500_000, D, U1)],
              [("b1", D, f"NEFT-{U1}-PAYFLOW", 500_000), ("b2", D, "IMPS/603412345678/CUSTOMER", 500_000)])
    assert len(out.matches) == 1 and types(out) == []


def test_duplicate_credit_for_matched_utr_is_unmatched(cfg, cal):
    out = run(cfg, cal, [("s1", 500_000, D, U1)],
              [("b1", D, f"NEFT-{U1}-PAYFLOW", 500_000), ("b2", date(2026, 2, 5), f"NEFT-{U1}-PAYFLOW", 500_000)])
    # both credits carry the UTR: grouped, the sum is double -> excess, never silently accepted
    assert ("excess_credit", "s1") in types(out)


def test_zero_or_negative_net_is_no_payout(cfg, cal):
    out = run(cfg, cal, [("s1", -5_000, D, U1)], [])
    assert out.settlement_status["s1"] == "no_payout" and out.exceptions == []


def test_unique_subset():
    items = [("a", 100), ("b", 250), ("c", 400)]
    assert unique_subset(600, items, 0, (2,)) is None
    assert unique_subset(350, items, 0, (2,)) == ["a", "b"]
    assert unique_subset(650, items, 0, (1, 2)) == ["b", "c"]
    assert unique_subset(500, [("a", 250), ("b", 250), ("c", 250)], 0, (2,)) is None  # ambiguous


class FakeExtractor:
    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def extract_many(self, narrations):
        self.calls += 1
        return [self.answers.get(n) for n in narrations]


HARD = "PAYFLOW/U.T.R/UTIB.N26034.000001/SETTL"


def test_llm_extracted_utr_match_needs_review(cfg, cal):
    fake = FakeExtractor({HARD: NarrationExtraction(index=0, utr="UTIB.N26034.000001", confidence=0.93)})
    out = run(cfg, cal, [("s1", 500_000, D, U1), ("s2", 500_000, D, U2)], [("b1", D, HARD, 500_000)], fake)
    assert out.matches[0].method == "llm_utr" and out.matches[0].status == "needs_review"
    assert out.settlement_status["s1"] == "needs_review"
    exc = out.exceptions[0]
    assert exc.type == T.NARRATION_NO_UTR and exc.needs_review
    assert out.narration_stats["llm_matched"] == 1


def test_llm_low_confidence_is_ignored(cfg, cal):
    fake = FakeExtractor({HARD: NarrationExtraction(index=0, utr=U1, confidence=0.3)})
    out = run(cfg, cal, [("s1", 500_000, D, U1), ("s2", 500_000, D, U2)], [("b1", D, HARD, 500_000)], fake)
    assert out.matches == []


def test_llm_utr_with_wrong_amount_is_not_trusted(cfg, cal):
    fake = FakeExtractor({HARD: NarrationExtraction(index=0, utr=U1, confidence=0.99)})
    out = run(cfg, cal, [("s1", 900_000, D, U1)], [("b1", D, HARD, 500_000)], fake)
    assert all(m.method != "llm_utr" for m in out.matches)


def test_llm_failure_degrades_gracefully(cfg, cal):
    class Boom:
        def extract_many(self, narrations):
            raise RuntimeError("provider down")

    out = run(cfg, cal, [("s1", 500_000, D, U1)], [("b1", D, HARD, 500_000)], Boom())
    assert out.matches[0].method == "amount_date"  # rules still match it
    assert out.narration_stats["llm_errors"] == 1
