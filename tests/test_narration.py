import pytest

from reconlens.narration import UTRIndex, extract_utr, is_gateway_narration

U = "UTIBN26034001234"


@pytest.mark.parametrize("text", [
    f"NEFT CR-UTIB0000123-PAYFLOW PAYMENTS PVT LTD-SETTL 0302-{U}",
    f"NEFT-{U}-PAYFLOW PAYMENTS PRIVATE LIMITED-SETTLEMENT",
    f"BY TRANSFER NEFT*{U}*PAYFLOW SETTL",
    f"neft cr {U.lower()} payflow setl 0302",
])
def test_standard_formats_extract_full_utr(text):
    p = extract_utr(text)
    assert p.utr == U and p.rule == "full"


def test_labelled_fragment():
    p = extract_utr("NEFT CR-PAYFLOW SETTL 0302 UTR 6034001234")
    assert p.utr == "6034001234" and p.rule == "labelled"


def test_imps_reference():
    assert extract_utr("IMPS/603412345678/PAYFLOW").utr == "603412345678"


@pytest.mark.parametrize("text", [
    "NEFT CR-PAYFLOW PAYMENTS PVT LTD SETTLEMENT",
    "PAYFLOW SETTLEMENT REF UTIB N26 034 001234 NEFT CR",
    "NEFT CR PAYFLOW UTIBN260 REF 34001234",
    "PAYFLOW/U.T.R/UTIB.N26034.001234/SETTL",
    "UTR NOT AVAILABLE PAYFLOW",
])
def test_no_or_unparseable_utr(text):
    assert extract_utr(text).utr is None


def test_utr_index_exact_suffix_and_ambiguity():
    idx = UTRIndex({U: ["s1"], "HDFCN26034001234": ["s2"], "UTIBN26035999999": ["s3", "s4"]})
    assert idx.resolve(U) == (U, ["s1"])
    assert idx.resolve("35999999") == ("UTIBN26035999999", ["s3", "s4"])
    # suffix shared by two UTRs is ambiguous -> unresolved
    assert idx.resolve("26034001234") == (None, [])
    assert idx.resolve("1234") == (None, [])  # too short
    assert idx.resolve(None) == (None, [])


def test_gateway_keyword():
    assert is_gateway_narration("neft cr pay  flow settl", ["PAY FLOW"])
    assert not is_gateway_narration("NEFT CR-ACME RETAIL", ["PAYFLOW"])
