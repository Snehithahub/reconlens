"""Exception taxonomy and result containers produced by the reconciliation engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd


class ExceptionType(StrEnum):
    MISSING_SETTLEMENT = "missing_settlement"
    SHORT_CREDIT = "short_credit"
    EXCESS_CREDIT = "excess_credit"
    MDR_OVERCHARGE = "mdr_overcharge"
    GST_MISMATCH = "gst_mismatch"
    DUPLICATE_REFUND = "duplicate_refund"
    REFUND_WITHOUT_ORDER = "refund_without_order"
    ORDER_PAID_NO_PAYMENT = "order_paid_no_payment"
    PAYMENT_NOT_RECORDED = "payment_not_recorded"
    PAYMENT_WITHOUT_ORDER = "payment_without_order"
    AMOUNT_MISMATCH = "amount_mismatch"
    DUPLICATE_CAPTURE = "duplicate_capture"
    SETTLEMENT_DELAYED = "settlement_delayed"
    NARRATION_NO_UTR = "narration_no_utr"
    SPLIT_CREDIT = "split_credit"
    MERGED_CREDIT = "merged_credit"
    ROUNDING_DIFFERENCE = "rounding_difference"
    UNMATCHED_BANK_CREDIT = "unmatched_bank_credit"


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class TypeInfo:
    title: str
    severity: Severity
    recoverable: bool  # impact is money the merchant can claim back
    action: str


TAXONOMY: dict[ExceptionType, TypeInfo] = {
    ExceptionType.MISSING_SETTLEMENT: TypeInfo(
        "Settlement not credited to bank", Severity.HIGH, True,
        "Raise a ticket with the gateway quoting settlement_id and UTR; ask for the payout status and bank proof.",
    ),
    ExceptionType.SHORT_CREDIT: TypeInfo(
        "Bank credit lower than settlement", Severity.HIGH, True,
        "Ask the gateway for a breakup of the deduction; claim the shortfall if no adjustment explains it.",
    ),
    ExceptionType.EXCESS_CREDIT: TypeInfo(
        "Bank credit higher than settlement", Severity.MEDIUM, False,
        "Confirm with the gateway whether this includes another payout; expect a recovery debit otherwise.",
    ),
    ExceptionType.MDR_OVERCHARGE: TypeInfo(
        "MDR charged above contracted rate", Severity.HIGH, True,
        "Email gateway support with the payment ids, contracted vs charged rate, and request a fee reversal.",
    ),
    ExceptionType.GST_MISMATCH: TypeInfo(
        "GST on fee mis-computed", Severity.MEDIUM, True,
        "Ask the gateway to correct the tax invoice; excess GST must be reversed or a credit note issued.",
    ),
    ExceptionType.DUPLICATE_REFUND: TypeInfo(
        "Duplicate refund", Severity.HIGH, True,
        "Check whether the customer received two refunds; recover from the customer or ask the gateway to reverse.",
    ),
    ExceptionType.REFUND_WITHOUT_ORDER: TypeInfo(
        "Refund with no matching order", Severity.HIGH, False,
        "Find who initiated the refund in the gateway dashboard; link it to an order or treat as a loss.",
    ),
    ExceptionType.ORDER_PAID_NO_PAYMENT: TypeInfo(
        "Order marked paid, no captured payment", Severity.HIGH, False,
        "Hold fulfilment if not shipped; verify the payment with the customer before marking paid.",
    ),
    ExceptionType.PAYMENT_NOT_RECORDED: TypeInfo(
        "Payment captured but order not updated", Severity.MEDIUM, False,
        "Update the order to paid and fulfil, or refund the customer if the order was cancelled.",
    ),
    ExceptionType.PAYMENT_WITHOUT_ORDER: TypeInfo(
        "Payment with no order in ledger", Severity.MEDIUM, False,
        "Find the order (payment link / manual sale?) and record it, or refund the customer.",
    ),
    ExceptionType.AMOUNT_MISMATCH: TypeInfo(
        "Payment amount differs from order amount", Severity.MEDIUM, False,
        "Check discounts or partial payments on the order; correct the ledger or collect the difference.",
    ),
    ExceptionType.DUPLICATE_CAPTURE: TypeInfo(
        "Order paid twice", Severity.HIGH, False,
        "Refund the extra payment to the customer.",
    ),
    ExceptionType.SETTLEMENT_DELAYED: TypeInfo(
        "Settlement later than contracted SLA", Severity.LOW, False,
        "Track with the gateway; repeated delays are a working-capital cost worth escalating.",
    ),
    ExceptionType.NARRATION_NO_UTR: TypeInfo(
        "Bank narration has no usable UTR", Severity.LOW, False,
        "Matched on amount and date only; confirm the pairing once, then accept.",
    ),
    ExceptionType.SPLIT_CREDIT: TypeInfo(
        "Settlement paid as several bank credits", Severity.LOW, False,
        "Confirm the parts add up to the settlement and accept the grouped match.",
    ),
    ExceptionType.MERGED_CREDIT: TypeInfo(
        "Several settlements paid as one bank credit", Severity.LOW, False,
        "Confirm the settlements in the group and accept the grouped match.",
    ),
    ExceptionType.ROUNDING_DIFFERENCE: TypeInfo(
        "Paise-level rounding difference", Severity.LOW, False,
        "Within tolerance; write off to rounding.",
    ),
    ExceptionType.UNMATCHED_BANK_CREDIT: TypeInfo(
        "Gateway-looking bank credit with no settlement", Severity.MEDIUM, False,
        "Identify the source of the credit; it may belong to a settlement outside this report period.",
    ),
}


@dataclass
class ReconException:
    type: ExceptionType
    key: str  # primary entity id (payment, settlement, refund, order or bank txn)
    impact_paise: int
    event_date: date | None
    evidence: list[dict[str, Any]]
    detail: str
    needs_review: bool = False
    exception_id: str = ""

    @property
    def info(self) -> TypeInfo:
        return TAXONOMY[self.type]

    def assign_id(self, run_id: str) -> None:
        self.exception_id = "exc_" + hashlib.sha1(f"{run_id}|{self.type}|{self.key}".encode()).hexdigest()[:16]


@dataclass
class Match:
    kind: str  # "settlement_bank"
    settlement_ids: list[str]
    bank_txn_ids: list[str]
    method: str  # utr | utr_split | utr_merged | llm_utr | amount_date | amount_split | amount_merged
    status: str  # auto | needs_review
    diff_paise: int  # bank total - settlement total


@dataclass
class EngineResult:
    payments: pd.DataFrame
    gateway: pd.DataFrame
    settlements: pd.DataFrame
    bank: pd.DataFrame
    orders: pd.DataFrame
    matches: list[Match]
    exceptions: list[ReconException]
    summary: dict[str, Any] = field(default_factory=dict)
