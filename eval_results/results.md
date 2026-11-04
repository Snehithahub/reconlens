### 10,000 payments (seed 7, LLM off)

- rows: {'payments': 10000, 'gateway_rows': 10438, 'settlements': 286, 'bank_rows': 511, 'ledger_rows': 13005}
- runtime: ingest 0.12s + reconcile 0.33s = 0.46s (data generation 0.29s, not counted)
- payment auto-match rate: 93.28%; settlement auto-match rate: 95.59%
- false-match rate (settlement-to-bank edges): 0.00% (0/272); gold edges recovered 272/272
- narration UTR extraction (credits whose narration carries a UTR): rule-only 96.84%, LLM-assisted: pending a live run

| exception type | gold | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| duplicate_refund | 5 | 5 | 0 | 0 | 1.000 | 1.000 |
| gst_mismatch | 10 | 10 | 0 | 0 | 1.000 | 1.000 |
| mdr_overcharge | 20 | 20 | 0 | 0 | 1.000 | 1.000 |
| merged_credit | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| missing_settlement | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| narration_no_utr | 15 | 15 | 0 | 0 | 1.000 | 1.000 |
| order_paid_no_payment | 5 | 5 | 0 | 0 | 1.000 | 1.000 |
| payment_not_recorded | 10 | 10 | 0 | 0 | 1.000 | 1.000 |
| refund_without_order | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| rounding_difference | 9 | 9 | 0 | 0 | 1.000 | 1.000 |
| settlement_delayed | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| short_credit | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| split_credit | 6 | 6 | 0 | 0 | 1.000 | 1.000 |

### 100,000 payments (seed 7, LLM off)

- rows: {'payments': 100000, 'gateway_rows': 104380, 'settlements': 314, 'bank_rows': 553, 'ledger_rows': 130050}
- runtime: ingest 0.95s + reconcile 1.89s = 2.84s (data generation 2.44s, not counted)
- payment auto-match rate: 93.24%; settlement auto-match rate: 95.60%
- false-match rate (settlement-to-bank edges): 0.00% (0/273); gold edges recovered 273/273
- narration UTR extraction (credits whose narration carries a UTR): rule-only 98.03%, LLM-assisted: pending a live run

| exception type | gold | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| duplicate_refund | 50 | 50 | 0 | 0 | 1.000 | 1.000 |
| gst_mismatch | 100 | 100 | 0 | 0 | 1.000 | 1.000 |
| mdr_overcharge | 200 | 200 | 0 | 0 | 1.000 | 1.000 |
| merged_credit | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| missing_settlement | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| narration_no_utr | 12 | 12 | 0 | 0 | 1.000 | 1.000 |
| order_paid_no_payment | 50 | 50 | 0 | 0 | 1.000 | 1.000 |
| payment_not_recorded | 100 | 100 | 0 | 0 | 1.000 | 1.000 |
| refund_without_order | 30 | 30 | 0 | 0 | 1.000 | 1.000 |
| rounding_difference | 9 | 9 | 0 | 0 | 1.000 | 1.000 |
| settlement_delayed | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| short_credit | 6 | 6 | 0 | 0 | 1.000 | 1.000 |
| split_credit | 6 | 6 | 0 | 0 | 1.000 | 1.000 |

### 10,000 payments (seed 7, LLM off, stress: 80% of narrations without a parseable UTR, 6 accounts)

- rows: {'payments': 10000, 'gateway_rows': 10438, 'settlements': 563, 'bank_rows': 737, 'ledger_rows': 13005}
- runtime: ingest 0.12s + reconcile 0.5s = 0.63s (data generation 0.26s, not counted)
- payment auto-match rate: 94.72%; settlement auto-match rate: 95.93%
- false-match rate (settlement-to-bank edges): 0.00% (0/541); gold edges recovered 541/541
- narration UTR extraction (credits whose narration carries a UTR): rule-only 50.17%, LLM-assisted: pending a live run

| exception type | gold | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| duplicate_refund | 5 | 5 | 0 | 0 | 1.000 | 1.000 |
| gst_mismatch | 10 | 10 | 0 | 0 | 1.000 | 1.000 |
| mdr_overcharge | 20 | 20 | 0 | 0 | 1.000 | 1.000 |
| merged_credit | 11 | 11 | 0 | 0 | 1.000 | 1.000 |
| missing_settlement | 11 | 11 | 0 | 0 | 1.000 | 1.000 |
| narration_no_utr | 373 | 373 | 0 | 0 | 1.000 | 1.000 |
| order_paid_no_payment | 5 | 5 | 0 | 0 | 1.000 | 1.000 |
| payment_not_recorded | 10 | 10 | 0 | 0 | 1.000 | 1.000 |
| refund_without_order | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| rounding_difference | 16 | 16 | 0 | 0 | 1.000 | 1.000 |
| settlement_delayed | 11 | 11 | 0 | 0 | 1.000 | 1.000 |
| short_credit | 11 | 11 | 0 | 0 | 1.000 | 1.000 |
| split_credit | 11 | 11 | 0 | 0 | 1.000 | 1.000 |
