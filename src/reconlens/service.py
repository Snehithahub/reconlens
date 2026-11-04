"""Run orchestration shared by the API and CLI: fingerprint, reuse, ingest, reconcile, persist."""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Any

from reconlens.config import ReconConfig, Settings
from reconlens.datagen import GenConfig, generate
from reconlens.engine import ENGINE_VERSION, reconcile
from reconlens.engine.bank_match import NarrationExtractor
from reconlens.ingest import RawInput, load_bank, load_gateway, load_ledger
from reconlens.store import Store

log = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    run_id: str
    reused: bool
    run: dict[str, Any]


def demo_inputs(settings: Settings, cfg: ReconConfig, n: int | None = None, seed: int | None = None
                ) -> tuple[RawInput, RawInput, RawInput]:
    data = generate(GenConfig(n_payments=n or settings.demo_payments, seed=seed or settings.demo_seed), cfg)

    def csv(df, name: str) -> RawInput:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return RawInput(name, buf.getvalue().encode())

    return (csv(data.gateway, "gateway_report.csv"), csv(data.bank, "bank_statement.csv"),
            csv(data.ledger, "order_ledger.csv"))


def fingerprint(gw: RawInput, bank: RawInput, ledger: RawInput, cfg: ReconConfig, use_llm: bool) -> str:
    h = hashlib.sha256()
    for part in (gw.sha256, bank.sha256, ledger.sha256, cfg.fingerprint(), ENGINE_VERSION, str(use_llm)):
        h.update(part.encode())
    return h.hexdigest()


def execute_run(store: Store, cfg: ReconConfig, gw: RawInput, bank: RawInput, ledger: RawInput, *,
                source: str, force: bool = False, extractor: NarrationExtractor | None = None,
                min_confidence: float = 0.7) -> RunOutcome:
    fp = fingerprint(gw, bank, ledger, cfg, extractor is not None)
    run_id = "run_" + fp[:12]
    existing = store.get_run(run_id)
    if existing and not force:
        log.info("run %s reused (same inputs and config)", run_id)
        return RunOutcome(run_id, True, existing)
    result = reconcile(load_gateway(gw), load_bank(bank), load_ledger(ledger), cfg, extractor=extractor,
                       min_confidence=min_confidence)
    for e in result.exceptions:
        e.assign_id(run_id)
    inputs = {k: {"name": r.name, "sha256": r.sha256, "bytes": len(r.content)}
              for k, r in (("gateway", gw), ("bank", bank), ("ledger", ledger))}
    store.save_run(run_id, fp, source, inputs, result)
    return RunOutcome(run_id, False, store.get_run(run_id) or {})
