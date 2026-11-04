"""DuckDB persistence for runs, exceptions, settlements, resolutions and explanations.

Re-saving a run deletes and rewrites its rows in one transaction, so re-runs are idempotent.
Resolutions are keyed by the deterministic exception id and survive forced re-runs.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from reconlens.engine.types import EngineResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id VARCHAR PRIMARY KEY, fingerprint VARCHAR, created_at TIMESTAMP, source VARCHAR,
  status VARCHAR, summary JSON, inputs JSON
);
CREATE TABLE IF NOT EXISTS exceptions (
  run_id VARCHAR, exception_id VARCHAR, type VARCHAR, title VARCHAR, severity VARCHAR, key VARCHAR,
  impact_paise BIGINT, recoverable BOOLEAN, needs_review BOOLEAN, event_date DATE, detail VARCHAR,
  suggested_action VARCHAR, evidence JSON
);
CREATE TABLE IF NOT EXISTS settlements (
  run_id VARCHAR, settlement_id VARCHAR, account VARCHAR, settled_at DATE, utr VARCHAR, n_payments INTEGER,
  gross_paise BIGINT, fees_paise BIGINT, tax_paise BIGINT, refunds_paise BIGINT, chargebacks_paise BIGINT,
  net_paise BIGINT, expected_by DATE, status VARCHAR, bank_txn_ids VARCHAR, match_method VARCHAR
);
CREATE TABLE IF NOT EXISTS resolutions (
  exception_id VARCHAR PRIMARY KEY, status VARCHAR, note VARCHAR, resolved_by VARCHAR, updated_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS explanations (
  run_id VARCHAR, cluster VARCHAR, source VARCHAR, payload JSON, created_at TIMESTAMP
);
"""

SETTLEMENT_COLS = ["settlement_id", "account", "settled_at", "utr", "n_payments", "gross_paise", "fees_paise",
                   "tax_paise", "refunds_paise", "chargebacks_paise", "net_paise", "expected_by", "status",
                   "bank_txn_ids", "match_method"]


class Store:
    def __init__(self, path: Path | str) -> None:
        p = str(path)
        if p != ":memory:":
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(p)
        self._lock = threading.RLock()
        with self._lock:
            self._con.execute(SCHEMA)

    def close(self) -> None:
        self._con.close()

    def ping(self) -> bool:
        with self._lock:
            return self._con.execute("SELECT 1").fetchone() == (1,)

    def _df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        with self._lock:
            return self._con.execute(sql, params or []).df()

    # ---- runs ---------------------------------------------------------------------------------
    def save_run(self, run_id: str, fingerprint: str, source: str, inputs: dict[str, Any],
                 result: EngineResult) -> None:
        exc = pd.DataFrame([{
            "run_id": run_id, "exception_id": e.exception_id, "type": str(e.type), "title": e.info.title,
            "severity": str(e.info.severity), "key": e.key, "impact_paise": int(e.impact_paise),
            "recoverable": e.info.recoverable, "needs_review": e.needs_review, "event_date": e.event_date,
            "detail": e.detail, "suggested_action": e.info.action, "evidence": json.dumps(e.evidence, default=str),
        } for e in result.exceptions], columns=["run_id", "exception_id", "type", "title", "severity", "key",
                                                "impact_paise", "recoverable", "needs_review", "event_date",
                                                "detail", "suggested_action", "evidence"])
        setl = result.settlements[SETTLEMENT_COLS].copy()
        setl.insert(0, "run_id", run_id)
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            con = self._con
            con.execute("BEGIN")
            try:
                for t in ("runs", "exceptions", "settlements", "explanations"):
                    con.execute(f"DELETE FROM {t} WHERE run_id = ?", [run_id])
                con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, 'completed', ?, ?)",
                            [run_id, fingerprint, now, source, json.dumps(result.summary, default=str),
                             json.dumps(inputs)])
                con.register("_exc", exc)
                con.execute("INSERT INTO exceptions SELECT * FROM _exc")
                con.unregister("_exc")
                con.register("_setl", setl)
                con.execute("INSERT INTO settlements SELECT * FROM _setl")
                con.unregister("_setl")
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        df = self._df("SELECT * FROM runs WHERE run_id = ?", [run_id])
        if df.empty:
            return None
        r = df.iloc[0]
        return {"run_id": r.run_id, "fingerprint": r.fingerprint, "created_at": r.created_at.isoformat(),
                "source": r.source, "status": r.status, "summary": json.loads(r.summary),
                "inputs": json.loads(r.inputs)}

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        df = self._df("SELECT run_id, created_at, source, status FROM runs ORDER BY created_at DESC LIMIT ?", [limit])
        return [{**r, "created_at": r["created_at"].isoformat()} for r in df.to_dict("records")]

    # ---- exceptions ---------------------------------------------------------------------------
    def exceptions(self, run_id: str, *, types: list[str] | None = None, severity: str | None = None,
                   status: str | None = None, min_impact_paise: int | None = None, limit: int = 100,
                   offset: int = 0) -> tuple[int, pd.DataFrame]:
        where = ["e.run_id = ?"]
        params: list[Any] = [run_id]
        if types:
            where.append(f"e.type IN ({','.join('?' * len(types))})")
            params += types
        if severity:
            where.append("e.severity = ?")
            params.append(severity)
        if status:
            where.append("coalesce(r.status, 'open') = ?")
            params.append(status)
        if min_impact_paise is not None:
            where.append("abs(e.impact_paise) >= ?")
            params.append(min_impact_paise)
        base = (f"FROM exceptions e LEFT JOIN resolutions r USING (exception_id) WHERE {' AND '.join(where)}")
        total = int(self._df(f"SELECT count(*) AS n {base}", params)["n"].iat[0])
        df = self._df(
            f"SELECT e.*, coalesce(r.status, 'open') AS status, r.note, r.resolved_by, r.updated_at AS resolved_at "
            f"{base} ORDER BY CASE e.severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, "
            f"abs(e.impact_paise) DESC, e.exception_id LIMIT ? OFFSET ?", [*params, limit, offset])
        return total, df

    def get_exception(self, exception_id: str) -> pd.Series | None:
        df = self._df("SELECT e.*, coalesce(r.status, 'open') AS status, r.note, r.resolved_by, "
                      "r.updated_at AS resolved_at FROM exceptions e LEFT JOIN resolutions r USING (exception_id) "
                      "WHERE e.exception_id = ?", [exception_id])
        return None if df.empty else df.iloc[0]

    def resolve(self, exception_id: str, status: str, note: str | None, resolved_by: str | None) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            self._con.execute(
                "INSERT INTO resolutions VALUES (?, ?, ?, ?, ?) ON CONFLICT (exception_id) DO UPDATE SET "
                "status = excluded.status, note = excluded.note, resolved_by = excluded.resolved_by, "
                "updated_at = excluded.updated_at", [exception_id, status, note, resolved_by, now])

    def settlements(self, run_id: str) -> pd.DataFrame:
        return self._df("SELECT * EXCLUDE (run_id) FROM settlements WHERE run_id = ? ORDER BY settled_at, "
                        "settlement_id", [run_id])

    # ---- explanations -------------------------------------------------------------------------
    def save_explanations(self, run_id: str, items: list[dict[str, Any]]) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._lock:
            self._con.execute("DELETE FROM explanations WHERE run_id = ?", [run_id])
            for it in items:
                self._con.execute("INSERT INTO explanations VALUES (?, ?, ?, ?, ?)",
                                  [run_id, it["cluster"], it["source"], json.dumps(it), now])

    def explanations(self, run_id: str) -> list[dict[str, Any]]:
        df = self._df("SELECT payload FROM explanations WHERE run_id = ? ORDER BY cluster", [run_id])
        return [json.loads(p) for p in df["payload"]]
