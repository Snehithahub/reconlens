"""FastAPI service: create runs from uploads or demo data, browse/resolve exceptions, export reports."""

from __future__ import annotations

import io
import json
import logging
import time
import uuid
from datetime import date
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from reconlens import __version__
from reconlens.config import ReconConfig, Settings, get_settings, load_recon_config
from reconlens.engine.types import TAXONOMY, ExceptionType
from reconlens.ingest import InputError, RawInput
from reconlens.llm.client import JSONLLM, build_client
from reconlens.llm.explain import explain_clusters
from reconlens.llm.narration import LLMNarrationExtractor
from reconlens.logging import configure_logging
from reconlens.money import to_rupees
from reconlens.service import demo_inputs, execute_run
from reconlens.store import Store

log = logging.getLogger("reconlens.api")


class ResolveIn(BaseModel):
    status: Literal["resolved", "ignored", "open", "in_progress"]
    note: str | None = Field(None, max_length=2000)
    resolved_by: str | None = Field(None, max_length=120)


class ExceptionOut(BaseModel):
    exception_id: str
    run_id: str
    type: str
    title: str
    severity: str
    key: str
    impact_paise: int
    impact_inr: float
    recoverable: bool
    needs_review: bool
    event_date: date | None
    age_days: int | None
    detail: str
    suggested_action: str
    evidence: list[dict[str, Any]]
    status: str
    note: str | None = None
    resolved_by: str | None = None


def _err(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
                                                               "details": details}})


def _row_to_exception(r: pd.Series, as_of: date | None) -> ExceptionOut:
    ev_date = r["event_date"].date() if pd.notna(r["event_date"]) else None
    return ExceptionOut(
        exception_id=r["exception_id"], run_id=r["run_id"], type=r["type"], title=r["title"],
        severity=r["severity"], key=r["key"], impact_paise=int(r["impact_paise"]),
        impact_inr=to_rupees(int(r["impact_paise"])), recoverable=bool(r["recoverable"]),
        needs_review=bool(r["needs_review"]), event_date=ev_date,
        age_days=(as_of - ev_date).days if as_of and ev_date else None, detail=r["detail"],
        suggested_action=r["suggested_action"], evidence=json.loads(r["evidence"]), status=r["status"],
        note=None if pd.isna(r["note"]) else r["note"],
        resolved_by=None if pd.isna(r["resolved_by"]) else r["resolved_by"],
    )


def create_app(settings: Settings | None = None, store: Store | None = None, llm: JSONLLM | None = None,
               recon_config: ReconConfig | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    cfg = recon_config or load_recon_config(settings.config_dir)
    store = store or Store(settings.database_path)
    llm = llm if llm is not None else build_client(settings)

    app = FastAPI(title="ReconLens", version=__version__,
                  description="Payment settlement reconciliation: gateway report vs bank statement vs order ledger")
    app.state.store = store

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info("request", extra={"request_id": rid, "path": request.url.path, "status": response.status_code,
                                   "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        return response

    @app.exception_handler(HTTPException)
    async def http_exc(_: Request, e: HTTPException) -> JSONResponse:
        code = {400: "bad_request", 404: "not_found", 413: "too_large", 422: "invalid_input",
                503: "unavailable"}.get(e.status_code, "error")
        return _err(e.status_code, code, str(e.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_exc(_: Request, e: RequestValidationError) -> JSONResponse:
        return _err(422, "invalid_request", "request validation failed", json.loads(json.dumps(e.errors(),
                                                                                              default=str)))

    @app.exception_handler(InputError)
    async def input_exc(_: Request, e: InputError) -> JSONResponse:
        return _err(422, "invalid_input", f"{e.source} could not be read", e.problems)

    def run_or_404(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, f"run {run_id} not found")
        return run

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready() -> JSONResponse:
        try:
            ok = store.ping()
        except Exception as e:  # report, don't crash
            return _err(503, "not_ready", f"database unavailable: {e}")
        return JSONResponse({"status": "ready" if ok else "not_ready", "llm_enabled": llm is not None,
                             "gateway": cfg.rate_card.gateway})

    @app.get("/config")
    def config() -> dict[str, Any]:
        return {"rate_card": cfg.rate_card.model_dump(mode="json"), "tolerances": cfg.tolerances.model_dump(),
                "llm_enabled": llm is not None, "llm_provider": settings.llm_provider,
                "exception_types": {str(t): {"title": i.title, "severity": str(i.severity),
                                             "recoverable": i.recoverable} for t, i in TAXONOMY.items()}}

    async def _read(f: UploadFile | None, label: str) -> RawInput | None:
        if f is None:
            return None
        content = await f.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"{label} exceeds {settings.max_upload_mb} MB")
        return RawInput(f.filename or label, content)

    @app.post("/runs", status_code=201)
    async def create_run(
        gateway_file: Annotated[UploadFile | None, File()] = None,
        bank_file: Annotated[UploadFile | None, File()] = None,
        ledger_file: Annotated[UploadFile | None, File()] = None,
        demo: Annotated[bool, Form()] = False,
        demo_payments: Annotated[int | None, Form(ge=100, le=200_000)] = None,
        demo_seed: Annotated[int | None, Form()] = None,
        use_llm: Annotated[bool, Form()] = False,
        force: Annotated[bool, Query()] = False,
    ) -> JSONResponse:
        """Upload all three files (CSV/XLSX) or set demo=true to use generated demo data."""
        files = [await _read(gateway_file, "gateway_file"), await _read(bank_file, "bank_file"),
                 await _read(ledger_file, "ledger_file")]
        if demo:
            gw, bank, led = demo_inputs(settings, cfg, demo_payments, demo_seed)
            source = f"demo:{demo_payments or settings.demo_payments}:{demo_seed or settings.demo_seed}"
        elif all(files):
            gw, bank, led = files  # type: ignore[misc]
            source = "upload"
        else:
            raise HTTPException(400, "upload gateway_file, bank_file and ledger_file, or set demo=true")
        extractor = None
        if use_llm:
            if llm is None:
                raise HTTPException(503, "use_llm requested but no LLM is configured (set LLM_PROVIDER/LLM_API_KEY)")
            extractor = LLMNarrationExtractor(llm, settings.llm_narration_batch_size)
        out = execute_run(store, cfg, gw, bank, led, source=source, force=force, extractor=extractor,
                          min_confidence=settings.llm_narration_min_confidence)
        return JSONResponse(status_code=200 if out.reused else 201, content={"reused": out.reused, **out.run})

    @app.get("/runs")
    def list_runs(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        return store.list_runs(limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return run_or_404(run_id)

    @app.get("/runs/{run_id}/settlements")
    def get_settlements(run_id: str) -> list[dict[str, Any]]:
        run_or_404(run_id)
        df = store.settlements(run_id)
        return json.loads(df.to_json(orient="records", date_format="iso"))

    def _filtered(run_id: str, type_: list[str] | None, severity: str | None, status: str | None,
                  min_impact_inr: float | None, limit: int, offset: int) -> tuple[int, list[ExceptionOut]]:
        run = run_or_404(run_id)
        bad = [t for t in type_ or [] if t not in ExceptionType._value2member_map_]
        if bad:
            raise HTTPException(422, f"unknown exception type(s): {bad}")
        total, df = store.exceptions(run_id, types=type_, severity=severity, status=status,
                                     min_impact_paise=None if min_impact_inr is None else int(min_impact_inr * 100),
                                     limit=limit, offset=offset)
        as_of = date.fromisoformat(run["summary"]["as_of"]) if run["summary"].get("as_of") else None
        return total, [_row_to_exception(r, as_of) for _, r in df.iterrows()]

    @app.get("/runs/{run_id}/exceptions")
    def list_exceptions(
        run_id: str,
        type: Annotated[list[str] | None, Query()] = None,
        severity: Literal["high", "medium", "low"] | None = None,
        status: Literal["open", "resolved", "ignored", "in_progress"] | None = None,
        min_impact_inr: float | None = Query(None, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        total, items = _filtered(run_id, type, severity, status, min_impact_inr, limit, offset)
        return {"total": total, "limit": limit, "offset": offset, "items": [i.model_dump(mode="json") for i in items]}

    @app.get("/exceptions/{exception_id}")
    def get_exception(exception_id: str) -> dict[str, Any]:
        r = store.get_exception(exception_id)
        if r is None:
            raise HTTPException(404, f"exception {exception_id} not found")
        run = store.get_run(r["run_id"]) or {"summary": {}}
        as_of = date.fromisoformat(run["summary"]["as_of"]) if run["summary"].get("as_of") else None
        return _row_to_exception(r, as_of).model_dump(mode="json")

    @app.post("/exceptions/{exception_id}/resolve")
    def resolve(exception_id: str, body: ResolveIn) -> dict[str, Any]:
        if store.get_exception(exception_id) is None:
            raise HTTPException(404, f"exception {exception_id} not found")
        store.resolve(exception_id, body.status, body.note, body.resolved_by)
        return get_exception(exception_id)

    @app.get("/runs/{run_id}/report")
    def export(run_id: str, format: Literal["csv", "xlsx"] = "csv",
               type: Annotated[list[str] | None, Query()] = None,
               status: Literal["open", "resolved", "ignored", "in_progress"] | None = None) -> StreamingResponse:
        _, items = _filtered(run_id, type, None, status, None, 1_000_000, 0)
        df = pd.DataFrame([{
            "exception_id": i.exception_id, "type": i.type, "title": i.title, "severity": i.severity,
            "key": i.key, "impact_inr": i.impact_inr, "recoverable": i.recoverable, "needs_review": i.needs_review,
            "event_date": i.event_date, "age_days": i.age_days, "status": i.status, "note": i.note,
            "detail": i.detail, "suggested_action": i.suggested_action,
            "evidence_ids": ", ".join(str(e.get("id")) for e in i.evidence),
        } for i in items])
        buf = io.BytesIO()
        if format == "csv":
            buf.write(df.to_csv(index=False).encode())
            media = "text/csv"
        else:
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                df.to_excel(xw, sheet_name="exceptions", index=False)
                store.settlements(run_id).to_excel(xw, sheet_name="settlements", index=False)
            media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        buf.seek(0)
        return StreamingResponse(buf, media_type=media, headers={
            "Content-Disposition": f'attachment; filename="reconlens_{run_id}_exceptions.{format}"'})

    @app.post("/runs/{run_id}/explain")
    def explain(run_id: str, use_llm: bool = Query(True)) -> dict[str, Any]:
        """Explain each exception cluster. Uses the LLM when configured, otherwise rule-based templates."""
        run_or_404(run_id)
        _, df = store.exceptions(run_id, limit=1_000_000)
        items = explain_clusters(df, cfg, llm if use_llm else None)
        store.save_explanations(run_id, items)
        return {"llm_used": bool(use_llm and llm is not None), "items": items}

    @app.get("/runs/{run_id}/explanations")
    def explanations(run_id: str) -> dict[str, Any]:
        run_or_404(run_id)
        return {"items": store.explanations(run_id)}

    return app
