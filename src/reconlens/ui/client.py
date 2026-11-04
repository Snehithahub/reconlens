"""Thin client for the ReconLens API, used by the Streamlit dashboard.

The dashboard always talks to the API contract. In ``api`` mode that is the HTTP service at
``API_URL``; in ``local`` mode the same FastAPI app runs in-process (no server needed), so the
dashboard behaves identically either way.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

from reconlens.config import Settings, get_settings

log = logging.getLogger(__name__)

Mode = Literal["api", "local"]
PAGE_SIZE = 1000  # API maximum for /runs/{id}/exceptions
RUN_TIMEOUT_S = 600.0  # large uploads can take a while to reconcile


class ApiError(RuntimeError):
    """An API call failed; ``message`` is safe to show to the user."""

    def __init__(self, status: int, message: str, details: Any = None) -> None:
        self.status = status
        self.message = message
        self.details = details
        super().__init__(f"{status}: {message}")


class HTTPLike(Protocol):
    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...


@dataclass(frozen=True)
class Upload:
    name: str
    content: bytes


def _raise_for(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    try:
        err = r.json().get("error", {})
        message, details = err.get("message") or r.text, err.get("details")
    except (ValueError, AttributeError):
        message, details = r.text or r.reason_phrase, None
    raise ApiError(r.status_code, message, details)


class ReconClient:
    def __init__(self, http: HTTPLike, mode: Mode, base_url: str) -> None:
        self.http = http
        self.mode = mode
        self.base_url = base_url
        # The in-process TestClient ignores (and warns about) per-request timeouts.
        self._long = {"timeout": RUN_TIMEOUT_S} if mode == "api" else {}

    def _get(self, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self.http.get(path, **kw)
        except httpx.HTTPError as e:
            raise ApiError(0, f"API unreachable at {self.base_url}: {e}") from e
        _raise_for(r)
        return r

    def _post(self, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self.http.post(path, **kw)
        except httpx.HTTPError as e:
            raise ApiError(0, f"API unreachable at {self.base_url}: {e}") from e
        _raise_for(r)
        return r

    # ---- service ------------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._get("/health").json()

    def ready(self) -> dict[str, Any]:
        return self._get("/ready").json()

    # ---- runs ---------------------------------------------------------------------------------
    def run_demo(self, n_payments: int, seed: int, *, use_llm: bool = False, force: bool = False
                 ) -> dict[str, Any]:
        data = {"demo": "true", "demo_payments": str(n_payments), "demo_seed": str(seed),
                "use_llm": str(use_llm).lower()}
        return self._post("/runs", data=data, params={"force": str(force).lower()}, **self._long).json()

    def run_upload(self, gateway: Upload, bank: Upload, ledger: Upload, *, use_llm: bool = False,
                   force: bool = False) -> dict[str, Any]:
        files = {"gateway_file": (gateway.name, gateway.content), "bank_file": (bank.name, bank.content),
                 "ledger_file": (ledger.name, ledger.content)}
        return self._post("/runs", files=files, data={"use_llm": str(use_llm).lower()},
                          params={"force": str(force).lower()}, **self._long).json()

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._get("/runs", params={"limit": limit}).json()

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/runs/{run_id}").json()

    # ---- exceptions ---------------------------------------------------------------------------
    def exceptions(self, run_id: str) -> list[dict[str, Any]]:
        """All exceptions of a run, following pagination."""
        items: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self._get(f"/runs/{run_id}/exceptions", params={"limit": PAGE_SIZE, "offset": offset}).json()
            items.extend(page["items"])
            offset += len(page["items"])
            if not page["items"] or offset >= page["total"]:
                return items

    def resolve(self, exception_id: str, status: str, note: str | None = None,
                resolved_by: str | None = None) -> dict[str, Any]:
        body = {"status": status, "note": note or None, "resolved_by": resolved_by or None}
        return self._post(f"/exceptions/{exception_id}/resolve", json=body).json()

    def report(self, run_id: str, fmt: Literal["csv", "xlsx"]) -> bytes:
        return self._get(f"/runs/{run_id}/report", params={"format": fmt}, **self._long).content

    # ---- explanations -------------------------------------------------------------------------
    def explanations(self, run_id: str) -> list[dict[str, Any]]:
        return self._get(f"/runs/{run_id}/explanations").json()["items"]

    def explain(self, run_id: str, *, use_llm: bool) -> dict[str, Any]:
        return self._post(f"/runs/{run_id}/explain", params={"use_llm": str(use_llm).lower()},
                          **self._long).json()


def remote_client(api_url: str, timeout_s: float = 30.0) -> ReconClient:
    base = api_url.rstrip("/")
    return ReconClient(httpx.Client(base_url=base, timeout=timeout_s), "api", base)


def local_client(db_path: str = ":memory:", settings: Settings | None = None) -> ReconClient:
    """Run the API app in-process. Used when no API server is reachable (local development)."""
    from reconlens.api.app import create_app
    from reconlens.store import Store

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # starlette's TestClient deprecation notice is irrelevant here
        from fastapi.testclient import TestClient

    app = create_app(settings=settings or get_settings(), store=Store(db_path))
    return ReconClient(TestClient(app, raise_server_exceptions=False), "local", "in-process")


def connect(api_url: str, mode: Literal["auto", "api", "local"] = "auto", *, local_db: str = ":memory:"
            ) -> ReconClient:
    """``api`` requires the server, ``local`` never uses it, ``auto`` prefers it and falls back."""
    if mode == "local":
        return local_client(local_db)
    client = remote_client(api_url)
    try:
        client.health()
        return client
    except ApiError:
        if mode == "api":
            raise
        log.warning("API at %s unreachable; running the API in-process", api_url)
        return local_client(local_db)
