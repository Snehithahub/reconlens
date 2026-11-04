"""Typed configuration: process settings from env, reconciliation rules from YAML."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

LLMProvider = Literal["none", "groq", "gemini", "openai", "azure", "custom"]

# Base URLs and default models for the OpenAI-compatible providers we ship presets for.
LLM_PRESETS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path | None = None
    config_dir: Path = DEFAULT_CONFIG_DIR
    log_level: str = "INFO"
    max_upload_mb: int = 50
    demo_payments: int = 5000
    demo_seed: int = 42

    llm_provider: LLMProvider = "none"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_api_version: str = "2024-10-21"  # Azure only
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3
    llm_narration_min_confidence: float = Field(0.7, ge=0, le=1)
    llm_narration_batch_size: int = Field(20, ge=1, le=100)

    api_url: str = "http://localhost:8601"  # used by the Streamlit UI

    @property
    def database_path(self) -> Path:
        return self.db_path or (self.data_dir / "reconlens.duckdb")

    def resolved_llm(self) -> tuple[str, str] | None:
        """(base_url, model) when an LLM is configured and has a key, else None."""
        if self.llm_provider == "none" or not self.llm_api_key:
            return None
        base, model = LLM_PRESETS.get(self.llm_provider, ("", ""))
        base = self.llm_base_url or base
        model = self.llm_model or model
        if not base or not model:
            return None
        return base.rstrip("/"), model


class MethodRate(BaseModel):
    percent: Decimal = Field(ge=0, le=100)
    flat_paise: int = Field(0, ge=0)
    min_fee_paise: int = Field(0, ge=0)
    settlement_days: int = Field(ge=0, le=30)

    @property
    def rate_ppm(self) -> int:
        """Rate in parts per million, so fee math stays in integers."""
        return int(self.percent * 10_000)


class RateCard(BaseModel):
    gateway: str
    gst_rate_percent: Decimal = Decimal(18)
    methods: dict[str, MethodRate]

    @field_validator("methods")
    @classmethod
    def _lower(cls, v: dict[str, MethodRate]) -> dict[str, MethodRate]:
        return {k.lower(): m for k, m in v.items()}

    @property
    def gst_bps(self) -> int:
        return int(self.gst_rate_percent * 100)


class Tolerances(BaseModel):
    fee_paise: int = 1
    gst_paise: int = 1
    settlement_paise: int = 100
    credit_window_business_days: int = 3
    sla_grace_business_days: int = 1
    missing_grace_business_days: int = 2


class ReconConfig(BaseModel):
    rate_card: RateCard
    tolerances: Tolerances = Tolerances()
    gateway_keywords: list[str] = ["PAYFLOW"]
    bank_holidays: list[date] = []

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def load_recon_config(config_dir: Path | None = None) -> ReconConfig:
    d = config_dir or DEFAULT_CONFIG_DIR
    rate_card = RateCard.model_validate(yaml.safe_load((d / "rate_card.yaml").read_text()))
    recon = yaml.safe_load((d / "recon.yaml").read_text()) or {}
    return ReconConfig(rate_card=rate_card, **recon)


@lru_cache
def get_settings() -> Settings:
    return Settings()
