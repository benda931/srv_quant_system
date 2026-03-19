"""
config/settings.py

Production-grade configuration for the SRV Quantamental Trading System.

Key design goals:
- Single source of truth for tickers, windows, scoring weights, paths.
- Strong typing via pydantic-settings.
- Safe defaults + environment-based overrides.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    # settings.py is typically at <root>/config/settings.py
    # parents[0] = settings.py, parents[1] = config, parents[2] = project root
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Environment-driven settings.

    Reads .env by default (expects FMP_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    # --- Secrets ---
    fmp_api_key: str = Field(..., alias="FMP_API_KEY")

    # --- FMP endpoints (legacy + stable fallback) ---
    fmp_base_url: str = Field(default="https://financialmodelingprep.com")
    fmp_api_v3_path: str = Field(default="/api/v3")
    fmp_stable_path: str = Field(default="/stable")

    # --- Paths ---
    project_root: Path = Field(default_factory=_project_root)
    data_dir: Path = Field(default_factory=lambda: _project_root() / "data_lake")
    parquet_dir: Path = Field(default_factory=lambda: _project_root() / "data_lake" / "parquet")
    log_dir: Path = Field(default_factory=lambda: _project_root() / "logs")

    # --- Universe ---
    # 11 GICS sectors represented by Select Sector SPDR ETFs.
    # Source mapping commonly used in practice; the tickers list is standard. (SSGA lists 11 funds)  citeturn6search4
    sector_tickers: Dict[str, str] = Field(
        default_factory=lambda: {
            "Communication Services": "XLC",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Financials": "XLF",
            "Health Care": "XLV",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Technology": "XLK",
            "Utilities": "XLU",
        }
    )
    spy_ticker: str = Field(default="SPY")

    macro_tickers: Dict[str, str] = Field(
        default_factory=lambda: {
            "TNX_10Y": "^TNX",
            "DXY_USD": "DX-Y.NYB",
        }
    )
    credit_tickers: Dict[str, str] = Field(default_factory=lambda: {"HYG": "HYG", "IEF": "IEF"})
    vol_tickers: Dict[str, str] = Field(default_factory=lambda: {"VIX": "^VIX"})

    # --- Core windows ---
    # 252 ~ 1y trading days; 60 ~ 3m; 21 ~ 1m.
    pca_window: int = Field(default=252, ge=126, le=756)
    zscore_window: int = Field(default=60, ge=20, le=252)
    macro_window: int = Field(default=60, ge=20, le=252)
    dispersion_window: int = Field(default=21, ge=10, le=126)

    # --- PCA configuration (research-driven enhancements) ---
    # Dynamic component selection to hit an explained variance target, bounded by [min, max].
    pca_explained_var_target: float = Field(default=0.80, ge=0.50, le=0.99)
    pca_min_components: int = Field(default=1, ge=1, le=10)
    pca_max_components: int = Field(default=5, ge=1, le=11)

    # --- EWMA volatility ---
    # EWMA decay factor used by RiskMetrics for daily data  citeturn2search5turn2search2
    ewma_lambda: float = Field(default=0.94, ge=0.80, le=0.99)

    # --- Scoring weights ---
    # Must sum to 1.0 in production.
    weight_stat: float = Field(default=0.4, ge=0.0, le=1.0)
    weight_macro: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_fund: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_vol: float = Field(default=0.2, ge=0.0, le=1.0)

    # Points by layer (0..100 total)
    points_stat: int = Field(default=40, ge=0, le=100)
    points_macro: int = Field(default=20, ge=0, le=100)
    points_fund: int = Field(default=20, ge=0, le=100)
    points_vol: int = Field(default=20, ge=0, le=100)

    # --- Data engineering knobs ---
    history_years: int = Field(default=10, ge=2, le=25)
    request_timeout_s: int = Field(default=30, ge=5, le=120)
    max_workers: int = Field(default=12, ge=2, le=64)
    cache_max_age_hours: int = Field(default=12, ge=0, le=168)
    quote_chunk_size: int = Field(default=100, ge=10, le=500)

    # --- Risk / regime thresholds (tunable, but not placeholders) ---
    # VIX gating: aggressive stat-arb usually degrades in extreme vol spikes.
    vix_level_soft: float = Field(default=25.0, ge=10.0, le=80.0)
    vix_level_hard: float = Field(default=35.0, ge=10.0, le=120.0)
    vix_percentile_hard: float = Field(default=0.90, ge=0.50, le=0.99)

    # Credit stress threshold on HYG-IEF z-score (more negative -> higher stress).
    credit_stress_z: float = Field(default=-1.0, ge=-5.0, le=0.0)

    # Fundamental guardrails
    pe_extreme: float = Field(default=30.0, ge=5.0, le=200.0)
    neg_earnings_weight_cap: float = Field(default=0.20, ge=0.0, le=1.0)

    # Sector mapping for FMP sector weightings (SPY sector allocation)
    # FMP sometimes uses alternative names; we normalize to our canonical sector names.
    fmp_sector_name_aliases: Dict[str, str] = Field(
        default_factory=lambda: {
            "Communication Services": "Communication Services",
            "Consumer Cyclical": "Consumer Discretionary",
            "Consumer Defensive": "Consumer Staples",
            "Energy": "Energy",
            "Financial Services": "Financials",
            "Financial": "Financials",
            "Healthcare": "Health Care",
            "Health Care": "Health Care",
            "Industrials": "Industrials",
            "Basic Materials": "Materials",
            "Materials": "Materials",
            "Real Estate": "Real Estate",
            "Technology": "Technology",
            "Utilities": "Utilities",
        }
    )

    @field_validator("parquet_dir", "data_dir", "log_dir", mode="after")
    @classmethod
    def _ensure_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("pca_max_components", mode="after")
    @classmethod
    def _pca_bounds(cls, v: int, info) -> int:
        min_c = info.data.get("pca_min_components", 1)
        if v < min_c:
            raise ValueError("pca_max_components must be >= pca_min_components")
        return v

    @field_validator("weight_vol", mode="after")
    @classmethod
    def _weights_sum(cls, v: float, info) -> float:
        # Validate sum to ~1.0, but allow tiny floating error.
        w_stat = info.data.get("weight_stat", 0.4)
        w_macro = info.data.get("weight_macro", 0.2)
        w_fund = info.data.get("weight_fund", 0.2)
        total = w_stat + w_macro + w_fund + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Layer weights must sum to 1.0. Got {total:.6f}")
        return v

    def sector_list(self) -> List[str]:
        return list(self.sector_tickers.values())

    def all_price_tickers(self) -> List[str]:
        # Prices required for analytics
        return (
            self.sector_list()
            + [self.spy_ticker]
            + list(self.macro_tickers.values())
            + list(self.credit_tickers.values())
            + list(self.vol_tickers.values())
        )

    def canonical_sector_by_ticker(self) -> Dict[str, str]:
        return {v: k for k, v in self.sector_tickers.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
