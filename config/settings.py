from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """
    settings.py is located at <project_root>/config/settings.py
    so parents[1] resolves to <project_root>.
    """
    return Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
    Central configuration object for the SRV Quantamental DSS.

    Design goals:
    - Single source of truth for universe, windows, thresholds and paths
    - Environment-driven secrets and overrides
    - Pydantic-validated configuration for production safety
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
        populate_by_name=True,
    )

    # =====================================================
    # Secrets / API
    # =====================================================
    fmp_api_key: str = Field(..., alias="FMP_API_KEY")

    # =====================================================
    # Vendor endpoints
    # =====================================================
    fmp_base_url: str = Field(default="https://financialmodelingprep.com")
    fmp_api_v3_path: str = Field(default="/api/v3")
    fmp_stable_path: str = Field(default="/stable")

    # =====================================================
    # Paths
    # =====================================================
    project_root: Path = Field(default_factory=_project_root)
    data_dir: Path = Field(default_factory=lambda: _project_root() / "data_lake")
    parquet_dir: Path = Field(default_factory=lambda: _project_root() / "data_lake" / "parquet")
    db_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "srv_quant.duckdb"
    )
    log_dir: Path = Field(default_factory=lambda: _project_root() / "logs")

    # =====================================================
    # Universe
    # =====================================================
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

    credit_tickers: Dict[str, str] = Field(
        default_factory=lambda: {
            "HYG": "HYG",
            "IEF": "IEF",
        }
    )

    vol_tickers: Dict[str, str] = Field(
        default_factory=lambda: {
            "VIX": "^VIX",
        }
    )

    # =====================================================
    # Extended Universe (optional — add tickers beyond sector ETFs)
    # Comma-separated list, fetched alongside sector_tickers
    # Examples: "AAPL,MSFT,GOOGL" or "EWJ,EWZ,FXI" (country ETFs)
    # =====================================================
    extended_tickers: str = Field(default="")
    # Universe mode: "sectors" = 11 sector ETFs only, "extended" = sectors + extended_tickers
    universe_mode: str = Field(default="sectors")

    # =====================================================
    # Data engineering
    # =====================================================
    history_years: int = Field(default=10, ge=2, le=25)
    request_timeout_s: int = Field(default=30, ge=5, le=180)
    max_workers: int = Field(default=12, ge=2, le=64)
    cache_max_age_hours: int = Field(default=12, ge=0, le=168)
    quote_chunk_size: int = Field(default=100, ge=10, le=500)

    # =====================================================
    # Core analytical windows
    # =====================================================
    pca_window: int = Field(default=252, ge=126, le=756)
    zscore_window: int = Field(default=60, ge=20, le=252)
    macro_window: int = Field(default=60, ge=20, le=252)
    dispersion_window: int = Field(default=21, ge=10, le=126)

    # Correlation regime windows
    corr_window: int = Field(default=60, ge=20, le=252)
    corr_baseline_window: int = Field(default=252, ge=60, le=756)

    # =====================================================
    # Correlation Structure Measurement Engine
    # (Short Vol via Correlation/Dispersion research brief)
    # =====================================================
    # Frobenius distortion z-score lookback
    corr_distortion_z_lookback: int = Field(default=252, ge=60, le=756)
    # CoC instability z-score lookback
    coc_z_lookback: int = Field(default=252, ge=60, le=756)

    # =====================================================
    # Signal Stack (Short Vol / Dispersion)
    # Layer 1: Distortion Score logistic coefficients
    # S^dist = σ(a1·z_D + a2·rank(m_t) + a3·z_CoC)
    # =====================================================
    # CALIBRATED via grid search (2026-03-22): Sharpe -0.542 → +0.053
    signal_a1_frob: float = Field(default=0.3, ge=0.0, le=5.0)
    signal_a2_mode: float = Field(default=0.2, ge=0.0, le=5.0)
    signal_a3_coc: float = Field(default=0.3, ge=0.0, le=5.0)

    # Layer 2: Dislocation z-score parameters
    signal_z_lookback: int = Field(default=60, ge=20, le=252)
    signal_z_cap: float = Field(default=2.5, ge=1.0, le=10.0)

    # Entry/exit thresholds — CALIBRATED: lower threshold captures more opportunities
    signal_entry_threshold: float = Field(default=0.05, ge=0.01, le=1.0)

    # Sector Mean-Reversion Whitelist — CALIBRATED 2026-03-24
    # Only these sectors show statistically significant mean reversion (Sharpe > 0.1)
    # XLC=0.12, XLF=0.12, XLI=0.25, XLU=0.40. Others: XLE=-0.28, XLP=-0.01, XLK=-0.02
    sector_mr_whitelist: str = Field(default="XLC,XLF,XLI,XLU")
    # Enable whitelist filter (set False to trade all sectors)
    sector_mr_filter_enabled: bool = Field(default=True)
    # Optimal hold period (days) — CALIBRATED: lb=60 hold=5 → Sharpe 0.456
    signal_optimal_hold: int = Field(default=5, ge=1, le=60)

    # Regime-adaptive sizing — CALIBRATED 2026-03-24
    # CALM: MR Sharpe=0.66 → full size + bonus
    # NORMAL: MR Sharpe=0.23 → full size
    # TENSION: MR Sharpe=0.68 → reduced but active (surprise: MR works in TENSION!)
    # CRISIS: MR Sharpe=-0.78 → KILL (no trading)
    regime_size_calm: float = Field(default=1.3, ge=0.0, le=2.0)
    regime_size_normal: float = Field(default=1.0, ge=0.0, le=2.0)
    regime_size_tension: float = Field(default=0.6, ge=0.0, le=2.0)
    regime_size_crisis: float = Field(default=0.0, ge=0.0, le=2.0)

    # Regime-adaptive z-threshold — tighter in volatile regimes
    regime_z_calm: float = Field(default=0.6, ge=0.1, le=3.0)
    regime_z_normal: float = Field(default=0.8, ge=0.1, le=3.0)
    regime_z_tension: float = Field(default=1.0, ge=0.1, le=3.0)
    regime_z_crisis: float = Field(default=99.0, ge=0.1, le=100.0)  # effectively disabled

    # Layer 3: Mean-Reversion Quality weights
    signal_mr_w_hl: float = Field(default=0.35, ge=0.0, le=1.0)
    signal_mr_w_adf: float = Field(default=0.40, ge=0.0, le=1.0)
    signal_mr_w_hurst: float = Field(default=0.25, ge=0.0, le=1.0)

    # Layer 4: Regime Safety — CALIBRATED from 10yr VIX distribution (2016-2026)
    # VIX soft=75th pctl, hard=95th pctl, kill=99th pctl
    signal_vix_soft: float = Field(default=21.0, ge=10.0, le=30.0)   # Was 18 (75th=21.3)
    signal_vix_hard: float = Field(default=31.0, ge=20.0, le=50.0)   # Was 30 (95th=30.8)
    signal_vix_kill: float = Field(default=45.0, ge=25.0, le=80.0)   # Was 35 (99th=44.9)

    # =====================================================
    # Trade Structure & Execution
    # =====================================================
    trade_dte_short_index: int = Field(default=30, ge=7, le=90)
    trade_dte_long_sector: int = Field(default=45, ge=14, le=120)
    trade_max_loss_pct: float = Field(default=0.02, ge=0.005, le=0.10)
    trade_profit_target_pct: float = Field(default=0.015, ge=0.005, le=0.10)
    # CALIBRATED: shorter hold (25d) + tighter sizing improves Sharpe
    trade_max_holding_days: int = Field(default=25, ge=7, le=180)
    trade_max_gross_dispersion: float = Field(default=0.40, ge=0.05, le=1.0)

    # =====================================================
    # Trade Monitoring & Exit Management
    # =====================================================
    monitor_z_compression_exit: float = Field(default=0.75, ge=0.3, le=1.0)
    # CALIBRATED: wider stop (2.5x) reduces premature stop-outs
    monitor_z_extension_stop: float = Field(default=2.50, ge=1.1, le=3.0)
    monitor_time_decay_warning: float = Field(default=0.60, ge=0.3, le=0.9)
    monitor_time_decay_exit: float = Field(default=0.90, ge=0.7, le=1.0)
    monitor_safety_floor: float = Field(default=0.15, ge=0.0, le=0.5)
    monitor_hl_exit_multiple: float = Field(default=2.5, ge=1.0, le=5.0)

    # =====================================================
    # PCA configuration — CALIBRATED: more components = richer factor structure
    # =====================================================
    pca_explained_var_target: float = Field(default=0.85, ge=0.50, le=0.99)  # Was 0.80
    pca_min_components: int = Field(default=2, ge=1, le=10)   # Was 1 — at least market+1 factor
    pca_max_components: int = Field(default=8, ge=1, le=11)   # Was 5 — capture more factors
    # PCA refit interval: reuse PCA model for N days before refitting (~5x speedup)
    pca_refit_interval: int = Field(default=5, ge=1, le=20)

    # =====================================================
    # Volatility model
    # =====================================================
    ewma_lambda: float = Field(default=0.94, ge=0.80, le=0.99)

    # =====================================================
    # Layer weights (must sum to 1.0)
    # =====================================================
    weight_stat: float = Field(default=0.40, ge=0.0, le=1.0)
    weight_macro: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_fund: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_vol: float = Field(default=0.20, ge=0.0, le=1.0)

    # =====================================================
    # Layer points (0..100 total)
    # =====================================================
    points_stat: int = Field(default=40, ge=0, le=100)
    points_macro: int = Field(default=20, ge=0, le=100)
    points_fund: int = Field(default=20, ge=0, le=100)
    points_vol: int = Field(default=20, ge=0, le=100)

    # =====================================================
    # Regime / risk thresholds
    # =====================================================
    vix_level_soft: float = Field(default=25.0, ge=10.0, le=80.0)
    vix_level_hard: float = Field(default=35.0, ge=10.0, le=120.0)
    vix_percentile_hard: float = Field(default=0.90, ge=0.50, le=0.99)

    credit_stress_z: float = Field(default=-1.0, ge=-5.0, le=0.0)
    # =====================================================
    # Regime engine thresholds
    # =====================================================
    # Correlation thresholds — CALIBRATED from avg_corr by VIX regime (2016-2026)
    # CALM(VIX<15)=0.33, NORMAL(15-20)=0.39, TENSION(20-25)=0.56, CRISIS(30+)=0.77
    calm_avg_corr_max: float = Field(default=0.40, ge=0.0, le=1.0)     # Was 0.45
    tension_avg_corr_min: float = Field(default=0.55, ge=0.0, le=1.0)  # Was 0.60
    crisis_avg_corr_min: float = Field(default=0.75, ge=0.0, le=1.0)   # Confirmed

    calm_mode_strength_max: float = Field(default=0.18, ge=0.0, le=1.0)
    tension_mode_strength_min: float = Field(default=0.28, ge=0.0, le=1.0)
    crisis_mode_strength_min: float = Field(default=0.38, ge=0.0, le=1.0)

    tension_corr_dist_min: float = Field(default=0.18, ge=0.0, le=5.0)
    crisis_corr_dist_min: float = Field(default=0.32, ge=0.0, le=5.0)

    tension_delta_corr_dist_min: float = Field(default=0.08, ge=0.0, le=5.0)
    crisis_delta_corr_dist_min: float = Field(default=0.18, ge=0.0, le=5.0)

    transition_score_caution: float = Field(default=0.45, ge=0.0, le=1.0)
    transition_score_danger: float = Field(default=0.70, ge=0.0, le=1.0)
    # =====================================================
    # Fundamental guardrails
    # =====================================================
    pe_extreme: float = Field(default=30.0, ge=5.0, le=200.0)
    neg_earnings_weight_cap: float = Field(default=0.20, ge=0.0, le=1.0)

    # =====================================================
    # Attribution scoring thresholds
    # =====================================================
    # SDS z-score normalization band: score = clip((|z| - lo) / (hi - lo))
    sds_z_lo: float = Field(default=0.75, ge=0.3, le=2.5)  # lower bound (entry threshold)
    sds_z_hi: float = Field(default=2.75, ge=1.0, le=5.0)  # upper bound (full score)

    # MSS normalization denominators
    mss_beta_mag_norm: float = Field(default=1.25, ge=0.3, le=5.0)   # |β_tnx, β_dxy| normalization
    mss_beta_delta_norm: float = Field(default=0.35, ge=0.05, le=2.0) # Δβ_spy instability normalization
    mss_corr_delta_norm: float = Field(default=0.20, ge=0.05, le=1.0) # Δcorr_to_spy instability normalization

    # STF slope normalization denominator
    stf_slope_norm: float = Field(default=0.12, ge=0.01, le=1.0)  # adverse slope → full structural risk

    # =====================================================
    # Attribution / book construction
    # =====================================================
    target_portfolio_vol: float = Field(default=0.12, ge=0.02, le=0.60)
    max_single_name_weight: float = Field(default=0.20, ge=0.01, le=0.50)
    mc_floor_trade: float = Field(default=0.10, ge=0.0, le=1.0)

    # Regime-aware gross leverage caps (multiplier on vol-scaled portfolio)
    # Applied AFTER volatility targeting to prevent catastrophic exposure in stress regimes.
    # CALM: full leverage allowed; NORMAL: 2x cap; TENSION: 1.2x; CRISIS: flat
    max_leverage_calm:    float = Field(default=3.0, ge=0.5, le=5.0)
    max_leverage_normal:  float = Field(default=2.0, ge=0.5, le=4.0)
    max_leverage_tension: float = Field(default=1.2, ge=0.1, le=3.0)
    max_leverage_crisis:  float = Field(default=0.0, ge=0.0, le=1.0)

    # Direction-dependent Z-score entry thresholds per regime
    # In higher-corr regimes, signals must be stronger to warrant entry
    zscore_threshold_calm:    float = Field(default=0.75, ge=0.3, le=2.5)
    zscore_threshold_normal:  float = Field(default=0.90, ge=0.3, le=2.5)
    zscore_threshold_tension: float = Field(default=1.25, ge=0.5, le=3.0)
    zscore_threshold_crisis:  float = Field(default=9.99, ge=1.0, le=10.0)  # effectively disabled

    # =====================================================
    # UI / dashboard defaults
    # =====================================================
    dashboard_title: str = Field(default="SRV Quantamental DSS")
    dashboard_host: str = Field(default="0.0.0.0")
    dashboard_port: int = Field(default=8050, ge=1, le=65535)
    dashboard_debug: bool = Field(default=False)

    # =====================================================
    # FMP sector alias normalization
    # =====================================================
    fmp_sector_name_aliases: Dict[str, str] = Field(
        default_factory=lambda: {
            "Communication Services": "Communication Services",
            "Consumer Cyclical": "Consumer Discretionary",
            "Consumer Discretionary": "Consumer Discretionary",
            "Consumer Defensive": "Consumer Staples",
            "Consumer Staples": "Consumer Staples",
            "Energy": "Energy",
            "Financial Services": "Financials",
            "Financial": "Financials",
            "Financials": "Financials",
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

    # =====================================================
    # Validators
    # =====================================================
    @field_validator("data_dir", "parquet_dir", "log_dir", mode="after")
    @classmethod
    def _ensure_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("pca_max_components", mode="after")
    @classmethod
    def _validate_pca_component_bounds(cls, v: int, info) -> int:
        min_c = info.data.get("pca_min_components", 1)
        if v < min_c:
            raise ValueError("pca_max_components must be >= pca_min_components")
        return v

    @field_validator("weight_vol", mode="after")
    @classmethod
    def _validate_layer_weight_sum(cls, v: float, info) -> float:
        w_stat = info.data.get("weight_stat", 0.40)
        w_macro = info.data.get("weight_macro", 0.20)
        w_fund = info.data.get("weight_fund", 0.20)
        total = w_stat + w_macro + w_fund + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Layer weights must sum to 1.0. Got {total:.6f}")
        return v

    @field_validator("points_vol", mode="after")
    @classmethod
    def _validate_points_sum(cls, v: int, info) -> int:
        p_stat = info.data.get("points_stat", 40)
        p_macro = info.data.get("points_macro", 20)
        p_fund = info.data.get("points_fund", 20)
        total = p_stat + p_macro + p_fund + v
        if total != 100:
            raise ValueError(f"Layer points must sum to 100. Got {total}")
        return v

    @field_validator("corr_baseline_window", mode="after")
    @classmethod
    def _validate_corr_windows(cls, v: int, info) -> int:
        corr_w = info.data.get("corr_window", 60)
        if v < corr_w:
            raise ValueError("corr_baseline_window must be >= corr_window")
        return v

    # =====================================================
    # Convenience helpers
    # =====================================================
    def sector_list(self) -> List[str]:
        return list(self.sector_tickers.values())

    def extended_ticker_list(self) -> List[str]:
        """Parse extended_tickers comma string into list."""
        if not self.extended_tickers.strip():
            return []
        return [t.strip() for t in self.extended_tickers.split(",") if t.strip()]

    def tradeable_universe(self) -> List[str]:
        """All tickers that the system actively trades/analyzes.
        In 'sectors' mode: 11 sector ETFs.
        In 'extended' mode: sectors + extended_tickers.
        """
        base = self.sector_list()
        if self.universe_mode == "extended" and self.extended_tickers.strip():
            base = base + self.extended_ticker_list()
        return base

    def all_price_tickers(self) -> List[str]:
        return (
            self.tradeable_universe()
            + [self.spy_ticker]
            + list(self.macro_tickers.values())
            + list(self.credit_tickers.values())
            + list(self.vol_tickers.values())
        )

    def canonical_sector_by_ticker(self) -> Dict[str, str]:
        return {ticker: sector for sector, ticker in self.sector_tickers.items()}

    def sector_name_by_ticker(self, ticker: str) -> str:
        return self.canonical_sector_by_ticker().get(ticker, ticker)

    def cyclical_sector_tickers(self) -> List[str]:
        names = [
            "Energy",
            "Financials",
            "Industrials",
            "Materials",
            "Consumer Discretionary",
        ]
        return [self.sector_tickers[n] for n in names if n in self.sector_tickers]

    def defensive_sector_tickers(self) -> List[str]:
        names = [
            "Utilities",
            "Consumer Staples",
            "Health Care",
        ]
        return [self.sector_tickers[n] for n in names if n in self.sector_tickers]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()