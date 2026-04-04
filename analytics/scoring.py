"""
analytics/scoring.py
======================
Typed scoring interface on top of QuantEngine's master_df output.

Purpose:
  QuantEngine.calculate_conviction_score() produces a raw master_df with
  50+ columns. This module provides typed accessor classes that give
  clean domain-oriented access to the data without coupling to column names.

Architecture:
  QuantEngine (stat_arb.py) → master_df (raw DataFrame)
      ↓
  ScoringResult.from_master_df(master_df) → typed accessors
      ↓
  SectorScore, RegimeState, PortfolioView, RiskMetrics

Usage:
    from analytics.scoring import ScoringResult
    result = ScoringResult.from_master_df(master_df)

    for sector in result.sectors:
        print(f"{sector.ticker}: {sector.direction} conv={sector.conviction:.3f}")

    print(f"Regime: {result.regime.state} VIX={result.regime.vix:.1f}")
    print(f"Portfolio: {result.portfolio.n_longs}L / {result.portfolio.n_shorts}S")

This avoids the need to break QuantEngine apart while providing clean interfaces
for downstream consumers (DSS, UI, agents, execution).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SectorScore:
    """Typed score for a single sector."""
    ticker: str
    sector_name: str
    direction: str                   # LONG / SHORT / NEUTRAL
    conviction: float                # 0–1 combined conviction score
    z_score: float                   # PCA residual z-score
    half_life: float                 # Mean-reversion half-life (days)

    # Factor scores
    sds: float = 0.0                 # Statistical Dislocation Score
    fjs: float = 0.0                 # Fundamental Justification Score
    mss: float = 0.0                 # Macro Shift Score
    stf: float = 0.0                 # Structural Trend Filter
    mc: float = 0.0                  # Mispricing Confidence

    # Weight
    weight: float = 0.0             # Final portfolio weight (after sizing)

    # Macro betas
    beta_spy: float = 0.0
    beta_tnx: float = 0.0
    beta_dxy: float = 0.0

    # Fundamentals
    rel_pe_vs_spy: float = 0.0

    # Risk label
    risk_label: str = ""
    action_bias: str = ""
    interpretation: str = ""


@dataclass
class RegimeState:
    """Current market regime with all components."""
    state: str                       # CALM / NORMAL / TENSION / CRISIS
    vix: float
    avg_correlation: float
    credit_z: float
    transition_score: float
    crisis_probability: float

    # Component scores
    vol_score: float = 0.0
    corr_score: float = 0.0
    credit_score: float = 0.0

    # Regime alert
    alert: str = ""
    state_bias: str = ""
    mean_reversion_allowed: bool = True

    @property
    def is_safe(self) -> bool:
        return self.state in ("CALM", "NORMAL")

    @property
    def is_crisis(self) -> bool:
        return self.state == "CRISIS"


@dataclass
class PortfolioView:
    """Portfolio-level summary from scoring."""
    n_sectors: int
    n_longs: int
    n_shorts: int
    n_neutral: int
    gross_exposure: float            # Sum of |weights|
    net_exposure: float              # Sum of weights (long - short)
    avg_conviction: float
    top_long: Optional[SectorScore] = None
    top_short: Optional[SectorScore] = None


@dataclass
class ScoringResult:
    """
    Complete typed scoring output.

    Single point of access for all scoring data produced by QuantEngine.
    Replaces ad-hoc master_df column access throughout the codebase.
    """
    sectors: List[SectorScore]
    regime: RegimeState
    portfolio: PortfolioView
    raw_df: pd.DataFrame             # Original master_df for backward compat
    timestamp: str = ""

    @classmethod
    def from_master_df(cls, master_df: pd.DataFrame) -> "ScoringResult":
        """
        Convert raw master_df into typed ScoringResult.

        This is the main factory method — call it after
        QuantEngine.calculate_conviction_score().
        """
        if master_df is None or master_df.empty:
            return cls(
                sectors=[],
                regime=RegimeState(state="UNKNOWN", vix=0, avg_correlation=0, credit_z=0, transition_score=0, crisis_probability=0),
                portfolio=PortfolioView(n_sectors=0, n_longs=0, n_shorts=0, n_neutral=0, gross_exposure=0, net_exposure=0, avg_conviction=0),
                raw_df=pd.DataFrame(),
            )

        def _sf(val, default=0.0):
            try:
                v = float(val)
                return v if math.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        # Extract sectors
        sectors = []
        for _, row in master_df.iterrows():
            sectors.append(SectorScore(
                ticker=str(row.get("sector_ticker", "")),
                sector_name=str(row.get("sector_name", "")),
                direction=str(row.get("direction", "NEUTRAL")),
                conviction=_sf(row.get("conviction_score")),
                z_score=_sf(row.get("pca_residual_z")),
                half_life=_sf(row.get("half_life_days_est")),
                sds=_sf(row.get("sds_score")),
                fjs=_sf(row.get("fjs_score")),
                mss=_sf(row.get("mss_score")),
                stf=_sf(row.get("stf_score")),
                mc=_sf(row.get("mc_score_raw")),
                weight=_sf(row.get("w_final")),
                beta_spy=_sf(row.get("beta_spy_delta")),
                beta_tnx=_sf(row.get("beta_tnx_60d")),
                beta_dxy=_sf(row.get("beta_dxy_60d")),
                rel_pe_vs_spy=_sf(row.get("rel_pe_vs_spy")),
                risk_label=str(row.get("risk_label", "")),
                action_bias=str(row.get("action_bias", "")),
                interpretation=str(row.get("interpretation", "")),
            ))

        # Extract regime
        row0 = master_df.iloc[0]
        regime = RegimeState(
            state=str(row0.get("market_state", "UNKNOWN")),
            vix=_sf(row0.get("vix_level")),
            avg_correlation=_sf(row0.get("sector_corr_avg")),
            credit_z=_sf(row0.get("credit_z")),
            transition_score=_sf(row0.get("transition_score")),
            crisis_probability=_sf(row0.get("crisis_probability")),
            vol_score=_sf(row0.get("vol_score")),
            corr_score=_sf(row0.get("corr_score")),
            credit_score=_sf(row0.get("credit_score")),
            alert=str(row0.get("regime_alert", "")),
            state_bias=str(row0.get("state_bias", "")),
            mean_reversion_allowed=bool(row0.get("mean_reversion_allowed", True)),
        )

        # Portfolio summary
        longs = [s for s in sectors if s.direction == "LONG"]
        shorts = [s for s in sectors if s.direction == "SHORT"]
        neutrals = [s for s in sectors if s.direction == "NEUTRAL"]

        gross = sum(abs(s.weight) for s in sectors)
        net = sum(s.weight for s in sectors)
        avg_conv = float(np.mean([s.conviction for s in sectors])) if sectors else 0

        top_long = max(longs, key=lambda s: s.conviction) if longs else None
        top_short = max(shorts, key=lambda s: s.conviction) if shorts else None

        portfolio = PortfolioView(
            n_sectors=len(sectors),
            n_longs=len(longs),
            n_shorts=len(shorts),
            n_neutral=len(neutrals),
            gross_exposure=round(gross, 4),
            net_exposure=round(net, 4),
            avg_conviction=round(avg_conv, 4),
            top_long=top_long,
            top_short=top_short,
        )

        return cls(
            sectors=sectors,
            regime=regime,
            portfolio=portfolio,
            raw_df=master_df,
        )

    def get_sector(self, ticker: str) -> Optional[SectorScore]:
        """Get score for a specific sector."""
        for s in self.sectors:
            if s.ticker == ticker:
                return s
        return None

    def longs(self) -> List[SectorScore]:
        return [s for s in self.sectors if s.direction == "LONG"]

    def shorts(self) -> List[SectorScore]:
        return [s for s in self.sectors if s.direction == "SHORT"]

    def ranked_by_conviction(self) -> List[SectorScore]:
        return sorted(self.sectors, key=lambda s: s.conviction, reverse=True)

    def to_dict(self) -> Dict:
        """Serialize for API/JSON output."""
        return {
            "regime": {
                "state": self.regime.state,
                "vix": self.regime.vix,
                "correlation": self.regime.avg_correlation,
                "crisis_prob": self.regime.crisis_probability,
                "is_safe": self.regime.is_safe,
            },
            "portfolio": {
                "n_longs": self.portfolio.n_longs,
                "n_shorts": self.portfolio.n_shorts,
                "gross": self.portfolio.gross_exposure,
                "net": self.portfolio.net_exposure,
                "avg_conviction": self.portfolio.avg_conviction,
            },
            "sectors": [
                {
                    "ticker": s.ticker,
                    "direction": s.direction,
                    "conviction": s.conviction,
                    "z": s.z_score,
                    "weight": s.weight,
                }
                for s in self.ranked_by_conviction()[:10]
            ],
        }
