"""
analytics/stress.py

Deterministic stress-test engine for the SRV Quantamental DSS.

Ten institutionally calibrated scenarios cover the primary risk dimensions
that can disrupt a cross-sectional sector-rotation / stat-arb strategy:
rates, risk-off, stagflation, factor rotation, credit, dollar, earnings,
and model-structural breakdown.

PnL estimation methodology
---------------------------
Each scenario defines a set of factor shocks (SPY return, TNX change,
DXY % change) and idiosyncratic sector returns.  Portfolio P&L is computed
as a linear factor decomposition using the synthetic greeks already present
in master_df:

    pnl_sector_i = delta_spy_i  * spy_return
                 + delta_tnx_i  * tnx_change
                 + delta_dxy_i  * dxy_change
                 + w_final_i    * sector_idio_shock_i

    portfolio_pnl = Σ pnl_sector_i

where delta_*_i = w_final_i * beta_*_i  (pre-computed in QuantEngine).

Signal reliability score
------------------------
Measures how much the current model signals can be trusted under the
stressed environment.  Derived from:
  - Baseline quality: median mc_score_raw across sectors
  - VIX dampening:    elevated VIX compresses mean-reversion opportunities
  - Correlation dampening: high cross-sector correlation renders PCA
                            residuals less informative
  - Credit dampening: spread blowout breaks fundamental anchors
  - Model penalty:    scenario-specific direct signal degradation

All shocks are one-shot (not time-scaled).  No Monte Carlo or path
simulation is performed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


# ==========================================================================
# Private helpers
# ==========================================================================

def _sf(x: Any, default: float = 0.0) -> float:
    """Safe float coercion; returns default when x is None / NaN / inf."""
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return lo
    return max(lo, min(hi, x))


def _col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series from df[col], filling missing with default."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


# ==========================================================================
# Data containers
# ==========================================================================

@dataclass(frozen=True)
class StressScenario:
    """
    Specification of a single stress scenario.

    shocks keys (all optional, default 0.0 / current):
        spy_return          : float  – SPY total return in the scenario (e.g., -0.08)
        tnx_change          : float  – Absolute change in TNX yield index (e.g., 1.50 = +150bp)
        dxy_change          : float  – DXY % change (e.g., 0.10 = +10%)
        vix_implied         : float  – Implied VIX level (e.g., 45.0)
        avg_corr_implied    : float  – Implied avg cross-sector correlation
        credit_z_shock      : float  – Additive shock to current credit z-score
        model_signal_penalty: float  – Direct model degradation factor [0, 0.9]

    expected_sector_impacts:
        Idiosyncratic return per sector ETF ticker (e.g., {"XLK": -0.25}).
        Applied on top of factor channels as w_final * idio.
    """
    name: str
    description: str
    shocks: Dict[str, float]
    expected_sector_impacts: Dict[str, float]


@dataclass
class StressResult:
    """Output of one scenario evaluation against the current portfolio."""
    scenario_name: str
    portfolio_pnl_estimate: float   # estimated total P&L (gross=1 normalised)
    worst_sector: str               # ticker with largest individual loss
    best_sector: str                # ticker with largest individual gain
    signal_reliability_score: float # 0 = model useless, 1 = fully trustworthy
    regime_label: str               # implied regime under scenario
    notes: str                      # PM-facing narrative summary

    # Factor attribution (available for drill-down)
    pnl_spy_channel: float = 0.0
    pnl_tnx_channel: float = 0.0
    pnl_dxy_channel: float = 0.0
    pnl_idio_channel: float = 0.0

    # Per-sector P&L (dict ticker → float)
    sector_pnl: Dict[str, float] = field(default_factory=dict)


# ==========================================================================
# Scenario library
# ==========================================================================

def _build_scenarios() -> List[StressScenario]:
    """
    Construct the 10 canonical stress scenarios.

    Calibration rationale is documented inline.  All magnitudes reflect
    plausible but severe one-month instantaneous shocks unless noted.
    """
    return [

        # ------------------------------------------------------------------
        # 1. RATES_SHOCK_UP
        # Parallel +150bp shift: inflation surprise or Fed hawkish pivot.
        # Duration-heavy sectors (Utilities, Real Estate, Tech) re-price
        # their earnings multiples downward; Financials benefit via NIM.
        # DXY strengthens on rate differential; SPY loses ~3.5%.
        # ------------------------------------------------------------------
        StressScenario(
            name="RATES_SHOCK_UP",
            description=(
                "+150bps parallel rate shock: TNX beta stress, duration repricing. "
                "Growth sectors hit via earnings multiple compression; Financials benefit."
            ),
            shocks={
                "spy_return":           -0.035,
                "tnx_change":            1.50,
                "dxy_change":            0.03,
                "vix_implied":          24.0,
                "avg_corr_implied":      0.55,
                "credit_z_shock":       -0.50,
                "model_signal_penalty":  0.10,
            },
            expected_sector_impacts={
                "XLU":  -0.060,   # Utilities: bond proxy, long-duration proxies re-price
                "XLRE": -0.055,   # Real Estate: debt cost, cap-rate expansion
                "XLK":  -0.040,   # Technology: long-duration growth, multiple compression
                "XLC":  -0.030,   # Communication: growth-like duration, ad spend cyclical
                "XLY":  -0.025,   # Consumer Disc: rate-sensitive consumer credit
                "XLP":  -0.015,   # Consumer Staples: mild bond proxy, defensive offset
                "XLV":  -0.010,   # Health Care: moderate duration; defensive partially offset
                "XLI":  -0.005,   # Industrials: mild rate headwind, growth offset
                "XLB":   0.010,   # Materials: real asset / inflation hedge premium
                "XLE":   0.020,   # Energy: rates + inflation signal, commodity hedge
                "XLF":   0.025,   # Financials: NIM expansion, re-pricing tailwind
            },
        ),

        # ------------------------------------------------------------------
        # 2. RATES_SHOCK_DOWN
        # -100bp: recession signal, yield curve inversion deepens.
        # Flight to defensive sectors; cyclicals and Financials re-price.
        # DXY weakens on dovish Fed; VIX elevated on growth fears.
        # ------------------------------------------------------------------
        StressScenario(
            name="RATES_SHOCK_DOWN",
            description=(
                "-100bps rate shock: recession signal, yield curve inversion. "
                "Defensives bid up; cyclicals and Financials pressured."
            ),
            shocks={
                "spy_return":           -0.055,
                "tnx_change":           -1.00,
                "dxy_change":           -0.025,
                "vix_implied":          32.0,
                "avg_corr_implied":      0.65,
                "credit_z_shock":       -1.20,
                "model_signal_penalty":  0.20,
            },
            expected_sector_impacts={
                "XLE":  -0.060,   # Energy: demand collapse pricing
                "XLF":  -0.045,   # Financials: NIM compression, recession loan losses
                "XLC":  -0.030,   # Communication: ad spend recession sensitivity
                "XLY":  -0.030,   # Consumer Disc: consumer pullback
                "XLI":  -0.020,   # Industrials: capex cycle contraction
                "XLB":  -0.020,   # Materials: demand collapse
                "XLK":  -0.010,   # Technology: mild; recession offset by flight to quality
                "XLP":   0.020,   # Consumer Staples: defensive rotation bid
                "XLV":   0.025,   # Health Care: defensive, non-discretionary demand
                "XLU":   0.030,   # Utilities: bond proxy, flights to income
                "XLRE":  0.005,   # Real Estate: conflicting—lower rates help but recession hurts
            },
        ),

        # ------------------------------------------------------------------
        # 3. RISK_OFF_ACUTE
        # VIX spike to 45, correlation to 0.85, credit +300bp.
        # Acute market shock: geopolitical event, flash crash, liquidity seizure.
        # All sectors sell off; cross-sector correlations collapse to near 1.
        # PCA residuals lose informational content entirely.
        # ------------------------------------------------------------------
        StressScenario(
            name="RISK_OFF_ACUTE",
            description=(
                "Acute risk-off: VIX 45, cross-sector corr 0.85, credit +300bps. "
                "All sectors sell off; model signals degrade severely."
            ),
            shocks={
                "spy_return":           -0.120,
                "tnx_change":           -0.50,   # flight to safety
                "dxy_change":            0.05,   # safe-haven bid
                "vix_implied":          45.0,
                "avg_corr_implied":      0.85,
                "credit_z_shock":       -3.00,
                "model_signal_penalty":  0.65,   # severe: correlation spike kills PCA residuals
            },
            expected_sector_impacts={
                "XLF":  -0.060,   # Financials: liquidity premium, credit risk
                "XLE":  -0.050,   # Energy: demand shock + commodity sell-off
                "XLY":  -0.045,   # Consumer Disc: risk asset sell-off
                "XLC":  -0.040,   # Communication: correlated with growth sell-off
                "XLB":  -0.040,   # Materials: commodity liquidation
                "XLI":  -0.035,   # Industrials: growth risk premium
                "XLK":  -0.035,   # Technology: high beta, risk-off de-rating
                "XLRE": -0.030,   # Real Estate: credit spread widening
                "XLV":  -0.010,   # Health Care: modest flight to defensive
                "XLP":  -0.008,   # Consumer Staples: mild defensive buffer
                "XLU":  -0.005,   # Utilities: near-zero; defensive plus rate flight
            },
        ),

        # ------------------------------------------------------------------
        # 4. RISK_OFF_CHRONIC
        # 6-month slow grind: VIX 30–35, avg corr 0.70, growth compression.
        # Persistent uncertainty creates elevated but stable stress; mean
        # reversion opportunities exist but with high false-positive rate.
        # ------------------------------------------------------------------
        StressScenario(
            name="RISK_OFF_CHRONIC",
            description=(
                "Chronic risk-off grind: VIX 30-35, avg corr 0.70, 6-month drawdown. "
                "Persistent regime; mean-reversion signals unreliable."
            ),
            shocks={
                "spy_return":           -0.080,
                "tnx_change":           -0.30,
                "dxy_change":            0.02,
                "vix_implied":          32.0,
                "avg_corr_implied":      0.70,
                "credit_z_shock":       -1.50,
                "model_signal_penalty":  0.40,
            },
            expected_sector_impacts={
                "XLE":  -0.030,
                "XLF":  -0.025,
                "XLY":  -0.025,
                "XLC":  -0.020,
                "XLI":  -0.020,
                "XLB":  -0.018,
                "XLK":  -0.015,
                "XLRE": -0.015,
                "XLV":   0.010,
                "XLP":   0.015,
                "XLU":   0.012,
            },
        ),

        # ------------------------------------------------------------------
        # 5. STAGFLATION
        # Inflation rising, growth falling: energy/materials outperform,
        # tech and consumer heavily hit by margin compression + rate + demand.
        # Historical comps: 1973–74, 2022 partial pattern.
        # ------------------------------------------------------------------
        StressScenario(
            name="STAGFLATION",
            description=(
                "Stagflation: inflation up, growth down, supply shock. "
                "Energy/Materials outperform; Tech/Consumer Disc/Staples compressed."
            ),
            shocks={
                "spy_return":           -0.050,
                "tnx_change":            0.80,   # inflation expectations push yields up
                "dxy_change":            0.015,
                "vix_implied":          28.0,
                "avg_corr_implied":      0.60,
                "credit_z_shock":       -0.80,
                "model_signal_penalty":  0.25,
            },
            expected_sector_impacts={
                "XLK":  -0.060,   # Tech: margin compression + rate headwind
                "XLY":  -0.050,   # Consumer Disc: real income erosion + rates
                "XLP":  -0.025,   # Consumer Staples: cost pass-through incomplete
                "XLV":  -0.015,   # Health Care: cost pressure, moderate
                "XLC":  -0.020,   # Communication: ad spend + consumer pullback
                "XLF":  -0.010,   # Financials: NIM partially offset by credit risk
                "XLRE": -0.020,   # Real Estate: higher rates + slower growth
                "XLI":   0.005,   # Industrials: infrastructure / defense offset
                "XLU":  -0.005,   # Utilities: cost inflation, cap structure stress
                "XLB":   0.035,   # Materials: commodity price surge (direct beneficiary)
                "XLE":   0.055,   # Energy: primary beneficiary of commodity shock
            },
        ),

        # ------------------------------------------------------------------
        # 6. TECH_SELLOFF
        # XLK -25%: AI bubble deflation, regulatory risk, or rate catalyst.
        # Since XLK ~ 30% of S&P, SPY falls ~8%.  Growth-to-value rotation:
        # capital flows into Energy, Financials, Materials.
        # ------------------------------------------------------------------
        StressScenario(
            name="TECH_SELLOFF",
            description=(
                "XLK drawdown -25%: AI/growth bubble deflation or regulatory shock. "
                "Factor reversal drives growth-to-value rotation."
            ),
            shocks={
                "spy_return":           -0.080,   # XLK ~30% of S&P
                "tnx_change":            0.50,    # growth → value rotation catalyst
                "dxy_change":            0.00,
                "vix_implied":          32.0,
                "avg_corr_implied":      0.62,
                "credit_z_shock":       -0.30,
                "model_signal_penalty":  0.30,    # factor reversal disrupts PCA
            },
            expected_sector_impacts={
                "XLK":  -0.250,   # direct shock
                "XLC":  -0.080,   # high tech/mega-cap correlation
                "XLY":  -0.030,   # consumer cyclical correlated
                "XLRE": -0.020,   # mild duration sell-off
                "XLV":  -0.005,   # near-neutral
                "XLP":   0.005,   # mild defensive bid
                "XLU":   0.010,   # mild defensive / yield bid
                "XLI":   0.015,   # value rotation
                "XLB":   0.020,   # value rotation, real assets
                "XLF":   0.025,   # value rotation, NIM benefit from rates
                "XLE":   0.030,   # value rotation, inflation hedge
            },
        ),

        # ------------------------------------------------------------------
        # 7. CREDIT_CRISIS
        # HYG/IEF blow-out, financial sector stress, liquidity premium spike.
        # Historical comps: 2008 GFC, 2011 Euro crisis, 2020 March liquidity.
        # ------------------------------------------------------------------
        StressScenario(
            name="CREDIT_CRISIS",
            description=(
                "Credit crisis: HYG/IEF blow-out, Financials stress, liquidity freeze. "
                "Credit z-score collapses; correlation spikes to crisis level."
            ),
            shocks={
                "spy_return":           -0.100,
                "tnx_change":           -0.80,   # flight to quality: govts rally
                "dxy_change":            0.04,   # dollar safe-haven bid
                "vix_implied":          40.0,
                "avg_corr_implied":      0.80,
                "credit_z_shock":       -3.50,   # +300bp credit spread shock
                "model_signal_penalty":  0.60,
            },
            expected_sector_impacts={
                "XLF":  -0.150,   # Financials: direct credit / counterparty risk
                "XLRE": -0.060,   # Real Estate: credit market access impaired
                "XLY":  -0.040,   # Consumer Disc: credit availability, consumer stress
                "XLC":  -0.030,
                "XLI":  -0.030,
                "XLB":  -0.025,
                "XLE":  -0.025,
                "XLK":  -0.020,
                "XLV":   0.005,
                "XLP":   0.010,
                "XLU":   0.015,
            },
        ),

        # ------------------------------------------------------------------
        # 8. DOLLAR_SURGE
        # DXY +10%: Fed hawkish surprise or EM currency crisis.
        # Commodity prices fall in USD terms; international revenue headwind.
        # Energy and Materials hardest hit via commodity price channel.
        # ------------------------------------------------------------------
        StressScenario(
            name="DOLLAR_SURGE",
            description=(
                "DXY +10%: Fed hawkish surprise or EM crisis. "
                "Commodity prices compressed; international revenue headwind."
            ),
            shocks={
                "spy_return":           -0.030,
                "tnx_change":            0.50,
                "dxy_change":            0.10,
                "vix_implied":          22.0,
                "avg_corr_implied":      0.52,
                "credit_z_shock":       -0.40,
                "model_signal_penalty":  0.10,
            },
            expected_sector_impacts={
                "XLE":  -0.055,   # Energy: oil/gas prices fall in USD terms
                "XLB":  -0.045,   # Materials: commodity prices, EM demand
                "XLC":  -0.020,   # Communication: international revenue exposure
                "XLK":  -0.015,   # Technology: international revenue (mitigated)
                "XLY":  -0.010,   # Consumer Disc: import cost headwind
                "XLI":  -0.010,   # Industrials: export competitiveness loss
                "XLRE":  0.000,
                "XLP":  -0.005,   # Consumer Staples: slight international revenue
                "XLV":   0.005,   # Health Care: limited dollar exposure
                "XLF":   0.015,   # Financials: USD strength, dollar assets bid
                "XLU":   0.005,   # Utilities: domestic, mild positive
            },
        ),

        # ------------------------------------------------------------------
        # 9. EARNINGS_MISS
        # Broad EPS disappointment across sectors: guidance cuts, margin
        # compression, FJS scores compressed.  Fundamentals channel fails;
        # signal reliability degrades through the fjs dimension specifically.
        # ------------------------------------------------------------------
        StressScenario(
            name="EARNINGS_MISS",
            description=(
                "Broad EPS disappointment: guidance cuts, margin compression. "
                "Fundamental justification scores compressed; fjs-anchored signals degrade."
            ),
            shocks={
                "spy_return":           -0.060,
                "tnx_change":           -0.20,   # growth expectations lower
                "dxy_change":           -0.01,
                "vix_implied":          30.0,
                "avg_corr_implied":      0.63,
                "credit_z_shock":       -0.70,
                "model_signal_penalty":  0.35,   # FJS-anchored signals lose validity
            },
            expected_sector_impacts={
                "XLK":  -0.050,   # Tech: highest multiple compression on EPS miss
                "XLC":  -0.040,   # Communication: ad revenue, guidance cuts
                "XLY":  -0.040,   # Consumer Disc: margin pressure
                "XLI":  -0.030,   # Industrials: revenue miss on capex slowdown
                "XLB":  -0.025,   # Materials: demand + price softening
                "XLE":  -0.025,   # Energy: demand + capex cuts
                "XLF":  -0.025,   # Financials: loan growth, NII guidance cuts
                "XLRE": -0.020,   # Real Estate: NOI miss on vacancy / rent
                "XLP":  -0.015,   # Consumer Staples: cost-through miss
                "XLV":  -0.020,   # Health Care: drug pricing / volume guidance
                "XLU":  -0.010,   # Utilities: rate case outcomes
            },
        ),

        # ------------------------------------------------------------------
        # 10. CORRELATION_BREAKDOWN
        # PCA regime shift: cross-sector correlations collapse toward 1,
        # market mode dominates all variance.  PCA residuals no longer capture
        # idiosyncratic mean reversion; alpha decay is severe.
        # This is a model-structural stress: the strategy itself is at risk.
        # ------------------------------------------------------------------
        StressScenario(
            name="CORRELATION_BREAKDOWN",
            description=(
                "PCA regime shift: correlation spike to 0.82, market mode dominates. "
                "PCA residuals lose informational content; alpha decay stress."
            ),
            shocks={
                "spy_return":            0.000,   # market-neutral surface; direction uncertain
                "tnx_change":            0.00,
                "dxy_change":            0.00,
                "vix_implied":          35.0,
                "avg_corr_implied":      0.82,    # PCA becomes near-trivial decomposition
                "credit_z_shock":       -0.50,
                "model_signal_penalty":  0.75,   # direct model structural degradation
            },
            expected_sector_impacts={
                # Under correlation breakdown, idiosyncratic shocks are indeterminate;
                # all sectors move with the common factor.
                # We apply mild uniform pressure to reflect deleveraging flows.
                "XLK":  -0.015,
                "XLC":  -0.012,
                "XLY":  -0.010,
                "XLF":  -0.010,
                "XLI":  -0.008,
                "XLB":  -0.008,
                "XLE":  -0.008,
                "XLRE": -0.008,
                "XLV":  -0.005,
                "XLP":  -0.005,
                "XLU":  -0.005,
            },
        ),
    ]


# ==========================================================================
# StressEngine
# ==========================================================================

class StressEngine:
    """
    Deterministic stress-test engine for the SRV Quantamental DSS.

    Uses the portfolio greeks and macro betas already computed by QuantEngine
    (available in master_df) to estimate P&L under each scenario without any
    random simulation.

    Usage
    -----
    engine = StressEngine()
    results = engine.run_all(master_df, settings)
    df = engine.summary_table(results)
    """

    SCENARIOS: List[StressScenario] = _build_scenarios()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(
        self,
        master_df: pd.DataFrame,
        settings: Settings,
    ) -> List[StressResult]:
        """
        Run all 10 scenarios against the current portfolio state.

        Parameters
        ----------
        master_df : pd.DataFrame
            Output of QuantEngine.calculate_conviction_score().
        settings : Settings
            Validated settings instance.

        Returns
        -------
        List[StressResult]
            Results sorted from worst to best estimated P&L.
        """
        if master_df is None or master_df.empty:
            raise ValueError("master_df is empty; run QuantEngine.calculate_conviction_score() first.")

        results = [self.run_scenario(s, master_df, settings) for s in self.SCENARIOS]
        results.sort(key=lambda r: r.portfolio_pnl_estimate)
        logger.info(
            "Stress test complete: %d scenarios | worst=%s (%.2f%%) | best=%s (%.2f%%)",
            len(results),
            results[0].scenario_name, results[0].portfolio_pnl_estimate * 100,
            results[-1].scenario_name, results[-1].portfolio_pnl_estimate * 100,
        )
        return results

    def run_scenario(
        self,
        scenario: StressScenario,
        master_df: pd.DataFrame,
        settings: Settings,
    ) -> StressResult:
        """
        Evaluate one scenario against the current portfolio state.

        Parameters
        ----------
        scenario : StressScenario
            Scenario specification (from SCENARIOS or custom).
        master_df : pd.DataFrame
            Output of QuantEngine.calculate_conviction_score().
        settings : Settings
            Validated settings instance.

        Returns
        -------
        StressResult
        """
        shocks = scenario.shocks
        spy_ret  = _sf(shocks.get("spy_return",  0.0))
        tnx_chg  = _sf(shocks.get("tnx_change",  0.0))
        dxy_chg  = _sf(shocks.get("dxy_change",  0.0))

        # Per-sector factor P&L channels
        d_spy = _col(master_df, "delta_spy_i")
        d_tnx = _col(master_df, "delta_tnx_i")
        d_dxy = _col(master_df, "delta_dxy_i")
        w_fin = _col(master_df, "w_final")

        pnl_spy = d_spy * spy_ret
        pnl_tnx = d_tnx * tnx_chg
        pnl_dxy = d_dxy * dxy_chg

        # Idiosyncratic channel
        tickers = master_df["sector_ticker"].astype(str).values if "sector_ticker" in master_df.columns else []
        idio_shocks = pd.Series(
            [_sf(scenario.expected_sector_impacts.get(t, 0.0)) for t in tickers],
            index=master_df.index,
        )
        pnl_idio = w_fin * idio_shocks

        sector_pnl_series = pnl_spy + pnl_tnx + pnl_dxy + pnl_idio
        portfolio_pnl = float(sector_pnl_series.sum())

        # Portfolio-level channel attribution
        pnl_spy_total  = float(pnl_spy.sum())
        pnl_tnx_total  = float(pnl_tnx.sum())
        pnl_dxy_total  = float(pnl_dxy.sum())
        pnl_idio_total = float(pnl_idio.sum())

        # Per-sector P&L dict for drill-down
        sector_pnl_dict: Dict[str, float] = {}
        for i, t in enumerate(tickers):
            sector_pnl_dict[str(t)] = float(sector_pnl_series.iloc[i])

        # Worst / best sector
        worst_sector, best_sector = self._worst_best(sector_pnl_dict)

        # Signal reliability
        reliability = self._signal_reliability(master_df, scenario, settings)

        # Implied regime label
        regime_label = self._implied_regime(scenario, settings)

        # PM narrative
        notes = self._build_notes(
            scenario=scenario,
            portfolio_pnl=portfolio_pnl,
            pnl_spy_total=pnl_spy_total,
            pnl_tnx_total=pnl_tnx_total,
            pnl_dxy_total=pnl_dxy_total,
            pnl_idio_total=pnl_idio_total,
            worst_sector=worst_sector,
            best_sector=best_sector,
            reliability=reliability,
            regime_label=regime_label,
        )

        logger.debug(
            "Scenario %-25s  pnl=%+.2f%%  regime=%-8s  reliability=%.2f",
            scenario.name, portfolio_pnl * 100, regime_label, reliability,
        )

        return StressResult(
            scenario_name=scenario.name,
            portfolio_pnl_estimate=portfolio_pnl,
            worst_sector=worst_sector,
            best_sector=best_sector,
            signal_reliability_score=reliability,
            regime_label=regime_label,
            notes=notes,
            pnl_spy_channel=pnl_spy_total,
            pnl_tnx_channel=pnl_tnx_total,
            pnl_dxy_channel=pnl_dxy_total,
            pnl_idio_channel=pnl_idio_total,
            sector_pnl=sector_pnl_dict,
        )

    def summary_table(self, results: List[StressResult]) -> pd.DataFrame:
        """
        Build a Dash-ready summary DataFrame from a list of StressResult.

        One row per scenario, sorted by portfolio_pnl_estimate ascending
        (worst scenarios first).

        Columns
        -------
        scenario, pnl_pct, spy_channel_pct, tnx_channel_pct, dxy_channel_pct,
        idio_channel_pct, worst_sector, best_sector, signal_reliability,
        regime_label, notes
        """
        rows = []
        for r in sorted(results, key=lambda x: x.portfolio_pnl_estimate):
            rows.append({
                "scenario":           r.scenario_name,
                "pnl_pct":            round(r.portfolio_pnl_estimate * 100, 3),
                "spy_channel_pct":    round(r.pnl_spy_channel * 100, 3),
                "tnx_channel_pct":    round(r.pnl_tnx_channel * 100, 3),
                "dxy_channel_pct":    round(r.pnl_dxy_channel * 100, 3),
                "idio_channel_pct":   round(r.pnl_idio_channel * 100, 3),
                "worst_sector":       r.worst_sector,
                "best_sector":        r.best_sector,
                "signal_reliability": round(r.signal_reliability_score, 3),
                "regime_label":       r.regime_label,
                "notes":              r.notes,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _signal_reliability(
        self,
        master_df: pd.DataFrame,
        scenario: StressScenario,
        settings: Settings,
    ) -> float:
        """
        Estimate signal reliability under the scenario environment.

        Combines current baseline signal quality with scenario-specific
        dampening from VIX, correlation, credit, and model penalties.
        """
        # Baseline: median mc_score_raw (current model quality)
        mc_vals = pd.to_numeric(
            master_df.get("mc_score_raw", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        base_mc = float(mc_vals.median()) if len(mc_vals) > 0 else 0.50
        base_mc = _clip(base_mc, 0.20, 0.90)

        shocks = scenario.shocks
        vix_implied = _sf(shocks.get("vix_implied", settings.vix_level_soft))
        avg_corr    = _sf(shocks.get("avg_corr_implied", settings.calm_avg_corr_max))
        credit_shock = _sf(shocks.get("credit_z_shock", 0.0))
        model_pen   = _sf(shocks.get("model_signal_penalty", 0.0))

        # Current credit z (from master_df, first row — broadcast column)
        current_credit_z = _sf(
            master_df["credit_z"].iloc[0] if "credit_z" in master_df.columns and len(master_df) else 0.0
        )
        net_credit_z = current_credit_z + credit_shock

        # VIX dampening: 0 at vix_soft (25), 0.55 at vix_hard (35), capped at 0.65
        vix_range = max(1.0, settings.vix_level_hard - settings.vix_level_soft)
        vix_stress = _clip((vix_implied - settings.vix_level_soft) / vix_range, 0.0, 1.0)
        vix_dampening = 1.0 - 0.55 * vix_stress

        # Correlation dampening: 0 at calm_max (0.45), 0.55 at crisis_min (0.75)
        corr_range = max(0.01, settings.crisis_avg_corr_min - settings.calm_avg_corr_max)
        corr_stress = _clip((avg_corr - settings.calm_avg_corr_max) / corr_range, 0.0, 1.0)
        corr_dampening = 1.0 - 0.55 * corr_stress

        # Credit dampening: 0 above credit_stress_z threshold, 0.30 at z = -3.5
        credit_stress_val = _clip((-net_credit_z - 0.5) / 3.0, 0.0, 1.0)
        credit_dampening = 1.0 - 0.30 * credit_stress_val

        # Direct model penalty (scenario-specific)
        model_factor = 1.0 - _clip(model_pen, 0.0, 0.90)

        raw = base_mc * vix_dampening * corr_dampening * credit_dampening * model_factor
        return round(_clip(raw, 0.03, 0.95), 4)

    def _implied_regime(
        self,
        scenario: StressScenario,
        settings: Settings,
    ) -> str:
        """Classify the scenario environment using the same thresholds as QuantEngine."""
        shocks = scenario.shocks
        vix = _sf(shocks.get("vix_implied", settings.vix_level_soft))
        avg_corr = _sf(shocks.get("avg_corr_implied", settings.calm_avg_corr_max))

        if vix >= settings.vix_level_hard and avg_corr >= settings.crisis_avg_corr_min:
            return "CRISIS"
        if vix >= settings.vix_level_hard or avg_corr >= settings.crisis_avg_corr_min:
            return "CRISIS"
        if vix >= settings.vix_level_soft and avg_corr >= settings.tension_avg_corr_min:
            return "TENSION"
        if vix >= settings.vix_level_soft or avg_corr >= settings.tension_avg_corr_min:
            return "TENSION"
        if avg_corr >= settings.calm_avg_corr_max:
            return "NORMAL"
        return "CALM"

    @staticmethod
    def _worst_best(sector_pnl: Dict[str, float]) -> Tuple[str, str]:
        """Return (worst_sector, best_sector) ticker by P&L."""
        if not sector_pnl:
            return "N/A", "N/A"
        worst = min(sector_pnl, key=lambda k: sector_pnl[k])
        best  = max(sector_pnl, key=lambda k: sector_pnl[k])
        return worst, best

    @staticmethod
    def _build_notes(
        *,
        scenario: StressScenario,
        portfolio_pnl: float,
        pnl_spy_total: float,
        pnl_tnx_total: float,
        pnl_dxy_total: float,
        pnl_idio_total: float,
        worst_sector: str,
        best_sector: str,
        reliability: float,
        regime_label: str,
    ) -> str:
        """Generate a concise PM-facing narrative for the stress result."""
        direction = "gain" if portfolio_pnl >= 0 else "loss"
        dom_channels: List[str] = []
        for label, val in [
            ("SPY", pnl_spy_total),
            ("TNX", pnl_tnx_total),
            ("DXY", pnl_dxy_total),
            ("idio", pnl_idio_total),
        ]:
            if abs(val) >= 0.001:
                dom_channels.append(f"{label} {val*100:+.2f}%")

        channels_str = ", ".join(dom_channels) if dom_channels else "negligible factor exposure"

        if reliability >= 0.60:
            sig_note = "Signals remain moderately reliable."
        elif reliability >= 0.35:
            sig_note = "Signal reliability degraded; reduce position sizing."
        else:
            sig_note = "Severe signal degradation — model output unreliable under this scenario."

        return (
            f"[{regime_label}] Estimated portfolio {direction} {portfolio_pnl*100:+.2f}%. "
            f"Key channels: {channels_str}. "
            f"Worst: {worst_sector}, Best: {best_sector}. "
            f"Reliability {reliability:.0%}. {sig_note}"
        )


# ==========================================================================
# Monte Carlo Stress Engine
# ==========================================================================

@dataclass
class MCStressResult:
    """
    Output of Monte Carlo portfolio stress simulation.

    P&L distribution is generated by sampling correlated factor shocks
    (SPY, TNX, DXY) from a multivariate normal calibrated on historical
    returns, then propagating through the same linear factor model used
    by the deterministic engine.
    """
    n_simulations: int
    var_95: float               # 5th percentile P&L (95% VaR)
    var_99: float               # 1st percentile P&L (99% VaR)
    cvar_95: float              # Expected Shortfall at 95%
    cvar_99: float              # Expected Shortfall at 99%
    mean_pnl: float
    median_pnl: float
    std_pnl: float
    skewness: float
    kurtosis: float
    max_loss: float
    max_gain: float
    pnl_distribution: np.ndarray    # Full array of simulated portfolio P&Ls
    # Per-channel attribution (mean across simulations)
    mean_spy_channel: float = 0.0
    mean_tnx_channel: float = 0.0
    mean_dxy_channel: float = 0.0
    mean_idio_channel: float = 0.0
    # Percentile grid for plotting (e.g., [1,5,10,25,50,75,90,95,99])
    percentiles: Dict[int, float] = field(default_factory=dict)


class MonteCarloStressEngine:
    """
    Monte Carlo stress engine: simulates N correlated factor paths and
    computes portfolio P&L distribution using the same linear factor
    decomposition as StressEngine.

    Calibration
    -----------
    Factor returns (SPY daily, TNX daily change, DXY daily %) are estimated
    from historical prices. The covariance matrix is scaled to a 1-month
    horizon (21 trading days) via √T scaling.

    Idiosyncratic sector shocks are sampled independently from each sector's
    residual volatility after removing factor exposure.

    Usage
    -----
        mc = MonteCarloStressEngine()
        result = mc.run(master_df, prices, settings)
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        horizon_days: int = 21,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.seed = seed

    def run(
        self,
        master_df: pd.DataFrame,
        prices: pd.DataFrame,
        settings: Settings,
    ) -> MCStressResult:
        """
        Run Monte Carlo simulation and return P&L distribution.

        Parameters
        ----------
        master_df : DataFrame from QuantEngine (needs w_final, delta_spy_i, delta_tnx_i, delta_dxy_i)
        prices    : Historical price DataFrame (columns include SPY, ^TNX, sector ETFs)
        settings  : Settings instance
        """
        rng = np.random.default_rng(self.seed)
        horizon = self.horizon_days

        # ── Step 1: Extract portfolio exposures from master_df ────────────
        w_fin   = _col(master_df, "w_final").values
        d_spy   = _col(master_df, "delta_spy_i").values
        d_tnx   = _col(master_df, "delta_tnx_i").values
        d_dxy   = _col(master_df, "delta_dxy_i").values
        tickers = master_df["sector_ticker"].astype(str).values if "sector_ticker" in master_df.columns else []
        n_sectors = len(w_fin)

        # ── Step 2: Calibrate factor distribution from historical prices ──
        factor_mu, factor_cov, sector_idio_vol = self._calibrate(
            prices, tickers, settings, horizon,
        )

        # ── Step 3: Sample factor shocks (N × 3) ─────────────────────────
        # factor_shocks[:, 0] = SPY return, [:, 1] = TNX change, [:, 2] = DXY % change
        factor_shocks = rng.multivariate_normal(
            factor_mu, factor_cov, size=self.n_simulations,
        )  # shape (N, 3)

        # ── Step 4: Sample idiosyncratic sector shocks (N × n_sectors) ────
        idio_shocks = rng.normal(0.0, sector_idio_vol, size=(self.n_simulations, n_sectors))

        # ── Step 5: Compute P&L per simulation ───────────────────────────
        # pnl_spy[i] = sum_j(d_spy_j * factor_shocks[i, 0])
        pnl_spy = factor_shocks[:, 0:1] * d_spy[np.newaxis, :]    # (N, n_sectors)
        pnl_tnx = factor_shocks[:, 1:2] * d_tnx[np.newaxis, :]
        pnl_dxy = factor_shocks[:, 2:3] * d_dxy[np.newaxis, :]
        pnl_idio = idio_shocks * w_fin[np.newaxis, :]

        sector_pnl = pnl_spy + pnl_tnx + pnl_dxy + pnl_idio       # (N, n_sectors)
        portfolio_pnl = sector_pnl.sum(axis=1)                      # (N,)

        # ── Step 6: Compute statistics ────────────────────────────────────
        sorted_pnl = np.sort(portfolio_pnl)
        n = self.n_simulations

        var_95 = float(np.percentile(portfolio_pnl, 5))
        var_99 = float(np.percentile(portfolio_pnl, 1))
        cvar_95 = float(sorted_pnl[:max(1, int(n * 0.05))].mean())
        cvar_99 = float(sorted_pnl[:max(1, int(n * 0.01))].mean())

        mean_pnl = float(portfolio_pnl.mean())
        std_pnl = float(portfolio_pnl.std())

        # Skewness and kurtosis (excess)
        if std_pnl > 1e-12:
            skewness = float(((portfolio_pnl - mean_pnl) ** 3).mean() / std_pnl ** 3)
            kurtosis = float(((portfolio_pnl - mean_pnl) ** 4).mean() / std_pnl ** 4 - 3.0)
        else:
            skewness, kurtosis = 0.0, 0.0

        # Percentile grid
        pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {p: round(float(np.percentile(portfolio_pnl, p)), 6) for p in pct_keys}

        logger.info(
            "MC Stress: %d sims | VaR95=%.2f%% | CVaR95=%.2f%% | mean=%.2f%% | skew=%.2f | kurt=%.2f",
            n, var_95 * 100, cvar_95 * 100, mean_pnl * 100, skewness, kurtosis,
        )

        return MCStressResult(
            n_simulations=n,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            mean_pnl=mean_pnl,
            median_pnl=float(np.median(portfolio_pnl)),
            std_pnl=std_pnl,
            skewness=skewness,
            kurtosis=kurtosis,
            max_loss=float(portfolio_pnl.min()),
            max_gain=float(portfolio_pnl.max()),
            pnl_distribution=portfolio_pnl,
            mean_spy_channel=float(pnl_spy.sum(axis=1).mean()),
            mean_tnx_channel=float(pnl_tnx.sum(axis=1).mean()),
            mean_dxy_channel=float(pnl_dxy.sum(axis=1).mean()),
            mean_idio_channel=float(pnl_idio.sum(axis=1).mean()),
            percentiles=percentiles,
        )

    def _calibrate(
        self,
        prices: pd.DataFrame,
        tickers: np.ndarray,
        settings: Settings,
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calibrate factor means, covariance, and per-sector idiosyncratic vol.

        Returns
        -------
        factor_mu  : (3,) — expected 1-month factor returns [SPY, TNX, DXY]
        factor_cov : (3, 3) — covariance matrix scaled to horizon
        sector_idio_vol : (n_sectors,) — residual vol per sector (1-month)
        """
        # Identify factor columns
        spy_col = settings.spy_ticker if settings.spy_ticker in prices.columns else "SPY"
        tnx_col = next((c for c in prices.columns if "TNX" in c.upper()), None)
        dxy_col = next((c for c in prices.columns if "DX" in c.upper() and "Y" in c.upper()), None)

        lookback = min(504, len(prices))  # 2 years max
        p = prices.iloc[-lookback:].copy()

        # Factor daily returns
        spy_ret = np.log(p[spy_col] / p[spy_col].shift(1)).dropna().values if spy_col in p.columns else np.zeros(lookback - 1)
        tnx_ret = p[tnx_col].diff().dropna().values / 100.0 if tnx_col and tnx_col in p.columns else np.zeros(lookback - 1)
        dxy_ret = (p[dxy_col].pct_change()).dropna().values if dxy_col and dxy_col in p.columns else np.zeros(lookback - 1)

        # Align lengths
        min_len = min(len(spy_ret), len(tnx_ret), len(dxy_ret))
        factor_daily = np.column_stack([
            spy_ret[-min_len:],
            tnx_ret[-min_len:],
            dxy_ret[-min_len:],
        ])

        # Daily stats → scale to horizon
        sqrt_h = math.sqrt(horizon)
        daily_mu = factor_daily.mean(axis=0)
        daily_cov = np.cov(factor_daily.T)
        if daily_cov.ndim == 0:
            daily_cov = np.array([[float(daily_cov)]])

        factor_mu = daily_mu * horizon
        factor_cov = daily_cov * horizon

        # Ensure PSD
        ev = np.linalg.eigvalsh(factor_cov)
        if ev.min() < 0:
            factor_cov += (-ev.min() + 1e-10) * np.eye(factor_cov.shape[0])

        # Per-sector idiosyncratic vol (residual after factor regression)
        n_sectors = len(tickers)
        sector_idio_vol = np.full(n_sectors, 0.03 * sqrt_h)  # default 3%/month

        for i, t in enumerate(tickers):
            if str(t) in p.columns:
                sec_ret = np.log(p[str(t)] / p[str(t)].shift(1)).dropna().values
                sec_len = min(len(sec_ret), min_len)
                if sec_len > 30:
                    # Regress sector on factors → residual std
                    X = factor_daily[-sec_len:]
                    y = sec_ret[-sec_len:]
                    try:
                        beta, _, _, _ = np.linalg.lstsq(
                            np.column_stack([X, np.ones(sec_len)]), y, rcond=None,
                        )
                        resid = y - (X @ beta[:3] + beta[3])
                        sector_idio_vol[i] = float(resid.std()) * sqrt_h
                    except Exception:
                        sector_idio_vol[i] = float(sec_ret[-252:].std()) * sqrt_h

        # Floor idiosyncratic vol
        sector_idio_vol = np.maximum(sector_idio_vol, 0.005 * sqrt_h)

        return factor_mu, factor_cov, sector_idio_vol
