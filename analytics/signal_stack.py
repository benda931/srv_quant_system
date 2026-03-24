"""
analytics/signal_stack.py
==========================
Multi-Layer Signal Stack for Short Volatility via Correlation/Dispersion Trades

Implements the 4-layer scoring system from the research brief:

  Layer 1: Distortion Score (S^dist)
    - Frobenius distortion z-score
    - Market-mode share rank
    - CoC instability z-score
    - Combined via logistic: S^dist = σ(a1·z_D + a2·rank(m_t) + a3·z_CoC)

  Layer 2: Residual Dislocation Score (S^disloc)
    - Per candidate trade residual z-score
    - Capped and normalized: S^disloc = min(1, |z| / Z_cap)

  Layer 3: Mean-Reversion Score (S^mr) — see signal_mean_reversion.py
  Layer 4: Regime Safety Score (S^safe) — see signal_regime_safety.py

  Combined conviction: Score_j = S^dist × S^disloc × S^mr × S^safe

Entry/exit logic:
  - Enter if Score_j >= Θ_enter AND layer gates pass
  - Exit if residual compresses OR regime safety collapses OR hedge unstable

Ref: Cboe Implied Correlation Index methodology
Ref: Random Matrix / PCA (Laloux et al.)
Ref: Dispersion P&L decomposition (Jacquier & Slaoui)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.signal_mean_reversion import (
    MeanReversionResult,
    batch_mean_reversion_scores,
    compute_mean_reversion_score,
)
from analytics.signal_regime_safety import (
    RegimeSafetyResult,
    compute_regime_safety_score,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DistortionScoreResult:
    """Layer 1 output: correlation structure distortion scoring."""
    # Raw inputs
    frob_distortion_z: float       # z_D — Frobenius distortion z-score
    market_mode_rank: float        # rank(m_t) ∈ [0, 1] — percentile of market-mode share
    coc_instability_z: float       # z_CoC — correlation-of-correlations instability z-score

    # Logistic combination
    logit_input: float             # a1·z_D + a2·rank(m_t) + a3·z_CoC
    distortion_score: float        # S^dist ∈ [0, 1] — final distortion score

    # Interpretation
    label: str                     # "HIGH_DISTORTION" / "MODERATE" / "LOW"
    rationale: str


@dataclass
class DislocationScoreResult:
    """Layer 2 output: per-candidate residual dislocation scoring."""
    ticker: str                    # Trade candidate identifier
    trade_type: str                # "rv_spread" / "pca_residual" / "dispersion" / "sector"

    # Residual process
    residual_value: float          # x_t^(j) — current residual level
    residual_z: float              # z_t^(j) — z-score of residual
    residual_z_abs: float          # |z_t^(j)|
    z_lookback: int                # k — lookback window used for z-score

    # Dislocation score
    dislocation_score: float       # S^disloc = min(1, |z| / Z_cap)
    z_cap: float                   # Z_cap used

    # Direction
    direction: str                 # "LONG" (z < 0) / "SHORT" (z > 0)

    # Metadata
    spread_name: Optional[str] = None  # e.g., "XLK-XLU" for RV spread


@dataclass
class SignalStackResult:
    """Combined output: all layers for one trade candidate."""
    ticker: str
    trade_type: str

    # Layer scores
    distortion_score: float        # S^dist ∈ [0, 1]
    dislocation_score: float       # S^disloc ∈ [0, 1]
    mean_reversion_score: float    # S^mr ∈ [0, 1] (from Layer 3, default 1.0)
    regime_safety_score: float     # S^safe ∈ [0, 1] (from Layer 4, default 1.0)

    # Combined
    conviction_score: float        # S^dist × S^disloc × S^mr × S^safe
    entry_threshold: float         # Θ_enter
    passes_entry: bool             # conviction >= threshold

    # Direction & sizing hint
    direction: str
    residual_z: float
    size_multiplier: float         # 0-1, decreases with regime stress

    # Detail objects
    distortion_detail: Optional[DistortionScoreResult] = None
    dislocation_detail: Optional[DislocationScoreResult] = None
    mean_reversion_detail: Optional[MeanReversionResult] = None
    regime_safety_detail: Optional[RegimeSafetyResult] = None

    # Layer gate status
    gates: Dict[str, bool] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Distortion Score
# ─────────────────────────────────────────────────────────────────────────────

def _logistic(x: float) -> float:
    """Logistic sigmoid σ(x) = 1 / (1 + exp(-x))."""
    if x > 20:
        return 1.0
    if x < -20:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_distortion_score(
    frob_distortion_z: float,
    market_mode_share: float,
    coc_instability_z: float,
    *,
    # Logistic coefficients — CALIBRATED via grid search
    a1_frob: float = 0.3,
    a2_mode: float = 0.2,
    a3_coc: float = 0.3,
    # Cross-asset features (new — Wave 5)
    credit_z: float = 0.0,       # HYG/IEF spread z-score (negative = stress)
    vix_term_slope: float = 0.0, # VIX term structure slope (negative = backwardation)
    a4_credit: float = 0.15,     # Credit spread coefficient
    a5_term: float = 0.10,       # Term structure coefficient
    # Market-mode ranking: percentile within historical distribution
    market_mode_history: Optional[pd.Series] = None,
) -> DistortionScoreResult:
    """
    Layer 1: Distortion Score.

    S^dist = σ(a1·z_D + a2·rank(m_t) + a3·z_CoC)

    Higher score = more distortion detected = potentially more opportunity
    (but must be gated by regime safety in Layer 4).

    Parameters
    ----------
    frob_distortion_z   : z-score of Frobenius distortion D_t
    market_mode_share   : m_t = λ1/N (current)
    coc_instability_z   : z-score of CoC instability
    a1_frob, a2_mode, a3_coc : logistic coefficients
    market_mode_history : historical m_t series for ranking

    Returns
    -------
    DistortionScoreResult
    """
    # Default fallbacks for NaN inputs
    z_D = frob_distortion_z if math.isfinite(frob_distortion_z) else 0.0
    z_CoC = coc_instability_z if math.isfinite(coc_instability_z) else 0.0
    z_credit = credit_z if math.isfinite(credit_z) else 0.0
    z_term = vix_term_slope if math.isfinite(vix_term_slope) else 0.0

    # Market-mode rank: percentile of current m_t in historical distribution
    if market_mode_history is not None and len(market_mode_history.dropna()) >= 20:
        hist = market_mode_history.dropna().values
        rank_pct = float(np.mean(hist <= market_mode_share))
    else:
        # Heuristic: map m_t to [0,1] assuming typical range [0.1, 0.7]
        rank_pct = max(0.0, min(1.0, (market_mode_share - 0.1) / 0.6))

    # Logistic combination (5-feature: frob + mode + coc + credit + term structure)
    logit = (a1_frob * z_D + a2_mode * rank_pct + a3_coc * z_CoC
             + a4_credit * abs(z_credit) + a5_term * max(0, -z_term))  # Backwardation = opportunity
    score = _logistic(logit)

    # Interpretation
    if score >= 0.7:
        label = "HIGH_DISTORTION"
        rationale = (
            f"Correlation structure highly distorted: "
            f"Frob z={z_D:.2f}, mode rank={rank_pct:.0%}, CoC z={z_CoC:.2f}"
        )
    elif score >= 0.4:
        label = "MODERATE_DISTORTION"
        rationale = (
            f"Moderate distortion detected: "
            f"Frob z={z_D:.2f}, mode rank={rank_pct:.0%}"
        )
    else:
        label = "LOW_DISTORTION"
        rationale = "Correlation structure near baseline — limited opportunity"

    return DistortionScoreResult(
        frob_distortion_z=round(z_D, 4),
        market_mode_rank=round(rank_pct, 4),
        coc_instability_z=round(z_CoC, 4),
        logit_input=round(logit, 4),
        distortion_score=round(score, 4),
        label=label,
        rationale=rationale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Residual Dislocation Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_dislocation_score(
    residual_series: pd.Series,
    ticker: str,
    trade_type: str = "sector",
    *,
    z_lookback: int = 60,
    z_cap: float = 3.0,
    spread_name: Optional[str] = None,
) -> DislocationScoreResult:
    """
    Layer 2: Residual Dislocation Score for a single trade candidate.

    S^disloc_j = min(1, |z_t^(j)| / Z_cap)

    Higher score = residual further from mean = stronger trade opportunity.

    Parameters
    ----------
    residual_series : Time series of residual process x_t^(j)
    ticker          : Identifier for this candidate
    trade_type      : "sector" / "rv_spread" / "pca_residual" / "dispersion"
    z_lookback      : Window for z-score calculation (k)
    z_cap           : Maximum z-score for normalization (Z_cap)
    spread_name     : Optional label for RV spread

    Returns
    -------
    DislocationScoreResult
    """
    x = residual_series.dropna()
    if len(x) < max(20, z_lookback):
        return DislocationScoreResult(
            ticker=ticker,
            trade_type=trade_type,
            residual_value=float("nan"),
            residual_z=float("nan"),
            residual_z_abs=0.0,
            z_lookback=z_lookback,
            dislocation_score=0.0,
            z_cap=z_cap,
            direction="NEUTRAL",
            spread_name=spread_name,
        )

    # Current value
    x_t = float(x.iloc[-1])

    # Rolling z-score
    lookback = x.iloc[-z_lookback:]
    mu = float(lookback.mean())
    sigma = float(lookback.std(ddof=1))

    if sigma < 1e-10:
        z_t = 0.0
    else:
        z_t = (x_t - mu) / sigma

    z_abs = abs(z_t)

    # Dislocation score: logarithmic mapping (preserves extreme signal info)
    # S = log(1 + |z|/z_cap) / log(1 + 3.0/z_cap) — saturates smoothly, doesn't clip
    import math as _math
    if z_cap > 0 and z_abs > 0:
        score = min(1.0, _math.log(1 + z_abs / z_cap) / _math.log(1 + 3.0 / z_cap))
    else:
        score = 0.0

    # Direction: negative z = below mean = LONG; positive z = above mean = SHORT
    if z_abs < 0.3:
        direction = "NEUTRAL"
    elif z_t < 0:
        direction = "LONG"
    else:
        direction = "SHORT"

    return DislocationScoreResult(
        ticker=ticker,
        trade_type=trade_type,
        residual_value=round(x_t, 6),
        residual_z=round(z_t, 4),
        residual_z_abs=round(z_abs, 4),
        z_lookback=z_lookback,
        dislocation_score=round(score, 4),
        z_cap=z_cap,
        direction=direction,
        spread_name=spread_name,
    )


def compute_dislocation_from_zscore(
    ticker: str,
    z_score: float,
    residual_value: float = float("nan"),
    trade_type: str = "sector",
    z_lookback: int = 60,
    z_cap: float = 3.0,
    spread_name: Optional[str] = None,
) -> DislocationScoreResult:
    """
    Layer 2 shortcut: compute dislocation from a pre-computed z-score.
    Used when QuantEngine already provides residual z-scores.
    """
    z = z_score if math.isfinite(z_score) else 0.0
    z_abs = abs(z)
    score = min(1.0, z_abs / z_cap) if z_cap > 0 else 0.0

    if z_abs < 0.3:
        direction = "NEUTRAL"
    elif z < 0:
        direction = "LONG"
    else:
        direction = "SHORT"

    return DislocationScoreResult(
        ticker=ticker,
        trade_type=trade_type,
        residual_value=round(residual_value, 6) if math.isfinite(residual_value) else float("nan"),
        residual_z=round(z, 4),
        residual_z_abs=round(z_abs, 4),
        z_lookback=z_lookback,
        dislocation_score=round(score, 4),
        z_cap=z_cap,
        direction=direction,
        spread_name=spread_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RV Spread residual generation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rv_spread_residual(
    prices_a: pd.Series,
    prices_b: pd.Series,
    hedge_ratio: Optional[float] = None,
    hedge_window: int = 60,
) -> Tuple[pd.Series, float]:
    """
    Compute relative-value spread residual:
      x_t = ln(P_A) - h · ln(P_B)

    If hedge_ratio not provided, estimated via rolling OLS.

    Parameters
    ----------
    prices_a     : Price series for leg A
    prices_b     : Price series for leg B
    hedge_ratio  : Fixed h, or None for rolling OLS
    hedge_window : Window for hedge ratio estimation

    Returns
    -------
    spread : pd.Series — residual x_t
    h      : float — hedge ratio used
    """
    log_a = np.log(prices_a.dropna())
    log_b = np.log(prices_b.dropna())

    # Align
    common = log_a.index.intersection(log_b.index)
    log_a = log_a.loc[common]
    log_b = log_b.loc[common]

    if hedge_ratio is None:
        # Estimate beta-neutral hedge ratio from recent window
        ret_a = log_a.diff().dropna().tail(hedge_window)
        ret_b = log_b.diff().dropna().tail(hedge_window)
        common_rets = ret_a.index.intersection(ret_b.index)
        ret_a = ret_a.loc[common_rets]
        ret_b = ret_b.loc[common_rets]

        if len(ret_a) >= 20:
            cov_ab = float(ret_a.cov(ret_b))
            var_b = float(ret_b.var())
            hedge_ratio = cov_ab / var_b if var_b > 1e-12 else 1.0
        else:
            hedge_ratio = 1.0

    spread = log_a - hedge_ratio * log_b
    return spread, hedge_ratio


def generate_all_rv_spreads(
    prices: pd.DataFrame,
    tickers: List[str],
    hedge_window: int = 60,
) -> Dict[str, Tuple[pd.Series, float, str, str]]:
    """
    Generate all pairwise RV spread residuals for a universe.

    Returns
    -------
    dict : {spread_name: (spread_series, hedge_ratio, ticker_a, ticker_b)}
    """
    spreads = {}
    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tickers[i], tickers[j]
            if a not in prices.columns or b not in prices.columns:
                continue
            name = f"{a}-{b}"
            try:
                spread, h = compute_rv_spread_residual(
                    prices[a], prices[b], hedge_window=hedge_window
                )
                if len(spread.dropna()) >= 60:
                    spreads[name] = (spread, h, a, b)
            except Exception as e:
                log.debug("RV spread %s failed: %s", name, e)
    return spreads


# ─────────────────────────────────────────────────────────────────────────────
# Signal Stack Combiner
# ─────────────────────────────────────────────────────────────────────────────

class SignalStackEngine:
    """
    Combines all 4 layers into unified conviction scores per trade candidate.

    Usage:
        engine = SignalStackEngine(settings)
        results = engine.score_candidates(
            corr_snapshot=...,  # from CorrelationStructureEngine
            residuals=...,      # dict of residual series or pre-computed z-scores
            mean_reversion_scores=...,  # from Layer 3 (optional)
            regime_safety_score=...,    # from Layer 4 (optional)
        )
    """

    def __init__(self, settings=None):
        self.settings = settings

        # Layer 1 coefficients (configurable)
        self.a1_frob = getattr(settings, "signal_a1_frob", 1.0) if settings else 1.0
        self.a2_mode = getattr(settings, "signal_a2_mode", 0.5) if settings else 0.5
        self.a3_coc = getattr(settings, "signal_a3_coc", 0.3) if settings else 0.3

        # Layer 2 parameters
        self.z_lookback = getattr(settings, "signal_z_lookback", 60) if settings else 60
        self.z_cap = getattr(settings, "signal_z_cap", 3.0) if settings else 3.0

        # Entry threshold
        self.entry_threshold = getattr(settings, "signal_entry_threshold", 0.15) if settings else 0.15

        # Sector MR whitelist — only trade sectors with proven mean reversion
        self.mr_filter_enabled = getattr(settings, "sector_mr_filter_enabled", True) if settings else True
        mr_str = getattr(settings, "sector_mr_whitelist", "XLC,XLF,XLI,XLU") if settings else "XLC,XLF,XLI,XLU"
        self.mr_whitelist = set(t.strip() for t in mr_str.split(",") if t.strip())

        # Regime-adaptive sizing & z-thresholds
        self.regime_size = {
            "CALM": getattr(settings, "regime_size_calm", 1.3) if settings else 1.3,
            "NORMAL": getattr(settings, "regime_size_normal", 1.0) if settings else 1.0,
            "TENSION": getattr(settings, "regime_size_tension", 0.6) if settings else 0.6,
            "CRISIS": getattr(settings, "regime_size_crisis", 0.0) if settings else 0.0,
        }
        self.regime_z_thresh = {
            "CALM": getattr(settings, "regime_z_calm", 0.6) if settings else 0.6,
            "NORMAL": getattr(settings, "regime_z_normal", 0.8) if settings else 0.8,
            "TENSION": getattr(settings, "regime_z_tension", 1.0) if settings else 1.0,
            "CRISIS": getattr(settings, "regime_z_crisis", 99.0) if settings else 99.0,
        }

    def get_regime_multiplier(self, regime_label: str = "NORMAL") -> float:
        """Get position size multiplier for current regime."""
        return self.regime_size.get(regime_label.upper(), 1.0)

    def get_regime_z_threshold(self, regime_label: str = "NORMAL") -> float:
        """Get minimum z-score threshold for current regime."""
        return self.regime_z_thresh.get(regime_label.upper(), 0.8)

    def score_sector_candidates(
        self,
        frob_distortion_z: float,
        market_mode_share: float,
        coc_instability_z: float,
        sector_residuals: Dict[str, pd.Series],
        *,
        market_mode_history: Optional[pd.Series] = None,
        mean_reversion_scores: Optional[Dict[str, float]] = None,
        regime_safety_score: float = 1.0,
        regime_safety_result: Optional[RegimeSafetyResult] = None,
    ) -> List[SignalStackResult]:
        """
        Score all sector-level trade candidates with full 4-layer stack.

        Parameters
        ----------
        frob_distortion_z    : From measurement engine
        market_mode_share    : From measurement engine
        coc_instability_z    : From measurement engine
        sector_residuals     : {ticker: residual_series} — PCA residuals per sector
        market_mode_history  : Historical m_t for percentile ranking
        mean_reversion_scores: {ticker: S^mr} — override from externally computed Layer 3
        regime_safety_score  : S^safe scalar override
        regime_safety_result : Full Layer 4 result (takes precedence over scalar)

        Returns
        -------
        List[SignalStackResult] sorted by conviction descending
        """
        # Layer 1: Distortion (same for all candidates)
        dist_result = compute_distortion_score(
            frob_distortion_z=frob_distortion_z,
            market_mode_share=market_mode_share,
            coc_instability_z=coc_instability_z,
            a1_frob=self.a1_frob,
            a2_mode=self.a2_mode,
            a3_coc=self.a3_coc,
            market_mode_history=market_mode_history,
        )

        # Layer 3: batch mean-reversion scores (unless pre-computed)
        mr_results: Dict[str, MeanReversionResult] = {}
        if mean_reversion_scores is None:
            mr_results = batch_mean_reversion_scores(sector_residuals)
        else:
            mr_results = {}

        # Layer 4: use provided result or scalar
        safe_result = regime_safety_result
        safe_score = safe_result.regime_safety_score if safe_result else regime_safety_score

        results = []
        for ticker, resid in sector_residuals.items():
            # MR whitelist filter: penalize non-MR sectors
            mr_whitelisted = (not self.mr_filter_enabled) or (ticker in self.mr_whitelist)

            # Layer 2: Dislocation
            disloc_result = compute_dislocation_score(
                residual_series=resid,
                ticker=ticker,
                trade_type="sector",
                z_lookback=self.z_lookback,
                z_cap=self.z_cap,
            )

            # Layer 3: Mean reversion
            if mean_reversion_scores is not None:
                mr_score = mean_reversion_scores.get(ticker, 1.0)
                mr_detail = None
            elif ticker in mr_results:
                mr_detail = mr_results[ticker]
                mr_score = mr_detail.mean_reversion_score
            else:
                mr_score = 1.0
                mr_detail = None

            # MR whitelist penalty: non-whitelisted sectors get severely reduced score
            if not mr_whitelisted:
                mr_score = mr_score * 0.15  # 85% penalty for non-MR sectors

            # Combined conviction: weighted additive with safety gate
            # Changed from multiplicative (one zero kills all) to additive (more robust)
            # Score = w1·S_dist + w2·S_disloc + w3·S_mr, then multiply by S_safe
            # This way: safety gates the whole score, but layers don't kill each other
            raw_conviction = (
                0.25 * dist_result.distortion_score
                + 0.45 * disloc_result.dislocation_score
                + 0.30 * mr_score
            )
            # Safety still multiplicative (regime gate must reduce/kill)
            conviction = raw_conviction * safe_score

            # Gates — including Layer 3 & 4 gates + MR whitelist
            gates = {
                "distortion_above_min": dist_result.distortion_score >= 0.15,
                "dislocation_nonzero": disloc_result.dislocation_score > 0.05,
                "mean_reversion_ok": mr_score >= 0.15,
                "regime_safe": safe_score >= 0.1,
                "no_hard_kill": not (safe_result.any_hard_kill if safe_result else False),
                "mr_whitelisted": mr_whitelisted,
            }
            all_gates_pass = all(gates.values())
            passes_entry = conviction >= self.entry_threshold and all_gates_pass

            results.append(SignalStackResult(
                ticker=ticker,
                trade_type="sector",
                distortion_score=dist_result.distortion_score,
                dislocation_score=disloc_result.dislocation_score,
                mean_reversion_score=round(mr_score, 4),
                regime_safety_score=round(safe_score, 4),
                conviction_score=round(conviction, 4),
                entry_threshold=self.entry_threshold,
                passes_entry=passes_entry,
                direction=disloc_result.direction,
                residual_z=disloc_result.residual_z,
                size_multiplier=round(
                    safe_score * min(1.0, conviction / max(1e-9, self.entry_threshold)), 4
                ),
                distortion_detail=dist_result,
                dislocation_detail=disloc_result,
                mean_reversion_detail=mr_detail,
                regime_safety_detail=safe_result,
                gates=gates,
            ))

        # Sort by conviction descending
        results.sort(key=lambda r: r.conviction_score, reverse=True)
        return results

    def score_rv_spread_candidates(
        self,
        frob_distortion_z: float,
        market_mode_share: float,
        coc_instability_z: float,
        rv_spreads: Dict[str, Tuple[pd.Series, float, str, str]],
        *,
        market_mode_history: Optional[pd.Series] = None,
        mean_reversion_scores: Optional[Dict[str, float]] = None,
        regime_safety_score: float = 1.0,
        regime_safety_result: Optional[RegimeSafetyResult] = None,
    ) -> List[SignalStackResult]:
        """
        Score all RV spread trade candidates with full 4-layer stack.

        Parameters
        ----------
        rv_spreads : {spread_name: (spread_series, hedge_ratio, ticker_a, ticker_b)}
                     from generate_all_rv_spreads()
        """
        # Layer 1: Distortion (same for all)
        dist_result = compute_distortion_score(
            frob_distortion_z=frob_distortion_z,
            market_mode_share=market_mode_share,
            coc_instability_z=coc_instability_z,
            a1_frob=self.a1_frob,
            a2_mode=self.a2_mode,
            a3_coc=self.a3_coc,
            market_mode_history=market_mode_history,
        )

        # Layer 3: batch MR scores for all spreads
        spread_series_map = {name: s for name, (s, h, a, b) in rv_spreads.items()}
        mr_results: Dict[str, MeanReversionResult] = {}
        if mean_reversion_scores is None:
            mr_results = batch_mean_reversion_scores(spread_series_map)

        # Layer 4
        safe_result = regime_safety_result
        safe_score = safe_result.regime_safety_score if safe_result else regime_safety_score

        results = []
        for spread_name, (spread, h, a, b) in rv_spreads.items():
            # Layer 2: Dislocation on spread residual
            disloc_result = compute_dislocation_score(
                residual_series=spread,
                ticker=spread_name,
                trade_type="rv_spread",
                z_lookback=self.z_lookback,
                z_cap=self.z_cap,
                spread_name=spread_name,
            )

            # Layer 3
            if mean_reversion_scores is not None:
                mr_score = mean_reversion_scores.get(spread_name, 1.0)
                mr_detail = None
            elif spread_name in mr_results:
                mr_detail = mr_results[spread_name]
                mr_score = mr_detail.mean_reversion_score
            else:
                mr_score = 1.0
                mr_detail = None

            conviction = (
                dist_result.distortion_score
                * disloc_result.dislocation_score
                * mr_score
                * safe_score
            )

            gates = {
                "distortion_above_min": dist_result.distortion_score >= 0.2,
                "dislocation_nonzero": disloc_result.dislocation_score > 0.05,
                "mean_reversion_ok": mr_score >= 0.15,
                "regime_safe": safe_score >= 0.1,
                "no_hard_kill": not (safe_result.any_hard_kill if safe_result else False),
            }
            all_gates_pass = all(gates.values())
            passes_entry = conviction >= self.entry_threshold and all_gates_pass

            results.append(SignalStackResult(
                ticker=spread_name,
                trade_type="rv_spread",
                distortion_score=dist_result.distortion_score,
                dislocation_score=disloc_result.dislocation_score,
                mean_reversion_score=round(mr_score, 4),
                regime_safety_score=round(safe_score, 4),
                conviction_score=round(conviction, 4),
                entry_threshold=self.entry_threshold,
                passes_entry=passes_entry,
                direction=disloc_result.direction,
                residual_z=disloc_result.residual_z,
                size_multiplier=round(
                    safe_score * min(1.0, conviction / max(1e-9, self.entry_threshold)), 4
                ),
                distortion_detail=dist_result,
                dislocation_detail=disloc_result,
                mean_reversion_detail=mr_detail,
                regime_safety_detail=safe_result,
                gates=gates,
            ))

        results.sort(key=lambda r: r.conviction_score, reverse=True)
        return results

    def score_from_master_df(
        self,
        frob_distortion_z: float,
        market_mode_share: float,
        coc_instability_z: float,
        master_df: pd.DataFrame,
        *,
        market_mode_history: Optional[pd.Series] = None,
        mean_reversion_scores: Optional[Dict[str, float]] = None,
        regime_safety_score: float = 1.0,
        regime_safety_result: Optional[RegimeSafetyResult] = None,
    ) -> List[SignalStackResult]:
        """
        Score sector candidates using pre-computed z-scores from QuantEngine's master_df.
        This is the primary integration point with the existing pipeline.

        Expects master_df to have columns:
        - sector_ticker
        - pca_residual_z
        - pca_residual_level (optional)
        - half_life_days_est (optional — used for Layer 3 approximation)
        """
        dist_result = compute_distortion_score(
            frob_distortion_z=frob_distortion_z,
            market_mode_share=market_mode_share,
            coc_instability_z=coc_instability_z,
            a1_frob=self.a1_frob,
            a2_mode=self.a2_mode,
            a3_coc=self.a3_coc,
            market_mode_history=market_mode_history,
        )

        # Layer 4
        safe_result = regime_safety_result
        safe_score = safe_result.regime_safety_score if safe_result else regime_safety_score

        results = []
        for _, row in master_df.iterrows():
            ticker = str(row.get("sector_ticker", ""))
            if not ticker:
                continue

            # MR whitelist filter
            mr_whitelisted = (not self.mr_filter_enabled) or (ticker in self.mr_whitelist)

            z = float(row.get("pca_residual_z", 0.0))
            resid_val = float(row.get("pca_residual_level", float("nan")))

            disloc_result = compute_dislocation_from_zscore(
                ticker=ticker,
                z_score=z,
                residual_value=resid_val,
                trade_type="sector",
                z_lookback=self.z_lookback,
                z_cap=self.z_cap,
            )

            # Layer 3: use pre-computed scores or half-life approximation
            if mean_reversion_scores is not None:
                mr_score = mean_reversion_scores.get(ticker, 1.0)
            else:
                # Approximate from half_life if available in master_df
                hl = float(row.get("half_life_days_est", float("nan")))
                if math.isfinite(hl) and 5 <= hl <= 90:
                    mr_score = 0.55 + 0.45 * math.exp(-abs(hl - 35) / 45)
                elif math.isfinite(hl):
                    mr_score = max(0.15, 0.50 * math.exp(-0.01 * abs(hl - 45)))
                else:
                    mr_score = 0.5  # Unknown → neutral

            # MR whitelist penalty: non-whitelisted sectors get severely reduced score
            if not mr_whitelisted:
                mr_score = mr_score * 0.15

            conviction = (
                dist_result.distortion_score
                * disloc_result.dislocation_score
                * mr_score
                * safe_score
            )

            gates = {
                "distortion_above_min": dist_result.distortion_score >= 0.2,
                "dislocation_nonzero": disloc_result.dislocation_score > 0.05,
                "mean_reversion_ok": mr_score >= 0.15,
                "regime_safe": safe_score >= 0.1,
                "no_hard_kill": not (safe_result.any_hard_kill if safe_result else False),
                "mr_whitelisted": mr_whitelisted,
            }
            all_gates_pass = all(gates.values())
            passes_entry = conviction >= self.entry_threshold and all_gates_pass

            results.append(SignalStackResult(
                ticker=ticker,
                trade_type="sector",
                distortion_score=dist_result.distortion_score,
                dislocation_score=disloc_result.dislocation_score,
                mean_reversion_score=round(mr_score, 4),
                regime_safety_score=round(safe_score, 4),
                conviction_score=round(conviction, 4),
                entry_threshold=self.entry_threshold,
                passes_entry=passes_entry,
                direction=disloc_result.direction,
                residual_z=disloc_result.residual_z,
                size_multiplier=round(
                    safe_score * min(1.0, conviction / max(1e-9, self.entry_threshold)), 4
                ),
                distortion_detail=dist_result,
                dislocation_detail=disloc_result,
                regime_safety_detail=safe_result,
                gates=gates,
            ))

        results.sort(key=lambda r: r.conviction_score, reverse=True)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compact summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────

def signal_stack_summary(results: List[SignalStackResult]) -> dict:
    """Compact serializable summary for Claude / AgentBus."""
    if not results:
        return {"candidates": [], "n_passing": 0}

    passing = [r for r in results if r.passes_entry]
    dist_score = results[0].distortion_score if results else 0.0

    return {
        "distortion_score": dist_score,
        "distortion_label": results[0].distortion_detail.label if results and results[0].distortion_detail else "N/A",
        "n_candidates": len(results),
        "n_passing_entry": len(passing),
        "top_candidates": [
            {
                "ticker": r.ticker,
                "type": r.trade_type,
                "direction": r.direction,
                "conviction": r.conviction_score,
                "z": r.residual_z,
                "disloc": r.dislocation_score,
                "mr": r.mean_reversion_score,
                "safe": r.regime_safety_score,
                "entry": r.passes_entry,
            }
            for r in results[:10]
        ],
    }
