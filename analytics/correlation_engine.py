"""
analytics/correlation_engine.py
---------------------------------
Correlation Structure & Volatility Pricing Engine

מטרה:
  1. מחשב correlation implicite מתוך פיזור הסקטורים + variance decomposition
  2. מתמחר את הקורלציה (Fair Value vs Realized)
  3. מייצר Signal מובנה לshort volatility דרך dispersion trade
  4. מזהה עיוותים ספציפיים בין זוגות סקטורים
  5. מחשב time series של avg corr, implied corr, CRP לאורך זמן

==============================================================================
המתמטיקה (SPY real sector weights wᵢ):
==============================================================================

Variance decomposition:
  var_port = var_spy (proxy)
  indep_var     = Σwᵢ² × σᵢ²                (zero-correlation baseline)
  cross_var_term = (Σwᵢσᵢ)² − indep_var      (weight-correct denominator)

  var_spy = indep_var + ρ_implied × cross_var_term

  => ρ_implied = (var_spy − indep_var) / cross_var_term

When ρ_implied > 0: sectors moving together, index variance > idiosyncratic
When ρ_implied < 0: sectors diverging, dispersion regime

Correlation Risk Premium (CRP):
  CRP = ρ_implied - ρ_fair_value
  ρ_fair_value = EWMA(ρ_implied_history, λ=0.97)

  CRP > 0 → correlation is "expensive" → dispersion trade attractive
  CRP < 0 → correlation is "cheap" → avoid selling correlation

Short Vol Signal (0-100):
  Conditions for attractive short correlation / vol:
  1. ρ_implied significantly above ρ_fair_value (CRP > threshold)
  2. avg_corr_t above long-run baseline (z > 1.5)
  3. Dispersion is LOW (sectors co-moving: dispersion_ratio_z < 0)
  4. NOT in CRISIS (crisis correlation = real systemic risk)
  5. VIX regime not extreme (avoid after-the-fact)
==============================================================================
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# N sectors (SPDR universe)
_N_SECTORS = 11
_EWMA_LAMBDA_CORR = 0.97    # for fair-value smoothing
_SHORT_VOL_LOOKBACK = 252   # for z-score of implied corr history
_MIN_HISTORY = 60


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrVolAnalysis:
    """
    Full correlation structure and volatility signal snapshot.
    All float values are NaN-safe.
    """

    # ── Current correlation matrices ─────────────────────────────────────────
    corr_current: pd.DataFrame      # C_t  (short window, e.g. 60d)
    corr_baseline: pd.DataFrame     # C_b  (long window, e.g. 252d)
    corr_delta: pd.DataFrame        # C_t - C_b  (distortion matrix)

    # ── Scalar correlation metrics ────────────────────────────────────────────
    avg_corr_current: float         # mean off-diagonal of C_t
    avg_corr_baseline: float        # mean off-diagonal of C_b
    avg_corr_delta: float           # current - baseline

    # ── Implied correlation from variance decomposition ───────────────────────
    implied_corr: float             # ρ_implied (from var_spy / indep_var)
    implied_corr_z: float           # z-score vs rolling history
    fair_value_corr: float          # EWMA(implied_corr_history)
    corr_risk_premium: float        # implied_corr - fair_value_corr
    crp_z: float                    # CRP z-score vs history

    # ── Dispersion metrics ────────────────────────────────────────────────────
    dispersion_ratio: float         # indep_var / var_spy  (>1 = dispersion, <1 = correlation)
    dispersion_ratio_z: float       # z-score vs rolling
    dispersion_index: float         # current cross-sectional std of sector returns (%)
    dispersion_history: pd.Series   # rolling dispersion_ratio series (for charts)

    # ── Time series for charts ────────────────────────────────────────────────
    implied_corr_ts: pd.Series      # implied_corr over time
    fair_value_ts: pd.Series        # EWMA fair value over time
    avg_corr_ts: pd.Series          # avg_corr_t over time
    crp_ts: pd.Series               # CRP over time

    # ── Short vol signal ─────────────────────────────────────────────────────
    short_vol_score: float          # 0-100 (100 = very attractive to short vol/corr)
    short_vol_label: str            # "STRONG SHORT" / "MILD SHORT" / "NEUTRAL" / "AVOID"
    short_vol_rationale: str        # plain-language explanation

    # ── Eigenvalue / market mode ──────────────────────────────────────────────
    market_mode_strength: float     # λ1/N  (fraction of variance in first PC)
    market_mode_loadings: Dict[str, float]
    eigenvalue_concentration: float # Herfindahl of eigenvalue distribution (0-1)

    # ── Anomalous sector pairs ────────────────────────────────────────────────
    anomalous_pairs: List[Dict]     # top pairs sorted by |delta| descending
    # Each: {"sector_a", "sector_b", "corr_current", "corr_baseline", "delta", "direction"}

    # ── Regime context ────────────────────────────────────────────────────────
    corr_regime: str                # "LOW_CORR" / "NORMAL" / "HIGH_CORR" / "CRISIS_CORR"
    market_state: str               # from master_df


# ─────────────────────────────────────────────────────────────────────────────
# Engine class
# ─────────────────────────────────────────────────────────────────────────────
class CorrVolEngine:
    """
    Correlation Volatility Engine.

    Usage:
        from analytics.correlation_engine import CorrVolEngine
        engine = CorrVolEngine()
        analysis = engine.run(quant_engine, master_df, settings)
    """

    def run(self, quant_engine, master_df: pd.DataFrame, settings) -> CorrVolAnalysis:
        """
        Compute full correlation-volatility analysis.

        Parameters
        ----------
        quant_engine : QuantEngine (must have been through load() + calculate_conviction_score())
        master_df    : pd.DataFrame from calculate_conviction_score()
        settings     : Settings instance
        """
        # ── Extract existing data from QuantEngine ────────────────────────────
        corr_m = quant_engine.corr_metrics
        disp_df = quant_engine.dispersion_df
        prices = quant_engine.prices
        sectors = settings.sector_list()
        spy = settings.spy_ticker

        if corr_m is None or disp_df is None:
            raise RuntimeError("QuantEngine must complete calculate_conviction_score() first")

        # ── Matrices ──────────────────────────────────────────────────────────
        C_t = corr_m.C_t
        C_b = corr_m.C_b
        C_delta = corr_m.C_delta

        avg_corr_current  = corr_m.avg_corr_t
        avg_corr_baseline = corr_m.avg_corr_b
        avg_corr_delta    = corr_m.avg_corr_delta

        # ── Implied Correlation from Variance Decomposition ───────────────────
        # ρ_implied = (var_spy - indep_var) / cross_var_term
        # where cross_var_term = (Σwᵢσᵢ)² − Σwᵢ²σᵢ²  (weight-correct denominator)
        disp_last = disp_df.dropna().iloc[-1] if not disp_df.dropna().empty else None
        if disp_last is not None:
            var_spy       = float(disp_last.get("var_spy", float("nan")))
            indep_var     = float(disp_last.get("indep_var", float("nan")))
            cross_var_term = float(disp_last.get("cross_var_term", float("nan")))
            disp_ratio    = float(disp_last.get("dispersion_ratio", float("nan")))
            disp_ratio_z  = float(disp_last.get("dispersion_ratio_z", float("nan")))
        else:
            var_spy = indep_var = cross_var_term = disp_ratio = disp_ratio_z = float("nan")

        implied_corr = self._calc_implied_corr(var_spy, indep_var, cross_var_term)

        # ── Build rolling implied corr + fair value time series ───────────────
        implied_corr_ts, fair_value_ts, crp_ts = self._build_rolling_ts(disp_df)

        # ── Implied corr z-score + CRP z-score ───────────────────────────────
        implied_corr_z = self._zscore_last(implied_corr_ts, window=_SHORT_VOL_LOOKBACK)
        crp_last = float(crp_ts.iloc[-1]) if not crp_ts.empty and not crp_ts.isna().all() else float("nan")
        crp_z = self._zscore_last(crp_ts, window=_SHORT_VOL_LOOKBACK)
        fair_value_last = float(fair_value_ts.iloc[-1]) if not fair_value_ts.empty else float("nan")

        # ── Average corr time series (from existing QuantEngine method) ───────
        try:
            corr_ts_df = quant_engine.get_correlation_regime_timeseries()
            avg_corr_ts = corr_ts_df["avg_corr_t"].dropna() if "avg_corr_t" in corr_ts_df.columns else pd.Series(dtype=float)
        except Exception:
            avg_corr_ts = pd.Series(dtype=float)

        # ── Dispersion index (cross-sectional std of recent returns) ──────────
        dispersion_index = self._calc_dispersion_index(prices, sectors, window=20)
        dispersion_history = disp_df["dispersion_ratio"].dropna() if "dispersion_ratio" in disp_df.columns else pd.Series(dtype=float)

        # ── Eigenvalue concentration ──────────────────────────────────────────
        eigenvalue_concentration = self._calc_eigenvalue_hhi(C_t)

        # ── Anomalous pairs ───────────────────────────────────────────────────
        anomalous_pairs = self._find_anomalous_pairs(C_t, C_b, C_delta, settings)

        # ── Market state ──────────────────────────────────────────────────────
        market_state = str(master_df["market_state"].iloc[0]) if "market_state" in master_df.columns else "UNKNOWN"

        # ── Short vol signal ──────────────────────────────────────────────────
        corr_regime = self._classify_corr_regime(avg_corr_current, avg_corr_baseline, settings)
        short_vol_score, short_vol_label, short_vol_rationale = self._compute_short_vol_signal(
            avg_corr_current=avg_corr_current,
            avg_corr_baseline=avg_corr_baseline,
            implied_corr=implied_corr,
            implied_corr_z=implied_corr_z,
            crp=crp_last,
            crp_z=crp_z,
            disp_ratio_z=disp_ratio_z,
            market_state=market_state,
            corr_regime=corr_regime,
        )

        return CorrVolAnalysis(
            corr_current=C_t,
            corr_baseline=C_b,
            corr_delta=C_delta,
            avg_corr_current=avg_corr_current,
            avg_corr_baseline=avg_corr_baseline,
            avg_corr_delta=avg_corr_delta,
            implied_corr=implied_corr,
            implied_corr_z=implied_corr_z,
            fair_value_corr=fair_value_last,
            corr_risk_premium=crp_last,
            crp_z=crp_z,
            dispersion_ratio=disp_ratio,
            dispersion_ratio_z=disp_ratio_z,
            dispersion_index=dispersion_index,
            dispersion_history=dispersion_history,
            implied_corr_ts=implied_corr_ts,
            fair_value_ts=fair_value_ts,
            avg_corr_ts=avg_corr_ts,
            crp_ts=crp_ts,
            short_vol_score=short_vol_score,
            short_vol_label=short_vol_label,
            short_vol_rationale=short_vol_rationale,
            market_mode_strength=corr_m.market_mode_strength,
            market_mode_loadings=corr_m.market_mode_loadings,
            eigenvalue_concentration=eigenvalue_concentration,
            anomalous_pairs=anomalous_pairs,
            corr_regime=corr_regime,
            market_state=market_state,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Implied correlation math
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _calc_implied_corr(var_spy: float, indep_var: float, cross_var_term: float) -> float:
        """
        ρ_implied = (var_spy - indep_var) / cross_var_term

        cross_var_term = (Σwᵢσᵢ)² − Σwᵢ²σᵢ²  is the weight-correct denominator
        that accounts for real SPY sector weights (replaces equal-weight (N-1) formula).

        Interpretation:
          ρ_implied > avg_corr_baseline → correlation elevated → premium
          ρ_implied < 0                 → strong dispersion regime
        """
        if not (math.isfinite(var_spy) and math.isfinite(indep_var)
                and math.isfinite(cross_var_term) and cross_var_term > 1e-20):
            return float("nan")
        return (var_spy - indep_var) / cross_var_term

    def _build_rolling_ts(
        self, disp_df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Build rolling implied_corr, EWMA fair_value, and CRP time series.
        Returns (implied_corr_ts, fair_value_ts, crp_ts) as pd.Series.
        Uses weight-correct formula: ρ = (var_spy − indep_var) / cross_var_term.
        """
        if disp_df is None or disp_df.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, empty

        needed = ["var_spy", "indep_var", "cross_var_term"]
        available = [c for c in needed if c in disp_df.columns]
        df = disp_df[available].dropna()
        if len(df) < _MIN_HISTORY or "cross_var_term" not in df.columns:
            empty = pd.Series(dtype=float)
            return empty, empty, empty

        implied = df.apply(
            lambda row: self._calc_implied_corr(
                row["var_spy"], row["indep_var"], row["cross_var_term"]
            ), axis=1
        )
        implied.name = "implied_corr"
        implied = implied.clip(-0.5, 1.0)  # physical bounds

        # EWMA fair value
        fair_value = implied.ewm(alpha=1 - _EWMA_LAMBDA_CORR, adjust=False).mean()
        fair_value.name = "fair_value_corr"

        crp = (implied - fair_value).rename("crp")
        return implied, fair_value, crp

    @staticmethod
    def _zscore_last(ts: pd.Series, window: int) -> float:
        """Z-score of the last observation vs rolling window."""
        if ts is None or len(ts) < 20:
            return float("nan")
        recent = ts.dropna().tail(window)
        if len(recent) < 20:
            return float("nan")
        mu = float(recent.mean())
        sd = float(recent.std(ddof=1))
        if sd < 1e-10:
            return 0.0
        return float((recent.iloc[-1] - mu) / sd)

    # ─────────────────────────────────────────────────────────────────────────
    # Short vol signal
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_short_vol_signal(
        avg_corr_current: float,
        avg_corr_baseline: float,
        implied_corr: float,
        implied_corr_z: float,
        crp: float,
        crp_z: float,
        disp_ratio_z: float,
        market_state: str,
        corr_regime: str,
    ) -> Tuple[float, str, str]:
        """
        Compute short-volatility/dispersion-trade signal score 0-100.

        Logic (each component max 33 pts):
          A. Correlation elevation (how far current corr is above fair value)
          B. CRP (implied - EWMA fair value, z-score)
          C. Dispersion suppression (sectors co-moving = dispersion_ratio_z negative)

        Regime multiplier:
          CRISIS → 0.0  (never short vol in crisis — systemic risk)
          HIGH_CORR → 0.8 (elevated, but approaching limits)
          NORMAL / LOW_CORR → 1.0

        Returns (score_0_100, label, rationale_text)
        """
        factors = []
        rationale_parts = []

        # ── A. Correlation elevation ──────────────────────────────────────────
        # z-score of current avg_corr vs baseline
        if math.isfinite(avg_corr_current) and math.isfinite(avg_corr_baseline):
            delta_pct = avg_corr_current - avg_corr_baseline
            # Score: each 0.05 of elevation above baseline = ~10 pts (max 33)
            a_raw = min(max(delta_pct / 0.05 * 10, 0), 33)
        else:
            a_raw = 0.0
        factors.append(a_raw)
        if a_raw > 15:
            rationale_parts.append(
                f"קורלציה נוכחית ({avg_corr_current:.2f}) גבוהה מ-baseline ({avg_corr_baseline:.2f}) "
                f"ב-{(avg_corr_current - avg_corr_baseline)*100:.1f}pp"
            )

        # ── B. Correlation Risk Premium ───────────────────────────────────────
        if math.isfinite(crp_z):
            b_raw = min(max(crp_z / 2.0 * 33, 0), 33)
        elif math.isfinite(crp) and math.isfinite(implied_corr):
            b_raw = min(max(crp / 0.02 * 10, 0), 33)
        else:
            b_raw = 0.0
        factors.append(b_raw)
        if b_raw > 15 and math.isfinite(crp):
            rationale_parts.append(
                f"Correlation Risk Premium = {crp:.3f} (z={crp_z:.1f}): "
                "implied corr מתומחרת יקר מול fair value"
            )

        # ── C. Dispersion suppression ─────────────────────────────────────────
        if math.isfinite(disp_ratio_z):
            # Negative disp_ratio_z = sectors co-moving = good for short corr
            c_raw = min(max(-disp_ratio_z / 2.0 * 33, 0), 33)
        else:
            c_raw = 0.0
        factors.append(c_raw)
        if c_raw > 15:
            rationale_parts.append(
                f"פיזור נמוך (disp_z={disp_ratio_z:.1f}): סקטורים נעים יחד → הזדמנות dispersion"
            )

        raw_score = sum(factors)  # 0-99

        # ── Regime multiplier ─────────────────────────────────────────────────
        regime_multiplier = {
            "CRISIS_CORR": 0.0,
            "HIGH_CORR": 0.8,
            "NORMAL": 1.0,
            "LOW_CORR": 1.0,
        }.get(corr_regime, 1.0)

        if market_state == "CRISIS":
            regime_multiplier = 0.0
            rationale_parts.append("⛔ CRISIS regime — אסור לshort vol, סיכון סיסטמי אמיתי")
        elif market_state == "TENSION":
            regime_multiplier *= 0.7
            rationale_parts.append("⚠ TENSION regime — הפחתת score ב-30%")

        score = min(raw_score * regime_multiplier, 100)
        score = max(score, 0)

        # ── Label ─────────────────────────────────────────────────────────────
        if score >= 70:
            label = "SHORT VOL חזק"
        elif score >= 45:
            label = "SHORT VOL מתון"
        elif score >= 20:
            label = "ניטרלי"
        else:
            label = "הימנע מ-short vol"

        if not rationale_parts:
            rationale_parts.append("תנאים לא מספיקים לsignal משמעותי")

        rationale = "\n".join(rationale_parts)
        return round(score, 1), label, rationale

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_corr_regime(avg_corr_t: float, avg_corr_b: float, settings) -> str:
        if not math.isfinite(avg_corr_t):
            return "UNKNOWN"
        if avg_corr_t >= getattr(settings, "crisis_avg_corr_min", 0.75):
            return "CRISIS_CORR"
        if avg_corr_t >= getattr(settings, "tension_avg_corr_min", 0.60):
            return "HIGH_CORR"
        if avg_corr_t <= getattr(settings, "calm_avg_corr_max", 0.45):
            return "LOW_CORR"
        return "NORMAL"

    @staticmethod
    def _calc_dispersion_index(prices: pd.DataFrame, sectors: list, window: int = 20) -> float:
        """Cross-sectional std of recent sector returns (annualized %)."""
        if prices is None or prices.empty:
            return float("nan")
        avail = [s for s in sectors if s in prices.columns]
        if len(avail) < 3:
            return float("nan")
        try:
            rets = np.log(prices[avail] / prices[avail].shift(1)).dropna().tail(window)
            if len(rets) < 5:
                return float("nan")
            # Average daily cross-sectional std, annualized
            cs_std = rets.std(axis=1, ddof=1)
            return round(float(cs_std.mean() * math.sqrt(252) * 100), 2)
        except Exception:
            return float("nan")

    @staticmethod
    def _calc_eigenvalue_hhi(C: pd.DataFrame) -> float:
        """Herfindahl-Hirschman Index of eigenvalue distribution (concentration)."""
        if C is None or C.empty:
            return float("nan")
        try:
            Cv = 0.5 * (C.values + C.values.T)
            Cv = np.nan_to_num(Cv.astype(float))
            evals = np.linalg.eigvalsh(Cv)
            evals = np.clip(evals, 0, None)
            total = evals.sum()
            if total < 1e-10:
                return float("nan")
            shares = evals / total
            return round(float((shares ** 2).sum()), 4)
        except Exception:
            return float("nan")

    @staticmethod
    def _find_anomalous_pairs(
        C_t: pd.DataFrame, C_b: pd.DataFrame, C_delta: pd.DataFrame,
        settings
    ) -> List[Dict]:
        """
        Find top sector pairs with largest correlation distortion |C_t - C_b|.
        Returns list of dicts sorted by |delta| descending.
        """
        ticker_to_name = settings.canonical_sector_by_ticker()
        pairs = []
        try:
            n = len(C_delta.columns)
            cols = list(C_delta.columns)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = cols[i], cols[j]
                    delta = float(C_delta.iloc[i, j])
                    if not math.isfinite(delta):
                        continue
                    corr_curr = float(C_t.iloc[i, j])
                    corr_base = float(C_b.iloc[i, j])
                    pairs.append({
                        "sector_a": a,
                        "name_a": ticker_to_name.get(a, a),
                        "sector_b": b,
                        "name_b": ticker_to_name.get(b, b),
                        "corr_current": round(corr_curr, 3),
                        "corr_baseline": round(corr_base, 3),
                        "delta": round(delta, 3),
                        "direction": "↑ elevated" if delta > 0 else "↓ suppressed",
                    })
        except Exception as e:
            log.warning("_find_anomalous_pairs error: %s", e)

        pairs.sort(key=lambda x: abs(x["delta"]), reverse=True)
        return pairs[:10]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience summary for agents / API
# ─────────────────────────────────────────────────────────────────────────────
def corr_vol_summary(analysis: CorrVolAnalysis) -> dict:
    """Compact serializable summary for Claude / AgentBus."""
    top3 = analysis.anomalous_pairs[:3]
    return {
        "avg_corr_current":    round(analysis.avg_corr_current, 3)     if math.isfinite(analysis.avg_corr_current)    else None,
        "avg_corr_baseline":   round(analysis.avg_corr_baseline, 3)    if math.isfinite(analysis.avg_corr_baseline)   else None,
        "avg_corr_delta":      round(analysis.avg_corr_delta, 3)       if math.isfinite(analysis.avg_corr_delta)      else None,
        "implied_corr":        round(analysis.implied_corr, 3)         if math.isfinite(analysis.implied_corr)        else None,
        "fair_value_corr":     round(analysis.fair_value_corr, 3)      if math.isfinite(analysis.fair_value_corr)     else None,
        "corr_risk_premium":   round(analysis.corr_risk_premium, 3)    if math.isfinite(analysis.corr_risk_premium)   else None,
        "dispersion_ratio":    round(analysis.dispersion_ratio, 3)     if math.isfinite(analysis.dispersion_ratio)    else None,
        "dispersion_index_pct":analysis.dispersion_index,
        "short_vol_score":     analysis.short_vol_score,
        "short_vol_label":     analysis.short_vol_label,
        "market_mode_strength":round(analysis.market_mode_strength, 3) if math.isfinite(analysis.market_mode_strength) else None,
        "eigenvalue_hhi":      analysis.eigenvalue_concentration,
        "corr_regime":         analysis.corr_regime,
        "top_distorted_pairs": [
            f"{p['sector_a']}-{p['sector_b']}: Δ={p['delta']:+.3f} ({p['direction']})"
            for p in top3
        ],
        "rationale":           analysis.short_vol_rationale,
    }
