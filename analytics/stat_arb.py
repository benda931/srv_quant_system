"""
analytics/stat_arb.py

QuantEngine: mathematical core of the SRV Quantamental Decision Support System.

Implements:
- Log returns & EWMA volatility (RiskMetrics style).
- Volatility parity hedge ratios (HR = Vol(SPY)/Vol(Sector)).
- Realized Dispersion via variance difference method:
    indep_var = Σ (w_i^2 * Var(r_i)), dispersion = indep_var - Var(SPY)
- Strict Out-of-Sample Rolling PCA residual generation:
    train on [t-window:t), transform on [t], no look-ahead.
- Macro betas vs ^TNX and DX-Y.NYB (rolling covariance beta).
- Correlation Structure Engine:
    - Current corr matrix C_t (60d)
    - Baseline corr matrix C_b (252d)
    - Delta corr matrix ΔC = C_t - C_b
    - Distortion scalar D_t = ||ΔC_offdiag||_F and ΔD (t - t-1)
    - Market mode: λ1/N + eigenvector loadings per sector
    - Sector distortion contributions (row-norm of ΔC)
- Delta-1 RV portfolio construction:
    signal -> vol scaling -> beta-neutral projection -> normalize gross=1
- Synthetic Greeks & exposures:
    - Δ_SPY, Δ_TNX, Δ_DXY (beta exposures)
    - Γ_synth (short convexity intensity proxy)
    - Vega_synth (realized vol intensity proxy)
    - ρ_mode (market-mode/corr proxy)
    - ρ_dist (corr distortion risk proxy)
- Risk decomposition:
    portfolio variance split into diagonal vs off-diagonal components.

Outputs:
- master_df: sector-level signals + conviction + execution directives + book/greeks
- residual_level_ts: time series per sector (for tear sheets)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import Settings
from data.pipeline import ParquetArtifacts
from analytics.attribution import compute_attribution_row
from analytics.fundamentals_engine import (
    SectorFundamentals,
    aggregate_sector_fundamentals,
    compute_all_sectors_fjs,
)


# -----------------------------
# Utilities
# -----------------------------
def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    sd = sd.replace(0.0, np.nan)
    return (series - mu) / sd


def _rolling_percentile_of_last(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank of the last element within its rolling window.
    Returns values in [0..1]. Uses an O(window) apply; acceptable for small series.
    """

    def pct_rank(x: np.ndarray) -> float:
        if len(x) == 0:
            return float("nan")
        last = x[-1]
        return float(np.sum(x <= last) / len(x))

    return series.rolling(window=window, min_periods=window).apply(lambda x: pct_rank(np.asarray(x)), raw=False)

def _clip(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return lo
    return max(lo, min(hi, float(x)))

def ewma_volatility(returns: pd.Series, lam: float, min_periods: int = 20) -> pd.Series:
    """
    EWMA volatility estimator (RiskMetrics style):
        sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2
    """
    r = returns.astype(float).values
    n = len(r)
    var = np.full(n, np.nan, dtype=float)

    if n >= min_periods and np.isfinite(r[:min_periods]).sum() >= int(0.8 * min_periods):
        var[min_periods - 1] = np.nanvar(r[:min_periods], ddof=0)

    for t in range(min_periods, n):
        prev = var[t - 1]
        if not np.isfinite(prev):
            window = r[max(0, t - min_periods) : t]
            if np.isfinite(window).sum() >= int(0.8 * len(window)):
                prev = np.nanvar(window, ddof=0)
            else:
                continue

        rt_1 = r[t - 1]
        if not np.isfinite(rt_1):
            var[t] = prev
            continue

        var[t] = lam * prev + (1.0 - lam) * (rt_1**2)

    return pd.Series(np.sqrt(var), index=returns.index, name=returns.name)


def estimate_ar1_half_life(series: pd.Series, min_obs: int = 120) -> float:
    """
    Estimate mean reversion half-life using AR(1):
        x_t = a + phi x_{t-1} + eps
        half-life = -ln(2) / ln(phi), for 0 < phi < 1
    """
    x = series.dropna().astype(float)
    if len(x) < min_obs:
        return float("nan")

    x_lag = x.shift(1).dropna()
    x_now = x.loc[x_lag.index]
    if len(x_lag) < min_obs:
        return float("nan")

    X = np.vstack([np.ones(len(x_lag)), x_lag.values]).T
    y = x_now.values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        phi = float(beta[1])
    except Exception:
        return float("nan")

    if phi <= 0.0 or phi >= 0.999:
        return float("nan")

    return float(-math.log(2.0) / math.log(phi))


def _upper_triangle_mean(C: pd.DataFrame) -> float:
    """Mean of off-diagonal correlations (upper triangle)."""
    if C is None or C.empty:
        return float("nan")
    a = C.values.astype(float)
    n = a.shape[0]
    if n <= 1:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    v = a[iu]
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _fro_offdiag(C: pd.DataFrame) -> float:
    """Frobenius norm of off-diagonal part (diagonal zeroed)."""
    if C is None or C.empty:
        return float("nan")
    a = C.values.astype(float).copy()
    np.fill_diagonal(a, 0.0)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.linalg.norm(a, ord="fro"))


def _row_norm_offdiag(C: pd.DataFrame) -> Dict[str, float]:
    """Per-row (per-sector) norm of off-diagonal values."""
    if C is None or C.empty:
        return {}
    a = C.values.astype(float).copy()
    np.fill_diagonal(a, 0.0)
    out: Dict[str, float] = {}
    for i, s in enumerate(C.index):
        v = a[i, :]
        v = v[np.isfinite(v)]
        out[str(s)] = float(np.linalg.norm(v)) if v.size else float("nan")
    return out


def _series_last(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return float("nan")
    x = float(s.iloc[-1])
    return x if math.isfinite(x) else float("nan")


def _get_int_setting(settings: Settings, name: str, default: int) -> int:
    v = getattr(settings, name, None)
    try:
        iv = int(v)
        return iv if iv > 0 else default
    except Exception:
        return default


# -----------------------------
# Correlation metrics container
# -----------------------------
@dataclass(frozen=True)
class CorrMetrics:
    C_t: pd.DataFrame
    C_b: pd.DataFrame
    C_delta: pd.DataFrame

    avg_corr_t: float
    avg_corr_b: float
    avg_corr_delta: float

    distortion_t: float
    distortion_prev: float
    delta_distortion: float

    market_mode_strength: float
    market_mode_loadings: Dict[str, float]

    sector_corr_avg_t: Dict[str, float]
    sector_corr_avg_b: Dict[str, float]
    sector_corr_avg_delta: Dict[str, float]
    sector_distortion_contrib: Dict[str, float]

@dataclass(frozen=True)
class RegimeMetrics:
    market_state: str
    state_bias: str
    mean_reversion_allowed: bool
    regime_alert: str

    regime_vol_score: float
    regime_credit_score: float
    regime_corr_score: float
    regime_transition_score: float

    transition_probability: float
    crisis_probability: float
# -----------------------------
# Quant Engine
# -----------------------------
class QuantEngine:
    """
    Core analytics engine.

    Inputs:
    - prices.parquet: wide daily price panel
    - fundamentals.parquet: quote table (at least ETFs; ideally holdings too)
    - weights.parquet: ETF holdings + SPY sector allocations (optional)

    Outputs:
    - master_df: sector-level signal + conviction + execution directives + book/greeks
    - residual_level_ts: time series per sector (for tear sheets)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.artifacts = ParquetArtifacts.from_settings(settings)

        self.prices: Optional[pd.DataFrame] = None
        self.fundamentals: Optional[pd.DataFrame] = None
        self.ext_fundamentals: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.DataFrame] = None

        # Analytics outputs
        self.residual_returns: Optional[pd.DataFrame] = None
        self.residual_levels: Optional[pd.DataFrame] = None
        self.residual_zscores: Optional[pd.DataFrame] = None
        self.dispersion_df: Optional[pd.DataFrame] = None
        self.ewma_vol: Optional[pd.DataFrame] = None
        self.macro_betas: Optional[pd.DataFrame] = None
        self.master_df: Optional[pd.DataFrame] = None

        # Corr engine storage for UI (heatmaps etc.)
        self.corr_matrix_current: Optional[pd.DataFrame] = None
        self.corr_matrix_baseline: Optional[pd.DataFrame] = None
        self.corr_matrix_delta: Optional[pd.DataFrame] = None
        self.corr_metrics: Optional[CorrMetrics] = None

    def _compute_regime_metrics(
        self,
        *,
        avg_corr_t: float,
        avg_corr_delta: float,
        distortion_t: float,
        delta_distortion: float,
        market_mode_strength: float,
        vix_level: float,
        vix_percentile: float,
        credit_z: float,
    ) -> RegimeMetrics:
        s = self.settings

        # -------------------------
        # Vol regime score
        # -------------------------
        vol_score = 0.0
        if math.isfinite(vix_level):
            if vix_level <= s.vix_level_soft:
                vol_score = 0.0
            elif vix_level >= s.vix_level_hard:
                vol_score = 1.0
            else:
                vol_score = (vix_level - s.vix_level_soft) / max(1e-9, (s.vix_level_hard - s.vix_level_soft))

        if math.isfinite(vix_percentile):
            vol_score = max(vol_score, _clip((vix_percentile - 0.70) / 0.25, 0.0, 1.0))

        # -------------------------
        # Credit regime score
        # -------------------------
        credit_score = 0.0
        if math.isfinite(credit_z):
            credit_score = _clip((-credit_z - 0.25) / 1.75, 0.0, 1.0)

        # -------------------------
        # Correlation regime score
        # -------------------------
        corr_level_score = _clip(
            (avg_corr_t - s.calm_avg_corr_max) / max(1e-9, (s.crisis_avg_corr_min - s.calm_avg_corr_max)),
            0.0,
            1.0,
        ) if math.isfinite(avg_corr_t) else 0.0

        mode_score = _clip(
            (market_mode_strength - s.calm_mode_strength_max) /
            max(1e-9, (s.crisis_mode_strength_min - s.calm_mode_strength_max)),
            0.0,
            1.0,
        ) if math.isfinite(market_mode_strength) else 0.0

        dist_score = _clip(
            (distortion_t - s.tension_corr_dist_min) /
            max(1e-9, (s.crisis_corr_dist_min - s.tension_corr_dist_min)),
            0.0,
            1.0,
        ) if math.isfinite(distortion_t) else 0.0

        corr_score = _clip(
            0.45 * corr_level_score +
            0.30 * mode_score +
            0.25 * dist_score,
            0.0,
            1.0,
        )

        # -------------------------
        # Transition score
        # -------------------------
        avg_corr_delta_score = _clip(abs(avg_corr_delta) / 0.18, 0.0, 1.0) if math.isfinite(avg_corr_delta) else 0.0
        delta_dist_score = _clip(
            (delta_distortion - 0.03) / max(1e-9, (s.crisis_delta_corr_dist_min - 0.03)),
            0.0,
            1.0,
        ) if math.isfinite(delta_distortion) else 0.0

        transition_score = _clip(
            0.45 * delta_dist_score +
            0.25 * avg_corr_delta_score +
            0.15 * vol_score +
            0.15 * credit_score,
            0.0,
            1.0,
        )

        crisis_probability = _clip(
            0.40 * corr_score +
            0.25 * vol_score +
            0.20 * credit_score +
            0.15 * transition_score,
            0.0,
            1.0,
        )

        transition_probability = _clip(
            0.50 * transition_score +
            0.30 * corr_score +
            0.20 * vol_score,
            0.0,
            1.0,
        )

        # -------------------------
        # State classification
        # -------------------------
        crisis_hits = 0
        if math.isfinite(vix_level) and vix_level >= s.vix_level_hard:
            crisis_hits += 1
        if math.isfinite(avg_corr_t) and avg_corr_t >= s.crisis_avg_corr_min:
            crisis_hits += 1
        if math.isfinite(market_mode_strength) and market_mode_strength >= s.crisis_mode_strength_min:
            crisis_hits += 1
        if math.isfinite(distortion_t) and distortion_t >= s.crisis_corr_dist_min:
            crisis_hits += 1
        if math.isfinite(delta_distortion) and delta_distortion >= s.crisis_delta_corr_dist_min:
            crisis_hits += 1
        if math.isfinite(credit_z) and credit_z <= s.credit_stress_z:
            crisis_hits += 1

        tension_hits = 0
        if math.isfinite(vix_level) and vix_level >= s.vix_level_soft:
            tension_hits += 1
        if math.isfinite(avg_corr_t) and avg_corr_t >= s.tension_avg_corr_min:
            tension_hits += 1
        if math.isfinite(market_mode_strength) and market_mode_strength >= s.tension_mode_strength_min:
            tension_hits += 1
        if math.isfinite(distortion_t) and distortion_t >= s.tension_corr_dist_min:
            tension_hits += 1
        if math.isfinite(delta_distortion) and delta_distortion >= s.tension_delta_corr_dist_min:
            tension_hits += 1

        if crisis_hits >= 3 or crisis_probability >= 0.78:
            market_state = "CRISIS"
        elif tension_hits >= 2 or transition_score >= s.transition_score_danger:
            market_state = "TENSION"
        elif transition_score >= s.transition_score_caution or corr_score >= 0.40 or vol_score >= 0.35:
            market_state = "NORMAL"
        else:
            market_state = "CALM"

        if market_state == "CALM":
            state_bias = "MEAN_REVERSION_FRIENDLY"
            mean_reversion_allowed = True
            regime_alert = "LOW_STRESS"
        elif market_state == "NORMAL":
            state_bias = "SELECTIVE_MEAN_REVERSION"
            mean_reversion_allowed = True
            regime_alert = "WATCH_TRANSITIONS"
        elif market_state == "TENSION":
            state_bias = "REDUCE_AND_BE_SELECTIVE"
            mean_reversion_allowed = True
            regime_alert = "INSTABILITY_RISING"
        else:
            state_bias = "DEFENSIVE_OR_NO_TRADE"
            mean_reversion_allowed = False
            regime_alert = "REGIME_BREAK_RISK"

        return RegimeMetrics(
            market_state=market_state,
            state_bias=state_bias,
            mean_reversion_allowed=mean_reversion_allowed,
            regime_alert=regime_alert,
            regime_vol_score=vol_score,
            regime_credit_score=credit_score,
            regime_corr_score=corr_score,
            regime_transition_score=transition_score,
            transition_probability=transition_probability,
            crisis_probability=crisis_probability,
        )
    

    def _state_color(self, state: str) -> str:
        state_key = str(state).upper().strip()
        state_to_color = {
            "CALM": "success",
            "NORMAL": "primary",
            "TENSION": "warning",
            "CRISIS": "danger",
        }
        return state_to_color.get(state_key, "secondary")
    
    # -----------------------------
    # Data loading
    # -----------------------------
    def load(self) -> None:
        self.prices = pd.read_parquet(self.artifacts.prices_path)
        self.prices.index = pd.to_datetime(self.prices.index)
        self.prices = self.prices.sort_index()

        self.fundamentals = pd.read_parquet(self.artifacts.fundamentals_path)

        self.ext_fundamentals = None
        ext_path = self.artifacts.extended_fundamentals_path
        if ext_path.exists():
            try:
                df_ext = pd.read_parquet(ext_path)
                self.ext_fundamentals = df_ext if not df_ext.empty else None
            except Exception as e:
                self.logger.warning("Failed to load extended_fundamentals parquet: %s", repr(e))

        self.weights = None
        if self.artifacts.weights_path.exists():
            try:
                self.weights = pd.read_parquet(self.artifacts.weights_path)
            except Exception as e:
                self.logger.warning("Failed to load weights parquet: %s", repr(e))
                self.weights = None

        # Basic cleaning
        self.prices = self.prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

    # -----------------------------
    # Core transforms
    # -----------------------------
    def _log_returns(self, px: pd.DataFrame) -> pd.DataFrame:
        lp = np.log(px)
        return lp.diff()

    def _compute_ewma_vol_panel(self, returns: pd.DataFrame) -> pd.DataFrame:
        vols: List[pd.Series] = []
        for col in returns.columns:
            vols.append(ewma_volatility(returns[col], lam=self.settings.ewma_lambda, min_periods=30).rename(col))
        vol = pd.concat(vols, axis=1)
        return vol * math.sqrt(252.0)  # annualized

    # -----------------------------
    # Dispersion
    # -----------------------------
    def _get_spy_sector_weights(self) -> Dict[str, float]:
        """
        Sector weights for SPY across sector ETFs (for dispersion approximation).
        Uses weights parquet if available; otherwise equal weights.
        """
        sector_etfs = self.settings.sector_list()

        if self.weights is None or self.weights.empty or "record_type" not in self.weights.columns:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        df_sw = self.weights[self.weights["record_type"] == "sector_weight"].copy()
        if df_sw.empty or "sector" not in df_sw.columns or "weightPercentage" not in df_sw.columns:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        alias = self.settings.fmp_sector_name_aliases
        canonical_to_ticker = self.settings.sector_tickers

        sector_w: Dict[str, float] = {}
        for _, row in df_sw.iterrows():
            raw_name = str(row.get("sector", "")).strip()
            canon = alias.get(raw_name, None)
            if not canon:
                continue
            etf = canonical_to_ticker.get(canon)
            if not etf:
                continue
            w = _safe_float(row.get("weightPercentage"))
            if math.isfinite(w) and w > 0:
                sector_w[etf] = sector_w.get(etf, 0.0) + w

        total = sum(sector_w.values())
        if total <= 0:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        # If weights came as percents (sum ~100)
        if total > 1.5:
            sector_w = {k: v / 100.0 for k, v in sector_w.items()}
            total = sum(sector_w.values())

        sector_w = {k: v / total for k, v in sector_w.items()}
        for t in sector_etfs:
            sector_w.setdefault(t, 0.0)

        # Avoid extremely sparse mapping
        if sum(1 for v in sector_w.values() if v > 0) < 6:
            return {t: 1.0 / len(sector_etfs) for t in sector_etfs}

        return sector_w

    def _compute_realized_dispersion(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Variance difference method (rolling window):
            var_spy = Var(r_spy)
            indep_var = Σ (w_i^2 * Var(r_sector_i))
            dispersion = indep_var - var_spy
            correlation_component = var_spy - indep_var
            dispersion_ratio = indep_var / var_spy
        """
        w = self._get_spy_sector_weights()
        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker

        rv = returns[sectors + [spy]].rolling(
            window=self.settings.dispersion_window,
            min_periods=self.settings.dispersion_window,
        ).var(ddof=0)
        var_spy = rv[spy]

        indep_var = None
        for s in sectors:
            wi = float(w.get(s, 0.0))
            term = (wi**2) * rv[s]
            indep_var = term if indep_var is None else (indep_var + term)

        indep_var = indep_var.rename("indep_var")
        dispersion = (indep_var - var_spy).rename("dispersion")
        corr_component = (var_spy - indep_var).rename("correlation_component")
        dispersion_ratio = (indep_var / var_spy).rename("dispersion_ratio")

        # Correct implied-correlation denominator: (Σwᵢσᵢ)² − Σwᵢ²σᵢ²
        # This removes the equal-weight assumption from _calc_implied_corr.
        weighted_vol_sum = None
        for s in sectors:
            wi = float(w.get(s, 0.0))
            vol_s = rv[s].apply(lambda x: math.sqrt(max(x, 0.0)) if math.isfinite(x) else 0.0)
            term = wi * vol_s
            weighted_vol_sum = term if weighted_vol_sum is None else (weighted_vol_sum + term)

        cross_var_term = ((weighted_vol_sum ** 2) - indep_var).rename("cross_var_term").clip(lower=1e-20)

        out = pd.concat([var_spy.rename("var_spy"), indep_var, dispersion, corr_component, dispersion_ratio, cross_var_term], axis=1)
        out["dispersion_ratio_z"] = _zscore(out["dispersion_ratio"], window=max(126, self.settings.pca_window))
        return out

    # -----------------------------
    # OOS PCA residuals
    # -----------------------------
    def _oos_rolling_pca_residuals(self, rel_returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        OOS rolling PCA on relative returns (sector - SPY).

        For each t >= window:
          - Fit scaler+PCA on rel_returns[t-window : t]  (excluding t)
          - Transform and reconstruct rel_returns[t]
          - residual_return[t] = rel_returns[t] - reconstruction[t]

        Uses joblib parallel processing (4 workers) for ~3x speedup on multi-core.
        """
        window = self.settings.pca_window
        zwin = self.settings.zscore_window
        sectors = rel_returns.columns.tolist()

        n = len(rel_returns)
        idx = rel_returns.index

        # Pre-extract numpy arrays — forward-fill NaN (not zeros) to avoid bias
        values = rel_returns.astype(float).ffill().fillna(0.0).values
        nan_mask_row = rel_returns.isna().values
        nan_mask_train_pct = np.array([
            rel_returns.iloc[t - window : t].isna().mean().mean()
            for t in range(window, n)
        ])
        nan_mask_x_pct = np.array([
            rel_returns.iloc[t].isna().mean() for t in range(window, n)
        ])

        pca_max = self.settings.pca_max_components
        pca_min = self.settings.pca_min_components
        pca_var_target = self.settings.pca_explained_var_target
        n_cols = values.shape[1]

        # PCA refit interval: reuse PCA model for N days before refitting
        # This gives ~5x speedup with negligible accuracy loss (PCA stable over short periods)
        pca_refit_interval = getattr(self.settings, "pca_refit_interval", 5)
        _cached_scaler = [None]
        _cached_pca = [None]
        _last_refit = [0]

        def _process_date(t_offset):
            """Process a single date (t_offset is index into range(window, n))."""
            t = t_offset + window
            if nan_mask_train_pct[t_offset] > 0.05 or nan_mask_x_pct[t_offset] > 0.10:
                return t, None

            x_row = values[t : t + 1]

            # Refit PCA every pca_refit_interval days (or on first call)
            need_refit = (
                _cached_scaler[0] is None
                or (t_offset - _last_refit[0]) >= pca_refit_interval
            )

            if need_refit:
                train_f = values[t - window : t]
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs = scaler.fit_transform(train_f)

                pca_full = PCA(n_components=min(pca_max, n_cols), svd_solver="full")
                pca_full.fit(Xs)

                csum = np.cumsum(pca_full.explained_variance_ratio_)
                k = int(np.searchsorted(csum, pca_var_target) + 1)
                k = max(pca_min, min(k, pca_max))

                pca = PCA(n_components=k, svd_solver="full")
                pca.fit(Xs)

                _cached_scaler[0] = scaler
                _cached_pca[0] = pca
                _last_refit[0] = t_offset
            else:
                scaler = _cached_scaler[0]
                pca = _cached_pca[0]

            x_s = scaler.transform(x_row)
            scores = pca.transform(x_s)
            x_hat_s = pca.inverse_transform(scores)
            resid_s = (x_s - x_hat_s).reshape(-1)
            resid = resid_s * scaler.scale_
            return t, resid

        # Sequential with PCA caching (refit every N days) — faster than parallel for cached mode
        # The cache makes sequential ~5x faster, and avoids joblib serialization overhead
        n_dates = n - window
        if pca_refit_interval > 1:
            # Sequential mode with cache — most efficient
            self.logger.debug(
                "PCA residuals: sequential with refit every %d days (%d dates)",
                pca_refit_interval, n_dates,
            )
            results = [_process_date(t_off) for t_off in range(n_dates)]
        else:
            # Full refit every day — use parallel for speedup
            try:
                from joblib import Parallel, delayed
                n_jobs = min(4, max(1, n_dates // 100))
                if n_jobs > 1 and n_dates > 200:
                    self.logger.debug("PCA residuals: parallel with %d workers (%d dates)", n_jobs, n_dates)
                    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
                        delayed(_process_date)(t_off) for t_off in range(n_dates)
                    )
                else:
                    results = [_process_date(t_off) for t_off in range(n_dates)]
            except ImportError:
                results = [_process_date(t_off) for t_off in range(n_dates)]

        # Assemble results
        resid_ret = pd.DataFrame(index=idx, columns=sectors, dtype=float)
        for t, resid in results:
            if resid is not None:
                resid_ret.iloc[t] = resid

        resid_level = resid_ret.fillna(0.0).cumsum()
        resid_z = resid_level.apply(lambda s: _zscore(s, window=zwin), axis=0)

        return resid_ret, resid_level, resid_z.rename_axis(index="date")

    # -----------------------------
    # Macro betas
    # -----------------------------
    def _compute_macro_betas(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling covariance betas (macro_window) vs:
        - ^TNX (daily change)
        - DX-Y.NYB (log return)

        beta = Cov(r_sector, r_macro) / Var(r_macro)
        """
        sectors = self.settings.sector_list()
        tnx = self.settings.macro_tickers["TNX_10Y"]
        dxy = self.settings.macro_tickers["DXY_USD"]
        w = self.settings.macro_window

        tnx_change = self.prices[tnx].diff()
        dxy_ret = returns[dxy]

        out_rows: List[pd.DataFrame] = []
        for s in sectors:
            r_s = returns[s]

            var_tnx = tnx_change.rolling(w, min_periods=w).var(ddof=0).replace(0.0, np.nan)
            var_dxy = dxy_ret.rolling(w, min_periods=w).var(ddof=0).replace(0.0, np.nan)

            beta_tnx = r_s.rolling(w, min_periods=w).cov(tnx_change) / var_tnx
            beta_dxy = r_s.rolling(w, min_periods=w).cov(dxy_ret) / var_dxy

            out = pd.DataFrame({"beta_tnx": beta_tnx, "beta_dxy": beta_dxy}, index=returns.index)
            out["sector"] = s
            out_rows.append(out)

        betas = pd.concat(out_rows, axis=0).reset_index().rename(columns={"index": "date"})
        betas["date"] = pd.to_datetime(betas["date"])
        betas = betas.set_index(["date", "sector"]).sort_index()
        return betas

    # -----------------------------
    # Fundamentals helpers
    # -----------------------------
    def _build_symbol_to_quote(self) -> Dict[str, Dict[str, Any]]:
        if self.fundamentals is None or self.fundamentals.empty:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for _, row in self.fundamentals.iterrows():
            sym = str(row.get("symbol", "")).strip()
            if sym:
                out[sym] = dict(row)
        return out

    def _get_holdings_for_etf(self, etf: str) -> pd.DataFrame:
        if self.weights is None or self.weights.empty or "record_type" not in self.weights.columns:
            return pd.DataFrame(columns=["asset", "weightPercentage"])
        df = self.weights
        h = df[(df["record_type"] == "holding") & (df["etfSymbol"] == etf)].copy()
        if h.empty:
            return pd.DataFrame(columns=["asset", "weightPercentage"])
        h["weightPercentage"] = pd.to_numeric(h["weightPercentage"], errors="coerce")
        h = h.dropna(subset=["asset", "weightPercentage"])
        return h[["asset", "weightPercentage"]].copy()

    def _aggregate_portfolio_pe(self, holdings: pd.DataFrame, quote_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute portfolio PE from holdings and quotes:
            EY = Σ w_i / PE_i  (PE_i > 0)
            PE_portfolio = 1 / EY
        Returns also coverage and weighted median EPS (display).
        """
        if holdings.empty:
            return {
                "pe_portfolio": float("nan"),
                "earnings_yield": float("nan"),
                "covered_weight": 0.0,
                "neg_or_missing_weight": 1.0,
                "weighted_median_eps": float("nan"),
            }

        w = holdings.copy()
        w["w"] = pd.to_numeric(w["weightPercentage"], errors="coerce") / 100.0
        w = w.dropna(subset=["asset", "w"])
        w = w[w["w"] > 0]
        if w.empty:
            return {
                "pe_portfolio": float("nan"),
                "earnings_yield": float("nan"),
                "covered_weight": 0.0,
                "neg_or_missing_weight": 1.0,
                "weighted_median_eps": float("nan"),
            }

        w_sum = float(w["w"].sum())
        if w_sum <= 0:
            return {
                "pe_portfolio": float("nan"),
                "earnings_yield": float("nan"),
                "covered_weight": 0.0,
                "neg_or_missing_weight": 1.0,
                "weighted_median_eps": float("nan"),
            }
        w["w"] = w["w"] / w_sum

        earnings_yield = 0.0
        covered_weight = 0.0
        eps_vals: List[Tuple[float, float]] = []

        for _, row in w.iterrows():
            sym = str(row["asset"]).strip()
            wi = float(row["w"])
            q = quote_map.get(sym, {})
            pe = _safe_float(q.get("pe"))
            eps = _safe_float(q.get("eps"))

            if math.isfinite(eps):
                eps_vals.append((eps, wi))

            if math.isfinite(pe) and pe > 0:
                earnings_yield += wi / pe
                covered_weight += wi

        neg_missing_weight = 1.0 - covered_weight
        pe_portfolio = (1.0 / earnings_yield) if earnings_yield > 0 else float("nan")

        weighted_median_eps = float("nan")
        if eps_vals:
            eps_vals.sort(key=lambda t: t[0])
            cum = 0.0
            for eps, wi in eps_vals:
                cum += wi
                if cum >= 0.5:
                    weighted_median_eps = float(eps)
                    break

        return {
            "pe_portfolio": pe_portfolio,
            "earnings_yield": earnings_yield,
            "covered_weight": covered_weight,
            "neg_or_missing_weight": neg_missing_weight,
            "weighted_median_eps": weighted_median_eps,
        }

    def _etf_proxy_valuation(self, etf: str, quote_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Fallback valuation when holdings quotes are not available:
        use ETF's own quote (pe/eps).
        """
        q = quote_map.get(etf, {})
        pe = _safe_float(q.get("pe"))
        eps = _safe_float(q.get("eps"))

        # We don't fabricate PE if missing; keep NaN.
        ey = (1.0 / pe) if (math.isfinite(pe) and pe > 0) else float("nan")
        neg_w = 0.0 if (math.isfinite(pe) and pe > 0) else 1.0

        return {
            "pe_portfolio": pe,
            "earnings_yield": ey,
            "covered_weight": 1.0 if (math.isfinite(pe) and pe > 0) else 0.0,
            "neg_or_missing_weight": neg_w,
            "weighted_median_eps": eps,
        }

    # -----------------------------
    # Correlation Structure Engine
    # -----------------------------
    def _compute_corr_metrics(self, returns: pd.DataFrame, sectors: Sequence[str]) -> CorrMetrics:
        corr_w = _get_int_setting(self.settings, "corr_window", 60)
        base_w = _get_int_setting(self.settings, "corr_baseline_window", 252)

        R = returns[list(sectors)].dropna(how="all")

        if len(R) >= corr_w:
            C_t = R.iloc[-corr_w:].corr()
        else:
            C_t = pd.DataFrame(np.nan, index=sectors, columns=sectors)

        if len(R) >= base_w:
            C_b = R.iloc[-base_w:].corr()
        elif len(R) >= corr_w:
            C_b = C_t.copy()
        else:
            C_b = pd.DataFrame(np.nan, index=sectors, columns=sectors)

        C_delta = C_t - C_b

        avg_corr_t = _upper_triangle_mean(C_t)
        avg_corr_b = _upper_triangle_mean(C_b)
        avg_corr_delta = avg_corr_t - avg_corr_b if (math.isfinite(avg_corr_t) and math.isfinite(avg_corr_b)) else float("nan")

        distortion_t = _fro_offdiag(C_delta)

        distortion_prev = float("nan")
        if len(R) >= corr_w + 1:
            C_prev = R.iloc[-(corr_w + 1) : -1].corr()
            C_prev_delta = C_prev - C_b
            distortion_prev = _fro_offdiag(C_prev_delta)

        delta_distortion = (
            float(distortion_t - distortion_prev)
            if (math.isfinite(distortion_t) and math.isfinite(distortion_prev))
            else float("nan")
        )

        market_mode_strength = float("nan")
        market_mode_loadings: Dict[str, float] = {str(s): float("nan") for s in sectors}
        try:
            if C_t.notna().sum().sum() > 0:
                Ct = C_t.fillna(0.0).values.astype(float)
                Ct = 0.5 * (Ct + Ct.T)
                evals, evecs = np.linalg.eigh(Ct)
                i_max = int(np.argmax(evals))
                lam1 = float(evals[i_max])
                v1 = evecs[:, i_max].astype(float)

                # stable sign anchoring
                if v1[np.argmax(np.abs(v1))] < 0:
                    v1 = -v1

                market_mode_strength = float(lam1 / len(sectors))
                market_mode_loadings = {str(sectors[i]): float(v1[i]) for i in range(len(sectors))}
        except Exception:
            pass

        # Sector-level correlation averages and distortion contributions
        sector_corr_avg_t: Dict[str, float] = {}
        sector_corr_avg_b: Dict[str, float] = {}
        sector_corr_avg_delta: Dict[str, float] = {}
        for s in sectors:
            others = [o for o in sectors if o != s]
            vt = C_t.loc[s, others].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            vb = C_b.loc[s, others].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
            mt = float(np.mean(vt)) if vt.size else float("nan")
            mb = float(np.mean(vb)) if vb.size else float("nan")
            sector_corr_avg_t[str(s)] = mt
            sector_corr_avg_b[str(s)] = mb
            sector_corr_avg_delta[str(s)] = (mt - mb) if (math.isfinite(mt) and math.isfinite(mb)) else float("nan")

        sector_dist_contrib = _row_norm_offdiag(C_delta)

        return CorrMetrics(
            C_t=C_t,
            C_b=C_b,
            C_delta=C_delta,
            avg_corr_t=avg_corr_t,
            avg_corr_b=avg_corr_b,
            avg_corr_delta=avg_corr_delta,
            distortion_t=distortion_t,
            distortion_prev=distortion_prev,
            delta_distortion=delta_distortion,
            market_mode_strength=market_mode_strength,
            market_mode_loadings=market_mode_loadings,
            sector_corr_avg_t=sector_corr_avg_t,
            sector_corr_avg_b=sector_corr_avg_b,
            sector_corr_avg_delta=sector_corr_avg_delta,
            sector_distortion_contrib=sector_dist_contrib,
        )

    # -----------------------------
    # SPY beta/corr snapshots
    # -----------------------------
    def _beta_corr_to_spy_snapshots(
        self,
        returns: pd.DataFrame,
        sectors: Sequence[str],
        spy: str,
        window_short: int,
        window_long: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns per-sector snapshots:
        - beta_spy_short, corr_spy_short
        - beta_spy_long,  corr_spy_long
        - deltas: short - long
        """
        r_spy = returns[spy]
        out: Dict[str, Dict[str, float]] = {}

        var_spy_s = r_spy.rolling(window_short, min_periods=window_short).var(ddof=0).replace(0.0, np.nan)
        var_spy_l = r_spy.rolling(window_long, min_periods=window_long).var(ddof=0).replace(0.0, np.nan)

        for s in sectors:
            cov_s_s = returns[s].rolling(window_short, min_periods=window_short).cov(r_spy)
            cov_s_l = returns[s].rolling(window_long, min_periods=window_long).cov(r_spy)

            beta_s = cov_s_s / var_spy_s
            beta_l = cov_s_l / var_spy_l

            corr_s = returns[s].rolling(window_short, min_periods=window_short).corr(r_spy)
            corr_l = returns[s].rolling(window_long, min_periods=window_long).corr(r_spy)

            b_s = _series_last(beta_s)
            b_l = _series_last(beta_l)
            c_s = _series_last(corr_s)
            c_l = _series_last(corr_l)

            out[str(s)] = {
                "beta_spy_short": b_s,
                "beta_spy_long": b_l,
                "beta_spy_delta": (b_s - b_l) if (math.isfinite(b_s) and math.isfinite(b_l)) else float("nan"),
                "corr_spy_short": c_s,
                "corr_spy_long": c_l,
                "corr_spy_delta": (c_s - c_l) if (math.isfinite(c_s) and math.isfinite(c_l)) else float("nan"),
            }

        return out

    def _compute_ratio_trend_slopes(
        self,
        prices: pd.DataFrame,
        sectors: Sequence[str],
        spy: str,
    ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        log_spy = np.log(prices[spy].astype(float))

        def _slope_last(y: pd.Series, window: int) -> float:
            z = y.dropna()
            if len(z) < window:
                return float("nan")
            yy = z.iloc[-window:].values.astype(float)
            xx = np.arange(len(yy), dtype=float)
            xx = xx - xx.mean()
            yy = yy - yy.mean()
            denom = float((xx * xx).sum())
            if denom <= 1e-12:
                return float("nan")
            return float((xx * yy).sum() / denom)

        for s in sectors:
            ratio = (np.log(prices[s].astype(float)) - log_spy).replace([np.inf, -np.inf], np.nan)
            out[str(s)] = {
                "trend_ratio_slope_63d": _slope_last(ratio, 63),
                "trend_ratio_slope_126d": _slope_last(ratio, 126),
            }
        return out

    def _apply_target_portfolio_vol(
        self,
        w_raw: pd.Series,
        sector_returns: pd.DataFrame,
        target_vol: float,
        cov_window: int,
        market_state: str = "NORMAL",
    ) -> pd.Series:
        """
        Volatility-target the portfolio weights, then apply regime-aware gross leverage cap.

        Regime caps (configurable in Settings):
          CALM    → max_leverage_calm    (default 3.0x)
          NORMAL  → max_leverage_normal  (default 2.0x)
          TENSION → max_leverage_tension (default 1.2x)
          CRISIS  → max_leverage_crisis  (default 0.0x — flat book)
        """
        w = w_raw.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w.abs().sum() <= 1e-12:
            return w * 0.0

        # Regime-aware gross leverage cap
        _state = str(market_state).upper()
        _cap_map = {
            "CALM":    float(getattr(self.settings, "max_leverage_calm",    3.0)),
            "NORMAL":  float(getattr(self.settings, "max_leverage_normal",  2.0)),
            "TENSION": float(getattr(self.settings, "max_leverage_tension", 1.2)),
            "CRISIS":  float(getattr(self.settings, "max_leverage_crisis",  0.0)),
        }
        max_gross_leverage = _cap_map.get(_state, 2.0)

        # Early exit for CRISIS: return flat book immediately
        if max_gross_leverage <= 0.0:
            self.logger.info(
                "vol_target: CRISIS regime → max_leverage=0.0, returning flat book"
            )
            return w * 0.0

        R = sector_returns.dropna(how="all")
        if len(R) < cov_window:
            gross = float(w.abs().sum())
            return (w / gross) if gross > 1e-12 else (w * 0.0)

        Sigma = R.iloc[-cov_window:].cov(ddof=0).reindex(index=w.index, columns=w.index).fillna(0.0)
        vec = w.values.reshape(-1, 1)
        try:
            port_var_daily = float((vec.T @ Sigma.values @ vec)[0, 0])
        except Exception:
            port_var_daily = float("nan")

        if not math.isfinite(port_var_daily) or port_var_daily <= 1e-12:
            gross = float(w.abs().sum())
            return (w / gross) if gross > 1e-12 else (w * 0.0)

        port_vol_ann = math.sqrt(port_var_daily * 252.0)

        # Vol-targeting scale, capped at the regime-specific gross leverage max
        scale = float(target_vol / port_vol_ann) if port_vol_ann > 1e-12 else 1.0
        scale = max(0.0, min(max_gross_leverage, scale))

        w_scaled = w * scale

        max_single = float(getattr(self.settings, "max_single_name_weight", 0.20))
        if max_single > 0:
            w_scaled = w_scaled.clip(lower=-max_single, upper=max_single)

        gross = float(w_scaled.abs().sum())
        self.logger.debug(
            "vol_target: state=%s scale=%.2f gross=%.2f cap=%.1f",
            _state, scale, gross, max_gross_leverage,
        )
        return (w_scaled / gross) if gross > 1e-12 else (w_scaled * 0.0)

    def get_correlation_regime_timeseries(self) -> pd.DataFrame:
        if self.prices is None:
            raise RuntimeError("Run load() first.")

        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker
        returns = self._log_returns(self.prices[[*sectors, spy]])

        corr_w = _get_int_setting(self.settings, "corr_window", 60)
        base_w = _get_int_setting(self.settings, "corr_baseline_window", 252)

        R = returns[sectors].dropna(how="all")
        if len(R) < base_w + 5:
            return pd.DataFrame(columns=["date", "avg_corr_t", "distortion_t", "market_mode_strength"]).set_index("date")

        last_date = R.index[-1]

        # ── Disk cache: avoid recomputing ~2300 iterations on every startup ──
        cache_dir = self.settings.project_root / "data" / "cache"
        cache_path = cache_dir / "corr_regime_ts.parquet"
        try:
            if cache_path.exists():
                cached = pd.read_parquet(cache_path)
                if not cached.empty and "date" in cached.columns:
                    cached_last = pd.Timestamp(cached["date"].max())
                    if cached_last >= pd.Timestamp(last_date) - pd.Timedelta(days=1):
                        self.logger.info(
                            "corr_regime_ts: cache HIT (last=%s, data=%s)",
                            cached_last.date(), last_date.date() if hasattr(last_date, "date") else last_date,
                        )
                        cached["date"] = pd.to_datetime(cached["date"])
                        return cached.set_index("date").sort_index()
                    else:
                        self.logger.info(
                            "corr_regime_ts: cache STALE (cached=%s, need=%s) — recomputing",
                            cached_last.date(), last_date.date() if hasattr(last_date, "date") else last_date,
                        )
        except Exception as _ce:
            self.logger.debug("corr_regime_ts: cache read failed: %s", _ce)

        self.logger.info("corr_regime_ts: computing %d dates (this may take ~15-20s)...", len(R) - base_w)

        rows: List[Dict[str, Any]] = []
        for t in range(base_w, len(R)):
            hist = R.iloc[: t + 1]
            try:
                cm = self._compute_corr_metrics(hist, sectors)
                rows.append(
                    {
                        "date": hist.index[-1],
                        "avg_corr_t": cm.avg_corr_t,
                        "distortion_t": cm.distortion_t,
                        "market_mode_strength": cm.market_mode_strength,
                    }
                )
            except Exception:
                continue

        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=["date", "avg_corr_t", "distortion_t", "market_mode_strength"]).set_index("date")

        # ── Write cache ──
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            out.to_parquet(cache_path, index=False)
            self.logger.info("corr_regime_ts: cached %d rows to %s", len(out), cache_path.name)
        except Exception as _we:
            self.logger.debug("corr_regime_ts: cache write failed: %s", _we)

        out["date"] = pd.to_datetime(out["date"])
        return out.set_index("date").sort_index()

    # -----------------------------
    # Risk decomposition
    # -----------------------------
    def _portfolio_risk_decomposition(
        self,
        sector_returns: pd.DataFrame,
        w_final: pd.Series,
        cov_window: int,
    ) -> Dict[str, float]:
        """
        Portfolio variance decomposition:
            Var = w' Σ w
            diag_var = w' diag(Σ) w
            offdiag_var = w' (Σ - diag(Σ)) w
        """
        R = sector_returns.dropna(how="all")
        if len(R) < cov_window:
            return {
                "port_var": float("nan"),
                "port_vol": float("nan"),
                "port_diag_var": float("nan"),
                "port_offdiag_var": float("nan"),
                "port_offdiag_share": float("nan"),
            }

        Sigma = R.iloc[-cov_window:].cov(ddof=0).fillna(0.0)
        Sigma = Sigma.reindex(index=w_final.index, columns=w_final.index).fillna(0.0)

        w = w_final.values.reshape(-1, 1)
        S = Sigma.values.astype(float)

        try:
            port_var = float((w.T @ S @ w)[0, 0])
            D = np.diag(np.diag(S))
            O = S - D
            diag_var = float((w.T @ D @ w)[0, 0])
            off_var = float((w.T @ O @ w)[0, 0])

            port_vol = float(np.sqrt(port_var * 252.0)) if port_var > 0 else float("nan")
            off_share = float(off_var / port_var) if abs(port_var) > 1e-12 else float("nan")

            return {
                "port_var": port_var,
                "port_vol": port_vol,
                "port_diag_var": diag_var,
                "port_offdiag_var": off_var,
                "port_offdiag_share": off_share,
            }
        except Exception:
            return {
                "port_var": float("nan"),
                "port_vol": float("nan"),
                "port_diag_var": float("nan"),
                "port_offdiag_var": float("nan"),
                "port_offdiag_share": float("nan"),
            }

    def _decision_sort_rank(self, decision_label: str) -> int:
        rank_map = {
            "ENTER": 0,
            "WATCH": 1,
            "REDUCE": 2,
            "AVOID": 3,
        }
        return rank_map.get(str(decision_label).upper().strip(), 9)

    def _compute_decision_fields(
        self,
        *,
        direction: str,
        conviction_score: float,
        mc_score_raw: float,
        market_state: str,
        regime_transition_score: float,
        crisis_probability: float,
        mss_score: float,
        stf_score: float,
        interpretation: str,
        action_bias: str,
        risk_label: str,
    ) -> Dict[str, Any]:
        """
        Convert model outputs into PM-facing decision fields.

        Returns:
        - decision_score
        - decision_label
        - size_bucket
        - entry_quality
        - risk_override
        - pm_note
        """
        direction = str(direction or "NEUTRAL").upper().strip()
        market_state = str(market_state or "NORMAL").upper().strip()
        action_bias = str(action_bias or "").upper().strip()

        if direction not in {"LONG", "SHORT"}:
            return {
                "decision_score": 0.0,
                "decision_label": "AVOID",
                "size_bucket": "ZERO",
                "entry_quality": "NO_SIGNAL",
                "risk_override": "NO_DIRECTION",
                "pm_note": "No directional signal: residual dislocation not strong enough for action.",
            }

        conviction_norm = max(0.0, min(1.0, float(conviction_score) / 100.0)) if math.isfinite(conviction_score) else 0.0
        mc_norm = max(0.0, min(1.0, float(mc_score_raw))) if math.isfinite(mc_score_raw) else 0.0
        regime_penalty = 0.0

        if market_state == "TENSION":
            regime_penalty = 0.18
        elif market_state == "CRISIS":
            regime_penalty = 0.45

        transition_penalty = 0.15 * max(0.0, min(1.0, float(regime_transition_score))) if math.isfinite(regime_transition_score) else 0.0
        crisis_penalty = 0.20 * max(0.0, min(1.0, float(crisis_probability))) if math.isfinite(crisis_probability) else 0.0
        macro_penalty = 0.20 * max(0.0, min(1.0, float(mss_score))) if math.isfinite(mss_score) else 0.0
        structural_penalty = 0.20 * max(0.0, min(1.0, float(stf_score))) if math.isfinite(stf_score) else 0.0

        decision_score = (
            0.45 * mc_norm
            + 0.35 * conviction_norm
            - regime_penalty
            - transition_penalty
            - crisis_penalty
            - macro_penalty
            - structural_penalty
        )
        decision_score = max(0.0, min(1.0, decision_score))

        # --------------------------------------------------
        # Decision label
        # --------------------------------------------------
        if market_state == "CRISIS" or crisis_probability >= 0.75:
            decision_label = "AVOID"
        elif mc_norm < 0.15:
            decision_label = "AVOID"
        elif mss_score >= 0.75 or stf_score >= 0.75:
            decision_label = "AVOID"
        elif market_state == "TENSION":
            if mc_norm >= 0.45 and conviction_norm >= 0.50 and mss_score < 0.55 and stf_score < 0.55:
                decision_label = "REDUCE"
            else:
                decision_label = "AVOID"
        elif mc_norm >= 0.55 and conviction_norm >= 0.55 and mss_score < 0.45 and stf_score < 0.45:
            decision_label = "ENTER"
        elif mc_norm >= 0.30 and conviction_norm >= 0.40:
            decision_label = "WATCH"
        else:
            decision_label = "REDUCE"

        # --------------------------------------------------
        # Size bucket
        # --------------------------------------------------
        if decision_label == "ENTER":
            if decision_score >= 0.65:
                size_bucket = "FULL"
            elif decision_score >= 0.50:
                size_bucket = "MEDIUM"
            else:
                size_bucket = "SMALL"
        elif decision_label == "WATCH":
            size_bucket = "SMALL"
        elif decision_label == "REDUCE":
            size_bucket = "SMALL"
        else:
            size_bucket = "ZERO"

        # --------------------------------------------------
        # Entry quality
        # --------------------------------------------------
        if mc_norm >= 0.65 and mss_score < 0.35 and stf_score < 0.35:
            entry_quality = "HIGH_QUALITY"
        elif mc_norm >= 0.40 and mss_score < 0.55 and stf_score < 0.55:
            entry_quality = "MEDIUM_QUALITY"
        elif mc_norm >= 0.20:
            entry_quality = "SPECULATIVE"
        else:
            entry_quality = "POOR_QUALITY"

        # --------------------------------------------------
        # Risk override
        # --------------------------------------------------
        if market_state == "CRISIS":
            risk_override = "CRISIS_OVERRIDE"
        elif crisis_probability >= 0.60:
            risk_override = "HIGH_CRISIS_RISK"
        elif mss_score >= 0.60:
            risk_override = "MACRO_OVERRIDE"
        elif stf_score >= 0.60:
            risk_override = "STRUCTURAL_OVERRIDE"
        elif regime_transition_score >= 0.60:
            risk_override = "TRANSITION_OVERRIDE"
        else:
            risk_override = "NONE"

        # --------------------------------------------------
        # PM note
        # --------------------------------------------------
        if decision_label == "ENTER":
            pm_note = (
                f"{interpretation}. "
                f"Signal is actionable with {risk_label.lower()} and {action_bias.lower()} bias."
            )
        elif decision_label == "WATCH":
            pm_note = (
                f"{interpretation}. "
                f"Interesting setup, but not yet clean enough for full commitment."
            )
        elif decision_label == "REDUCE":
            pm_note = (
                f"{interpretation}. "
                f"Signal exists, but regime / macro / structural risk suggests smaller size."
            )
        else:
            pm_note = (
                f"{interpretation}. "
                f"Avoid active mean-reversion exposure under current conditions."
            )

        return {
            "decision_score": float(decision_score),
            "decision_label": decision_label,
            "size_bucket": size_bucket,
            "entry_quality": entry_quality,
            "risk_override": risk_override,
            "pm_note": pm_note,
        }
    # -----------------------------
    # Main scoring / master DF
    # -----------------------------
    def calculate_conviction_score(self) -> pd.DataFrame:
        """
        Build the sector-level decision-support master table.

        Returns
        -------
        pd.DataFrame
            Sector-level decision-support table including:
            - statistical / macro / fundamental / vol scores
            - attribution and mispricing confidence
            - market regime context
            - delta-1 portfolio construction
            - synthetic greeks and exposure proxies
            - execution regime and portfolio risk decomposition
        """
        if self.prices is None:
            raise RuntimeError("Run load() first.")

        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker
        tnx = self.settings.macro_tickers["TNX_10Y"]
        dxy = self.settings.macro_tickers["DXY_USD"]
        vix = self.settings.vol_tickers["VIX"]
        hyg = self.settings.credit_tickers["HYG"]
        ief = self.settings.credit_tickers["IEF"]

        prices = self.prices.copy()

        required_cols = list(sectors) + [spy, tnx, dxy, vix, hyg, ief]
        missing = [c for c in required_cols if c not in prices.columns]
        if missing:
            raise ValueError(f"Missing required tickers in prices panel: {missing}")

        latest_date = prices.index[-1]

        # ==========================================================
        # Core analytics blocks
        # ==========================================================
        returns = self._log_returns(prices)

        ewma_vol = self._compute_ewma_vol_panel(returns[list(sectors) + [spy]])
        self.ewma_vol = ewma_vol

        dispersion_df = self._compute_realized_dispersion(returns)
        self.dispersion_df = dispersion_df

        corr_metrics = self._compute_corr_metrics(returns, sectors)
        self.corr_metrics = corr_metrics
        self.corr_matrix_current = corr_metrics.C_t
        self.corr_matrix_baseline = corr_metrics.C_b
        self.corr_matrix_delta = corr_metrics.C_delta

        rel_returns = returns[list(sectors)].subtract(returns[spy], axis=0)
        resid_ret, resid_level, resid_z = self._oos_rolling_pca_residuals(rel_returns)
        self.residual_returns = resid_ret
        self.residual_levels = resid_level
        self.residual_zscores = resid_z

        betas = self._compute_macro_betas(returns)
        self.macro_betas = betas

        # ==========================================================
        # Market-wide context
        # ==========================================================
        credit_spread = (np.log(prices[hyg]) - np.log(prices[ief])).rename("log_hyg_minus_ief")
        credit_z = _zscore(credit_spread, window=self.settings.macro_window).rename("credit_z")
        latest_credit_z = _series_last(credit_z)
        credit_stress = bool(
            math.isfinite(latest_credit_z) and latest_credit_z < self.settings.credit_stress_z
        )

        vix_level = prices[vix]
        vix_pct = _rolling_percentile_of_last(
            vix_level,
            window=max(252, self.settings.pca_window),
        )
        latest_vix = _series_last(vix_level)
        latest_vix_pct = _series_last(vix_pct)

        regime = self._compute_regime_metrics(
            avg_corr_t=corr_metrics.avg_corr_t,
            avg_corr_delta=corr_metrics.avg_corr_delta,
            distortion_t=corr_metrics.distortion_t,
            delta_distortion=corr_metrics.delta_distortion,
            market_mode_strength=corr_metrics.market_mode_strength,
            vix_level=latest_vix,
            vix_percentile=latest_vix_pct,
            credit_z=latest_credit_z,
        )

        # ==========================================================
        # Fundamentals context
        # ==========================================================
        quote_map = self._build_symbol_to_quote()

        spy_holdings = self._get_holdings_for_etf(spy)
        spy_val_h = self._aggregate_portfolio_pe(spy_holdings, quote_map)

        if float(spy_val_h.get("covered_weight", 0.0)) < 0.15:
            spy_val = self._etf_proxy_valuation(spy, quote_map)
            spy_val["covered_weight"] = float(spy_val_h.get("covered_weight", 0.0))
            spy_val["fund_source"] = "ETF_PROXY"
        else:
            spy_val = spy_val_h
            spy_val["fund_source"] = "HOLDINGS"

        spy_pe = float(spy_val["pe_portfolio"])
        spy_ey = float(spy_val["earnings_yield"])

        cyclicals = {
            self.settings.sector_tickers["Energy"],
            self.settings.sector_tickers["Financials"],
            self.settings.sector_tickers["Industrials"],
            self.settings.sector_tickers["Materials"],
            self.settings.sector_tickers["Consumer Discretionary"],
        }
        defensives = {
            self.settings.sector_tickers["Utilities"],
            self.settings.sector_tickers["Consumer Staples"],
            self.settings.sector_tickers["Health Care"],
        }

        # ==========================================================
        # FJS pre-computation (cross-sectional fundamentals scoring)
        # ==========================================================
        # Regime-adaptive Z-score entry threshold — higher in stressed regimes
        # to avoid false signals when sector correlations break down.
        _zscore_threshold_map = {
            "CALM":    float(getattr(self.settings, "zscore_threshold_calm",    0.75)),
            "NORMAL":  float(getattr(self.settings, "zscore_threshold_normal",  0.90)),
            "TENSION": float(getattr(self.settings, "zscore_threshold_tension", 1.25)),
            "CRISIS":  float(getattr(self.settings, "zscore_threshold_crisis",  9.99)),
        }
        _zscore_threshold = _zscore_threshold_map.get(str(regime.market_state).upper(), 0.90)
        self.logger.info(
            "Z-score threshold: %.2f (regime=%s)", _zscore_threshold, regime.market_state
        )

        # Pre-compute directions from z-scores so FJS can use them
        _direction_map: Dict[str, str] = {}
        for _s in sectors:
            _z = float(resid_z[_s].iloc[-1]) if _s in resid_z.columns and len(resid_z) else float("nan")
            _direction_map[_s] = ("LONG" if _z < 0 else "SHORT") if (math.isfinite(_z) and abs(_z) >= _zscore_threshold) else "NEUTRAL"

        # Current TNX level (yield in %)
        _latest_tnx = float(prices[tnx].iloc[-1]) if tnx in prices.columns else float("nan")
        # Normalise: FMP gives level in % (e.g. 4.5 means 4.5%) — convert to decimal
        if math.isfinite(_latest_tnx) and _latest_tnx > 1.0:
            _latest_tnx = _latest_tnx / 100.0

        _holdings_df = self.weights if self.weights is not None else pd.DataFrame()
        _fund_df = self.fundamentals if self.fundamentals is not None else pd.DataFrame()
        _ext_df = self.ext_fundamentals if self.ext_fundamentals is not None else pd.DataFrame()

        _sector_fundamentals: Dict[str, SectorFundamentals] = {}
        for _s in sectors:
            try:
                _sector_fundamentals[_s] = aggregate_sector_fundamentals(
                    etf=_s,
                    holdings_df=_holdings_df,
                    fund_df=_fund_df,
                    ext_fund_df=_ext_df,
                )
            except Exception as _exc:
                self.logger.debug("FundamentalsEngine: aggregate failed for %s: %s", _s, _exc)
                _sector_fundamentals[_s] = SectorFundamentals(etf=_s)

        _fjs_results: dict = {}
        try:
            _fjs_results = compute_all_sectors_fjs(
                sector_fundamentals=_sector_fundamentals,
                direction_map=_direction_map,
                tnx_10y=_latest_tnx,
                credit_z=latest_credit_z,
                regime=regime.market_state,
            )
            self.logger.info("FJS computed for %d sectors", len(_fjs_results))
        except Exception as _exc:
            self.logger.warning("FJS computation failed, falling back to PE scoring: %s", _exc)

        # ==========================================================
        # Sector rows
        # ==========================================================
        rows: List[Dict[str, Any]] = []

        for s in sectors:
            row: Dict[str, Any] = {
                "date": latest_date,
                "sector_ticker": s,
                "sector_name": self.settings.canonical_sector_by_ticker().get(s, s),
            }

            # ------------------------------------------------------
            # Residual / statistical layer
            # ------------------------------------------------------
            z = float(resid_z[s].iloc[-1]) if s in resid_z.columns and len(resid_z) else float("nan")
            mispricing = (
                float(resid_level[s].iloc[-1])
                if s in resid_level.columns and len(resid_level)
                else float("nan")
            )

            row["pca_residual_level"] = mispricing
            row["pca_residual_z"] = z

            if math.isfinite(z) and abs(z) >= _zscore_threshold:
                row["direction"] = "LONG" if z < 0 else "SHORT"
            else:
                row["direction"] = "NEUTRAL"
            row["zscore_threshold_used"] = _zscore_threshold

            hl = estimate_ar1_half_life(
                resid_level[s].iloc[-max(400, self.settings.pca_window):],
                min_obs=120,
            )
            row["half_life_days_est"] = hl

            # ------------------------------------------------------
            # Volatility / hedge ratio
            # ------------------------------------------------------
            vol_s = float(ewma_vol[s].iloc[-1]) if s in ewma_vol.columns else float("nan")
            vol_spy = float(ewma_vol[spy].iloc[-1]) if spy in ewma_vol.columns else float("nan")
            hr_s = (
                float(vol_spy / vol_s)
                if (math.isfinite(vol_s) and vol_s > 0 and math.isfinite(vol_spy))
                else float("nan")
            )

            row["ewma_vol_ann"] = vol_s
            row["hedge_ratio"] = hr_s

            # ------------------------------------------------------
            # Market dispersion snapshot
            # ------------------------------------------------------
            disp_ratio = (
                float(dispersion_df["dispersion_ratio"].iloc[-1])
                if not dispersion_df.empty
                else float("nan")
            )
            disp_z = (
                float(dispersion_df["dispersion_ratio_z"].iloc[-1])
                if not dispersion_df.empty
                else float("nan")
            )

            row["market_dispersion_ratio"] = disp_ratio
            row["market_dispersion_z"] = disp_z

            # ------------------------------------------------------
            # Macro betas
            # ------------------------------------------------------
            try:
                b_tnx = (
                    float(betas.loc[(latest_date, s), "beta_tnx"])
                    if (latest_date, s) in betas.index
                    else float("nan")
                )
                b_dxy = (
                    float(betas.loc[(latest_date, s), "beta_dxy"])
                    if (latest_date, s) in betas.index
                    else float("nan")
                )
            except Exception:
                b_tnx, b_dxy = float("nan"), float("nan")

            beta_mag = math.sqrt(
                (b_tnx if math.isfinite(b_tnx) else 0.0) ** 2
                + (b_dxy if math.isfinite(b_dxy) else 0.0) ** 2
            )

            row["beta_tnx_60d"] = b_tnx
            row["beta_dxy_60d"] = b_dxy
            row["beta_mag"] = float(beta_mag)

            # ------------------------------------------------------
            # Correlation structure
            # ------------------------------------------------------
            row["market_mode_loading"] = float(corr_metrics.market_mode_loadings.get(s, float("nan")))
            row["sector_corr_avg_t"] = float(corr_metrics.sector_corr_avg_t.get(s, float("nan")))
            row["sector_corr_avg_b"] = float(corr_metrics.sector_corr_avg_b.get(s, float("nan")))
            row["sector_corr_avg_delta"] = float(corr_metrics.sector_corr_avg_delta.get(s, float("nan")))
            row["sector_corr_dist_contrib"] = float(
                corr_metrics.sector_distortion_contrib.get(s, float("nan"))
            )

            # ------------------------------------------------------
            # Fundamentals
            # ------------------------------------------------------
            h = self._get_holdings_for_etf(s)
            val_h = self._aggregate_portfolio_pe(h, quote_map)

            if float(val_h.get("covered_weight", 0.0)) < 0.15:
                val = self._etf_proxy_valuation(s, quote_map)
                val["covered_weight"] = float(val_h.get("covered_weight", 0.0))
                fund_source = "ETF_PROXY"
            else:
                val = val_h
                fund_source = "HOLDINGS"

            pe = float(val["pe_portfolio"])
            ey = float(val["earnings_yield"])
            neg_w = float(val["neg_or_missing_weight"])
            eps_med = float(val["weighted_median_eps"])

            row["fund_source"] = fund_source
            row["pe_sector_portfolio"] = pe
            row["earnings_yield_sector"] = ey
            row["fund_covered_weight"] = float(val.get("covered_weight", 0.0))
            row["neg_or_missing_earnings_weight"] = neg_w
            row["eps_weighted_median"] = eps_med

            rel_pe = (
                pe / spy_pe
                if (math.isfinite(pe) and pe > 0 and math.isfinite(spy_pe) and spy_pe > 0)
                else float("nan")
            )
            rel_ey = (
                ey / spy_ey
                if (math.isfinite(ey) and math.isfinite(spy_ey) and spy_ey > 0)
                else float("nan")
            )

            row["rel_pe_vs_spy"] = rel_pe
            row["rel_earnings_yield_vs_spy"] = rel_ey

            # ------------------------------------------------------
            # Layer scoring
            # ------------------------------------------------------
            z_abs = abs(z) if math.isfinite(z) else 0.0
            # Normalize strength from threshold (entry point) to 2.5σ (max conviction)
            _z_max = max(_zscore_threshold + 0.01, 2.5)
            z_strength = _clip01((z_abs - _zscore_threshold) / (_z_max - _zscore_threshold))
            z_points = self.settings.points_stat * 0.75 * z_strength

            disp_strength = _clip01((disp_z + 1.0) / 3.0) if math.isfinite(disp_z) else 0.0
            disp_points = self.settings.points_stat * 0.25 * disp_strength

            hl_factor = 1.0
            if math.isfinite(hl):
                if hl < 5:
                    hl_factor = 0.75
                elif 5 <= hl <= 80:
                    hl_factor = 0.6 + 0.4 * math.exp(-abs(hl - 35.0) / 35.0)
                else:
                    hl_factor = 0.55

            stat_score = (z_points + disp_points) * hl_factor
            stat_score = max(0.0, min(float(self.settings.points_stat), stat_score))

            beta_threshold = 1.0
            beta_factor = math.exp(-beta_mag / beta_threshold) if math.isfinite(beta_mag) else 0.5

            macro_factor = beta_factor
            if credit_stress and row["direction"] != "NEUTRAL":
                if row["direction"] == "LONG" and s in cyclicals:
                    macro_factor *= 0.65
                if row["direction"] == "SHORT" and s in defensives:
                    macro_factor *= 0.70
                macro_factor *= 0.85

            if regime.market_state == "TENSION":
                macro_factor *= 0.85
            elif regime.market_state == "CRISIS":
                macro_factor *= 0.60

            macro_score = self.settings.points_macro * max(0.0, min(1.0, macro_factor))

            fund_score = 0.0
            if s in _fjs_results:
                # ── FundamentalsEngine (institutional FJS scoring) ──────────
                _fjs = _fjs_results[s]
                fund_score = self.settings.points_fund * max(0.0, min(1.0, _fjs.fjs))
                # Store rich FJS details for UI and attribution
                row["fjs_engine_score"]     = round(_fjs.fjs, 4)
                row["fjs_delta"]            = round(_fjs.delta, 4) if math.isfinite(_fjs.delta) else float("nan")
                row["fjs_z_delta"]          = round(_fjs.z_delta, 4) if math.isfinite(_fjs.z_delta) else float("nan")
                row["fjs_m_obs"]            = round(_fjs.m_obs, 3) if math.isfinite(_fjs.m_obs) else float("nan")
                row["fjs_m_hat"]            = round(_fjs.m_hat, 3) if math.isfinite(_fjs.m_hat) else float("nan")
                row["fjs_multiple_used"]    = _fjs.primary_multiple_used
                row["fjs_earnings_quality"] = round(_fjs.earnings_quality_score, 3)
                row["fjs_revision_score"]   = round(_fjs.revision_score, 3)
            elif row["direction"] != "NEUTRAL" and math.isfinite(rel_pe):
                # ── Fallback: simple relative-PE scoring ───────────────────
                if row["direction"] == "LONG":
                    raw = (1.0 - rel_pe) / 0.25
                else:
                    raw = (rel_pe - 1.0) / 0.25
                fund_score = self.settings.points_fund * _clip01(raw)
                row["fjs_engine_score"] = float("nan")
                row["fjs_multiple_used"] = "pe_fallback"

            # Earnings quality penalties (applied regardless of FJS / PE path)
            if math.isfinite(pe) and pe > self.settings.pe_extreme:
                fund_score *= 0.60
            if math.isfinite(neg_w) and neg_w > self.settings.neg_earnings_weight_cap:
                fund_score *= 0.50
            if not math.isfinite(pe) or pe <= 0:
                fund_score *= 0.35

            vix_factor = 1.0
            if math.isfinite(latest_vix):
                if latest_vix >= self.settings.vix_level_hard:
                    vix_factor = 0.25
                elif latest_vix >= self.settings.vix_level_soft:
                    vix_factor = 0.55

            if math.isfinite(latest_vix_pct) and latest_vix_pct >= self.settings.vix_percentile_hard:
                vix_factor *= 0.60

            hr_factor = 0.75
            if math.isfinite(hr_s) and hr_s > 0:
                hr_factor = 0.5 + 0.5 * math.exp(-abs(math.log(hr_s)) / math.log(2.0))

            vol_score = self.settings.points_vol * max(0.0, min(1.0, vix_factor * hr_factor))

            conviction = float(stat_score + macro_score + fund_score + vol_score)
            conviction = max(0.0, min(100.0, conviction))

            # ── MR Whitelist boost/penalty (CALIBRATED) ──────────────
            _mr_whitelist = set(
                t.strip() for t in getattr(self.settings, "sector_mr_whitelist", "XLC,XLF,XLI,XLU").split(",")
            )
            _mr_filter = getattr(self.settings, "sector_mr_filter_enabled", True)
            if _mr_filter and row["direction"] != "NEUTRAL":
                if s not in _mr_whitelist:
                    _penalty = getattr(self.settings, "non_whitelist_penalty", 0.3)
                    conviction *= _penalty

            # ── Regime-adaptive conviction scaling (CALIBRATED) ──────
            # Research: CALM Sharpe=+0.66, TENSION=+0.68, CRISIS=-0.78
            _regime_scale_map = {
                "CALM":    getattr(self.settings, "regime_conviction_scale_calm", 1.3),
                "NORMAL":  getattr(self.settings, "regime_conviction_scale_normal", 1.0),
                "TENSION": getattr(self.settings, "regime_conviction_scale_tension", 0.5),
                "CRISIS":  getattr(self.settings, "regime_conviction_scale_crisis", 0.0),
            }
            _regime_scale = float(_regime_scale_map.get(
                str(regime.market_state).upper(), 1.0
            ))
            conviction *= _regime_scale
            conviction = max(0.0, min(100.0, conviction))

            row["score_stat"] = float(stat_score)
            row["score_macro"] = float(macro_score)
            row["score_fund"] = float(fund_score)
            row["score_vol"] = float(vol_score)
            row["mr_whitelist_member"] = s in _mr_whitelist
            row["regime_conviction_scale"] = _regime_scale
            row["conviction_score"] = float(conviction)

            # ------------------------------------------------------
            # Execution directives
            # ------------------------------------------------------
            if row["direction"] == "LONG":
                row["trade_leg_sector"] = (
                    f"BUY {s} notional={hr_s:.2f}" if math.isfinite(hr_s) else f"BUY {s}"
                )
                row["trade_leg_spy"] = "SELL SPY notional=1.00"
            elif row["direction"] == "SHORT":
                row["trade_leg_sector"] = (
                    f"SELL {s} notional={hr_s:.2f}" if math.isfinite(hr_s) else f"SELL {s}"
                )
                row["trade_leg_spy"] = "BUY SPY notional=1.00"
            else:
                row["trade_leg_sector"] = "NO TRADE"
                row["trade_leg_spy"] = "NO TRADE"

            rv_align = 0.0
            if row["direction"] == "LONG":
                rv_align = z_strength * _clip01((1.0 - rel_pe) / 0.25) if math.isfinite(rel_pe) else 0.0
            elif row["direction"] == "SHORT":
                rv_align = z_strength * _clip01((rel_pe - 1.0) / 0.25) if math.isfinite(rel_pe) else 0.0

            row["rv_alignment"] = float(rv_align)

            rows.append(row)

        master = (
            pd.DataFrame(rows)
            .sort_values(["conviction_score", "pca_residual_z"], ascending=[False, True])
            .reset_index(drop=True)
        )

        # ==========================================================
        # Global market context
        # ==========================================================
        master["credit_z"] = latest_credit_z
        master["credit_stress"] = credit_stress
        master["vix_level"] = latest_vix
        master["vix_percentile"] = latest_vix_pct
        master["spy_pe_portfolio"] = spy_pe
        master["spy_earnings_yield"] = spy_ey
        master["spy_fund_source"] = spy_val.get("fund_source", "UNKNOWN")

        # GLD / TLT macro momentum signals (60d return)
        # Positive gold_momentum = risk-off sentiment
        # Positive bond_momentum = flight to safety
        _gld_ticker = self.settings.macro_tickers.get("GLD", "GLD")
        _tlt_ticker = self.settings.macro_tickers.get("TLT", "TLT")
        _macro_mom_window = 60

        if _gld_ticker in prices.columns:
            _gld_ret = prices[_gld_ticker].pct_change(_macro_mom_window)
            master["gold_momentum"] = float(_gld_ret.dropna().iloc[-1]) if len(_gld_ret.dropna()) > 0 else float("nan")
        else:
            master["gold_momentum"] = float("nan")

        if _tlt_ticker in prices.columns:
            _tlt_ret = prices[_tlt_ticker].pct_change(_macro_mom_window)
            master["bond_momentum"] = float(_tlt_ret.dropna().iloc[-1]) if len(_tlt_ret.dropna()) > 0 else float("nan")
        else:
            master["bond_momentum"] = float("nan")

        master["avg_corr_t"] = corr_metrics.avg_corr_t
        master["avg_corr_b"] = corr_metrics.avg_corr_b
        master["avg_corr_delta"] = corr_metrics.avg_corr_delta
        master["corr_matrix_dist_t"] = corr_metrics.distortion_t
        master["corr_matrix_dist_prev"] = corr_metrics.distortion_prev
        master["delta_corr_dist"] = corr_metrics.delta_distortion
        master["market_mode_strength"] = corr_metrics.market_mode_strength

        master["market_state"] = regime.market_state
        master["state_bias"] = regime.state_bias
        master["mean_reversion_allowed"] = regime.mean_reversion_allowed
        master["regime_alert"] = regime.regime_alert
        master["regime_vol_score"] = regime.regime_vol_score
        master["regime_credit_score"] = regime.regime_credit_score
        master["regime_corr_score"] = regime.regime_corr_score
        master["regime_transition_score"] = regime.regime_transition_score
        master["transition_probability"] = regime.transition_probability
        master["crisis_probability"] = regime.crisis_probability

        # ==========================================================
        # SPY beta / corr snapshots
        # ==========================================================
        beta_snap = self._beta_corr_to_spy_snapshots(
            returns=returns,
            sectors=sectors,
            spy=spy,
            window_short=self.settings.macro_window,
            window_long=max(252, self.settings.pca_window),
        )

        trend_slopes = self._compute_ratio_trend_slopes(
            prices=prices,
            sectors=sectors,
            spy=spy,
        )

        master["beta_spy_60d"] = master["sector_ticker"].map(
            lambda x: beta_snap.get(x, {}).get("beta_spy_short", float("nan"))
        )
        master["beta_spy_252d"] = master["sector_ticker"].map(
            lambda x: beta_snap.get(x, {}).get("beta_spy_long", float("nan"))
        )
        master["beta_spy_delta"] = master["sector_ticker"].map(
            lambda x: beta_snap.get(x, {}).get("beta_spy_delta", float("nan"))
        )

        master["corr_to_spy_60d"] = master["sector_ticker"].map(
            lambda x: beta_snap.get(x, {}).get("corr_spy_short", float("nan"))
        )
        master["corr_to_spy_252d"] = master["sector_ticker"].map(
            lambda x: beta_snap.get(x, {}).get("corr_spy_long", float("nan"))
        )
        master["corr_to_spy_delta"] = master["sector_ticker"].map(
            lambda x: beta_snap.get(x, {}).get("corr_spy_delta", float("nan"))
        )

        master["trend_ratio_slope_63d"] = master["sector_ticker"].map(
            lambda x: trend_slopes.get(x, {}).get("trend_ratio_slope_63d", float("nan"))
        )
        master["trend_ratio_slope_126d"] = master["sector_ticker"].map(
            lambda x: trend_slopes.get(x, {}).get("trend_ratio_slope_126d", float("nan"))
        )

        # ==========================================================
        # Attribution / MC
        # ==========================================================
        attrib_rows: List[Dict[str, Any]] = []
        for _, r in master.iterrows():
            a = compute_attribution_row(
                r.to_dict(),
                vix_soft=self.settings.vix_level_soft,
                vix_hard=self.settings.vix_level_hard,
                sds_z_lo=float(getattr(self.settings, "sds_z_lo", 0.75)),
                sds_z_hi=float(getattr(self.settings, "sds_z_hi", 2.75)),
                mss_beta_mag_norm=float(getattr(self.settings, "mss_beta_mag_norm", 1.25)),
                mss_beta_delta_norm=float(getattr(self.settings, "mss_beta_delta_norm", 0.35)),
                mss_corr_delta_norm=float(getattr(self.settings, "mss_corr_delta_norm", 0.20)),
                stf_slope_norm=float(getattr(self.settings, "stf_slope_norm", 0.12)),
            )
            attrib_rows.append(
                {
                    "sector_ticker": r["sector_ticker"],
                    "sds_score": a.sds,
                    "fjs_score": a.fjs,
                    "mss_score": a.mss,
                    "stf_score": a.stf,
                    "mc_score_raw": a.mc,
                    "beta_instability_score": a.beta_instability,
                    "corr_instability_score": a.corr_instability,
                    "corr_shift_score": a.corr_shift_score,
                    "trend_ratio_slope_63d": a.trend_ratio_slope_63d,
                    "trend_ratio_slope_126d": a.trend_ratio_slope_126d,
                    "dislocation_label": a.dislocation_label,
                    "fundamental_label": a.fundamental_label,
                    "macro_label": a.macro_label,
                    "structural_label": a.structural_label,
                    "mc_label": a.mc_label,
                    "action_bias": a.action_bias,
                    "risk_label": a.risk_label,
                    "interpretation": a.interpretation,
                    "explanation_tags": ", ".join(a.explanation_tags),
                }
            )

        attrib_df = pd.DataFrame(attrib_rows)
        master = master.merge(attrib_df, on="sector_ticker", how="left")

        mc_floor_trade = float(getattr(self.settings, "mc_floor_trade", 0.10))
        master["mc_score"] = (master["mc_score_raw"].astype(float) * 100.0).clip(0.0, 100.0)
        master["is_actionable_mc"] = master["mc_score_raw"].astype(float) >= mc_floor_trade

        # ==========================================================
        # Decision Layer (PM-facing action classification)
        # ==========================================================
        decision_rows: List[Dict[str, Any]] = []
        for _, r in master.iterrows():
            d = self._compute_decision_fields(
                direction=str(r.get("direction", "NEUTRAL")),
                conviction_score=float(r.get("conviction_score", float("nan"))),
                mc_score_raw=float(r.get("mc_score_raw", float("nan"))),
                market_state=str(r.get("market_state", "NORMAL")),
                regime_transition_score=float(r.get("regime_transition_score", float("nan"))),
                crisis_probability=float(r.get("crisis_probability", float("nan"))),
                mss_score=float(r.get("mss_score", float("nan"))),
                stf_score=float(r.get("stf_score", float("nan"))),
                interpretation=str(r.get("interpretation", "Mixed Signal / Requires PM Judgement")),
                action_bias=str(r.get("action_bias", "SELECTIVE")),
                risk_label=str(r.get("risk_label", "Moderate Risk")),
            )
            decision_rows.append(
                {
                    "sector_ticker": r["sector_ticker"],
                    "decision_score": d["decision_score"],
                    "decision_label": d["decision_label"],
                    "size_bucket": d["size_bucket"],
                    "entry_quality": d["entry_quality"],
                    "risk_override": d["risk_override"],
                    "pm_note": d["pm_note"],
                    "decision_rank": self._decision_sort_rank(d["decision_label"]),
                }
            )

        decision_df = pd.DataFrame(decision_rows)
        master = master.merge(decision_df, on="sector_ticker", how="left")

        # ==========================================================
        # Delta-1 portfolio construction
        # ==========================================================
        idx = master.set_index("sector_ticker")

        z_s = idx["pca_residual_z"].astype(float)
        mc = idx["mc_score_raw"].astype(float).clip(0.0, 1.0)

        signal_raw = (-z_s.clip(-3.0, 3.0) * mc).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if regime.market_state == "TENSION":
            signal_raw = signal_raw * 0.70
        elif regime.market_state == "CRISIS":
            signal_raw = signal_raw * 0.25

        sigma = idx["ewma_vol_ann"].astype(float).replace(0.0, np.nan)
        w_vol = (signal_raw / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        b_spy = idx["beta_spy_60d"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        denom = float((b_spy * b_spy).sum())
        if denom > 1e-12:
            alpha = float((w_vol * b_spy).sum() / denom)
            w_beta_neutral = w_vol - alpha * b_spy
        else:
            w_beta_neutral = w_vol.copy()

        w_final = self._apply_target_portfolio_vol(
            w_raw=w_beta_neutral,
            sector_returns=returns[list(sectors)],
            target_vol=float(getattr(self.settings, "target_portfolio_vol", 0.12)),
            cov_window=_get_int_setting(self.settings, "corr_window", 60),
            market_state=regime.market_state,   # regime-aware leverage cap applied inside
        )

        gross_exposure = float(w_final.abs().sum())
        net_exposure = float(w_final.sum())

        master["signal_raw"] = master["sector_ticker"].map(signal_raw.to_dict())
        master["w_vol"] = master["sector_ticker"].map(w_vol.to_dict())
        master["w_beta_neutral"] = master["sector_ticker"].map(w_beta_neutral.to_dict())
        master["w_final"] = master["sector_ticker"].map(w_final.to_dict())
        master["gross_exposure"] = gross_exposure
        master["net_exposure"] = net_exposure

        # ==========================================================
        # Synthetic greeks / exposure proxies
        # ==========================================================
        idx2 = master.set_index("sector_ticker")

        beta_tnx_s = idx2["beta_tnx_60d"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        beta_dxy_s = idx2["beta_dxy_60d"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        b_spy_2 = idx2["beta_spy_60d"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sigma_2 = idx2["ewma_vol_ann"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        z_s_2 = idx2["pca_residual_z"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        w_final_2 = idx2["w_final"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        delta_spy_i = (w_final_2 * b_spy_2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        delta_tnx_i = (w_final_2 * beta_tnx_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        delta_dxy_i = (w_final_2 * beta_dxy_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        gamma_synth_i = (w_final_2.abs() * z_s_2.abs()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vega_synth_i = (w_final_2.abs() * sigma_2).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        mm_loading = idx2["market_mode_loading"].astype(float).replace([np.inf, -np.inf], np.nan)
        corr_abs = idx2["corr_to_spy_60d"].astype(float).replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)
        loading_proxy = mm_loading.abs().where(mm_loading.notna(), corr_abs)

        rho_mode_i = (w_final_2.abs() * loading_proxy).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        dist_contrib = idx2["sector_corr_dist_contrib"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rho_dist_sector_i = (w_final_2.abs() * dist_contrib).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        master["delta_spy_i"] = master["sector_ticker"].map(delta_spy_i.to_dict())
        master["delta_tnx_i"] = master["sector_ticker"].map(delta_tnx_i.to_dict())
        master["delta_dxy_i"] = master["sector_ticker"].map(delta_dxy_i.to_dict())
        master["gamma_synth_i"] = master["sector_ticker"].map(gamma_synth_i.to_dict())
        master["vega_synth_i"] = master["sector_ticker"].map(vega_synth_i.to_dict())
        master["rho_mode_i"] = master["sector_ticker"].map(rho_mode_i.to_dict())
        master["rho_dist_sector_i"] = master["sector_ticker"].map(rho_dist_sector_i.to_dict())

        gamma_port = float(gamma_synth_i.sum())
        vega_port = float(vega_synth_i.sum())

        master["delta_spy_P"] = float(delta_spy_i.sum())
        master["delta_tnx_P"] = float(delta_tnx_i.sum())
        master["delta_dxy_P"] = float(delta_dxy_i.sum())
        master["gamma_synth_P"] = gamma_port
        master["vega_synth_P"] = vega_port
        master["rho_mode_P"] = float(rho_mode_i.sum())

        if math.isfinite(corr_metrics.delta_distortion):
            master["rho_dist_P"] = gamma_port * max(0.0, float(corr_metrics.delta_distortion))
        else:
            master["rho_dist_P"] = float("nan")

        master["rho_dist_sector_P"] = float(rho_dist_sector_i.sum())

        disp_z_latest = float(master["market_dispersion_z"].iloc[0]) if len(master) else float("nan")
        master["dispersion_ratio_z_latest"] = disp_z_latest
        master["dispersion_exposure_P"] = (
            gamma_port * disp_z_latest if math.isfinite(disp_z_latest) else float("nan")
        )

        # ==========================================================
        # Risk decomposition
        # ==========================================================
        cov_window = _get_int_setting(self.settings, "corr_window", 60)
        risk = self._portfolio_risk_decomposition(
            returns[list(sectors)],
            w_final.reindex(sectors).fillna(0.0),
            cov_window=cov_window,
        )
        for k, v in risk.items():
            master[k] = v

        # ==========================================================
        # Execution regime
        # ==========================================================
        avg_mss = (
            float(pd.to_numeric(master["mss_score"], errors="coerce").mean())
            if "mss_score" in master.columns
            else float("nan")
        )
        avg_mc = (
            float(pd.to_numeric(master["mc_score_raw"], errors="coerce").mean())
            if "mc_score_raw" in master.columns
            else float("nan")
        )

        if regime.market_state == "CALM":
            exec_regime = "OK"
        elif regime.market_state == "NORMAL":
            exec_regime = "OK"
        elif regime.market_state == "TENSION":
            exec_regime = "REDUCE"
        else:
            exec_regime = "STOP"

        if math.isfinite(avg_mss) and avg_mss >= 0.55 and exec_regime == "OK":
            exec_regime = "CAUTION"

        if math.isfinite(avg_mc) and avg_mc < 0.12 and exec_regime == "OK":
            exec_regime = "REDUCE"

        if math.isfinite(latest_vix) and latest_vix >= self.settings.vix_level_hard:
            exec_regime = "STOP"

        if (
            math.isfinite(corr_metrics.delta_distortion)
            and corr_metrics.delta_distortion > 0.25
            and gamma_port > 0.75
        ):
            exec_regime = "STOP"

        master["execution_regime"] = exec_regime

        master = master.sort_values(
            ["decision_rank", "decision_score", "mc_score", "conviction_score", "pca_residual_z"],
            ascending=[True, False, False, False, True],
        ).reset_index(drop=True)
        
        self.master_df = master
        return master

    # -----------------------------
    # Tear sheet API
    # -----------------------------
    def get_sector_tearsheet_series(self, sector_ticker: str) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        - residual_level
        - rolling mean
        - +/- 2 std bands
        - zscore
        """
        if self.residual_levels is None or self.residual_zscores is None:
            raise RuntimeError("Run calculate_conviction_score() first.")

        s = sector_ticker
        level = self.residual_levels[s].rename("residual_level")
        mu = level.rolling(window=self.settings.zscore_window, min_periods=self.settings.zscore_window).mean().rename("mean")
        sd = level.rolling(window=self.settings.zscore_window, min_periods=self.settings.zscore_window).std(ddof=0).rename("std")
        upper = (mu + 2.0 * sd).rename("upper_2s")
        lower = (mu - 2.0 * sd).rename("lower_2s")
        z = self.residual_zscores[s].rename("zscore")

        out = pd.concat([level, mu, upper, lower, z], axis=1).dropna()
        return out