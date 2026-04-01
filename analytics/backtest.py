"""
analytics/backtest.py

Walk-forward backtest engine for the SRV Quantamental DSS.

Evaluates the primary signal (OOS PCA residual z-score) against
actual sector forward returns using rolling walk-forward windows.

No look-ahead guarantee
-----------------------
Signal at date t:
    PCA trained on rel_returns[t-252 : t), never includes t.
    Cumulative residual and z-score use only data ≤ t-1.

Forward return for IC / Sharpe:
    Cumulative log return from close of t to close of t + fwd_period.
    Strictly future relative to signal generation date.

Regime label at date t:
    Computed from correlation structure, VIX, and credit spreads
    using data up to and including date t — contemporaneous context,
    no forward leakage.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import Settings

logger = logging.getLogger(__name__)


# ==========================================================================
# Private math helpers (self-contained; no import from stat_arb internals)
# ==========================================================================

def _zscore_series(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    sd = sd.replace(0.0, np.nan)
    return (series - mu) / sd


def _clip(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return lo
    return max(lo, min(hi, float(x)))


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _upper_triangle_mean(C: pd.DataFrame) -> float:
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
    if C is None or C.empty:
        return float("nan")
    a = C.values.astype(float).copy()
    np.fill_diagonal(a, 0.0)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.linalg.norm(a, ord="fro"))


def _rolling_pct_rank_last(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank of the last element in each window (→ [0, 1])."""
    def _pct(x: np.ndarray) -> float:
        if len(x) == 0:
            return float("nan")
        return float(np.sum(x <= x[-1]) / len(x))

    return series.rolling(window=window, min_periods=window).apply(
        lambda x: _pct(np.asarray(x)), raw=False
    )


def _max_drawdown(returns: List[float]) -> float:
    """Maximum drawdown of a cumulative P&L series built from per-period returns."""
    if not returns:
        return float("nan")
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    return float(dd.max()) if dd.size else float("nan")


def _sharpe(returns: List[float], ann_factor: float) -> float:
    """Annualised Sharpe ratio from a list of period returns."""
    r = np.array(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    std = float(np.std(r, ddof=1))
    if std < 1e-12:
        return float("nan")
    return float(np.mean(r) / std * ann_factor)


# ==========================================================================
# Result containers
# ==========================================================================

@dataclass
class WalkMetrics:
    """Metrics for a single walk-forward period."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp       # = signal generation date
    test_start: pd.Timestamp      # = first forward day
    test_end: pd.Timestamp        # = last forward day
    regime: str
    ic: float                     # cross-sectional Spearman IC
    hit_rate: float               # directional accuracy (this walk only)
    signal_return: float          # gross signal-weighted portfolio log-return
    net_signal_return: float      # net return after transaction cost deduction
    tc_cost: float                # transaction cost drag for this walk
    n_sectors: int                # sectors with valid signal + fwd return


@dataclass
class RegimeBreakdown:
    """Aggregated backtest statistics for a single regime label."""
    regime: str
    n_walks: int
    ic_mean: float
    ic_ir: float
    hit_rate: float
    sharpe: float


@dataclass
class BacktestResult:
    """Complete walk-forward backtest output."""

    # Per-walk IC time series (DatetimeIndex = test_start date)
    ic_series: pd.Series

    # Aggregate metrics over all walks — gross (before TC)
    ic_mean: float
    ic_ir: float           # IC_mean / std(IC) — information ratio of the signal
    hit_rate: float        # fraction of (sector, walk) pairs with correct direction
    sharpe: float          # annualised Sharpe of gross signal-weighted portfolio
    max_drawdown: float    # maximum peak-to-trough drawdown of cumulative P&L

    # Net-of-cost metrics
    net_sharpe: float      # annualised Sharpe after transaction cost deduction
    net_max_drawdown: float
    tc_bps: float          # round-trip TC assumption used (bps)
    annualized_tc_drag: float  # estimated annual TC drag (fraction)

    # Regime-conditional breakdown
    regime_breakdown: Dict[str, RegimeBreakdown]

    # Walk-level detail (one entry per valid walk)
    walk_metrics: List[WalkMetrics]

    # Dash-ready summary (one row per walk, sorted by date)
    summary_df: pd.DataFrame

    # Parameters used in this run
    train_window: int
    test_window: int
    step: int
    fwd_period: int
    n_walks: int
    n_sectors: int


# ==========================================================================
# WalkForwardBacktester
# ==========================================================================

class WalkForwardBacktester:
    """
    Rolling walk-forward evaluator for the SRV sector-rotation signal.

    Walk parameters (class-level defaults — overridden by settings at __init__):
        TRAIN_WINDOW = 252  days of in-sample training data
        TEST_WINDOW  = 21   days of out-of-sample evaluation
        STEP         = 21   days between consecutive walk anchors (monthly rebalance)
        FWD_PERIOD   = 20   days ahead used for IC / hit-rate / Sharpe
                            Aligns with signal_optimal_hold — medium-term sector trend

    Signal convention:
        signal_i = -pca_residual_z_i
        Positive z → sector rich vs PCA reconstruction → expect mean reversion
        down → SHORT → negative signal, so -z makes positive signal = LONG bias.
        IC > 0 means the signal correctly predicts the direction of forward return.

    This is a medium-term sector RV signal (20-day hold) evaluated monthly.
    """

    TRAIN_WINDOW: int = 252
    TEST_WINDOW: int = 21
    STEP: int = 21          # Monthly evaluation step (not weekly)
    FWD_PERIOD: int = 20    # 20-day forward return — matches signal_optimal_hold
    TC_BPS_ROUNDTRIP: float = 15.0  # Round-trip TC per gross unit (ETF spreads ~5bps/side)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        # Override class-level defaults with settings if available
        # FWD_PERIOD aligns with the PM's intended holding horizon
        optimal_hold = getattr(settings, "signal_optimal_hold", None)
        if optimal_hold and isinstance(optimal_hold, int) and 5 <= optimal_hold <= 60:
            self.FWD_PERIOD = optimal_hold  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        weights_df: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run walk-forward backtest on pre-loaded DataFrames.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Wide daily price panel (DatetimeIndex, columns include all tickers
            required by Settings: sectors, SPY, VIX, HYG, IEF, TNX, DXY).
        fundamentals_df : pd.DataFrame
            Fundamental snapshot — not consumed directly in signal computation
            but retained for interface consistency with QuantEngine callers.
        weights_df : pd.DataFrame
            ETF holdings and SPY sector weights — not consumed in this version
            of the backtest (signal uses price-only PCA residuals).

        Returns
        -------
        BacktestResult
            Full backtest output including per-walk IC series, aggregate
            IC_mean / IC_IR / hit_rate / Sharpe / max_drawdown, regime
            breakdown, and a Dash-ready summary DataFrame.
        """
        prices = self._prepare_prices(prices_df)
        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker

        self._validate_inputs(prices, sectors, spy)

        log_px = np.log(prices.astype(float))
        returns = log_px.diff()

        n = len(prices)
        min_start = self.TRAIN_WINDOW

        if n < min_start + self.FWD_PERIOD + 1:
            raise ValueError(
                f"prices_df has {n} rows; need at least "
                f"{min_start + self.FWD_PERIOD + 1} for walk-forward backtest."
            )

        self.logger.info(
            "Pre-computing OOS PCA residuals (%d sectors, window=%d, n=%d rows).",
            len(sectors), self.TRAIN_WINDOW, n,
        )
        _, resid_z = self._compute_oos_residuals(returns, sectors, spy)

        # Walk anchors: index positions where the test window begins
        anchors = list(range(min_start, n - self.FWD_PERIOD, self.STEP))
        self.logger.info(
            "Walk-forward: %d anchors, step=%d, fwd=%d.",
            len(anchors), self.STEP, self.FWD_PERIOD,
        )

        # Per-walk accumulation
        walk_metrics: List[WalkMetrics] = []
        all_signal_returns: List[float] = []
        all_net_signal_returns: List[float] = []
        hit_correct: int = 0
        hit_total: int = 0

        for anchor in anchors:
            signal_idx = anchor - 1
            signal_date = prices.index[signal_idx]

            # Signal: -z_score (positive = LONG bias for that sector)
            sig_row = resid_z.iloc[signal_idx][sectors]
            signal = -sig_row  # mean reversion: high z → expect decline → short

            # Forward cumulative log return: close(signal_date) → close(signal_date + FWD_PERIOD)
            fwd_end_idx = anchor + self.FWD_PERIOD - 1
            if fwd_end_idx >= n:
                continue

            fwd_ret = log_px.iloc[fwd_end_idx][sectors] - log_px.iloc[signal_idx][sectors]

            # Regime at signal_date (uses data up to and including signal_date)
            regime = self._regime_at(returns=returns.iloc[:anchor], prices=prices.iloc[:anchor], sectors=sectors)

            # Cross-sectional IC
            valid_mask = signal.notna() & fwd_ret.notna()
            n_valid = int(valid_mask.sum())
            if n_valid < 3:
                continue

            sig_v = signal[valid_mask].values.astype(float)
            fwd_v = fwd_ret[valid_mask].values.astype(float)

            try:
                rho, _ = spearmanr(sig_v, fwd_v)
                ic = float(rho) if math.isfinite(rho) else float("nan")
            except Exception:
                ic = float("nan")

            # Hit rate (only count positions with non-trivial signal magnitude)
            walk_hit_correct = 0
            walk_hit_total = 0
            for sv, rv in zip(sig_v, fwd_v):
                if math.isfinite(sv) and math.isfinite(rv) and abs(sv) >= 0.30:
                    walk_hit_total += 1
                    if sv * rv > 0:
                        walk_hit_correct += 1
            walk_hit_rate = (
                float(walk_hit_correct) / walk_hit_total
                if walk_hit_total > 0
                else float("nan")
            )
            hit_correct += walk_hit_correct
            hit_total += walk_hit_total

            # Signal-weighted portfolio return (gross=1 normalised)
            w = signal[valid_mask].copy().astype(float)
            gross = float(w.abs().sum())
            if gross > 1e-12:
                w_norm = w / gross
                port_ret = float((w_norm * fwd_ret[valid_mask]).sum())
            else:
                port_ret = 0.0

            # Transaction cost: round-trip at TC_BPS_ROUNDTRIP bps per unit of gross exposure.
            # Each walk opens and closes all positions → full round-trip cost.
            # Sector ETF bid-ask spread ~5 bps + SPY hedge ~3 bps = ~8 bps × 2 legs ≈ 15 bps rt.
            tc_cost = gross * (self.TC_BPS_ROUNDTRIP / 10_000.0)
            net_port_ret = port_ret - tc_cost

            all_signal_returns.append(port_ret)
            all_net_signal_returns.append(net_port_ret)

            test_end_idx = min(fwd_end_idx, n - 1)
            walk_metrics.append(WalkMetrics(
                train_start=prices.index[anchor - self.TRAIN_WINDOW],
                train_end=signal_date,
                test_start=prices.index[anchor],
                test_end=prices.index[test_end_idx],
                regime=regime,
                ic=ic,
                hit_rate=walk_hit_rate,
                signal_return=port_ret,
                net_signal_return=net_port_ret,
                tc_cost=tc_cost,
                n_sectors=n_valid,
            ))

        if not walk_metrics:
            raise RuntimeError(
                "No valid walk-forward periods produced. "
                "Check data length and PCA window settings."
            )

        # ------------------------------------------------------------------
        # Aggregate metrics
        # ------------------------------------------------------------------
        ic_vals = np.array(
            [w.ic for w in walk_metrics if math.isfinite(w.ic)], dtype=float
        )
        ic_mean = float(np.mean(ic_vals)) if ic_vals.size else float("nan")
        ic_std = float(np.std(ic_vals, ddof=1)) if ic_vals.size > 1 else float("nan")
        ic_ir = (
            float(ic_mean / ic_std)
            if (math.isfinite(ic_std) and ic_std > 1e-12)
            else float("nan")
        )

        hit_rate = float(hit_correct) / hit_total if hit_total > 0 else float("nan")

        ann_factor = math.sqrt(252.0 / self.STEP)
        sharpe = _sharpe(all_signal_returns, ann_factor)
        max_dd = _max_drawdown(all_signal_returns)

        # Net-of-cost metrics
        net_sharpe = _sharpe(all_net_signal_returns, ann_factor)
        net_max_dd = _max_drawdown(all_net_signal_returns)
        walks_per_year = 252.0 / self.STEP
        total_tc = sum(w.tc_cost for w in walk_metrics)
        annualized_tc_drag = (total_tc / max(1, len(walk_metrics))) * walks_per_year

        ic_series = pd.Series(
            [w.ic for w in walk_metrics],
            index=pd.DatetimeIndex([w.test_start for w in walk_metrics]),
            name="ic",
        )

        regime_breakdown = self._compute_regime_breakdown(walk_metrics, ann_factor)
        summary_df = self._build_summary_df(
            walk_metrics=walk_metrics,
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            sharpe=sharpe,
            max_drawdown=max_dd,
        )

        self.logger.info(
            "Backtest complete: %d walks | IC_mean=%.4f | IC_IR=%.2f | "
            "hit_rate=%.1f%% | Sharpe=%.2f (net=%.2f) | MaxDD=%.1f%% | TC_drag=%.0fbps/yr",
            len(walk_metrics),
            ic_mean,
            ic_ir if math.isfinite(ic_ir) else float("nan"),
            hit_rate * 100 if math.isfinite(hit_rate) else float("nan"),
            sharpe if math.isfinite(sharpe) else float("nan"),
            net_sharpe if math.isfinite(net_sharpe) else float("nan"),
            max_dd * 100 if math.isfinite(max_dd) else float("nan"),
            annualized_tc_drag * 10_000,
        )

        return BacktestResult(
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            sharpe=sharpe,
            max_drawdown=max_dd,
            net_sharpe=net_sharpe,
            net_max_drawdown=net_max_dd,
            tc_bps=self.TC_BPS_ROUNDTRIP,
            annualized_tc_drag=annualized_tc_drag,
            regime_breakdown=regime_breakdown,
            walk_metrics=walk_metrics,
            summary_df=summary_df,
            train_window=self.TRAIN_WINDOW,
            test_window=self.TEST_WINDOW,
            step=self.STEP,
            fwd_period=self.FWD_PERIOD,
            n_walks=len(walk_metrics),
            n_sectors=len(self.settings.sector_list()),
        )

    # ------------------------------------------------------------------
    # OOS PCA residuals (mirrors QuantEngine._oos_rolling_pca_residuals)
    # ------------------------------------------------------------------

    def _compute_oos_residuals(
        self,
        returns: pd.DataFrame,
        sectors: List[str],
        spy: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build out-of-sample rolling PCA residual levels and z-scores.

        For each date t >= TRAIN_WINDOW:
            Train scaler + PCA on rel_returns[t-TRAIN_WINDOW : t).
            Transform rel_returns[t] → residual at t.
        Residual level = cumsum(residual_returns).
        Residual z-score = rolling zscore of level.

        No look-ahead: training window excludes observation t.

        Returns
        -------
        resid_level : pd.DataFrame
            Cumulative PCA residual, columns = sectors.
        resid_z : pd.DataFrame
            Rolling z-score of resid_level, columns = sectors.
        """
        s = self.settings
        rel_returns = returns[sectors].subtract(returns[spy], axis=0)
        n = len(rel_returns)
        idx = rel_returns.index
        max_k = min(s.pca_max_components, len(sectors))

        resid_ret = pd.DataFrame(np.nan, index=idx, columns=sectors, dtype=float)

        for t in range(self.TRAIN_WINDOW, n):
            train = rel_returns.iloc[t - self.TRAIN_WINDOW : t].astype(float)
            x_t = rel_returns.iloc[t].astype(float)

            nan_frac_train = float(train.isna().mean().mean())
            nan_frac_x = float(x_t.isna().mean())
            if nan_frac_train > 0.05 or nan_frac_x > 0.10:
                continue

            train_arr = train.fillna(0.0).values
            x_row = x_t.fillna(0.0).values.reshape(1, -1)

            scaler = StandardScaler(with_mean=True, with_std=True)
            xs = scaler.fit_transform(train_arr)

            pca_full = PCA(n_components=max_k, svd_solver="full")
            pca_full.fit(xs)
            csum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(csum, s.pca_explained_var_target) + 1)
            k = max(s.pca_min_components, min(k, s.pca_max_components))

            pca = PCA(n_components=k, svd_solver="full")
            pca.fit(xs)

            x_s = scaler.transform(x_row)
            x_hat_s = pca.inverse_transform(pca.transform(x_s))
            resid_scaled = (x_s - x_hat_s).reshape(-1)
            resid_ret.iloc[t] = resid_scaled * scaler.scale_

        resid_level = resid_ret.fillna(0.0).cumsum()
        resid_z = resid_level.apply(
            lambda col: _zscore_series(col, window=s.zscore_window), axis=0
        )
        return resid_level, resid_z

    # ------------------------------------------------------------------
    # Regime classification (per anchor, uses data up to signal_date)
    # ------------------------------------------------------------------

    def _regime_at(
        self,
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        sectors: List[str],
    ) -> str:
        """
        Classify market regime using data up to the last row of the supplied slices.

        Uses the same scoring thresholds as QuantEngine._compute_regime_metrics.
        If optional tickers (VIX, HYG, IEF) are absent, those sub-scores default to 0.
        """
        s = self.settings
        corr_w = max(20, s.corr_window)
        base_w = max(corr_w, s.corr_baseline_window)
        macro_w = s.macro_window

        R = returns[sectors].dropna(how="all")
        n = len(R)

        # Correlation structure
        C_t = R.iloc[-corr_w:].corr() if n >= corr_w else R.corr()
        C_b = R.iloc[-base_w:].corr() if n >= base_w else C_t.copy()
        C_delta = C_t - C_b

        avg_corr_t = _upper_triangle_mean(C_t)
        avg_corr_b = _upper_triangle_mean(C_b)
        avg_corr_delta = (
            (avg_corr_t - avg_corr_b)
            if (math.isfinite(avg_corr_t) and math.isfinite(avg_corr_b))
            else float("nan")
        )

        distortion_t = _fro_offdiag(C_delta)
        distortion_prev = float("nan")
        if n >= corr_w + 1:
            C_prev = R.iloc[-(corr_w + 1) : -1].corr()
            distortion_prev = _fro_offdiag(C_prev - C_b)
        delta_distortion = (
            float(distortion_t - distortion_prev)
            if (math.isfinite(distortion_t) and math.isfinite(distortion_prev))
            else float("nan")
        )

        # Market mode strength (leading eigenvalue / N)
        market_mode_strength = float("nan")
        try:
            if C_t.notna().sum().sum() > 0:
                Ct = C_t.fillna(0.0).values.astype(float)
                Ct = 0.5 * (Ct + Ct.T)
                evals, _ = np.linalg.eigh(Ct)
                market_mode_strength = float(np.max(evals)) / len(sectors)
        except Exception:
            pass

        # VIX
        vix_col = s.vol_tickers.get("VIX", "^VIX")
        vix_level = float("nan")
        vix_percentile = float("nan")
        if vix_col in prices.columns:
            vix_s = prices[vix_col].dropna()
            if vix_s.size:
                vix_level = float(vix_s.iloc[-1])
                pct_win = max(252, s.pca_window)
                if len(vix_s) >= pct_win:
                    vix_percentile = float(
                        np.sum(vix_s.values[-pct_win:] <= vix_level) / pct_win
                    )

        # Credit z
        credit_z = float("nan")
        hyg_col = s.credit_tickers.get("HYG", "HYG")
        ief_col = s.credit_tickers.get("IEF", "IEF")
        if hyg_col in prices.columns and ief_col in prices.columns:
            spread = (np.log(prices[hyg_col]) - np.log(prices[ief_col])).dropna()
            if len(spread) >= macro_w:
                mu_s = float(spread.rolling(macro_w, min_periods=macro_w).mean().iloc[-1])
                sd_s = float(spread.rolling(macro_w, min_periods=macro_w).std(ddof=0).iloc[-1])
                if math.isfinite(mu_s) and math.isfinite(sd_s) and sd_s > 1e-12:
                    credit_z = float((spread.iloc[-1] - mu_s) / sd_s)

        return self._classify_regime(
            avg_corr_t=avg_corr_t,
            avg_corr_delta=avg_corr_delta,
            distortion_t=distortion_t,
            delta_distortion=delta_distortion,
            market_mode_strength=market_mode_strength,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            credit_z=credit_z,
        )

    def _classify_regime(
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
    ) -> str:
        """
        Reproduce QuantEngine._compute_regime_metrics classification logic.

        Returns one of "CALM", "NORMAL", "TENSION", "CRISIS".
        """
        s = self.settings

        # Vol score
        vol_score = 0.0
        if math.isfinite(vix_level):
            if vix_level >= s.vix_level_hard:
                vol_score = 1.0
            elif vix_level > s.vix_level_soft:
                vol_score = (vix_level - s.vix_level_soft) / max(
                    1e-9, s.vix_level_hard - s.vix_level_soft
                )
        if math.isfinite(vix_percentile):
            vol_score = max(vol_score, _clip((vix_percentile - 0.70) / 0.25, 0.0, 1.0))

        # Credit score
        credit_score = 0.0
        if math.isfinite(credit_z):
            credit_score = _clip((-credit_z - 0.25) / 1.75, 0.0, 1.0)

        # Corr score
        corr_level_score = (
            _clip(
                (avg_corr_t - s.calm_avg_corr_max)
                / max(1e-9, s.crisis_avg_corr_min - s.calm_avg_corr_max),
                0.0,
                1.0,
            )
            if math.isfinite(avg_corr_t)
            else 0.0
        )
        mode_score = (
            _clip(
                (market_mode_strength - s.calm_mode_strength_max)
                / max(1e-9, s.crisis_mode_strength_min - s.calm_mode_strength_max),
                0.0,
                1.0,
            )
            if math.isfinite(market_mode_strength)
            else 0.0
        )
        dist_score = (
            _clip(
                (distortion_t - s.tension_corr_dist_min)
                / max(1e-9, s.crisis_corr_dist_min - s.tension_corr_dist_min),
                0.0,
                1.0,
            )
            if math.isfinite(distortion_t)
            else 0.0
        )
        corr_score = _clip(
            0.45 * corr_level_score + 0.30 * mode_score + 0.25 * dist_score,
            0.0, 1.0,
        )

        # Transition score
        avg_corr_delta_score = (
            _clip(abs(avg_corr_delta) / 0.18, 0.0, 1.0)
            if math.isfinite(avg_corr_delta)
            else 0.0
        )
        delta_dist_score = (
            _clip(
                (delta_distortion - 0.03)
                / max(1e-9, s.crisis_delta_corr_dist_min - 0.03),
                0.0,
                1.0,
            )
            if math.isfinite(delta_distortion)
            else 0.0
        )
        transition_score = _clip(
            0.45 * delta_dist_score
            + 0.25 * avg_corr_delta_score
            + 0.15 * vol_score
            + 0.15 * credit_score,
            0.0, 1.0,
        )

        crisis_probability = _clip(
            0.40 * corr_score
            + 0.25 * vol_score
            + 0.20 * credit_score
            + 0.15 * transition_score,
            0.0, 1.0,
        )

        # Crisis hit count
        crisis_hits = sum([
            math.isfinite(vix_level) and vix_level >= s.vix_level_hard,
            math.isfinite(avg_corr_t) and avg_corr_t >= s.crisis_avg_corr_min,
            math.isfinite(market_mode_strength) and market_mode_strength >= s.crisis_mode_strength_min,
            math.isfinite(distortion_t) and distortion_t >= s.crisis_corr_dist_min,
            math.isfinite(delta_distortion) and delta_distortion >= s.crisis_delta_corr_dist_min,
            math.isfinite(credit_z) and credit_z <= s.credit_stress_z,
        ])
        tension_hits = sum([
            math.isfinite(vix_level) and vix_level >= s.vix_level_soft,
            math.isfinite(avg_corr_t) and avg_corr_t >= s.tension_avg_corr_min,
            math.isfinite(market_mode_strength) and market_mode_strength >= s.tension_mode_strength_min,
            math.isfinite(distortion_t) and distortion_t >= s.tension_corr_dist_min,
            math.isfinite(delta_distortion) and delta_distortion >= s.tension_delta_corr_dist_min,
        ])

        if crisis_hits >= 3 or crisis_probability >= 0.78:
            return "CRISIS"
        if tension_hits >= 2 or transition_score >= s.transition_score_danger:
            return "TENSION"
        if transition_score >= s.transition_score_caution or corr_score >= 0.40 or vol_score >= 0.35:
            return "NORMAL"
        return "CALM"

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _compute_regime_breakdown(
        self,
        walks: List[WalkMetrics],
        ann_factor: float,
    ) -> Dict[str, RegimeBreakdown]:
        """Compute per-regime IC / hit-rate / Sharpe from walk-level metrics."""
        regimes = sorted({w.regime for w in walks})
        breakdown: Dict[str, RegimeBreakdown] = {}

        for regime in regimes:
            subset = [w for w in walks if w.regime == regime]
            n_w = len(subset)

            ic_arr = np.array(
                [w.ic for w in subset if math.isfinite(w.ic)], dtype=float
            )
            ic_mean = float(np.mean(ic_arr)) if ic_arr.size else float("nan")
            ic_std = float(np.std(ic_arr, ddof=1)) if ic_arr.size > 1 else float("nan")
            ic_ir = (
                float(ic_mean / ic_std)
                if (math.isfinite(ic_std) and ic_std > 1e-12)
                else float("nan")
            )

            hr_arr = np.array(
                [w.hit_rate for w in subset if math.isfinite(w.hit_rate)], dtype=float
            )
            hit_rate = float(np.mean(hr_arr)) if hr_arr.size else float("nan")

            rets = [w.signal_return for w in subset if math.isfinite(w.signal_return)]
            sharpe = _sharpe(rets, ann_factor)

            breakdown[regime] = RegimeBreakdown(
                regime=regime,
                n_walks=n_w,
                ic_mean=ic_mean,
                ic_ir=ic_ir,
                hit_rate=hit_rate,
                sharpe=sharpe,
            )

        return breakdown

    def _build_summary_df(
        self,
        walk_metrics: List[WalkMetrics],
        ic_mean: float,
        ic_ir: float,
        hit_rate: float,
        sharpe: float,
        max_drawdown: float,
    ) -> pd.DataFrame:
        """
        Build a Dash-ready summary DataFrame from walk-level metrics.

        Rows: one per walk, sorted by test_start date.
        Columns:
            date, regime, ic, hit_rate, signal_return, cum_pnl,
            n_sectors, train_start, train_end, test_end.
        The final row contains aggregate stats (date=NaT, regime="ALL").
        """
        rows = []
        cum = 0.0
        net_cum = 0.0
        for w in sorted(walk_metrics, key=lambda x: x.test_start):
            cum += w.signal_return if math.isfinite(w.signal_return) else 0.0
            net_cum += w.net_signal_return if math.isfinite(w.net_signal_return) else 0.0
            rows.append({
                "date":             w.test_start,
                "regime":           w.regime,
                "ic":               round(w.ic, 4) if math.isfinite(w.ic) else float("nan"),
                "hit_rate":         round(w.hit_rate, 4) if math.isfinite(w.hit_rate) else float("nan"),
                "signal_return":    round(w.signal_return, 6) if math.isfinite(w.signal_return) else float("nan"),
                "net_signal_return":round(w.net_signal_return, 6) if math.isfinite(w.net_signal_return) else float("nan"),
                "tc_cost":          round(w.tc_cost, 6),
                "cum_pnl":          round(cum, 6),
                "net_cum_pnl":      round(net_cum, 6),
                "n_sectors":        w.n_sectors,
                "train_start":      w.train_start,
                "train_end":        w.train_end,
                "test_end":         w.test_end,
            })

        df = pd.DataFrame(rows)

        # Append aggregate summary row
        agg = {
            "date":             pd.NaT,
            "regime":           "ALL",
            "ic":               round(ic_mean, 4) if math.isfinite(ic_mean) else float("nan"),
            "hit_rate":         round(hit_rate, 4) if math.isfinite(hit_rate) else float("nan"),
            "signal_return":    round(float(df["signal_return"].sum(skipna=True)), 6),
            "net_signal_return":round(float(df["net_signal_return"].sum(skipna=True)), 6) if "net_signal_return" in df.columns else float("nan"),
            "tc_cost":          round(float(df["tc_cost"].sum(skipna=True)), 6) if "tc_cost" in df.columns else 0.0,
            "cum_pnl":          round(cum, 6),
            "net_cum_pnl":      round(net_cum, 6),
            "n_sectors":        int(df["n_sectors"].mean()) if not df.empty else 0,
            "train_start":      pd.NaT,
            "train_end":        pd.NaT,
            "test_end":         pd.NaT,
        }
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
        return df

    # ------------------------------------------------------------------
    # Input validation / preparation
    # ------------------------------------------------------------------

    def _prepare_prices(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        prices = prices_df.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)
        return prices

    def _validate_inputs(
        self,
        prices: pd.DataFrame,
        sectors: List[str],
        spy: str,
    ) -> None:
        required = sectors + [spy]
        missing = [c for c in required if c not in prices.columns]
        if missing:
            raise ValueError(
                f"prices_df is missing required tickers: {missing}. "
                f"Optional tickers (VIX, HYG, IEF) are used for regime "
                f"classification if present."
            )


# ==========================================================================
# Module-level convenience function
# ==========================================================================

def run_backtest(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    settings: Settings,
) -> BacktestResult:
    """
    Run a walk-forward backtest using default WalkForwardBacktester parameters.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Wide daily price panel (DatetimeIndex).
    fundamentals_df : pd.DataFrame
        Fundamental snapshot (passed through for interface consistency).
    weights_df : pd.DataFrame
        ETF holdings and SPY sector weights (passed through for interface consistency).
    settings : Settings
        Validated settings instance from config.settings.

    Returns
    -------
    BacktestResult
        Full backtest output.
    """
    return WalkForwardBacktester(settings).run_backtest(
        prices_df=prices_df,
        fundamentals_df=fundamentals_df,
        weights_df=weights_df,
    )
