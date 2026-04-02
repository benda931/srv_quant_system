"""
analytics/pnl_tracker.py

Live P&L Tracker for the SRV Quantamental DSS.

Computes historical and live portfolio P&L from signal-weighted positions:
  - Daily P&L per sector: w_final(t-1) × return(t)
  - Cumulative P&L with drawdown analysis
  - Hit rate per sector (directional accuracy)
  - Attribution: which sectors / regimes drove performance
  - Regime-conditional performance breakdown

Two modes:
  1. SIMULATED — reconstructs P&L from OOS PCA residuals (full history)
  2. LIVE — tracks P&L from actual analytics.sector_signals DB records

No look-ahead guarantee: weights at t use only data up to t-1.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _sf(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _max_drawdown_series(cum_pnl: pd.Series) -> pd.Series:
    """Running max drawdown (negative values = drawdown depth)."""
    running_max = cum_pnl.expanding().max()
    dd = cum_pnl - running_max
    return dd


def _sharpe(returns: pd.Series, ann_factor: float = np.sqrt(252)) -> float:
    r = returns.dropna()
    if len(r) < 20:
        return float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd < 1e-12:
        return float("nan")
    return mu / sd * ann_factor


def _calmar(cum_pnl: pd.Series, ann_factor: float = 252.0) -> float:
    """Calmar ratio = annualised return / max drawdown."""
    if len(cum_pnl) < 20:
        return float("nan")
    total_ret = float(cum_pnl.iloc[-1] - cum_pnl.iloc[0])
    n_days = len(cum_pnl)
    ann_ret = total_ret * ann_factor / n_days
    dd = _max_drawdown_series(cum_pnl)
    max_dd = float(dd.min())
    if abs(max_dd) < 1e-12:
        return float("nan")
    return ann_ret / abs(max_dd)


# ── Result containers ────────────────────────────────────────────────────

@dataclass
class SectorPnL:
    """P&L breakdown for a single sector."""
    sector: str
    total_pnl: float                  # Cumulative P&L
    avg_daily_pnl: float
    hit_rate: float                   # % days where sign(weight) × sign(return) > 0
    n_active_days: int                # Days where |weight| > 0
    n_hits: int
    n_misses: int
    sharpe: float
    max_drawdown: float
    best_day: float
    worst_day: float
    avg_weight: float                 # Mean |w_final| when active


@dataclass
class RegimePnL:
    """P&L breakdown for a single regime."""
    regime: str
    total_pnl: float
    n_days: int
    avg_daily_pnl: float
    sharpe: float
    hit_rate: float
    max_drawdown: float


@dataclass
class PnLResult:
    """Complete P&L tracking output."""

    # ── Aggregate portfolio metrics ──────────────────────────────────────
    total_pnl: float
    sharpe: float
    calmar: float
    max_drawdown: float
    max_drawdown_date: Optional[pd.Timestamp]
    hit_rate: float                   # Portfolio-level directional accuracy
    avg_daily_pnl: float
    pnl_volatility: float             # Daily P&L std
    best_day: float
    worst_day: float
    win_days: int
    loss_days: int
    n_trading_days: int

    # ── Time series (Dash-ready) ─────────────────────────────────────────
    daily_pnl: pd.Series              # DatetimeIndex, daily portfolio P&L
    cumulative_pnl: pd.Series         # DatetimeIndex, cumulative P&L
    drawdown: pd.Series               # DatetimeIndex, running drawdown
    rolling_sharpe: pd.Series         # 63-day rolling Sharpe

    # ── Per-sector breakdown ─────────────────────────────────────────────
    sector_pnl: Dict[str, SectorPnL]

    # ── Regime-conditional ───────────────────────────────────────────────
    regime_pnl: Dict[str, RegimePnL]

    # ── Monthly returns grid ─────────────────────────────────────────────
    monthly_returns: pd.DataFrame     # Year × Month matrix

    # ── Attribution: sector contribution time series ─────────────────────
    sector_contribution: pd.DataFrame  # DatetimeIndex × sector columns

    # ── Factor attribution ──────────────────────────────────────────────
    factor_attribution: Optional[Dict[str, float]] = None  # {factor: cumulative P&L contribution}
    # Keys: "spy_beta", "rates_tnx", "dollar_dxy", "credit_hyg", "idiosyncratic"
    factor_daily: Optional[pd.DataFrame] = None  # DatetimeIndex × factor columns

    # ── Performance ratios ──────────────────────────────────────────────
    sortino: float = 0.0
    information_ratio: float = 0.0    # vs SPY benchmark
    turnover_annual: float = 0.0

    # ── Summary table ────────────────────────────────────────────────────
    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ── Main engine ──────────────────────────────────────────────────────────

class PnLTracker:
    """
    Reconstructs portfolio P&L from OOS signals and actual sector returns.

    The tracker replicates the QuantEngine signal generation process
    (OOS PCA residual z-scores → vol-normalised weights) and computes
    actual P&L using next-day sector returns.

    Weight at close(t): signal generated from data up to t-1.
    P&L at t: weight(t-1) × return(t) — strict no-lookahead.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._log = logging.getLogger(self.__class__.__name__)

    def track(
        self,
        prices_df: pd.DataFrame,
        lookback_days: Optional[int] = None,
    ) -> PnLResult:
        """
        Run full P&L reconstruction.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Wide daily price panel (DatetimeIndex × tickers).
        lookback_days : int, optional
            If set, only compute P&L for the last N trading days.
            Default: full history (after PCA warm-up).

        Returns
        -------
        PnLResult
        """
        from analytics.backtest import WalkForwardBacktester

        prices = prices_df.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

        s = self.settings
        sectors = s.sector_list()
        spy = s.spy_ticker

        log_px = np.log(prices.astype(float))
        returns = log_px.diff()
        sector_returns = returns[sectors]

        # Compute OOS PCA residual z-scores
        bt = WalkForwardBacktester(s)
        _, resid_z = bt._compute_oos_residuals(returns, sectors, spy)

        # EWMA vol per sector (for vol-normalisation)
        ewma_lambda = s.ewma_lambda
        ewma_var = sector_returns.ewm(alpha=1 - ewma_lambda, adjust=False).var()
        ewma_vol = np.sqrt(ewma_var * 252)  # annualised
        ewma_vol = ewma_vol.clip(lower=0.01)  # floor at 1%

        n = len(prices)
        train_w = 252
        start_idx = train_w + s.zscore_window  # Need enough for z-score

        if lookback_days is not None:
            start_idx = max(start_idx, n - lookback_days)

        self._log.info(
            "P&L tracker: %d sectors, start_idx=%d, total=%d days",
            len(sectors), start_idx, n - start_idx,
        )

        # ── Build daily weights and P&L ──────────────────────────────────
        daily_pnl_list = []
        sector_pnl_daily: Dict[str, List[float]] = {s: [] for s in sectors}
        regime_labels: List[str] = []
        dates: List[pd.Timestamp] = []

        for t in range(start_idx, n):
            date = prices.index[t]
            signal_idx = t - 1  # Signal uses data up to yesterday

            # Signal: -z (mean reversion)
            z_row = resid_z.iloc[signal_idx][sectors]
            signal = -z_row.clip(-3, 3)

            # Skip if mostly NaN
            if signal.isna().sum() > len(sectors) * 0.5:
                continue

            # Regime at t-1 (for scaling)
            regime = bt._regime_at(
                returns=returns.iloc[:t],
                prices=prices.iloc[:t],
                sectors=sectors,
            )

            # Regime scaling
            regime_scale = {"CALM": 1.0, "NORMAL": 1.0, "TENSION": 0.70, "CRISIS": 0.0}
            scale = regime_scale.get(regime, 1.0)
            signal = signal * scale

            # Vol-normalise
            vol_t = ewma_vol.iloc[signal_idx][sectors]
            w_vol = signal / vol_t
            w_vol = w_vol.fillna(0.0)

            # Gross normalisation (target vol)
            gross = float(w_vol.abs().sum())
            if gross > 1e-12:
                max_lev = {
                    "CALM": s.max_leverage_calm,
                    "NORMAL": s.max_leverage_normal,
                    "TENSION": s.max_leverage_tension,
                    "CRISIS": s.max_leverage_crisis,
                }.get(regime, 2.0)
                if gross > max_lev and max_lev > 0:
                    w_vol = w_vol * (max_lev / gross)
                elif max_lev <= 0:
                    w_vol = w_vol * 0
            else:
                w_vol = w_vol * 0

            weights = w_vol

            # Today's actual returns
            ret_t = sector_returns.iloc[t][sectors]

            # P&L = weight(t-1) × return(t)
            pnl_per_sector = weights * ret_t
            pnl_per_sector = pnl_per_sector.fillna(0.0)
            port_pnl = float(pnl_per_sector.sum())

            daily_pnl_list.append(port_pnl)
            dates.append(date)
            regime_labels.append(regime)

            for sec in sectors:
                sector_pnl_daily[sec].append(float(pnl_per_sector.get(sec, 0.0)))

        if not dates:
            raise RuntimeError("No valid P&L days produced.")

        # ── Build time series ────────────────────────────────────────────
        idx = pd.DatetimeIndex(dates)
        daily_pnl = pd.Series(daily_pnl_list, index=idx, name="daily_pnl")
        cumulative_pnl = daily_pnl.cumsum()
        cumulative_pnl.name = "cumulative_pnl"
        drawdown = _max_drawdown_series(cumulative_pnl)
        drawdown.name = "drawdown"

        # Rolling 63-day Sharpe
        rolling_sharpe = daily_pnl.rolling(63, min_periods=30).apply(
            lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std(ddof=1) > 1e-12 else 0.0,
            raw=True,
        )
        rolling_sharpe.name = "rolling_sharpe_63d"

        # Sector contribution DataFrame
        sector_contrib = pd.DataFrame(sector_pnl_daily, index=idx)

        # ── Aggregate metrics ────────────────────────────────────────────
        total_pnl = float(cumulative_pnl.iloc[-1])
        n_days = len(daily_pnl)
        avg_pnl = float(daily_pnl.mean())
        pnl_vol = float(daily_pnl.std(ddof=1))
        sharpe = _sharpe(daily_pnl)
        calmar = _calmar(cumulative_pnl)
        max_dd = float(drawdown.min())
        max_dd_date = drawdown.idxmin() if not drawdown.empty else None

        win_days = int((daily_pnl > 0).sum())
        loss_days = int((daily_pnl < 0).sum())
        hit_rate = win_days / n_days if n_days > 0 else 0.0
        best_day = float(daily_pnl.max())
        worst_day = float(daily_pnl.min())

        # ── Per-sector breakdown ─────────────────────────────────────────
        sector_pnl_results: Dict[str, SectorPnL] = {}
        for sec in sectors:
            sec_series = sector_contrib[sec]
            active = sec_series[sec_series.abs() > 1e-10]
            n_active = len(active)
            sec_hits = int((active > 0).sum())
            sec_misses = int((active < 0).sum())
            sec_total = float(sec_series.sum())
            sec_cum = sec_series.cumsum()
            sec_dd = _max_drawdown_series(sec_cum)

            sector_pnl_results[sec] = SectorPnL(
                sector=sec,
                total_pnl=sec_total,
                avg_daily_pnl=float(active.mean()) if n_active > 0 else 0.0,
                hit_rate=sec_hits / n_active if n_active > 0 else 0.0,
                n_active_days=n_active,
                n_hits=sec_hits,
                n_misses=sec_misses,
                sharpe=_sharpe(sec_series),
                max_drawdown=float(sec_dd.min()),
                best_day=float(sec_series.max()),
                worst_day=float(sec_series.min()),
                avg_weight=float(active.abs().mean()) if n_active > 0 else 0.0,
            )

        # ── Regime-conditional P&L ───────────────────────────────────────
        regime_series = pd.Series(regime_labels, index=idx, name="regime")
        regime_pnl_results: Dict[str, RegimePnL] = {}
        for regime in ["CALM", "NORMAL", "TENSION", "CRISIS"]:
            mask = regime_series == regime
            if mask.sum() == 0:
                regime_pnl_results[regime] = RegimePnL(
                    regime=regime, total_pnl=0.0, n_days=0,
                    avg_daily_pnl=0.0, sharpe=float("nan"),
                    hit_rate=0.0, max_drawdown=0.0,
                )
                continue
            r_pnl = daily_pnl[mask]
            r_cum = r_pnl.cumsum()
            r_dd = _max_drawdown_series(r_cum)
            r_wins = int((r_pnl > 0).sum())

            regime_pnl_results[regime] = RegimePnL(
                regime=regime,
                total_pnl=float(r_pnl.sum()),
                n_days=int(mask.sum()),
                avg_daily_pnl=float(r_pnl.mean()),
                sharpe=_sharpe(r_pnl),
                hit_rate=r_wins / int(mask.sum()) if int(mask.sum()) > 0 else 0.0,
                max_drawdown=float(r_dd.min()),
            )

        # ── Monthly returns grid ─────────────────────────────────────────
        monthly = daily_pnl.resample("ME").sum()
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        monthly_pivot = monthly_df.pivot_table(
            index="year", columns="month", values="return", aggfunc="sum"
        ).fillna(0.0)
        # Add annual total
        monthly_pivot["Total"] = monthly_pivot.sum(axis=1)

        # ── Summary DataFrame ────────────────────────────────────────────
        summary_rows = []
        for sec in sectors:
            sp = sector_pnl_results[sec]
            summary_rows.append({
                "sector": sec,
                "total_pnl": round(sp.total_pnl * 100, 2),  # in %
                "hit_rate": round(sp.hit_rate * 100, 1),
                "sharpe": round(sp.sharpe, 2) if math.isfinite(sp.sharpe) else None,
                "max_dd": round(sp.max_drawdown * 100, 2),
                "n_days": sp.n_active_days,
                "avg_weight": round(sp.avg_weight, 4),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False)

        # ── Factor attribution (SPY/TNX/DXY/HYG/idiosyncratic) ─────────
        factor_attribution = None
        factor_daily_df = None
        sortino_ratio = 0.0
        info_ratio = 0.0
        turnover = 0.0

        try:
            factor_cols = {}
            spy_col = self._settings.spy_ticker if hasattr(self._settings, 'spy_ticker') else "SPY"
            if spy_col in log_rets.columns:
                factor_cols["spy_beta"] = log_rets[spy_col]
            for name, col in [("rates_tnx", "^TNX"), ("dollar_dxy", "DX-Y.NYB"), ("credit_hyg", "HYG")]:
                if col in prices_df.columns:
                    factor_cols[name] = prices_df[col].pct_change().reindex(daily_pnl.index).fillna(0)

            if factor_cols:
                F = pd.DataFrame(factor_cols).reindex(daily_pnl.index).fillna(0)
                # Regress portfolio P&L on factors
                F_with_const = np.column_stack([np.ones(len(F)), F.values])
                y = daily_pnl.values
                try:
                    betas, _, _, _ = np.linalg.lstsq(F_with_const, y, rcond=None)
                    factor_pnl = F.values @ betas[1:]
                    idio_pnl = y - factor_pnl

                    factor_daily_df = pd.DataFrame(index=daily_pnl.index)
                    for i, fname in enumerate(factor_cols.keys()):
                        factor_daily_df[fname] = F.values[:, i] * betas[i + 1]
                    factor_daily_df["idiosyncratic"] = idio_pnl

                    factor_attribution = {}
                    for col in factor_daily_df.columns:
                        factor_attribution[col] = round(float(factor_daily_df[col].sum()), 6)
                except np.linalg.LinAlgError:
                    pass

            # Sortino ratio
            downside = daily_pnl[daily_pnl < 0]
            dd_std = float(downside.std()) if len(downside) > 5 else pnl_vol
            sortino_ratio = float(avg_pnl / dd_std * np.sqrt(252)) if dd_std > 1e-10 else 0.0

            # Information ratio (vs SPY)
            if spy_col in log_rets.columns:
                spy_daily = log_rets[spy_col].reindex(daily_pnl.index).fillna(0)
                active_ret = daily_pnl - spy_daily
                te = float(active_ret.std())
                info_ratio = float(active_ret.mean() / te * np.sqrt(252)) if te > 1e-10 else 0.0

            # Turnover estimate (from weight changes)
            if not sector_contrib.empty:
                weight_changes = sector_contrib.diff().abs().sum(axis=1)
                turnover = float(weight_changes.mean() * 252)  # Annualized
        except Exception as _attr_err:
            self._log.debug("Factor attribution failed: %s", _attr_err)

        self._log.info(
            "P&L tracker: %d days | total=%.2f%% | Sharpe=%.2f | MaxDD=%.2f%% | HR=%.1f%%",
            n_days, total_pnl * 100, sharpe, max_dd * 100, hit_rate * 100,
        )

        return PnLResult(
            total_pnl=total_pnl,
            sharpe=sharpe,
            calmar=calmar,
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            hit_rate=hit_rate,
            avg_daily_pnl=avg_pnl,
            pnl_volatility=pnl_vol,
            best_day=best_day,
            worst_day=worst_day,
            win_days=win_days,
            loss_days=loss_days,
            n_trading_days=n_days,
            daily_pnl=daily_pnl,
            cumulative_pnl=cumulative_pnl,
            drawdown=drawdown,
            rolling_sharpe=rolling_sharpe,
            sector_pnl=sector_pnl_results,
            regime_pnl=regime_pnl_results,
            monthly_returns=monthly_pivot,
            sector_contribution=sector_contrib,
            factor_attribution=factor_attribution,
            factor_daily=factor_daily_df,
            sortino=round(sortino_ratio, 4),
            information_ratio=round(info_ratio, 4),
            turnover_annual=round(turnover, 2),
            summary_df=summary_df,
        )


# ── Module-level convenience ─────────────────────────────────────────────

def run_pnl_tracker(
    prices_df: pd.DataFrame,
    settings: Any,
    lookback_days: Optional[int] = None,
) -> PnLResult:
    """Run P&L tracking using default parameters."""
    return PnLTracker(settings).track(prices_df, lookback_days)
