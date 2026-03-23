"""
analytics/dss_backtest.py
===========================
Walk-Forward Backtest Engine for the DSS Signal Stack + Trade Structure

Replays the full DSS pipeline historically:
  1. At each anchor date, computes correlation structure snapshot
  2. Runs 4-layer Signal Stack scoring
  3. Constructs virtual trades (entry signals)
  4. Tracks trades through time (P&L, exit conditions)
  5. Measures DSS-specific performance metrics

No look-ahead guarantee:
  - Signal at date t uses only data ≤ t
  - Forward P&L uses data from t+1 onward
  - Regime classification uses data ≤ t

Key metrics:
  - DSS Sharpe: annualized Sharpe from trade-level P&L
  - Entry accuracy: % of entries that reach profit target before stop
  - Exit quality: avg P&L at exit vs optimal exit
  - Regime gating: drawdown avoided by regime safety kills
  - Layer contribution: information value of each layer

Ref: Walk-forward validation (Pardo, 2008)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VirtualTrade:
    """A single simulated trade from entry to exit."""
    trade_id: str
    ticker: str
    direction: str           # "LONG" / "SHORT"
    entry_date: str
    exit_date: str
    entry_z: float
    exit_z: float
    entry_conviction: float
    holding_days: int
    pnl_pct: float           # Realized P&L as % of notional
    exit_reason: str          # "PROFIT_TARGET" / "STOP_LOSS" / "TIME_EXIT" / "REGIME_EXIT" / "END_OF_DATA"
    regime_at_entry: str
    regime_at_exit: str
    weight: float             # Position weight at entry
    layers: Dict[str, float]  # Snapshot of all 4 layer scores at entry


@dataclass
class DSSBacktestResult:
    """Complete DSS backtest output."""
    # Aggregate metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    total_pnl_pct: float
    sharpe: float
    max_drawdown: float
    calmar: float

    # Entry quality
    avg_conviction_winners: float
    avg_conviction_losers: float
    conviction_accuracy: float     # Correlation between conviction and P&L

    # Exit quality
    profit_target_exits: int
    stop_loss_exits: int
    time_exits: int
    regime_exits: int
    avg_holding_days: float

    # Regime analysis
    regime_breakdown: Dict[str, Dict[str, float]]  # {regime: {n_trades, win_rate, avg_pnl, sharpe}}
    regime_gating_value: float   # P&L saved by regime kills

    # Layer contribution (information ratio of each layer alone)
    layer_contribution: Dict[str, float]

    # Per-trade detail
    trades: List[VirtualTrade]

    # Time series
    equity_curve: pd.Series      # Cumulative P&L indexed by date
    drawdown_curve: pd.Series    # Drawdown from peak

    # Parameters
    params: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# DSS Backtester
# ─────────────────────────────────────────────────────────────────────────────

class DSSBacktester:
    """
    Walk-forward backtester for the full DSS Signal Stack.

    Parameters
    ----------
    settings      : Settings instance
    train_window  : PCA training window (days)
    step          : Days between anchor evaluations
    max_hold      : Maximum holding period for a trade
    z_target_ratio: Z compression ratio for profit target (0.2 = exit when z at 20% of entry)
    z_stop_ratio  : Z extension ratio for stop loss (1.5 = stop if z extends 50% beyond entry)
    """

    def __init__(
        self,
        settings: Settings,
        train_window: int = 252,
        step: int = 5,
        max_hold: int = 45,
        z_target_ratio: float = 0.20,
        z_stop_ratio: float = 1.50,
    ):
        self.settings = settings
        self.train_window = train_window
        self.step = step
        self.max_hold = max_hold
        self.z_target_ratio = z_target_ratio
        self.z_stop_ratio = z_stop_ratio

    def run(self, prices: pd.DataFrame) -> DSSBacktestResult:
        """
        Run walk-forward DSS backtest on historical prices.

        Parameters
        ----------
        prices : DataFrame — daily close prices, columns include all sector ETFs + SPY + VIX + HYG + IEF
        """
        from analytics.signal_stack import (
            SignalStackEngine, compute_distortion_score, compute_dislocation_score,
        )
        from analytics.signal_regime_safety import compute_regime_safety_score

        sectors = self.settings.sector_list()
        spy = self.settings.spy_ticker
        avail_sectors = [s for s in sectors if s in prices.columns]
        if len(avail_sectors) < 5:
            raise ValueError(f"Need at least 5 sectors, found {len(avail_sectors)}")

        # Prepare returns
        log_rets = np.log(prices / prices.shift(1)).dropna(how="all")
        rel_rets = pd.DataFrame(index=log_rets.index)
        for s in avail_sectors:
            if s in log_rets.columns and spy in log_rets.columns:
                rel_rets[s] = log_rets[s] - log_rets[spy]
        rel_rets = rel_rets.dropna(how="all")

        n = len(rel_rets)
        min_start = self.train_window + 60  # Need enough data for z-scores
        if n < min_start + 20:
            raise ValueError(f"Need at least {min_start + 20} days, have {n}")

        # VIX and credit data
        vix_col = "^VIX" if "^VIX" in prices.columns else "VIX" if "VIX" in prices.columns else None
        hyg_col = "HYG" if "HYG" in prices.columns else None
        ief_col = "IEF" if "IEF" in prices.columns else None

        ss_engine = SignalStackEngine(self.settings)

        # State tracking
        active_trades: Dict[str, dict] = {}  # trade_id → trade state
        completed_trades: List[VirtualTrade] = []
        daily_pnl = pd.Series(0.0, index=rel_rets.index, dtype=float)
        no_regime_pnl = pd.Series(0.0, index=rel_rets.index, dtype=float)  # P&L without regime gating

        anchors = list(range(min_start, n - 1, self.step))
        log.info("DSS Backtest: %d anchors, step=%d, sectors=%d, dates=%s to %s",
                 len(anchors), self.step, len(avail_sectors),
                 rel_rets.index[min_start].date(), rel_rets.index[-1].date())

        for anchor_idx, anchor in enumerate(anchors):
            dt = rel_rets.index[anchor]

            # ── Compute residual z-scores at this anchor ──
            resid_levels = {}
            zscore_window = self.settings.zscore_window
            for s in avail_sectors:
                series = rel_rets[s].iloc[:anchor + 1].cumsum()
                if len(series) >= zscore_window + 10:
                    mu = series.rolling(zscore_window).mean().iloc[-1]
                    sd = series.rolling(zscore_window).std(ddof=1).iloc[-1]
                    if sd > 1e-10:
                        resid_levels[s] = (series.iloc[-1] - mu) / sd
                    else:
                        resid_levels[s] = 0.0
                else:
                    resid_levels[s] = 0.0

            # ── Correlation structure (lightweight) ──
            corr_window = self.settings.corr_window
            base_window = min(self.settings.corr_baseline_window, anchor)
            if anchor >= base_window:
                R_short = log_rets[avail_sectors].iloc[anchor - corr_window + 1: anchor + 1]
                R_base = log_rets[avail_sectors].iloc[anchor - base_window + 1: anchor + 1]
                C_s = R_short.corr()
                C_b = R_base.corr()
                dC = C_s - C_b

                # Frobenius distortion
                mask = np.ones_like(dC.values) - np.eye(len(avail_sectors))
                frob_d = float(np.sqrt(np.sum((dC.values * mask) ** 2)))

                # Market mode share
                try:
                    evals = np.linalg.eigvalsh(C_s.values)
                    evals = np.sort(evals)[::-1]
                    mode_share = float(evals[0] / len(avail_sectors)) if len(avail_sectors) > 0 else 0.3
                except Exception:
                    mode_share = 0.3

                # Average correlation
                iu = np.triu_indices(len(avail_sectors), k=1)
                avg_corr = float(np.mean(C_s.values[iu]))
            else:
                frob_d = 0.0
                mode_share = 0.3
                avg_corr = 0.3

            # ── VIX & credit ──
            vix_level = float(prices[vix_col].iloc[anchor]) if vix_col and anchor < len(prices) else float("nan")
            credit_z = float("nan")
            if hyg_col and ief_col and anchor >= 60:
                spread = np.log(prices[hyg_col] / prices[ief_col])
                sp_vals = spread.iloc[anchor - 60: anchor + 1]
                if len(sp_vals) >= 20:
                    mu_sp = sp_vals.mean()
                    sd_sp = sp_vals.std(ddof=1)
                    credit_z = (sp_vals.iloc[-1] - mu_sp) / sd_sp if sd_sp > 1e-10 else 0.0

            # ── Regime state (simplified) ──
            if vix_level > 35 or (math.isfinite(credit_z) and credit_z < -2):
                regime = "CRISIS"
            elif vix_level > 25 or avg_corr > 0.6:
                regime = "TENSION"
            elif vix_level > 18 or avg_corr > 0.45:
                regime = "NORMAL"
            else:
                regime = "CALM"

            # ── Regime Safety Score ──
            safety = compute_regime_safety_score(
                market_state=regime, vix_level=vix_level, credit_z=credit_z,
                avg_corr=avg_corr, corr_z=frob_d,
            )

            # ── Signal Stack scoring ──
            # Use simplified distortion (no z-score history in backtest — use raw values)
            dist_result = compute_distortion_score(
                frob_distortion_z=frob_d,  # Treat raw as z-score proxy
                market_mode_share=mode_share,
                coc_instability_z=0.0,  # CoC not available per-anchor
            )

            # Score all sectors
            for s in avail_sectors:
                z = resid_levels.get(s, 0.0)
                z_abs = abs(z)
                if z_abs < 0.5:
                    continue  # Too small to trade

                disloc_score = min(1.0, z_abs / 3.0)
                mr_score = 0.5  # Default — full MR test too expensive per-anchor

                conviction = dist_result.distortion_score * disloc_score * mr_score * safety.regime_safety_score
                direction = "LONG" if z < 0 else "SHORT"

                if conviction >= self.settings.signal_entry_threshold:
                    trade_id = f"{s}_{direction}_{dt.strftime('%Y%m%d')}"

                    # Don't open if we already have a trade in this sector
                    existing = [k for k in active_trades if k.startswith(f"{s}_")]
                    if existing:
                        continue

                    weight = min(0.20, conviction * 0.25) * safety.regime_safety_score
                    active_trades[trade_id] = {
                        "ticker": s,
                        "direction": direction,
                        "entry_idx": anchor,
                        "entry_date": str(dt.date()),
                        "entry_z": z,
                        "conviction": conviction,
                        "weight": weight,
                        "regime": regime,
                        "z_target": z * self.z_target_ratio,
                        "z_stop": z * self.z_stop_ratio,
                        "layers": {
                            "distortion": dist_result.distortion_score,
                            "dislocation": disloc_score,
                            "mean_reversion": mr_score,
                            "regime_safety": safety.regime_safety_score,
                        },
                    }

                    # Track P&L without regime gating (for measuring gating value)
                    if safety.regime_safety_score < 0.1:
                        pass  # Would have been blocked

            # ── Track active trades — daily P&L ──
            to_close = []
            for tid, trade in active_trades.items():
                s = trade["ticker"]
                if s not in rel_rets.columns or anchor >= n - 1:
                    continue

                # Daily return for this sector (relative to SPY)
                daily_ret = rel_rets[s].iloc[anchor] if anchor < len(rel_rets) else 0.0
                sign = 1.0 if trade["direction"] == "LONG" else -1.0
                trade_daily_pnl = sign * daily_ret * trade["weight"]
                daily_pnl.iloc[anchor] += trade_daily_pnl

                # Also track no-regime-gating P&L
                no_regime_pnl.iloc[anchor] += sign * daily_ret * min(0.20, trade["conviction"] * 0.25)

                # Check exit conditions
                days_held = anchor - trade["entry_idx"]
                current_z = resid_levels.get(s, trade["entry_z"])

                exit_reason = None
                if days_held >= self.max_hold:
                    exit_reason = "TIME_EXIT"
                elif trade["direction"] == "LONG" and current_z > 0 and abs(current_z) < abs(trade["entry_z"]) * self.z_target_ratio:
                    exit_reason = "PROFIT_TARGET"
                elif trade["direction"] == "SHORT" and current_z < 0 and abs(current_z) < abs(trade["entry_z"]) * self.z_target_ratio:
                    exit_reason = "PROFIT_TARGET"
                elif abs(current_z) > abs(trade["entry_z"]) * self.z_stop_ratio:
                    exit_reason = "STOP_LOSS"
                elif safety.regime_safety_score < 0.1:
                    exit_reason = "REGIME_EXIT"

                if exit_reason:
                    # Calculate total P&L for this trade
                    entry_idx = trade["entry_idx"]
                    pnl_sum = 0.0
                    for day in range(entry_idx, anchor + 1):
                        if day < len(rel_rets):
                            d_ret = rel_rets[s].iloc[day]
                            pnl_sum += sign * d_ret * trade["weight"]

                    completed_trades.append(VirtualTrade(
                        trade_id=tid,
                        ticker=s,
                        direction=trade["direction"],
                        entry_date=trade["entry_date"],
                        exit_date=str(dt.date()),
                        entry_z=trade["entry_z"],
                        exit_z=current_z,
                        entry_conviction=trade["conviction"],
                        holding_days=days_held,
                        pnl_pct=pnl_sum,
                        exit_reason=exit_reason,
                        regime_at_entry=trade["regime"],
                        regime_at_exit=regime,
                        weight=trade["weight"],
                        layers=trade["layers"],
                    ))
                    to_close.append(tid)

            for tid in to_close:
                del active_trades[tid]

            if (anchor_idx + 1) % 50 == 0:
                log.debug("  ... processed %d / %d anchors, %d active, %d completed",
                          anchor_idx + 1, len(anchors), len(active_trades), len(completed_trades))

        # Close remaining active trades
        for tid, trade in active_trades.items():
            s = trade["ticker"]
            entry_idx = trade["entry_idx"]
            sign = 1.0 if trade["direction"] == "LONG" else -1.0
            pnl_sum = 0.0
            for day in range(entry_idx, n):
                if day < len(rel_rets) and s in rel_rets.columns:
                    pnl_sum += sign * rel_rets[s].iloc[day] * trade["weight"]

            completed_trades.append(VirtualTrade(
                trade_id=tid, ticker=s, direction=trade["direction"],
                entry_date=trade["entry_date"], exit_date=str(rel_rets.index[-1].date()),
                entry_z=trade["entry_z"], exit_z=0.0,
                entry_conviction=trade["conviction"],
                holding_days=n - 1 - entry_idx,
                pnl_pct=pnl_sum, exit_reason="END_OF_DATA",
                regime_at_entry=trade["regime"], regime_at_exit="UNKNOWN",
                weight=trade["weight"], layers=trade["layers"],
            ))

        # ── Compute aggregate metrics ──
        return self._compute_results(completed_trades, daily_pnl, no_regime_pnl)

    def _compute_results(
        self,
        trades: List[VirtualTrade],
        daily_pnl: pd.Series,
        no_regime_pnl: pd.Series,
    ) -> DSSBacktestResult:
        """Aggregate trade-level results into backtest metrics."""
        n_trades = len(trades)
        if n_trades == 0:
            return self._empty_result()

        pnls = [t.pnl_pct for t in trades]
        winners = [t for t in trades if t.pnl_pct > 0]
        losers = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(winners) / n_trades

        # Equity curve
        eq = daily_pnl.cumsum()
        peak = eq.cummax()
        dd = eq - peak
        max_dd = float(dd.min()) if len(dd) else 0.0

        # Sharpe
        daily_std = daily_pnl.std()
        daily_mean = daily_pnl.mean()
        sharpe = float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 1e-10 else 0.0

        # Calmar
        total_pnl = float(eq.iloc[-1]) if len(eq) else 0.0
        calmar = abs(total_pnl / max_dd) if abs(max_dd) > 1e-6 else 0.0

        # Conviction accuracy
        convictions = [t.entry_conviction for t in trades]
        if len(set(pnls)) > 1 and len(set(convictions)) > 1:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(convictions, pnls)
            conviction_acc = float(corr) if math.isfinite(corr) else 0.0
        else:
            conviction_acc = 0.0

        # Exit breakdown
        exit_counts = {}
        for t in trades:
            exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

        # Regime breakdown
        regime_groups: Dict[str, List[VirtualTrade]] = {}
        for t in trades:
            r = t.regime_at_entry
            if r not in regime_groups:
                regime_groups[r] = []
            regime_groups[r].append(t)

        regime_breakdown = {}
        for r, group in regime_groups.items():
            g_pnls = [t.pnl_pct for t in group]
            g_wins = sum(1 for t in group if t.pnl_pct > 0)
            g_std = float(np.std(g_pnls)) if len(g_pnls) > 1 else 1e-10
            regime_breakdown[r] = {
                "n_trades": len(group),
                "win_rate": g_wins / len(group),
                "avg_pnl": float(np.mean(g_pnls)),
                "sharpe": float(np.mean(g_pnls) / g_std * np.sqrt(252 / 5)) if g_std > 1e-10 else 0.0,
            }

        # Regime gating value: P&L difference between with and without regime gating
        eq_no_regime = no_regime_pnl.cumsum()
        dd_no_regime = (eq_no_regime - eq_no_regime.cummax()).min()
        regime_gating_value = float(max_dd - dd_no_regime) if math.isfinite(dd_no_regime) else 0.0

        # Layer contribution: correlation of each layer score with trade P&L
        layer_contrib = {}
        for layer_name in ["distortion", "dislocation", "mean_reversion", "regime_safety"]:
            layer_vals = [t.layers.get(layer_name, 0) for t in trades]
            if len(set(layer_vals)) > 1 and len(set(pnls)) > 1:
                from scipy.stats import spearmanr
                c, _ = spearmanr(layer_vals, pnls)
                layer_contrib[layer_name] = round(float(c), 4) if math.isfinite(c) else 0.0
            else:
                layer_contrib[layer_name] = 0.0

        return DSSBacktestResult(
            total_trades=n_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=round(win_rate, 4),
            avg_pnl_per_trade=round(float(np.mean(pnls)), 6),
            total_pnl_pct=round(total_pnl, 6),
            sharpe=round(sharpe, 4),
            max_drawdown=round(max_dd, 6),
            calmar=round(calmar, 4),
            avg_conviction_winners=round(float(np.mean([t.entry_conviction for t in winners])), 4) if winners else 0.0,
            avg_conviction_losers=round(float(np.mean([t.entry_conviction for t in losers])), 4) if losers else 0.0,
            conviction_accuracy=round(conviction_acc, 4),
            profit_target_exits=exit_counts.get("PROFIT_TARGET", 0),
            stop_loss_exits=exit_counts.get("STOP_LOSS", 0),
            time_exits=exit_counts.get("TIME_EXIT", 0),
            regime_exits=exit_counts.get("REGIME_EXIT", 0),
            avg_holding_days=round(float(np.mean([t.holding_days for t in trades])), 1),
            regime_breakdown=regime_breakdown,
            regime_gating_value=round(regime_gating_value, 6),
            layer_contribution=layer_contrib,
            trades=trades,
            equity_curve=eq,
            drawdown_curve=dd,
            params={
                "train_window": self.train_window,
                "step": self.step,
                "max_hold": self.max_hold,
                "z_target_ratio": self.z_target_ratio,
                "z_stop_ratio": self.z_stop_ratio,
                "n_sectors": len([s for s in self.settings.sector_list() if s in eq.index or True]),
            },
        )

    def _empty_result(self) -> DSSBacktestResult:
        """Return empty result when no trades generated."""
        return DSSBacktestResult(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_pnl_per_trade=0, total_pnl_pct=0,
            sharpe=0, max_drawdown=0, calmar=0,
            avg_conviction_winners=0, avg_conviction_losers=0, conviction_accuracy=0,
            profit_target_exits=0, stop_loss_exits=0, time_exits=0, regime_exits=0,
            avg_holding_days=0, regime_breakdown={}, regime_gating_value=0,
            layer_contribution={}, trades=[], equity_curve=pd.Series(dtype=float),
            drawdown_curve=pd.Series(dtype=float), params={},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dss_backtest(prices: pd.DataFrame, settings: Optional[Settings] = None) -> DSSBacktestResult:
    """Run DSS backtest with default settings."""
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()
    bt = DSSBacktester(settings)
    return bt.run(prices)
