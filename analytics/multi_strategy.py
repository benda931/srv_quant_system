"""
analytics/multi_strategy.py
============================
Multi-strategy ensemble combiner for the SRV Quantamental DSS.

Combines three orthogonal alpha sources:
  1. Mean Reversion (MR) — PCA residual z-score based, whitelist sectors
  2. Dispersion Timing — long sector vol / short index vol when cheap
  3. Momentum + Mean Reversion Hybrid — MR in direction of medium-term momentum

Ensemble methods:
  - equal_weight: 1/N capital per strategy
  - sharpe_weighted: allocate proportional to rolling Sharpe
  - regime_adaptive: regime-dependent weights
  - risk_parity: equal risk contribution

Ref: DeMiguel et al. (2009) — "Optimal vs Naive Diversification"
Ref: Roncalli (2013) — "Introduction to Risk Parity and Budgeting"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# All 11 SPDR sector ETFs
ALL_SECTORS = [
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
]

# MR whitelist (defensive / low-beta sectors with stronger mean-reversion)
MR_WHITELIST = ["XLC", "XLF", "XLI", "XLP", "XLU"]


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A single directional signal from one strategy."""
    date: pd.Timestamp
    sector: str
    direction: int          # +1 LONG, -1 SHORT
    conviction: float       # [0, 1]
    strategy_source: str    # "MR" / "Dispersion" / "MomentumMR"
    weight: float = 0.0     # final portfolio weight (set by combiner)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of a series."""
    mu = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    sd = sd.replace(0.0, np.nan)
    return (series - mu) / sd


def _sharpe_from_returns(returns: np.ndarray, ann_factor: float = np.sqrt(252)) -> float:
    """Annualised Sharpe ratio."""
    r = returns[np.isfinite(returns)]
    if len(r) < 2:
        return 0.0
    std = float(np.std(r, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(r) / std * ann_factor)


def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from a return series."""
    r = returns[np.isfinite(returns)]
    if len(r) == 0:
        return 0.0
    cum = np.cumsum(r)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    return float(dd.max()) if dd.size else 0.0


def _regime_from_vix(vix_level: float) -> str:
    """Simple VIX-based regime classification."""
    if not math.isfinite(vix_level):
        return "NORMAL"
    if vix_level < 15:
        return "CALM"
    if vix_level < 22:
        return "NORMAL"
    if vix_level < 30:
        return "TENSION"
    return "CRISIS"


# ===========================================================================
# Strategy 1: Mean Reversion
# ===========================================================================

class MeanReversionStrategy:
    """
    PCA residual z-score based mean reversion.

    Only trades whitelist sectors (XLC, XLF, XLI, XLP, XLU).
    Entry when |z| > z_entry, exit when |z| < z_exit.
    Max hold = hold_days.
    """

    def __init__(
        self,
        z_entry: float = 0.7,
        z_exit: float = 0.25,
        hold_days: int = 15,
        pca_window: int = 252,
        zscore_window: int = 60,
        whitelist: Optional[List[str]] = None,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.hold_days = hold_days
        self.pca_window = pca_window
        self.zscore_window = zscore_window
        self.whitelist = whitelist or MR_WHITELIST

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date_idx: int,
        z_scores: pd.DataFrame,
        regime: str,
    ) -> List[Signal]:
        """Generate MR signals for a single date."""
        if regime == "CRISIS":
            return []

        date = prices.index[date_idx]
        signals = []

        for sector in self.whitelist:
            if sector not in z_scores.columns:
                continue
            z = float(z_scores[sector].iloc[date_idx])
            if not math.isfinite(z):
                continue
            if abs(z) < self.z_entry:
                continue

            direction = -1 if z > 0 else 1  # mean reversion: high z -> short
            conviction = min(1.0, abs(z) / 2.0)

            signals.append(Signal(
                date=date,
                sector=sector,
                direction=direction,
                conviction=conviction,
                strategy_source="MR",
            ))

        return signals

    def compute_z_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling PCA residual z-scores for all sectors."""
        sectors = [s for s in ALL_SECTORS if s in prices.columns]
        spy = "SPY"
        if spy not in prices.columns:
            return pd.DataFrame(index=prices.index, columns=sectors, dtype=float)

        log_px = np.log(prices[sectors + [spy]].astype(float))
        returns = log_px.diff()
        rel_returns = returns[sectors].subtract(returns[spy], axis=0)

        # Simplified: rolling z-score of cumulative relative return
        cum_rel = rel_returns.fillna(0).cumsum()
        z_scores = cum_rel.apply(
            lambda col: _zscore_rolling(col, self.zscore_window), axis=0
        )
        return z_scores


# ===========================================================================
# Strategy 2: Dispersion Timing
# ===========================================================================

class DispersionStrategy:
    """
    Long individual sector vol, short index vol when dispersion is cheap.

    Signal: dispersion_ratio_z < -0.5 (low dispersion = cheap to buy)
    Entry: buy sector vol (long sector ETF), sell index vol (short SPY)
    Exit: dispersion normalizes (z > 0)

    Only enters when VIX < 25 and regime != CRISIS.
    """

    def __init__(
        self,
        dispersion_z_entry: float = -0.5,
        dispersion_z_exit: float = 0.0,
        vix_max: float = 25.0,
        hold_days: int = 21,
        vol_lookback: int = 20,
    ):
        self.dispersion_z_entry = dispersion_z_entry
        self.dispersion_z_exit = dispersion_z_exit
        self.vix_max = vix_max
        self.hold_days = hold_days
        self.vol_lookback = vol_lookback

    def compute_dispersion_z(self, prices: pd.DataFrame) -> pd.Series:
        """
        Compute dispersion ratio z-score.

        Dispersion = avg(sector_vol) / index_vol.
        When dispersion is low relative to history, it is cheap to buy.
        """
        sectors = [s for s in ALL_SECTORS if s in prices.columns]
        spy = "SPY"
        if spy not in prices.columns or not sectors:
            return pd.Series(dtype=float, index=prices.index)

        returns = prices.pct_change()
        spy_vol = returns[spy].rolling(self.vol_lookback).std() * np.sqrt(252)
        sector_vols = returns[sectors].rolling(self.vol_lookback).std() * np.sqrt(252)
        avg_sector_vol = sector_vols.mean(axis=1)

        dispersion_ratio = avg_sector_vol / spy_vol.replace(0, np.nan)
        dispersion_z = _zscore_rolling(dispersion_ratio, window=60)
        return dispersion_z

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date_idx: int,
        dispersion_z: pd.Series,
        vix_level: float,
        regime: str,
    ) -> List[Signal]:
        """Generate dispersion timing signals for a single date."""
        if regime == "CRISIS":
            return []
        if not math.isfinite(vix_level) or vix_level > self.vix_max:
            return []

        dz = float(dispersion_z.iloc[date_idx]) if date_idx < len(dispersion_z) else float("nan")
        if not math.isfinite(dz):
            return []
        if dz > self.dispersion_z_entry:
            return []  # dispersion not cheap enough

        date = prices.index[date_idx]
        sectors = [s for s in ALL_SECTORS if s in prices.columns]
        signals = []

        # When dispersion is cheap, go long high-vol sectors, short SPY
        returns = prices.pct_change()
        sector_vols = returns[sectors].rolling(self.vol_lookback).std() * np.sqrt(252)

        if date_idx < self.vol_lookback:
            return []

        vol_row = sector_vols.iloc[date_idx]
        vol_median = vol_row.median()

        conviction = min(1.0, abs(dz) / 1.5)

        for sector in sectors:
            sv = float(vol_row[sector]) if sector in vol_row.index else float("nan")
            if not math.isfinite(sv):
                continue
            # Long high-vol sectors (they benefit most from dispersion increase)
            if sv > vol_median:
                signals.append(Signal(
                    date=date,
                    sector=sector,
                    direction=1,  # long the sector
                    conviction=conviction * 0.8,
                    strategy_source="Dispersion",
                ))

        return signals


# ===========================================================================
# Strategy 3: Momentum + Mean Reversion Hybrid
# ===========================================================================

class MomentumMRStrategy:
    """
    Mean reversion on relative strength, but only in direction of
    medium-term momentum. Avoids catching falling knives.

    Entry when:
      1. |z| > 0.7 (MR signal exists)
      2. 60d momentum of sector vs SPY agrees with MR direction
         (z < 0 AND sector outperforming SPY over 60d = LONG)
         (z > 0 AND sector underperforming SPY over 60d = SHORT)
      3. All 11 sectors eligible
    """

    def __init__(
        self,
        z_entry: float = 0.7,
        z_exit: float = 0.25,
        momentum_window: int = 60,
        hold_days: int = 15,
        zscore_window: int = 60,
    ):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.momentum_window = momentum_window
        self.hold_days = hold_days
        self.zscore_window = zscore_window

    def compute_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling momentum of sector vs SPY."""
        sectors = [s for s in ALL_SECTORS if s in prices.columns]
        spy = "SPY"
        if spy not in prices.columns:
            return pd.DataFrame(index=prices.index, columns=sectors, dtype=float)

        rel = prices[sectors].div(prices[spy], axis=0)
        momentum = rel.pct_change(self.momentum_window)
        return momentum

    def generate_signals(
        self,
        prices: pd.DataFrame,
        date_idx: int,
        z_scores: pd.DataFrame,
        momentum: pd.DataFrame,
        regime: str,
    ) -> List[Signal]:
        """Generate momentum-confirmed MR signals."""
        if regime == "CRISIS":
            return []

        date = prices.index[date_idx]
        sectors = [s for s in ALL_SECTORS if s in prices.columns]
        signals = []

        for sector in sectors:
            if sector not in z_scores.columns or sector not in momentum.columns:
                continue

            z = float(z_scores[sector].iloc[date_idx])
            mom = float(momentum[sector].iloc[date_idx])

            if not (math.isfinite(z) and math.isfinite(mom)):
                continue
            if abs(z) < self.z_entry:
                continue

            # MR direction
            mr_direction = -1 if z > 0 else 1

            # Momentum must agree: sector outperforming (mom > 0) -> LONG OK
            mom_direction = 1 if mom > 0 else -1
            if mr_direction != mom_direction:
                continue  # momentum disagrees with MR -> skip (falling knife)

            conviction = min(1.0, abs(z) / 2.0) * min(1.0, abs(mom) * 10)

            signals.append(Signal(
                date=date,
                sector=sector,
                direction=mr_direction,
                conviction=conviction,
                strategy_source="MomentumMR",
            ))

        return signals


# ===========================================================================
# Ensemble Combiner
# ===========================================================================

class StrategyEnsemble:
    """
    Combines multiple strategy signals into a single portfolio.

    Methods:
      - equal_weight: each strategy gets 1/N capital
      - sharpe_weighted: allocate more to higher-Sharpe strategies
      - regime_adaptive: different weights per regime
      - risk_parity: equal risk contribution
    """

    # Default regime-adaptive weights: {regime: {strategy: weight}}
    REGIME_WEIGHTS = {
        "CALM": {"MR": 0.40, "Dispersion": 0.35, "MomentumMR": 0.25},
        "NORMAL": {"MR": 0.35, "Dispersion": 0.30, "MomentumMR": 0.35},
        "TENSION": {"MR": 0.50, "Dispersion": 0.15, "MomentumMR": 0.35},
        "CRISIS": {"MR": 0.20, "Dispersion": 0.10, "MomentumMR": 0.70},
    }

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        method: str = "sharpe_weighted",
    ):
        self.strategy_names = strategies or ["MR", "Dispersion", "MomentumMR"]
        self.method = method
        self._rolling_returns: Dict[str, List[float]] = {s: [] for s in self.strategy_names}
        self._rolling_sharpes: Dict[str, float] = {s: 1.0 for s in self.strategy_names}

    def _equal_weights(self) -> Dict[str, float]:
        n = len(self.strategy_names)
        return {s: 1.0 / n for s in self.strategy_names}

    def _sharpe_weights(self) -> Dict[str, float]:
        """Allocate proportional to rolling Sharpe (floored at 0)."""
        sharpes = {s: max(0.01, self._rolling_sharpes.get(s, 0.01))
                   for s in self.strategy_names}
        total = sum(sharpes.values())
        if total < 1e-12:
            return self._equal_weights()
        return {s: v / total for s, v in sharpes.items()}

    def _regime_weights(self, regime: str) -> Dict[str, float]:
        """Regime-adaptive weights."""
        w = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS["NORMAL"])
        # Only include strategies we have
        filtered = {s: w.get(s, 0.0) for s in self.strategy_names}
        total = sum(filtered.values())
        if total < 1e-12:
            return self._equal_weights()
        return {s: v / total for s, v in filtered.items()}

    def _risk_parity_weights(self) -> Dict[str, float]:
        """Equal risk contribution (inverse-vol weighting)."""
        vols = {}
        for s in self.strategy_names:
            rets = np.array(self._rolling_returns.get(s, []))
            rets = rets[np.isfinite(rets)]
            vol = float(np.std(rets, ddof=1)) if len(rets) > 5 else 1.0
            vols[s] = max(vol, 1e-6)

        inv_vol = {s: 1.0 / v for s, v in vols.items()}
        total = sum(inv_vol.values())
        if total < 1e-12:
            return self._equal_weights()
        return {s: v / total for s, v in inv_vol.items()}

    def get_weights(self, regime: str = "NORMAL") -> Dict[str, float]:
        """Return strategy weights based on the chosen method."""
        if self.method == "equal_weight":
            return self._equal_weights()
        elif self.method == "sharpe_weighted":
            return self._sharpe_weights()
        elif self.method == "regime_adaptive":
            return self._regime_weights(regime)
        elif self.method == "risk_parity":
            return self._risk_parity_weights()
        else:
            return self._equal_weights()

    def update_rolling_stats(self, strategy_name: str, period_return: float) -> None:
        """Update rolling return history for a strategy."""
        if strategy_name not in self._rolling_returns:
            self._rolling_returns[strategy_name] = []
        self._rolling_returns[strategy_name].append(period_return)

        # Keep last 252 periods
        if len(self._rolling_returns[strategy_name]) > 252:
            self._rolling_returns[strategy_name] = self._rolling_returns[strategy_name][-252:]

        # Update rolling Sharpe
        rets = np.array(self._rolling_returns[strategy_name])
        self._rolling_sharpes[strategy_name] = _sharpe_from_returns(rets)

    def combine_signals(
        self,
        all_signals: Dict[str, List[Signal]],
        regime: str = "NORMAL",
    ) -> pd.DataFrame:
        """
        Combine signals from multiple strategies.

        Parameters
        ----------
        all_signals : {strategy_name: [Signal, ...]}
        regime : current market regime

        Returns
        -------
        pd.DataFrame with columns:
            sector, direction, conviction, strategy_source, weight
        """
        weights = self.get_weights(regime)
        rows = []

        for strategy_name, signals in all_signals.items():
            w = weights.get(strategy_name, 0.0)
            for sig in signals:
                rows.append({
                    "date": sig.date,
                    "sector": sig.sector,
                    "direction": sig.direction,
                    "conviction": sig.conviction,
                    "strategy_source": sig.strategy_source,
                    "weight": w * sig.conviction,
                })

        if not rows:
            return pd.DataFrame(columns=["date", "sector", "direction",
                                         "conviction", "strategy_source", "weight"])

        df = pd.DataFrame(rows)

        # Aggregate: if multiple strategies signal same sector, combine
        if len(df) > 0:
            agg = df.groupby("sector").agg({
                "date": "first",
                "direction": lambda x: int(np.sign(np.sum(x * df.loc[x.index, "weight"]))),
                "conviction": "mean",
                "strategy_source": lambda x: "+".join(sorted(set(x))),
                "weight": "sum",
            }).reset_index()
            # Remove zero-direction signals
            agg = agg[agg["direction"] != 0]
            return agg

        return df

    def backtest_ensemble(
        self,
        prices: pd.DataFrame,
        vix_col: str = "^VIX",
        step: int = 5,
        fwd_period: int = 5,
    ) -> Dict[str, Any]:
        """
        Walk-forward backtest of the full ensemble.

        Returns
        -------
        dict with: sharpe, max_dd, annual_return, total_trades,
                   equity_curve, strategy_sharpes
        """
        spy = "SPY"
        sectors = [s for s in ALL_SECTORS if s in prices.columns]
        if spy not in prices.columns or not sectors:
            raise ValueError("prices must contain SPY and at least one sector.")

        vix = prices[vix_col] if vix_col in prices.columns else pd.Series(
            20.0, index=prices.index
        )

        # Instantiate strategies
        mr = MeanReversionStrategy()
        disp = DispersionStrategy()
        mom_mr = MomentumMRStrategy()

        # Pre-compute signals infrastructure
        z_scores = mr.compute_z_scores(prices)
        dispersion_z = disp.compute_dispersion_z(prices)
        momentum = mom_mr.compute_momentum(prices)

        log_px = np.log(prices[sectors + [spy]].astype(float))

        warmup = max(252, mr.pca_window)
        n = len(prices)

        all_returns: List[float] = []
        strategy_returns: Dict[str, List[float]] = {
            "MR": [], "Dispersion": [], "MomentumMR": [],
        }
        total_trades = 0
        dates = []

        for i in range(warmup, n - fwd_period, step):
            vix_level = float(vix.iloc[i]) if i < len(vix) else 20.0
            regime = _regime_from_vix(vix_level)

            # Gather signals from each strategy
            mr_signals = mr.generate_signals(prices, i, z_scores, regime)
            disp_signals = disp.generate_signals(prices, i, dispersion_z, vix_level, regime)
            mom_signals = mom_mr.generate_signals(prices, i, z_scores, momentum, regime)

            all_sigs = {
                "MR": mr_signals,
                "Dispersion": disp_signals,
                "MomentumMR": mom_signals,
            }

            # Combine
            combined = self.combine_signals(all_sigs, regime)

            if combined.empty:
                all_returns.append(0.0)
                for sn in strategy_returns:
                    strategy_returns[sn].append(0.0)
                dates.append(prices.index[i])
                continue

            # Forward return
            fwd_end = min(i + fwd_period, n - 1)
            fwd_ret = log_px.iloc[fwd_end][sectors] - log_px.iloc[i][sectors]

            # Portfolio return from combined signals
            port_ret = 0.0
            total_weight = 0.0
            for _, row in combined.iterrows():
                sec = row["sector"]
                if sec in fwd_ret.index:
                    r = float(fwd_ret[sec])
                    if math.isfinite(r):
                        w = float(row["weight"]) * float(row["direction"])
                        port_ret += w * r
                        total_weight += abs(float(row["weight"]))

            if total_weight > 1e-12:
                port_ret /= total_weight

            all_returns.append(port_ret)
            total_trades += len(combined)
            dates.append(prices.index[i])

            # Per-strategy returns for rolling Sharpe updates
            for sn, sigs in all_sigs.items():
                strat_ret = 0.0
                strat_w = 0.0
                for sig in sigs:
                    if sig.sector in fwd_ret.index:
                        r = float(fwd_ret[sig.sector])
                        if math.isfinite(r):
                            strat_ret += sig.direction * sig.conviction * r
                            strat_w += sig.conviction
                if strat_w > 1e-12:
                    strat_ret /= strat_w
                strategy_returns[sn].append(strat_ret)
                self.update_rolling_stats(sn, strat_ret)

        # Compute metrics
        ret_arr = np.array(all_returns)
        ann_factor = np.sqrt(252.0 / step)
        sharpe = _sharpe_from_returns(ret_arr, ann_factor)
        max_dd = _max_drawdown(ret_arr)
        annual_return = float(np.sum(ret_arr)) / max(1, len(ret_arr)) * (252 / step)

        equity_curve = pd.Series(np.cumsum(ret_arr), index=dates[:len(ret_arr)])

        strat_sharpes = {
            sn: _sharpe_from_returns(np.array(rets), ann_factor)
            for sn, rets in strategy_returns.items()
        }

        return {
            "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 6),
            "annual_return": round(annual_return, 4),
            "total_trades": total_trades,
            "n_periods": len(all_returns),
            "equity_curve": equity_curve,
            "strategy_sharpes": strat_sharpes,
            "strategy_returns": strategy_returns,
            "ensemble_returns": all_returns,
        }


# ===========================================================================
# __main__: Run backtest comparison
# ===========================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load prices
    project_root = Path(__file__).resolve().parents[1]
    prices_path = project_root / "data_lake" / "parquet" / "prices.parquet"

    if not prices_path.exists():
        print(f"ERROR: prices file not found at {prices_path}")
        sys.exit(1)

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

    print("=" * 72)
    print("MULTI-STRATEGY ENSEMBLE BACKTEST")
    print("=" * 72)
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Rows: {len(prices)}")
    sectors_avail = [s for s in ALL_SECTORS if s in prices.columns]
    print(f"Sectors: {len(sectors_avail)} / 11")
    print()

    # Run ensemble with each method
    for method in ["equal_weight", "sharpe_weighted", "regime_adaptive", "risk_parity"]:
        print(f"--- Ensemble method: {method} ---")
        ensemble = StrategyEnsemble(method=method)
        try:
            result = ensemble.backtest_ensemble(prices, step=5, fwd_period=5)

            # Print comparison table
            print(f"{'Strategy':<20} {'Sharpe':>10} {'MaxDD':>10} {'AnnRet':>10}")
            print("-" * 52)

            for sn in ["MR", "Dispersion", "MomentumMR"]:
                s = result["strategy_sharpes"].get(sn, 0.0)
                # Compute per-strategy max_dd and annual_return
                rets = np.array(result["strategy_returns"].get(sn, []))
                mdd = _max_drawdown(rets)
                ann_ret = float(np.sum(rets)) / max(1, len(rets)) * (252 / 5)
                print(f"{sn:<20} {s:>10.3f} {mdd:>10.4f} {ann_ret:>10.4f}")

            print(f"{'ENSEMBLE':<20} {result['sharpe']:>10.3f} {result['max_dd']:>10.4f} {result['annual_return']:>10.4f}")
            print(f"  Total trades: {result['total_trades']}")
            print(f"  Periods: {result['n_periods']}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    print("=" * 72)
    print("Done.")
