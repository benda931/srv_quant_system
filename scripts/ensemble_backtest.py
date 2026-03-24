"""
scripts/ensemble_backtest.py
============================
Multi-strategy ensemble backtest with leverage scaling.

Compares:
  1. MR only (no leverage)
  2. MR + leverage
  3. Ensemble (no leverage)
  4. Ensemble + leverage

Saves best result to data/ensemble_results.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.multi_strategy import (
    MeanReversionStrategy,
    StrategyEnsemble,
    _regime_from_vix,
    _sharpe_from_returns,
    _max_drawdown,
    ALL_SECTORS,
)
from analytics.leverage_engine import LeverageEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
log = logging.getLogger("ensemble_backtest")


def _run_mr_backtest(
    prices: pd.DataFrame,
    vix: pd.Series,
    step: int = 5,
    fwd_period: int = 5,
    leverage_engine: LeverageEngine | None = None,
) -> dict:
    """Run MR-only backtest, optionally with leverage scaling."""
    spy = "SPY"
    sectors = [s for s in ALL_SECTORS if s in prices.columns]
    mr = MeanReversionStrategy()
    z_scores = mr.compute_z_scores(prices)
    log_px = np.log(prices[sectors + [spy]].astype(float))

    warmup = max(252, mr.pca_window)
    n = len(prices)
    all_returns: list[float] = []
    dates: list = []

    for i in range(warmup, n - fwd_period, step):
        vix_level = float(vix.iloc[i]) if i < len(vix) else 20.0
        regime = _regime_from_vix(vix_level)

        signals = mr.generate_signals(prices, i, z_scores, regime)
        if not signals:
            all_returns.append(0.0)
            dates.append(prices.index[i])
            continue

        fwd_end = min(i + fwd_period, n - 1)
        fwd_ret = log_px.iloc[fwd_end][sectors] - log_px.iloc[i][sectors]

        port_ret = 0.0
        total_weight = 0.0
        for sig in signals:
            if sig.sector in fwd_ret.index:
                r = float(fwd_ret[sig.sector])
                if np.isfinite(r):
                    w = sig.conviction * sig.direction
                    port_ret += w * r
                    total_weight += sig.conviction

        if total_weight > 1e-12:
            port_ret /= total_weight

        # Apply leverage scaling
        if leverage_engine is not None:
            cum_ret = sum(all_returns)
            dd_pct = 0.0
            if all_returns:
                cum_arr = np.cumsum(all_returns)
                running_max = np.maximum.accumulate(cum_arr)
                dd_pct = float(running_max[-1] - cum_arr[-1])

            lev_result = leverage_engine.compute_target_leverage(
                regime=regime,
                vix=vix_level if np.isfinite(vix_level) else 20.0,
                current_dd_pct=dd_pct,
                strategy_sharpe=0.885,
            )
            port_ret *= lev_result.target_leverage

        all_returns.append(port_ret)
        dates.append(prices.index[i])

    ret_arr = np.array(all_returns)
    ann_factor = np.sqrt(252.0 / step)
    sharpe = _sharpe_from_returns(ret_arr, ann_factor)
    max_dd = _max_drawdown(ret_arr)
    annual_return = float(np.sum(ret_arr)) / max(1, len(ret_arr)) * (252 / step)

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 6),
        "annual_return": round(annual_return, 4),
        "n_periods": len(all_returns),
    }


def _run_ensemble_backtest(
    prices: pd.DataFrame,
    vix: pd.Series,
    step: int = 5,
    fwd_period: int = 5,
    leverage_engine: LeverageEngine | None = None,
) -> dict:
    """Run full ensemble backtest with all 4 methods, pick best, optionally leverage."""
    ensemble = StrategyEnsemble(method="sharpe_weighted")
    result = ensemble.backtest_ensemble(prices, vix_col="^VIX", step=step, fwd_period=fwd_period)

    if leverage_engine is None:
        return {
            "sharpe": result["sharpe"],
            "max_dd": result["max_dd"],
            "annual_return": result["annual_return"],
            "n_periods": result["n_periods"],
            "total_trades": result["total_trades"],
            "strategy_sharpes": result["strategy_sharpes"],
        }

    # Apply leverage to ensemble returns
    ens_rets = result["ensemble_returns"]
    leveraged_rets: list[float] = []
    for idx, ret in enumerate(ens_rets):
        dd_pct = 0.0
        if leveraged_rets:
            cum_arr = np.cumsum(leveraged_rets)
            running_max = np.maximum.accumulate(cum_arr)
            dd_pct = float(running_max[-1] - cum_arr[-1])

        # Estimate VIX at this step
        vix_level = 20.0  # fallback
        lev_result = leverage_engine.compute_target_leverage(
            regime="NORMAL",
            vix=vix_level,
            current_dd_pct=dd_pct,
            strategy_sharpe=0.885,
        )
        leveraged_rets.append(ret * lev_result.target_leverage)

    ret_arr = np.array(leveraged_rets)
    ann_factor = np.sqrt(252.0 / step)
    sharpe = _sharpe_from_returns(ret_arr, ann_factor)
    max_dd = _max_drawdown(ret_arr)
    annual_return = float(np.sum(ret_arr)) / max(1, len(ret_arr)) * (252 / step)

    return {
        "sharpe": round(sharpe, 4),
        "max_dd": round(max_dd, 6),
        "annual_return": round(annual_return, 4),
        "n_periods": len(leveraged_rets),
        "total_trades": result["total_trades"],
        "strategy_sharpes": result["strategy_sharpes"],
    }


def main() -> None:
    # 1. Load prices
    prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
    if not prices_path.exists():
        log.error("Prices file not found at %s", prices_path)
        sys.exit(1)

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce").ffill(limit=5)

    vix = prices["^VIX"] if "^VIX" in prices.columns else pd.Series(20.0, index=prices.index)

    log.info("Loaded prices: %d rows, %d cols, %s to %s",
             len(prices), prices.shape[1],
             prices.index[0].date(), prices.index[-1].date())

    # 2. Leverage engine
    lev_engine = LeverageEngine(
        base_capital=1_000_000,
        max_leverage=5.0,
    )

    # 3. Run all 4 configurations
    step, fwd = 5, 5
    log.info("Running MR only (no leverage)...")
    mr_no_lev = _run_mr_backtest(prices, vix, step=step, fwd_period=fwd, leverage_engine=None)

    log.info("Running MR + leverage...")
    mr_with_lev = _run_mr_backtest(prices, vix, step=step, fwd_period=fwd, leverage_engine=lev_engine)

    log.info("Running Ensemble (no leverage)...")
    ens_no_lev = _run_ensemble_backtest(prices, vix, step=step, fwd_period=fwd, leverage_engine=None)

    log.info("Running Ensemble + leverage...")
    ens_with_lev = _run_ensemble_backtest(prices, vix, step=step, fwd_period=fwd, leverage_engine=lev_engine)

    # 4. Print results table
    print()
    print("=" * 72)
    print("MULTI-STRATEGY ENSEMBLE BACKTEST COMPARISON")
    print("=" * 72)
    print(f"{'Configuration':<30} {'Sharpe':>10} {'MaxDD':>10} {'AnnRet':>10} {'Periods':>10}")
    print("-" * 72)

    configs = {
        "MR only (no leverage)": mr_no_lev,
        "MR + leverage": mr_with_lev,
        "Ensemble (no leverage)": ens_no_lev,
        "Ensemble + leverage": ens_with_lev,
    }

    for name, res in configs.items():
        print(f"{name:<30} {res['sharpe']:>10.3f} {res['max_dd']:>10.4f} {res['annual_return']:>10.4f} {res['n_periods']:>10d}")

    print("=" * 72)

    # 5. Save best result
    best_name = max(configs, key=lambda k: configs[k]["sharpe"])
    best_result = configs[best_name]
    best_result["config_name"] = best_name

    # Also include all results for reference
    all_results = {name: res for name, res in configs.items()}
    output = {
        "best": best_result,
        "all_configs": all_results,
    }

    output_path = ROOT / "data" / "ensemble_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any non-serializable types
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    output_path.write_text(
        json.dumps(output, indent=2, default=_serialize),
        encoding="utf-8",
    )
    print(f"\nBest config: {best_name} (Sharpe={best_result['sharpe']:.3f})")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
