"""
research_sandbox/eval_regime_safety.py
=======================================
READ-ONLY — fixed evaluation harness for Target 1: Regime Safety Gate.
DO NOT MODIFY THIS FILE. Only modify regime_safety_candidate.py.

Usage:
  python research_sandbox/eval_regime_safety.py

Output (JSON, stdout):
  {
    "sharpe":          float   — annualised Sharpe of dispersion backtest (PRIMARY METRIC)
    "annual_return":   float   — total P&L divided by number of years
    "pass_rate":       float   — fraction of active days where S^safe > 0
    "n_trades":        int     — completed trades in backtest period
    "win_rate":        float   — fraction of trades with positive P&L
    "max_dd":          float   — max peak-to-trough drawdown of equity curve
    "gated_off_days":  int     — days in active window where S^safe == 0.0
    "guardrails_ok":   bool    — True if all guardrails pass
  }

Guardrails (any violation → guardrails_ok: false, change must be reverted):
  pass_rate    >= 0.30   (don't block more than 70% of trading days)
  n_trades     >= 50     (minimum for reliable Sharpe estimate)
  max_dd       >= -0.15  (drawdown must not exceed -15%)
  annual_return > 0.0    (strategy must be net profitable)

Cached data source: data_lake/parquet/prices.parquet
  — 2622 rows, 2016-03-28 to 2026-03-25, 19 columns
  — no network calls, no FMP API, fully deterministic

Design notes:
  - transition_score and crisis_probability are passed as nan (no proxy
    available from price data alone; penalty contribution = 0).
  - NaN safety scores (warmup period, <252 days of credit/corr data) are
    treated as 1.0 (open gate), consistent with no-data-means-no-penalty.
  - Sharpe is computed on non-zero daily P&L days only, matching the
    convention in analytics/dispersion_backtest.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Project root → analytics.*, config.* importable
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
# research_sandbox dir → regime_safety_candidate importable by short name
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from analytics.dispersion_backtest import (
    _dispersion_pnl,
    _implied_vol_proxy,
    _realized_vol,
)
from regime_safety_candidate import compute_regime_safety_score

# ── fixed configuration ───────────────────────────────────────────────────────
DATA_PATH   = _ROOT / "data_lake" / "parquet" / "prices.parquet"
SECTORS     = ["XLC", "XLY", "XLP", "XLE", "XLF",
               "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
LOOKBACK    = 20    # realized-vol and z-score correlation window
HOLD_PERIOD = 21    # max bars to hold a position
Z_ENTRY     = 0.8   # min |z| to open a position
Z_EXIT      = 0.3   # |z| below which position is closed (signal compressed)
MAX_POS     = 3     # max simultaneous open positions
STOP_LOSS   = 0.03  # max loss per trade (3% of notional)
WARMUP      = 252   # bars skipped before backtest begins

# ── guardrail thresholds ─────────────────────────────────────────────────────
G_PASS_RATE    = 0.30
G_N_TRADES     = 50
G_MAX_DD       = -0.15
G_ANN_RETURN   = 0.0


# ── helpers ───────────────────────────────────────────────────────────────────

def _market_state(vix: float) -> str:
    """Simple VIX-based regime label, consistent with DispersionBacktester."""
    if vix < 15:  return "CALM"
    if vix < 22:  return "NORMAL"
    if vix < 30:  return "TENSION"
    return "CRISIS"


def _build_safety_series(prices: pd.DataFrame) -> pd.Series:
    """
    Compute S^safe for every row using the candidate module.

    Inputs derived from prices.parquet — zero network calls:
      vix_level  : ^VIX column (direct)
      credit_z   : z-score of log(HYG/IEF) spread, 252-day rolling window
      avg_corr   : rolling 60-day mean of sector-SPY pairwise correlations
      corr_z     : z-score of avg_corr, 252-day rolling window
      market_state: inferred from VIX via _market_state()

    transition_score and crisis_probability are passed as nan because no
    clean price-only proxy exists; both penalty sub-components return 0.0
    when their inputs are nan (see _transition_penalty in candidate).

    NaN outputs (warmup / insufficient data) are filled with 1.0 (open gate).
    """
    vix      = prices["^VIX"]
    spy_ret  = prices["SPY"].pct_change()

    # credit z-score: HYG/IEF log spread normalised over trailing 252 days
    spread   = np.log(prices["HYG"] / prices["IEF"])
    cr_mean  = spread.rolling(252, min_periods=126).mean()
    cr_std   = spread.rolling(252, min_periods=126).std()
    credit_z = (spread - cr_mean) / cr_std.replace(0, np.nan)

    # avg sector-SPY rolling correlation (60 bars)
    corr_df  = pd.DataFrame({
        s: prices[s].pct_change().rolling(60, min_periods=30).corr(spy_ret)
        for s in SECTORS if s in prices.columns
    })
    avg_corr = corr_df.mean(axis=1)

    # z-score of avg_corr over trailing 252 days
    ac_mean  = avg_corr.rolling(252, min_periods=126).mean()
    ac_std   = avg_corr.rolling(252, min_periods=126).std()
    corr_z   = (avg_corr - ac_mean) / ac_std.replace(0, np.nan)

    scores = pd.Series(np.nan, index=prices.index, dtype=float)

    for i in range(len(prices)):
        v = float(vix.iloc[i])
        if not math.isfinite(v):
            scores.iloc[i] = 1.0
            continue

        cz  = float(credit_z.iloc[i])
        ac  = float(avg_corr.iloc[i])
        czz = float(corr_z.iloc[i])

        result = compute_regime_safety_score(
            market_state      = _market_state(v),
            vix_level         = v,
            credit_z          = cz  if math.isfinite(cz)  else float("nan"),
            avg_corr          = ac  if math.isfinite(ac)  else float("nan"),
            corr_z            = czz if math.isfinite(czz) else float("nan"),
            transition_score  = float("nan"),   # no price-only proxy
            crisis_probability= float("nan"),   # no price-only proxy
        )
        scores.iloc[i] = result.regime_safety_score

    return scores.fillna(1.0)


# ── backtest loop ─────────────────────────────────────────────────────────────

def _run_backtest(prices: pd.DataFrame, safety: pd.Series) -> dict:
    """
    Walk-forward dispersion backtest with regime safety gate injected.

    The sole difference from DispersionBacktester.run() is the entry condition:
      was:  regime != "CRISIS"
      now:  regime != "CRISIS"  AND  safety[date] > 0

    Uses helper functions imported from analytics.dispersion_backtest
    (_realized_vol, _implied_vol_proxy, _dispersion_pnl) so P&L maths
    are identical to the production backtest.
    """
    vix     = prices["^VIX"]
    spy     = prices["SPY"]
    spy_ret = spy.pct_change()
    spy_rv  = _realized_vol(spy_ret, LOOKBACK)
    spy_iv  = vix / 100.0

    sector_rets  = {}
    sector_rv    = {}
    sector_iv    = {}
    sector_corr  = {}
    z_scores     = {}

    for s in SECTORS:
        if s not in prices.columns:
            continue
        ret = prices[s].pct_change()
        sector_rets[s] = ret
        sector_rv[s]   = _realized_vol(ret, LOOKBACK)

        cov  = ret.rolling(60).cov(spy_ret)
        var  = spy_ret.rolling(60).var().replace(0, np.nan)
        beta = (cov / var).clip(0.3, 3.0)
        # Use last valid beta scalar for IV proxy (matches DispersionBacktester)
        last_beta = float(beta.dropna().iloc[-1]) if beta.dropna().size else 1.0
        sector_iv[s] = _implied_vol_proxy(vix, last_beta, sector_rv[s], spy_rv)

        sector_corr[s] = ret.rolling(LOOKBACK).corr(spy_ret)

        cum_rel = np.log(prices[s] / spy)
        mu      = cum_rel.rolling(60).mean()
        sig     = cum_rel.rolling(60).std().replace(0, np.nan)
        z       = (cum_rel - mu) / sig
        z_scores[s] = z.replace([np.inf, -np.inf], 0).fillna(0)

    avg_corr_bt = pd.DataFrame(sector_corr).mean(axis=1)

    daily_pnl      = pd.Series(0.0, index=prices.index)
    trades_list    = []   # (exit_date, sector, exit_reason, regime, pnl)
    open_positions = []   # list of dicts

    active_start = WARMUP
    active_end   = len(prices) - HOLD_PERIOD

    for i in range(active_start, active_end):
        date        = prices.index[i]
        v           = float(vix.iloc[i]) if math.isfinite(float(vix.iloc[i])) else 20.0
        safe_score  = float(safety.iloc[i])
        regime      = _market_state(v)

        # ── exits ────────────────────────────────────────────────────────────
        closed = []
        for pos in open_positions:
            days = i - pos["ei"]
            s    = pos["s"]

            z_now    = float(z_scores[s].iloc[i])       if i < len(z_scores[s])            else 0.0
            rv_s     = float(sector_rv[s].iloc[i])      if i < len(sector_rv[s])           else 0.0
            rv_spy   = float(spy_rv.iloc[i])            if i < len(spy_rv)                 else 0.0
            corr_now = float(sector_corr[s].iloc[i])   if i < len(sector_corr[s])         else 0.5
            impl_c   = float(avg_corr_bt.iloc[pos["ei"]]) if pos["ei"] < len(avg_corr_bt)  else 0.5

            total, _, _, _ = _dispersion_pnl(
                iv_index_entry  = pos["iv_spy"],
                iv_sector_entry = pos["iv_s"],
                rv_index        = rv_spy,
                rv_sector       = rv_s,
                implied_corr_entry = impl_c,
                realized_corr   = corr_now,
                T               = days / 252.0,
            )
            pos["pnl"] = total

            reason = None
            if v > 45.0:                   reason = "regime_kill"
            elif abs(z_now) < Z_EXIT:      reason = "signal_compress"
            elif days >= HOLD_PERIOD:      reason = "time_exit"
            elif total < -STOP_LOSS:       reason = "stop_loss"

            if reason:
                daily_pnl.iloc[i] += total / max(1, MAX_POS)
                trades_list.append((date, s, reason, regime, total))
                closed.append(pos)

        for p in closed:
            open_positions.remove(p)

        # ── entries ──────────────────────────────────────────────────────────
        # THE KEY GATE: safe_score > 0 is the additional condition controlled
        # entirely by regime_safety_candidate.py. Changing thresholds there
        # directly changes how many days are open for new entries.
        if (len(open_positions) < MAX_POS
                and regime != "CRISIS"
                and safe_score > 0):

            candidates = []
            for s in SECTORS:
                if s not in z_scores:
                    continue
                if any(p["s"] == s for p in open_positions):
                    continue
                z = float(z_scores[s].iloc[i])
                if math.isfinite(z) and abs(z) >= Z_ENTRY:
                    candidates.append((s, z, abs(z)))

            candidates.sort(key=lambda x: x[2], reverse=True)

            for s, z, _ in candidates[:MAX_POS - len(open_positions)]:
                iv_s   = float(sector_iv[s].iloc[i]) if i < len(sector_iv[s]) else 0.0
                iv_spy = float(spy_iv.iloc[i])        if i < len(spy_iv)        else 0.0
                rv_s   = float(sector_rv[s].iloc[i]) if i < len(sector_rv[s]) else 0.0
                rv_spy = float(spy_rv.iloc[i])        if i < len(spy_rv)        else 0.0
                open_positions.append({
                    "s": s, "ei": i,
                    "iv_s": iv_s, "iv_spy": iv_spy,
                    "rv_s": rv_s, "rv_spy": rv_spy,
                    "pnl": 0.0,
                })

    # ── aggregate metrics ─────────────────────────────────────────────────────
    n_trades  = len(trades_list)
    total_pnl = float(daily_pnl.sum())

    n_active  = active_end - active_start
    n_years   = n_active / 252.0
    annual_return = round(total_pnl / n_years, 6) if n_years > 0 else 0.0

    # Sharpe: non-zero daily P&L days only (matches DispersionBacktester)
    nz = daily_pnl[daily_pnl != 0.0]
    if len(nz) > 1 and float(nz.std()) > 1e-10:
        sharpe = round(float(nz.mean() / nz.std() * math.sqrt(252)), 4)
    else:
        sharpe = 0.0

    # Max drawdown
    eq     = daily_pnl.cumsum()
    peak   = eq.cummax()
    max_dd = round(float((eq - peak).min()), 6)

    # Win rate
    win_rate = round(
        sum(1 for *_, pnl in trades_list if pnl > 0) / max(1, n_trades), 4
    )

    # Pass rate and gated-off days (active window only)
    safety_active  = safety.iloc[active_start:active_end]
    gated_off_days = int((safety_active == 0.0).sum())
    pass_rate      = round(float((safety_active > 0.0).mean()), 4)

    return {
        "sharpe":         sharpe,
        "annual_return":  annual_return,
        "pass_rate":      pass_rate,
        "n_trades":       n_trades,
        "win_rate":       win_rate,
        "max_dd":         max_dd,
        "gated_off_days": gated_off_days,
    }


# ── guardrail checker ────────────────────────────────────────────────────────

def _check_guardrails(metrics: dict) -> tuple[bool, list[str]]:
    violations = []
    if metrics["pass_rate"] < G_PASS_RATE:
        violations.append(f"pass_rate {metrics['pass_rate']} < {G_PASS_RATE}")
    if metrics["n_trades"] < G_N_TRADES:
        violations.append(f"n_trades {metrics['n_trades']} < {G_N_TRADES}")
    if metrics["max_dd"] < G_MAX_DD:
        violations.append(f"max_dd {metrics['max_dd']} < {G_MAX_DD}")
    if metrics["annual_return"] <= G_ANN_RETURN:
        violations.append(f"annual_return {metrics['annual_return']} <= {G_ANN_RETURN}")
    return len(violations) == 0, violations


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    prices = pd.read_parquet(DATA_PATH)
    safety = _build_safety_series(prices)
    result = _run_backtest(prices, safety)

    ok, violations = _check_guardrails(result)
    result["guardrails_ok"] = ok
    if violations:
        result["guardrail_violations"] = violations

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
