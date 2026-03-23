"""
scripts/agent_portfolio_optimizer.py
--------------------------------------
סוכן 4: אופטימיזציית פורטפוליו מלאה מאפס

ROLE:    מריץ Constrained Mean-Variance Optimization, מחשב optimal weights,
         משווה מול heuristic w_final, ושולח תוצאות ל-Claude לניתוח והמלצות.

INPUTS:  master_df (QuantEngine)
         prices_df (data_lake parquet)
         config/settings.py (vol_target, max_sector_weight)

OUTPUTS: logs/optimization_<ts>.json
         → agent_bus["agent_portfolio_optimizer"]

מריץ Constrained Mean-Variance Optimization:
  - מחשב expected returns מ-Z-scores וציוני MC
  - Covariance: Ledoit-Wolf shrinkage
  - Constraints: max weight, net exposure, regime-aware leverage
  - Objective: maximize Sharpe ratio (adjustable)
  - מדווח ל-Claude עם תוצאות + ממתין להוראות המשך

הרצה:
  python scripts/agent_portfolio_optimizer.py
  python scripts/agent_portfolio_optimizer.py --objective min_risk
  python scripts/agent_portfolio_optimizer.py --no-claude   # without AI loop
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "optimizer.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("agent_portfolio_optimizer")


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OptimizationResult:
    objective:          str
    weights:            Dict[str, float]
    expected_return:    float
    expected_vol:       float
    sharpe:             float
    max_weight:         float
    net_exposure:       float
    gross_exposure:     float
    n_longs:            int
    n_shorts:           int
    regime:             str
    constraint_summary: str
    vs_heuristic:       Dict[str, float] = field(default_factory=dict)   # comparison vs w_final


# ─────────────────────────────────────────────────────────────────────────────
def _load_system_state() -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """Load master_df, prices_df, settings."""
    from config.settings import get_settings
    from data_ops.orchestrator import DataOrchestrator
    from analytics.stat_arb import QuantEngine

    settings = get_settings()
    orchestrator = DataOrchestrator(settings)
    data_state = orchestrator.run(force_refresh=False)

    engine = QuantEngine(settings)
    engine.load()
    master_df = engine.calculate_conviction_score()
    prices_df = data_state.artifacts.prices if data_state.artifacts else pd.DataFrame()

    return master_df, prices_df, settings


def _expected_returns(master_df: pd.DataFrame) -> pd.Series:
    """
    Proxy expected returns from signal strength:
    E[r_i] = direction_sign * mc_score * z_score_magnitude / scaling
    """
    df = master_df.copy()
    z = pd.to_numeric(df.get("pca_residual_z", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    mc = pd.to_numeric(df.get("mc_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0) / 100.0
    direction_sign = df["direction"].map({"LONG": 1.0, "SHORT": -1.0, "NEUTRAL": 0.0}).fillna(0.0)
    tickers = df["sector_ticker"].values

    # Signal: signed z-score weighted by MC confidence, normalized to ~ 1% annualized
    raw = direction_sign * mc * z.abs() * 0.01
    return pd.Series(raw.values, index=tickers)


def _ledoit_wolf_cov(prices_df: pd.DataFrame, tickers: list, window: int = 252) -> np.ndarray:
    """Compute Ledoit-Wolf shrunk covariance for given tickers."""
    from sklearn.covariance import LedoitWolf
    cols = [t for t in tickers if t in prices_df.columns]
    if len(cols) < 2:
        return np.eye(len(tickers)) * (0.01 ** 2)

    returns = np.log(prices_df[cols] / prices_df[cols].shift(1)).dropna().tail(window)
    if len(returns) < 30:
        return np.eye(len(tickers)) * (0.01 ** 2)

    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_
    # Annualize
    return cov * 252


def optimize(
    master_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    settings,
    objective: str = "max_sharpe",
) -> OptimizationResult:
    """
    Run constrained portfolio optimization.

    objective: "max_sharpe" | "min_risk" | "max_return"
    """
    from scipy.optimize import minimize

    # ── Universe: only tradable sectors ──────────────────────────────────────
    tradable = master_df[master_df["direction"].isin(["LONG", "SHORT"])].copy()
    if tradable.empty:
        log.warning("No tradable sectors — returning zero weights.")
        return OptimizationResult(
            objective=objective, weights={}, expected_return=0.0, expected_vol=0.0,
            sharpe=0.0, max_weight=0.0, net_exposure=0.0, gross_exposure=0.0,
            n_longs=0, n_shorts=0, regime="UNKNOWN", constraint_summary="No tradable sectors",
        )

    tickers = tradable["sector_ticker"].tolist()
    n = len(tickers)
    regime = str(master_df["market_state"].iloc[0]) if "market_state" in master_df.columns else "NORMAL"

    # ── Expected returns + covariance ─────────────────────────────────────────
    mu_all = _expected_returns(master_df)
    mu = np.array([float(mu_all.get(t, 0.0)) for t in tickers])
    Sigma = _ledoit_wolf_cov(prices_df, tickers)

    # ── Regime-dependent constraints ─────────────────────────────────────────
    regime_leverage = {"CALM": 1.0, "NORMAL": 0.9, "TENSION": 0.65, "CRISIS": 0.35}
    max_leverage    = regime_leverage.get(regime, 0.8)
    max_single      = float(getattr(settings, "max_sector_weight", 0.20))
    vol_target      = float(getattr(settings, "vol_target", 0.12))

    direction_signs = np.array([1.0 if d == "LONG" else -1.0
                                 for d in tradable["direction"].values])

    # ── Objective functions ───────────────────────────────────────────────────
    def portfolio_vol(w: np.ndarray) -> float:
        return float(np.sqrt(max(w @ Sigma @ w, 1e-10)))

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = portfolio_vol(w)
        return -(ret / vol) if vol > 1e-8 else 0.0

    def neg_return(w: np.ndarray) -> float:
        return -float(w @ mu)

    obj_fn = {"max_sharpe": neg_sharpe, "min_risk": portfolio_vol, "max_return": neg_return}[objective]

    # ── Constraints ───────────────────────────────────────────────────────────
    # Weights must be positive (we handle direction via sign adjustment below)
    bounds = [(0.0, max_single)] * n

    constraints = [
        # Gross exposure ≤ max_leverage
        {"type": "ineq", "fun": lambda w: max_leverage - w.sum()},
        # Gross exposure ≥ 0.1 (some exposure)
        {"type": "ineq", "fun": lambda w: w.sum() - 0.1},
        # Vol target: annualized portfolio vol ≤ vol_target
        {"type": "ineq", "fun": lambda w: vol_target - portfolio_vol(w * direction_signs)},
    ]

    # ── Solve ─────────────────────────────────────────────────────────────────
    w0 = np.ones(n) / n * max_leverage * 0.5
    result = minimize(obj_fn, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 500, "ftol": 1e-9})

    w_opt = np.clip(result.x, 0.0, max_single)
    # Apply direction signs → actual signed weights
    w_signed = w_opt * direction_signs

    opt_weights = {tickers[i]: round(float(w_signed[i]), 5) for i in range(n)}

    # ── Compare vs heuristic w_final ─────────────────────────────────────────
    vs_heuristic = {}
    if "w_final" in tradable.columns:
        for _, row in tradable.iterrows():
            t = row["sector_ticker"]
            wf = float(row.get("w_final", 0.0))
            wo = opt_weights.get(t, 0.0)
            vs_heuristic[t] = {"w_heuristic": round(wf, 5), "w_optimal": round(wo, 5), "delta": round(wo - wf, 5)}

    # ── Metrics ───────────────────────────────────────────────────────────────
    w_arr = np.array([opt_weights[t] for t in tickers])
    exp_ret = float(w_arr @ mu)
    exp_vol = portfolio_vol(w_arr)
    sharpe  = (exp_ret / exp_vol) if exp_vol > 1e-8 else 0.0

    constraint_summary = (
        f"regime={regime}, max_leverage={max_leverage:.0%}, "
        f"max_single={max_single:.0%}, vol_target={vol_target:.0%}, "
        f"converged={result.success}"
    )

    return OptimizationResult(
        objective=objective,
        weights=opt_weights,
        expected_return=round(exp_ret, 5),
        expected_vol=round(exp_vol, 5),
        sharpe=round(sharpe, 3),
        max_weight=round(float(np.abs(w_arr).max()), 5) if n > 0 else 0.0,
        net_exposure=round(float(w_arr.sum()), 5),
        gross_exposure=round(float(np.abs(w_arr).sum()), 5),
        n_longs=int((w_arr > 0).sum()),
        n_shorts=int((w_arr < 0).sum()),
        regime=regime,
        constraint_summary=constraint_summary,
        vs_heuristic=vs_heuristic,
    )


def build_claude_message(opt: OptimizationResult) -> str:
    """Build the message to send to Claude with optimization results."""
    weights_txt = "\n".join(
        f"  {t}: {w:+.2%}" for t, w in sorted(opt.weights.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    vs_txt = "\n".join(
        f"  {t}: heuristic={v['w_heuristic']:+.2%} → optimal={v['w_optimal']:+.2%} (Δ={v['delta']:+.2%})"
        for t, v in opt.vs_heuristic.items()
    )

    return f"""## Portfolio Optimization Results

**Objective:** {opt.objective}
**Regime:** {opt.regime}
**Constraints:** {opt.constraint_summary}

### Optimal Weights
{weights_txt}

### Key Metrics
- Expected Return (proxy): {opt.expected_return:.3%}
- Expected Vol (ann):      {opt.expected_vol:.2%}
- Implied Sharpe:          {opt.sharpe:.3f}
- Gross Exposure:          {opt.gross_exposure:.2%}
- Net Exposure:            {opt.net_exposure:.2%}
- Positions: {opt.n_longs} LONG, {opt.n_shorts} SHORT

### vs Heuristic Sizing (w_final)
{vs_txt}

---
כאן תוצאות האופטימיזציה. אנא:
1. בדוק אם המשקולות נראים הגיוניים ביחס למצב הנוכחי
2. אם Sharpe < 0.5 — הצע שינויי פרמטרים (חלון קוברינס, vol_target, max_single_weight)
3. אם יש gap גדול בין optimal ל-heuristic — הסבר למה ומה לעשות
4. שלח actions להמשך אם צריך
"""


SYSTEM_PROMPT = """אתה Portfolio Optimizer AI של מערכת SRV Quantamental DSS.
תפקידך: לנתח תוצאות אופטימיזציה, לזהות בעיות, ולתת הוראות ספציפיות לשיפור.

כאשר תרצה לבצע פעולה, שלח בלוק JSON:
```json
{"actions": [
  {"type": "edit_param", "file": "config/settings.py", "param": "vol_target", "value": 0.10},
  {"type": "run_tests", "path": "tests/"},
  {"type": "run_script", "script": "scripts/agent_daily_pipeline.py"},
  {"type": "log", "message": "הערה שלך"},
  {"type": "done", "summary": "סיכום הפעולות"}
]}
```

כללים:
- שמור על ממשקים קיימים — שנה רק פרמטרים ב-config/settings.py
- כל שינוי חייב לעבור tests
- הסבר כל החלטה מתמטית"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Optimizer Agent")
    parser.add_argument("--objective",   default="max_sharpe", choices=["max_sharpe", "min_risk", "max_return"])
    parser.add_argument("--no-claude",   action="store_true",   help="Skip Claude API loop, just print results")
    args = parser.parse_args()

    log.info("Loading system state...")
    master_df, prices_df, settings = _load_system_state()

    log.info("Running optimization (objective=%s)...", args.objective)
    opt = optimize(master_df, prices_df, settings, objective=args.objective)

    log.info("Results: Sharpe=%.3f, Vol=%.1f%%, Gross=%.1f%%",
             opt.sharpe, opt.expected_vol * 100, opt.gross_exposure * 100)

    # Save results
    out = ROOT / "logs" / f"optimization_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(opt.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("Results saved → %s", out.name)

    # ── Publish to AgentBus ───────────────────────────────────────────────────
    from scripts.agent_bus import get_bus
    get_bus().publish("agent_portfolio_optimizer", {
        "status": "ok",
        "objective": opt.objective,
        "sharpe": opt.sharpe,
        "expected_vol_pct": round(opt.expected_vol * 100, 2),
        "gross_exposure": opt.gross_exposure,
        "n_longs": opt.n_longs,
        "n_shorts": opt.n_shorts,
        "regime": opt.regime,
    })

    if args.no_claude:
        print(json.dumps(opt.__dict__, ensure_ascii=False, indent=2, default=str))
        sys.exit(0)

    # ── Claude feedback loop ──────────────────────────────────────────────────
    from scripts.claude_loop import run_agent_loop
    run_agent_loop(
        agent_name="portfolio_optimizer",
        system_prompt=SYSTEM_PROMPT,
        initial_message=build_claude_message(opt),
    )
