"""
Portfolio Construction Agent — Optimal allocation across signals.

Bridge between signals and money. Combines 3 methods:
1. Mean-Variance (Markowitz) — maximize risk-adjusted return
2. Risk Parity — equal risk contribution per position
3. Conviction-Weighted — proportional to signal strength

Regime-adaptive blending:
  CALM:    50% MV + 30% RP + 20% Conv
  NORMAL:  40% MV + 40% RP + 20% Conv
  TENSION: 20% MV + 60% RP + 20% Conv
  CRISIS:   0% MV + 80% RP + 20% Conv
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_DIR / "portfolio_construction.log", maxBytes=10_000_000, backupCount=3),
    ],
)
log = logging.getLogger("portfolio_construction")

OUTPUT_DIR = ROOT / "agents" / "portfolio_construction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]


class PortfolioConstructor:
    """Optimizes portfolio allocation across signals."""

    def __init__(self, settings=None):
        self._settings = settings
        self._prices: Optional[pd.DataFrame] = None

    @property
    def settings(self):
        if self._settings is None:
            from config.settings import get_settings
            self._settings = get_settings()
        return self._settings

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            p = ROOT / "data_lake" / "parquet" / "prices.parquet"
            if p.exists():
                self._prices = pd.read_parquet(p)
        return self._prices

    def load_current_signals(self) -> Optional[pd.DataFrame]:
        """Load latest signals from QuantEngine."""
        try:
            from analytics.stat_arb import QuantEngine
            qe = QuantEngine(self.settings)
            master_df, _ = qe.run(self.prices)
            return master_df
        except Exception as e:
            log.warning("Cannot load signals: %s", e)
            return None

    def compute_covariance(self, lookback: int = 60) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance matrix."""
        avail = [s for s in SECTORS if s in self.prices.columns]
        rets = np.log(self.prices[avail] / self.prices[avail].shift(1)).dropna().iloc[-lookback:]

        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(rets.values)
            cov = pd.DataFrame(lw.covariance_, index=avail, columns=avail)
        except ImportError:
            cov = rets.cov()

        return cov * 252  # Annualize

    def mean_variance_optimize(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                                risk_aversion: float = 2.0) -> Dict[str, float]:
        """Markowitz mean-variance optimization."""
        from scipy.optimize import minimize

        tickers = list(cov_matrix.columns)
        n = len(tickers)
        mu = np.array([expected_returns.get(t, 0.0) for t in tickers])
        sigma = cov_matrix.values

        def neg_utility(w):
            ret = w @ mu
            risk = w @ sigma @ w
            return -(ret - risk_aversion / 2 * risk)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 0.0},  # Market neutral
        ]
        bounds = [(-0.20, 0.20)] * n

        w0 = np.zeros(n)
        result = minimize(neg_utility, w0, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            return {t: round(float(w), 4) for t, w in zip(tickers, result.x)}
        else:
            log.warning("MV optimization failed: %s", result.message)
            return {t: 0.0 for t in tickers}

    def risk_parity_optimize(self, cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Equal risk contribution weights."""
        from scipy.optimize import minimize

        tickers = list(cov_matrix.columns)
        n = len(tickers)
        sigma = cov_matrix.values

        def risk_contrib_obj(w):
            w = np.abs(w)
            port_var = w @ sigma @ w
            if port_var < 1e-12:
                return 1e6
            marginal = sigma @ w
            risk_contrib = w * marginal / np.sqrt(port_var)
            target = np.sqrt(port_var) / n
            return np.sum((risk_contrib - target) ** 2)

        w0 = np.ones(n) / n
        bounds = [(0.01, 0.30)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(risk_contrib_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            return {t: round(float(w), 4) for t, w in zip(tickers, result.x)}
        else:
            return {t: round(1.0 / n, 4) for t in tickers}

    def conviction_weighted(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Weight proportional to conviction × direction."""
        weights = {}
        total_abs = 0

        for _, row in signals.iterrows():
            ticker = str(row.get("sector_ticker", ""))
            direction = str(row.get("direction", "NEUTRAL"))
            conviction = float(row.get("conviction_score", 0))

            if direction == "NEUTRAL" or conviction < 0.05:
                weights[ticker] = 0.0
                continue

            sign = 1.0 if direction == "LONG" else -1.0
            w = sign * conviction
            weights[ticker] = w
            total_abs += abs(w)

        # Normalize so gross = 1.0
        if total_abs > 0:
            for t in weights:
                weights[t] = round(weights[t] / total_abs, 4)

        return weights

    def blend_methods(self, mv_weights: Dict, rp_weights: Dict, conv_weights: Dict,
                       regime: str = "NORMAL") -> Dict[str, float]:
        """Regime-adaptive blending of 3 optimization methods."""
        blends = {
            "CALM":    (0.50, 0.30, 0.20),
            "NORMAL":  (0.40, 0.40, 0.20),
            "TENSION": (0.20, 0.60, 0.20),
            "CRISIS":  (0.00, 0.80, 0.20),
        }
        w_mv, w_rp, w_conv = blends.get(regime, (0.40, 0.40, 0.20))

        all_tickers = set(list(mv_weights.keys()) + list(rp_weights.keys()) + list(conv_weights.keys()))
        blended = {}

        for t in all_tickers:
            mv = mv_weights.get(t, 0.0)
            rp = rp_weights.get(t, 0.0)
            conv = conv_weights.get(t, 0.0)
            blended[t] = round(w_mv * mv + w_rp * rp + w_conv * conv, 4)

        return blended

    def apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce portfolio constraints."""
        max_single = 0.20
        max_gross = 2.0
        max_net = 0.40

        # Cap single positions
        for t in weights:
            weights[t] = max(-max_single, min(max_single, weights[t]))

        # Check gross
        gross = sum(abs(w) for w in weights.values())
        if gross > max_gross:
            scale = max_gross / gross
            for t in weights:
                weights[t] = round(weights[t] * scale, 4)

        # Check net
        net = sum(weights.values())
        if abs(net) > max_net:
            # Reduce the side with more exposure
            excess = net - (max_net if net > 0 else -max_net)
            direction = "LONG" if net > 0 else "SHORT"
            sorted_pos = sorted(
                [(t, w) for t, w in weights.items() if (w > 0) == (direction == "LONG")],
                key=lambda x: abs(x[1]), reverse=True
            )
            for t, w in sorted_pos:
                reduce = min(abs(w), abs(excess))
                weights[t] = round(w - (reduce if w > 0 else -reduce), 4)
                excess -= reduce
                if abs(excess) < 0.001:
                    break

        # Remove tiny weights
        weights = {t: w for t, w in weights.items() if abs(w) > 0.005}

        return weights

    def _get_regime(self) -> str:
        """Get current regime from prices."""
        vix = self.prices.get("^VIX", pd.Series(dtype=float)).dropna()
        if len(vix) == 0:
            return "NORMAL"
        v = float(vix.iloc[-1])
        if v < 15: return "CALM"
        if v < 22: return "NORMAL"
        if v < 30: return "TENSION"
        return "CRISIS"

    def run(self) -> Dict:
        """Full cycle: signals → covariance → optimize → blend → constrain → save."""
        log.info("=" * 60)
        log.info("PORTFOLIO CONSTRUCTION — %s", datetime.now(timezone.utc).isoformat()[:19])
        log.info("=" * 60)

        regime = self._get_regime()
        log.info("Regime: %s", regime)

        # Covariance
        cov = self.compute_covariance(lookback=60)
        log.info("Covariance: %d×%d matrix", cov.shape[0], cov.shape[1])

        # Expected returns from momentum
        avail = [s for s in SECTORS if s in self.prices.columns]
        spy = self.prices.get("SPY", pd.Series(dtype=float)).dropna()
        expected_returns = {}
        for s in avail:
            px = self.prices[s].dropna()
            if len(px) >= 63 and len(spy) >= 63:
                ret = float(np.log(px.iloc[-1] / px.iloc[-63]))
                spy_ret = float(np.log(spy.iloc[-1] / spy.iloc[-63]))
                expected_returns[s] = (ret - spy_ret) * 4  # Annualize 63d
            else:
                expected_returns[s] = 0.0
        er = pd.Series(expected_returns)

        # 3 optimization methods
        mv_weights = self.mean_variance_optimize(er, cov)
        log.info("MV weights: %s", {k: v for k, v in mv_weights.items() if abs(v) > 0.01})

        rp_weights = self.risk_parity_optimize(cov)
        log.info("RP weights: %s", {k: v for k, v in rp_weights.items() if abs(v) > 0.01})

        # Conviction from signals
        signals = self.load_current_signals()
        if signals is not None:
            conv_weights = self.conviction_weighted(signals)
        else:
            conv_weights = {t: 0.0 for t in avail}
        log.info("Conv weights: %s", {k: v for k, v in conv_weights.items() if abs(v) > 0.01})

        # Blend
        blended = self.blend_methods(mv_weights, rp_weights, conv_weights, regime)
        log.info("Blended (pre-constraint): gross=%.2f, net=%.2f",
                 sum(abs(w) for w in blended.values()), sum(blended.values()))

        # Constrain
        final = self.apply_constraints(blended)
        gross = sum(abs(w) for w in final.values())
        net = sum(final.values())

        # Metrics
        final_arr = np.array([final.get(t, 0) for t in avail])
        exp_ret = float(final_arr @ np.array([er.get(t, 0) for t in avail]))
        exp_vol = float(np.sqrt(final_arr @ cov.loc[avail, avail].values @ final_arr))
        sharpe_est = exp_ret / exp_vol if exp_vol > 0 else 0

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weights": final,
            "method_weights": {"mv": mv_weights, "rp": rp_weights, "conv": conv_weights},
            "blend_regime": regime,
            "gross_exposure": round(gross, 3),
            "net_exposure": round(net, 3),
            "n_positions": sum(1 for w in final.values() if abs(w) > 0.005),
            "expected_return": round(exp_ret, 4),
            "expected_vol": round(exp_vol, 4),
            "sharpe_estimate": round(sharpe_est, 3),
        }

        log.info("Final: %d positions, gross=%.2f, net=%.2f, Sharpe est=%.3f",
                 result["n_positions"], gross, net, sharpe_est)
        for t, w in sorted(final.items(), key=lambda x: -abs(x[1])):
            log.info("  %s: %+.1f%%", t, w * 100)

        # Save
        (OUTPUT_DIR / "portfolio_weights.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

        # Bus + registry
        try:
            from agents.shared.agent_bus import get_bus
            get_bus().publish("portfolio_construction", result)
        except Exception:
            pass
        try:
            from agents.shared.agent_registry import get_registry, AgentStatus
            get_registry().heartbeat("portfolio_construction", AgentStatus.COMPLETED)
        except Exception:
            pass

        return result


def main():
    parser = argparse.ArgumentParser(description="Portfolio Construction Agent")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        pc = PortfolioConstructor()
        result = pc.run()
        print(f"\n{result['n_positions']} positions, Sharpe est={result['sharpe_estimate']:.3f}")
    else:
        print("Usage: python -m agents.portfolio_construction --once")


if __name__ == "__main__":
    main()
