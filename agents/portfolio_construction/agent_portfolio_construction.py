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

import uuid

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

    # ── Institutional Methods ─────────────────────────────────────────

    def load_governance_inputs(self) -> Dict[str, Any]:
        """
        Consume methodology governance reports.

        Reads the latest methodology report from agents/methodology/reports/,
        extracts approval_matrix, scorecards, regime_fitness_map, machine_summary.
        Only APPROVED/CONDITIONAL strategies may receive allocation;
        REJECTED strategies are forced to weight = 0.
        """
        reports_dir = ROOT / "agents" / "methodology" / "reports"
        governance: Dict[str, Any] = {
            "approval_matrix": {},
            "scorecards": [],
            "regime_fitness_map": {},
            "machine_summary": {},
            "loaded": False,
        }

        if not reports_dir.exists():
            log.warning("Governance: reports dir not found at %s", reports_dir)
            return governance

        # Find latest methodology report (prefer v3 fields if present)
        candidates = sorted(reports_dir.glob("*.json"), reverse=True)
        report_data: Optional[Dict] = None
        for fpath in candidates:
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                # Must have approval_matrix or approval_matrix_v3
                if isinstance(data, dict) and (
                    "approval_matrix" in data or "approval_matrix_v3" in data
                ):
                    report_data = data
                    log.info("Governance: loaded report from %s", fpath.name)
                    break
            except (json.JSONDecodeError, OSError):
                continue

        if report_data is None:
            log.warning("Governance: no valid methodology report found")
            return governance

        # Extract approval matrix (prefer v3)
        governance["approval_matrix"] = (
            report_data.get("approval_matrix_v3")
            or report_data.get("approval_matrix", {})
        )

        # Extract scorecards
        governance["scorecards"] = report_data.get("scorecards", [])

        # Extract regime fitness map
        governance["regime_fitness_map"] = report_data.get("regime_fitness_map", {})

        # Extract machine summary (prefer v3)
        governance["machine_summary"] = (
            report_data.get("machine_summary_v3")
            or report_data.get("machine_summary", {})
        )

        governance["loaded"] = True
        n_approved = sum(
            1 for d in governance["approval_matrix"].values()
            if d in ("APPROVED", "CONDITIONAL")
            or (isinstance(d, dict) and d.get("decision") in ("APPROVED", "CONDITIONAL"))
        )
        log.info(
            "Governance: %d strategies in matrix, %d allocable (APPROVED/CONDITIONAL)",
            len(governance["approval_matrix"]),
            n_approved,
        )
        return governance

    def risk_budget_allocate(
        self, cov_matrix: pd.DataFrame, target_vol: float = 0.10,
        governance: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Allocate risk budget to each strategy given a target annual vol.

        Steps:
        1. Start with inverse-vol weights
        2. Apply regime-based tilts from methodology governance
        3. Cap individual risk contribution at 30%
        4. Scale to target vol

        Returns dict of ticker -> risk_budget_fraction.
        """
        tickers = list(cov_matrix.columns)
        n = len(tickers)
        if n == 0:
            return {}

        # Step 1: Inverse-vol weights
        vols = np.sqrt(np.diag(cov_matrix.values))
        vols = np.where(vols < 1e-8, 1e-8, vols)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()

        # Step 2: Regime tilts from governance
        if governance and governance.get("loaded"):
            regime_fitness = governance.get("regime_fitness_map", {})
            for i, t in enumerate(tickers):
                fitness = regime_fitness.get(t, {})
                if isinstance(fitness, dict):
                    tilt = fitness.get("fitness_score", 1.0)
                    weights[i] *= max(0.1, min(2.0, tilt))

            # Re-normalize
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum

        # Step 3: Cap individual risk contribution at 30%
        for _ in range(10):  # Iterative capping
            capped = False
            for i in range(n):
                if weights[i] > 0.30:
                    excess = weights[i] - 0.30
                    weights[i] = 0.30
                    # Redistribute excess proportionally
                    others = [j for j in range(n) if j != i and weights[j] < 0.30]
                    if others:
                        other_sum = sum(weights[j] for j in others)
                        if other_sum > 0:
                            for j in others:
                                weights[j] += excess * (weights[j] / other_sum)
                    capped = True
            if not capped:
                break

        # Normalize
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum

        # Step 4: Scale to target vol
        port_vol = float(np.sqrt(weights @ cov_matrix.values @ weights))
        if port_vol > 0:
            vol_scale = target_vol / port_vol
        else:
            vol_scale = 1.0

        result = {t: round(float(weights[i] * vol_scale), 4) for i, t in enumerate(tickers)}

        used = port_vol / target_vol if target_vol > 0 else 0.0
        log.info(
            "Risk budget: target_vol=%.2f, port_vol=%.4f, scale=%.2f, budget_used=%.2f",
            target_vol, port_vol, vol_scale, min(used, 1.0),
        )
        return result

    def compute_turnover(
        self, current_weights: Dict[str, float], new_weights: Dict[str, float],
    ) -> float:
        """
        Compute total turnover and cap at 20% if exceeded.

        Turnover = sum(|new - current|) / 2.
        If turnover > 20%, scale changes to cap at 20%.

        Returns actual turnover after any capping. Also modifies new_weights
        in-place if capping is applied.
        """
        all_tickers = set(list(current_weights.keys()) + list(new_weights.keys()))
        total_abs_diff = 0.0
        for t in all_tickers:
            total_abs_diff += abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0))

        turnover = total_abs_diff / 2.0

        if turnover > 0.20:
            scale = 0.20 / turnover
            log.warning(
                "Turnover %.2f%% exceeds 20%% cap — scaling changes by %.2f",
                turnover * 100, scale,
            )
            for t in all_tickers:
                curr = current_weights.get(t, 0.0)
                new = new_weights.get(t, 0.0)
                new_weights[t] = round(curr + (new - curr) * scale, 4)
            turnover = 0.20

        log.info("Turnover: %.2f%%", turnover * 100)
        return round(turnover, 4)

    def regime_adjust_weights(
        self, weights: Dict[str, float], regime: str,
    ) -> Dict[str, float]:
        """
        Regime-conditional allocation adjustments.

        CALM:    allow full allocation
        NORMAL:  cap gross exposure at 1.5x
        TENSION: reduce all weights by 40%, cap gross at 1.0x
        CRISIS:  keep only Risk Parity weights at 50% scale
        """
        adjusted = dict(weights)

        if regime == "CALM":
            # Full allocation, no adjustment
            pass

        elif regime == "NORMAL":
            gross = sum(abs(w) for w in adjusted.values())
            if gross > 1.5:
                scale = 1.5 / gross
                adjusted = {t: round(w * scale, 4) for t, w in adjusted.items()}
                log.info("NORMAL regime: capped gross from %.2f to 1.50", gross)

        elif regime == "TENSION":
            # Reduce all by 40%
            adjusted = {t: round(w * 0.6, 4) for t, w in adjusted.items()}
            # Cap gross at 1.0x
            gross = sum(abs(w) for w in adjusted.values())
            if gross > 1.0:
                scale = 1.0 / gross
                adjusted = {t: round(w * scale, 4) for t, w in adjusted.items()}
            log.info("TENSION regime: weights reduced 40%%, gross=%.2f",
                     sum(abs(w) for w in adjusted.values()))

        elif regime == "CRISIS":
            # Keep only Risk Parity weights at 50% scale — zero out everything
            # and rely on RP weights passed in via blend (RP-dominated in CRISIS)
            adjusted = {t: round(w * 0.5, 4) for t, w in adjusted.items()}
            log.info("CRISIS regime: scaled to 50%% of RP-dominated blend")

        else:
            log.warning("Unknown regime '%s' — no adjustment", regime)

        # Remove dust
        adjusted = {t: w for t, w in adjusted.items() if abs(w) > 0.005}
        return adjusted

    def _build_machine_summary(
        self,
        final_weights: Dict[str, float],
        regime: str,
        turnover: float,
        risk_budget_used: float,
        strategies_excluded: List[str],
    ) -> Dict[str, Any]:
        """Build institutional machine_summary for downstream consumption."""
        gross = sum(abs(w) for w in final_weights.values())
        net = sum(final_weights.values())
        return {
            "total_positions": sum(1 for w in final_weights.values() if abs(w) > 0.005),
            "gross_exposure": round(gross, 4),
            "net_exposure": round(net, 4),
            "regime": regime,
            "allocation_method": "regime_adaptive_blend",
            "turnover": turnover,
            "risk_budget_used": round(risk_budget_used, 4),
            "strategies_excluded": strategies_excluded,
            "execution_instructions": {
                t: round(w, 4) for t, w in final_weights.items() if abs(w) > 0.005
            },
        }

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
        """Full cycle: governance → signals → covariance → optimize → blend → constrain → save."""
        log.info("=" * 60)
        log.info("PORTFOLIO CONSTRUCTION — %s", datetime.now(timezone.utc).isoformat()[:19])
        log.info("=" * 60)

        regime = self._get_regime()
        log.info("Regime: %s", regime)

        # ── Governance inputs ──────────────────────────────────────────
        governance = self.load_governance_inputs()
        approval_matrix = governance.get("approval_matrix", {})
        strategies_excluded: List[str] = []
        for strat_name, decision in approval_matrix.items():
            dec = decision if isinstance(decision, str) else decision.get("decision", "REJECTED")
            if dec == "REJECTED":
                strategies_excluded.append(strat_name)

        # Covariance
        cov = self.compute_covariance(lookback=60)
        log.info("Covariance: %d×%d matrix", cov.shape[0], cov.shape[1])

        # ── Risk budget allocation ─────────────────────────────────────
        risk_budget = self.risk_budget_allocate(cov, target_vol=0.10, governance=governance)

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

        # ── Zero out REJECTED strategies in blended weights ────────────
        for strat in strategies_excluded:
            if strat in blended:
                blended[strat] = 0.0

        # ── Regime-conditional adjustment ──────────────────────────────
        blended = self.regime_adjust_weights(blended, regime)

        # Constrain
        final = self.apply_constraints(blended)

        # ── Turnover control ───────────────────────────────────────────
        prev_path = OUTPUT_DIR / "portfolio_weights.json"
        current_weights: Dict[str, float] = {}
        if prev_path.exists():
            try:
                prev = json.loads(prev_path.read_text(encoding="utf-8"))
                current_weights = prev.get("weights", {})
            except (json.JSONDecodeError, OSError):
                pass
        turnover = self.compute_turnover(current_weights, final)

        gross = sum(abs(w) for w in final.values())
        net = sum(final.values())

        # Metrics
        final_arr = np.array([final.get(t, 0) for t in avail])
        exp_ret = float(final_arr @ np.array([er.get(t, 0) for t in avail]))
        exp_vol = float(np.sqrt(final_arr @ cov.loc[avail, avail].values @ final_arr))
        sharpe_est = exp_ret / exp_vol if exp_vol > 0 else 0

        # ── Risk budget usage ──────────────────────────────────────────
        port_vol = exp_vol
        risk_budget_used = port_vol / 0.10 if 0.10 > 0 else 0.0
        risk_budget_used = min(risk_budget_used, 1.0)

        # ── Machine summary ────────────────────────────────────────────
        machine_summary = self._build_machine_summary(
            final, regime, turnover, risk_budget_used, strategies_excluded,
        )

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
            "risk_budget": risk_budget,
            "turnover": turnover,
            "governance_loaded": governance.get("loaded", False),
            "strategies_excluded": strategies_excluded,
            "machine_summary": machine_summary,
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
