"""
Portfolio Intelligence & Capital Allocation Engine.

Institutional-grade capital allocation committee engine that governs
portfolio construction through multi-layer allocation, role-based
budgeting, governance-aware haircuts, and regime-adaptive controls.

Legacy optimization methods (MV, RP, Conviction) are preserved as
building blocks. The new allocation pipeline layers on top:

  1. Universe construction from governance + health + regime data
  2. Role assignment (CORE / DIVERSIFIER / TACTICAL / SHADOW / DISABLED)
  3. Composite allocation scoring
  4. Deterministic haircuts (decay, regime, confidence, governance)
  5. Multi-layer allocation with role-based risk budgets
  6. Turnover, concentration, and dust controls
  7. Full diagnostics and machine-readable summary
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# ── Role-based risk budget allocation ─────────────────────────────────
ROLE_RISK_BUDGET = {
    "CORE": 0.60,
    "DIVERSIFIER": 0.20,
    "TACTICAL": 0.15,
    "SHADOW": 0.05,
}

# ── Regime-adaptive turnover caps ─────────────────────────────────────
REGIME_TURNOVER_CAP = {
    "CALM": 0.25,
    "NORMAL": 0.20,
    "TENSION": 0.10,
    "CRISIS": 0.05,
}


@dataclass
class AllocationSleeve:
    """Single allocation unit in the capital allocation committee."""

    sleeve_id: str
    sleeve_name: str
    sector: str
    strategy: str = ""
    allocation_role: str = "CORE"        # CORE/DIVERSIFIER/TACTICAL/SHADOW/DISABLED/WATCHLIST
    approval_status: str = "UNKNOWN"     # APPROVED/CONDITIONAL/REJECTED
    health_state: str = "UNKNOWN"        # HEALTHY/EARLY_DECAY/STRUCTURAL_DECAY/DEAD
    regime_eligibility: Dict[str, bool] = field(default_factory=dict)
    conviction: float = 0.0
    robustness_score: float = 0.5
    decay_hazard: float = 0.0
    implementation_cost: float = 0.0
    diversification_score: float = 0.5
    expected_alpha: float = 0.0
    risk_budget_target: float = 0.0
    allocable: bool = True
    exclusion_reason: str = ""
    confidence_haircut: float = 0.0
    decay_haircut: float = 0.0
    regime_haircut: float = 0.0
    final_weight: float = 0.0
    raw_allocation_score: float = 0.0


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

    # ══════════════════════════════════════════════════════════════════
    #  Capital Allocation Committee Engine — new institutional methods
    # ══════════════════════════════════════════════════════════════════

    def _load_decay_status(self) -> Dict[str, Any]:
        """Load Alpha Decay agent's decay_status.json."""
        path = ROOT / "agents" / "alpha_decay" / "decay_status.json"
        if not path.exists():
            log.warning("Decay status not found at %s", path)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cannot load decay status: %s", exc)
            return {}

    def _load_regime_forecast(self) -> Dict[str, Any]:
        """Load Regime Forecaster's regime_forecast.json."""
        path = ROOT / "agents" / "regime_forecaster" / "regime_forecast.json"
        if not path.exists():
            log.warning("Regime forecast not found at %s", path)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cannot load regime forecast: %s", exc)
            return {}

    def _load_optimizer_history(self) -> List[Dict]:
        """Load Optimizer agent's optimization_history.json for shadow candidates."""
        path = ROOT / "agents" / "optimizer" / "optimization_history.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _derive_health_state(self, decay_data: Dict[str, Any], strategy: str) -> str:
        """
        Derive a sleeve health state from decay status data.

        Maps decay_level -> health state:
          HEALTHY           -> HEALTHY
          EARLY_DECAY       -> EARLY_DECAY
          MODERATE_DECAY    -> EARLY_DECAY
          STRUCTURAL_DECAY  -> STRUCTURAL_DECAY
          DEAD              -> DEAD
        Also checks per-signal declining flags.
        """
        level = decay_data.get("decay_level", "HEALTHY")
        mapping = {
            "HEALTHY": "HEALTHY",
            "EARLY_DECAY": "EARLY_DECAY",
            "MODERATE_DECAY": "EARLY_DECAY",
            "STRUCTURAL_DECAY": "STRUCTURAL_DECAY",
            "DEAD": "DEAD",
        }
        return mapping.get(level, "HEALTHY")

    def _derive_regime_eligibility(
        self, regime_forecast: Dict[str, Any], sector: str
    ) -> Dict[str, bool]:
        """
        Determine which regimes a sleeve is eligible for.

        Uses regime probabilities: if predicted regime has P > 0.5,
        mark non-predicted regimes as ineligible for sectors with
        weak fitness in that regime.
        """
        probs = regime_forecast.get("probabilities", {})
        predicted = regime_forecast.get("predicted_regime", "NORMAL")
        eligibility: Dict[str, bool] = {}
        for regime_name in ("CALM", "NORMAL", "TENSION", "CRISIS"):
            eligibility[regime_name] = True
        # In CRISIS with high probability, mark CALM-only strategies ineligible
        crisis_prob = probs.get("CRISIS", 0.0)
        if crisis_prob > 0.5:
            eligibility["CALM"] = False
        return eligibility

    def _compute_diversification_score(
        self, sector: str, cov_matrix: pd.DataFrame, core_sectors: List[str]
    ) -> float:
        """
        Compute diversification value of this sector relative to current CORE positions.

        Low average correlation to CORE -> high diversification score.
        """
        if sector not in cov_matrix.columns or not core_sectors:
            return 0.5

        vols = np.sqrt(np.diag(cov_matrix.values))
        vol_map = {t: vols[i] for i, t in enumerate(cov_matrix.columns)}

        avg_corr = 0.0
        count = 0
        for cs in core_sectors:
            if cs in cov_matrix.columns and cs != sector:
                i = list(cov_matrix.columns).index(sector)
                j = list(cov_matrix.columns).index(cs)
                v_i = vol_map.get(sector, 1.0)
                v_j = vol_map.get(cs, 1.0)
                if v_i > 0 and v_j > 0:
                    corr = cov_matrix.values[i, j] / (v_i * v_j)
                    avg_corr += corr
                    count += 1
        if count > 0:
            avg_corr /= count

        # Score: 1.0 = uncorrelated, 0.0 = perfectly correlated
        return round(max(0.0, min(1.0, 1.0 - abs(avg_corr))), 4)

    def build_allocation_universe(
        self,
        governance: Dict[str, Any],
        decay_data: Dict[str, Any],
        regime_forecast: Dict[str, Any],
        signals: Optional[pd.DataFrame],
        cov_matrix: pd.DataFrame,
    ) -> List[AllocationSleeve]:
        """
        Construct the full allocation universe as AllocationSleeve objects.

        For each sector/strategy:
          - Load governance approval status
          - Load health state from decay agent
          - Load regime eligibility from forecaster
          - Load conviction from signals
          - Mark non-allocable for REJECTED / DEAD / STRUCTURAL_DECAY
        """
        approval_matrix = governance.get("approval_matrix", {})
        scorecards = governance.get("scorecards", [])
        regime_fitness = governance.get("regime_fitness_map", {})

        # Build conviction map from signals
        conviction_map: Dict[str, Tuple[float, str]] = {}
        if signals is not None:
            for _, row in signals.iterrows():
                ticker = str(row.get("sector_ticker", ""))
                conv = float(row.get("conviction_score", 0))
                direction = str(row.get("direction", "NEUTRAL"))
                conviction_map[ticker] = (conv, direction)

        # Build signal stack top candidates from methodology report
        methodology_ms = governance.get("machine_summary", {})

        health_state = self._derive_health_state(decay_data, "global")

        sleeves: List[AllocationSleeve] = []

        for sector in SECTORS:
            if sector not in cov_matrix.columns:
                continue

            # Governance decision
            raw_decision = approval_matrix.get(sector, "APPROVED")
            if isinstance(raw_decision, dict):
                approval = raw_decision.get("decision", "APPROVED")
            else:
                approval = str(raw_decision) if raw_decision else "APPROVED"

            # Conviction
            conv_val, direction = conviction_map.get(sector, (0.0, "NEUTRAL"))
            signed_alpha = conv_val if direction == "LONG" else -conv_val if direction == "SHORT" else 0.0

            # Regime eligibility
            regime_elig = self._derive_regime_eligibility(regime_forecast, sector)

            # Robustness from scorecards
            robustness = 0.5
            for sc in scorecards:
                if isinstance(sc, dict) and sc.get("sector") == sector:
                    robustness = float(sc.get("robustness", sc.get("composite_score", 0.5)))
                    break

            # Fitness from regime map
            fitness_entry = regime_fitness.get(sector, {})
            fitness_score = float(fitness_entry.get("fitness_score", 0.5)) if isinstance(fitness_entry, dict) else 0.5

            # Decay signals
            decay_signals = decay_data.get("signals", {})
            ic_declining = decay_signals.get("ic_declining", False)
            sharpe_declining = decay_signals.get("sharpe_declining", False)
            local_health = health_state
            decay_hazard = 0.0
            if ic_declining and sharpe_declining:
                local_health = "EARLY_DECAY"
                decay_hazard = 0.5
            elif ic_declining or sharpe_declining:
                decay_hazard = 0.25

            # Implementation cost (proxy: inverse liquidity via vol)
            vols = np.sqrt(np.diag(cov_matrix.values))
            vol_map = {t: vols[i] for i, t in enumerate(cov_matrix.columns)}
            sector_vol = vol_map.get(sector, 0.15)
            impl_cost = round(min(1.0, sector_vol / 0.40), 4)

            # Allocability
            allocable = True
            exclusion_reason = ""
            if approval == "REJECTED":
                allocable = False
                exclusion_reason = "GOVERNANCE_REJECTED"
            elif local_health in ("DEAD", "STRUCTURAL_DECAY"):
                allocable = False
                exclusion_reason = f"HEALTH_{local_health}"

            sleeve = AllocationSleeve(
                sleeve_id=f"{sector}_SECTOR",
                sleeve_name=f"{sector} Sector Allocation",
                sector=sector,
                strategy="SECTOR_STAT_ARB",
                allocation_role="CORE",
                approval_status=approval,
                health_state=local_health,
                regime_eligibility=regime_elig,
                conviction=conv_val,
                robustness_score=robustness,
                decay_hazard=decay_hazard,
                implementation_cost=impl_cost,
                diversification_score=0.5,
                expected_alpha=signed_alpha,
                risk_budget_target=0.0,
                allocable=allocable,
                exclusion_reason=exclusion_reason,
                confidence_haircut=0.0,
                decay_haircut=0.0,
                regime_haircut=0.0,
                final_weight=0.0,
                raw_allocation_score=0.0,
            )
            sleeves.append(sleeve)

        log.info(
            "Allocation universe: %d sleeves (%d allocable, %d excluded)",
            len(sleeves),
            sum(1 for s in sleeves if s.allocable),
            sum(1 for s in sleeves if not s.allocable),
        )
        return sleeves

    def assign_roles(
        self,
        sleeves: List[AllocationSleeve],
        cov_matrix: pd.DataFrame,
        optimizer_history: List[Dict],
    ) -> List[AllocationSleeve]:
        """
        Deterministic role assignment for each sleeve.

        Rules:
          APPROVED + HEALTHY + high conviction   -> CORE
          APPROVED + HEALTHY + low corr to CORE  -> DIVERSIFIER
          CONDITIONAL + moderate conviction       -> TACTICAL
          SHADOW candidate from Optimizer         -> SHADOW (max 2%)
          DEAD / STRUCTURAL_DECAY / REJECTED      -> DISABLED
          Under observation                       -> WATCHLIST
        """
        # Shadow candidates from optimizer proposals
        shadow_sectors: set = set()
        for entry in optimizer_history[-5:]:
            if isinstance(entry, dict) and entry.get("outcome") == "proposed":
                param = entry.get("param_name", "")
                if param:
                    shadow_sectors.add(param)

        # First pass: identify CORE candidates for diversification scoring
        core_sectors: List[str] = []
        for s in sleeves:
            if (
                s.allocable
                and s.approval_status == "APPROVED"
                and s.health_state == "HEALTHY"
                and s.conviction >= 0.05
            ):
                core_sectors.append(s.sector)

        # Second pass: assign roles
        for s in sleeves:
            if not s.allocable:
                s.allocation_role = "DISABLED"
                continue

            # Compute diversification score now that we know CORE candidates
            s.diversification_score = self._compute_diversification_score(
                s.sector, cov_matrix, core_sectors
            )

            if s.approval_status == "REJECTED":
                s.allocation_role = "DISABLED"
                s.allocable = False
                s.exclusion_reason = s.exclusion_reason or "GOVERNANCE_REJECTED"
            elif s.health_state in ("DEAD", "STRUCTURAL_DECAY"):
                s.allocation_role = "DISABLED"
                s.allocable = False
                s.exclusion_reason = s.exclusion_reason or f"HEALTH_{s.health_state}"
            elif s.approval_status == "APPROVED" and s.health_state == "HEALTHY":
                if s.conviction >= 0.05:
                    s.allocation_role = "CORE"
                elif s.diversification_score >= 0.6:
                    s.allocation_role = "DIVERSIFIER"
                else:
                    s.allocation_role = "DIVERSIFIER"
            elif s.approval_status == "CONDITIONAL":
                if s.conviction >= 0.03:
                    s.allocation_role = "TACTICAL"
                else:
                    s.allocation_role = "WATCHLIST"
            elif s.health_state == "EARLY_DECAY":
                s.allocation_role = "WATCHLIST"
            elif s.sector in shadow_sectors:
                s.allocation_role = "SHADOW"
            else:
                # Default: DIVERSIFIER for approved healthy low-conviction
                s.allocation_role = "DIVERSIFIER"

        role_counts = {}
        for s in sleeves:
            role_counts[s.allocation_role] = role_counts.get(s.allocation_role, 0) + 1
        log.info("Role assignment: %s", role_counts)
        return sleeves

    def compute_allocation_scores(
        self, sleeves: List[AllocationSleeve], regime: str
    ) -> List[AllocationSleeve]:
        """
        Composite allocation score per sleeve.

        score = (
            0.25 * alpha_quality +
            0.20 * robustness +
            0.15 * regime_alignment +
            0.15 * diversification_value -
            0.10 * decay_hazard -
            0.08 * implementation_cost -
            0.07 * uncertainty_penalty
        )
        """
        for s in sleeves:
            if not s.allocable:
                s.raw_allocation_score = 0.0
                continue

            # Alpha quality: normalized conviction [0,1]
            alpha_quality = min(1.0, s.conviction / 0.5) if s.conviction > 0 else 0.0

            # Regime alignment: is the sleeve eligible in current regime?
            regime_aligned = 1.0 if s.regime_eligibility.get(regime, True) else 0.3

            # Uncertainty penalty: inverse of conviction + robustness
            uncertainty = max(0.0, 1.0 - (s.conviction + s.robustness_score) / 2.0)

            score = (
                0.25 * alpha_quality
                + 0.20 * s.robustness_score
                + 0.15 * regime_aligned
                + 0.15 * s.diversification_score
                - 0.10 * s.decay_hazard
                - 0.08 * s.implementation_cost
                - 0.07 * uncertainty
            )
            s.raw_allocation_score = round(max(0.0, score), 6)

        # Log top scores
        ranked = sorted(
            [s for s in sleeves if s.allocable],
            key=lambda x: x.raw_allocation_score,
            reverse=True,
        )
        for s in ranked[:5]:
            log.info(
                "  Score %s: %.4f (conv=%.2f, rob=%.2f, div=%.2f, decay=%.2f)",
                s.sector, s.raw_allocation_score, s.conviction,
                s.robustness_score, s.diversification_score, s.decay_hazard,
            )
        return sleeves

    def apply_haircuts(
        self,
        sleeves: List[AllocationSleeve],
        regime: str,
        regime_forecast: Dict[str, Any],
    ) -> List[AllocationSleeve]:
        """
        Deterministic haircuts applied to allocation scores.

        Haircut rules:
          CONDITIONAL approval       -> 20% haircut
          WATCHLIST (under obs.)     -> 15% haircut
          EARLY_DECAY                -> 30% haircut
          REGIME_SUPPRESSED          -> 50% haircut
          High transition risk       -> 25% global haircut
          Low confidence / evidence  -> 20% haircut
          SHADOW role                -> 90% haircut (max 2% weight)
        """
        transition_prob = regime_forecast.get("transition_probability", 0.0)
        high_transition = transition_prob > 0.40

        for s in sleeves:
            if not s.allocable:
                continue

            base_score = s.raw_allocation_score

            # Governance haircuts
            if s.approval_status == "CONDITIONAL":
                s.confidence_haircut = 0.20
            elif s.allocation_role == "WATCHLIST":
                s.confidence_haircut = 0.15

            # Decay haircut
            if s.health_state == "EARLY_DECAY":
                s.decay_haircut = 0.30
            elif s.decay_hazard > 0.3:
                s.decay_haircut = 0.15

            # Regime haircut
            if not s.regime_eligibility.get(regime, True):
                s.regime_haircut = 0.50
            elif high_transition:
                s.regime_haircut = 0.25

            # Shadow haircut
            shadow_haircut = 0.90 if s.allocation_role == "SHADOW" else 0.0

            # Low conviction haircut
            low_conf_haircut = 0.20 if s.conviction < 0.03 and s.allocation_role != "SHADOW" else 0.0

            # Apply multiplicatively
            total_multiplier = (
                (1.0 - s.confidence_haircut)
                * (1.0 - s.decay_haircut)
                * (1.0 - s.regime_haircut)
                * (1.0 - shadow_haircut)
                * (1.0 - low_conf_haircut)
            )
            s.raw_allocation_score = round(base_score * max(0.0, total_multiplier), 6)

        haircut_count = sum(
            1 for s in sleeves
            if s.allocable and (s.confidence_haircut + s.decay_haircut + s.regime_haircut) > 0
        )
        log.info(
            "Haircuts applied: %d sleeves affected (transition_risk=%.2f)",
            haircut_count, transition_prob,
        )
        return sleeves

    def multi_layer_allocate(
        self,
        sleeves: List[AllocationSleeve],
        cov_matrix: pd.DataFrame,
        regime: str,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Ten-layer capital allocation engine.

        Layer 1:  Filter (remove DISABLED, non-allocable)
        Layer 2:  Role assignment (already done upstream)
        Layer 3:  Score and rank (already done upstream)
        Layer 4:  Risk budget by role
        Layer 5:  Optimize within each role bucket
        Layer 6:  Apply haircuts (already applied to scores)
        Layer 7:  Turnover control (regime-adaptive)
        Layer 8:  Concentration control (max 20% single, max 40% cluster)
        Layer 9:  Net/gross control
        Layer 10: Dust cleanup (< 0.5%)
        """
        # Layer 1: Filter
        active = [s for s in sleeves if s.allocable and s.allocation_role != "DISABLED"]
        if not active:
            log.warning("No allocable sleeves after filtering")
            return {}

        log.info("Layer 1 (Filter): %d active sleeves", len(active))

        # Layer 4: Risk budget by role
        role_buckets: Dict[str, List[AllocationSleeve]] = {}
        for s in active:
            role_buckets.setdefault(s.allocation_role, []).append(s)

        weights: Dict[str, float] = {}

        for role, bucket in role_buckets.items():
            budget = ROLE_RISK_BUDGET.get(role, 0.05)

            if not bucket:
                continue

            # Layer 5: Optimize within bucket
            total_score = sum(s.raw_allocation_score for s in bucket)
            if total_score <= 0:
                # Equal weight within bucket
                per_sleeve = budget / len(bucket)
                for s in bucket:
                    s.risk_budget_target = per_sleeve
                    signed = per_sleeve if s.expected_alpha >= 0 else -per_sleeve
                    weights[s.sector] = round(signed, 6)
            else:
                for s in bucket:
                    frac = s.raw_allocation_score / total_score
                    allocated = budget * frac
                    s.risk_budget_target = allocated

                    # For CORE: use risk-parity-like allocation
                    # For TACTICAL: use conviction-weighted
                    if role == "CORE":
                        # Scale by score fraction
                        signed = allocated if s.expected_alpha >= 0 else -allocated
                    elif role == "TACTICAL":
                        signed = allocated * (1.0 if s.expected_alpha >= 0 else -1.0)
                    elif role == "SHADOW":
                        signed = min(0.02, allocated) * (1.0 if s.expected_alpha >= 0 else -1.0)
                    else:
                        signed = allocated if s.expected_alpha >= 0 else -allocated

                    weights[s.sector] = round(signed, 6)

        log.info("Layer 5 (Within-role optimize): %d positions", len(weights))

        # Layer 7: Turnover control
        turnover_cap = REGIME_TURNOVER_CAP.get(regime, 0.20)
        all_tickers = set(list(current_weights.keys()) + list(weights.keys()))
        total_diff = sum(
            abs(weights.get(t, 0.0) - current_weights.get(t, 0.0)) for t in all_tickers
        )
        turnover = total_diff / 2.0

        if turnover > turnover_cap and turnover > 0:
            scale = turnover_cap / turnover
            for t in all_tickers:
                curr = current_weights.get(t, 0.0)
                new = weights.get(t, 0.0)
                weights[t] = round(curr + (new - curr) * scale, 6)
            log.info(
                "Layer 7 (Turnover): capped from %.2f%% to %.2f%% (%s regime)",
                turnover * 100, turnover_cap * 100, regime,
            )

        # Layer 8: Concentration control
        max_single = 0.20
        max_cluster = 0.40
        for t in weights:
            if abs(weights[t]) > max_single:
                weights[t] = round(max_single * (1.0 if weights[t] > 0 else -1.0), 6)

        # Cluster control: top 3 positions should not exceed 40%
        sorted_by_abs = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        top3_gross = sum(abs(w) for _, w in sorted_by_abs[:3])
        if top3_gross > max_cluster and top3_gross > 0:
            scale_cluster = max_cluster / top3_gross
            for t, w in sorted_by_abs[:3]:
                weights[t] = round(w * scale_cluster, 6)
            log.info("Layer 8 (Concentration): top-3 cluster scaled from %.2f to %.2f", top3_gross, max_cluster)

        # Layer 9: Net/gross control
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        max_gross = 2.0
        max_net = 0.40

        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            weights = {t: round(w * scale, 6) for t, w in weights.items()}

        net = sum(weights.values())
        if abs(net) > max_net:
            # Reduce the heavier side
            excess = net - (max_net if net > 0 else -max_net)
            direction = "LONG" if net > 0 else "SHORT"
            sorted_pos = sorted(
                [(t, w) for t, w in weights.items() if (w > 0) == (direction == "LONG")],
                key=lambda x: abs(x[1]), reverse=True,
            )
            for t, w in sorted_pos:
                reduce = min(abs(w), abs(excess))
                weights[t] = round(w - (reduce if w > 0 else -reduce), 6)
                excess -= reduce
                if abs(excess) < 0.001:
                    break

        # Layer 10: Dust cleanup
        weights = {t: round(w, 4) for t, w in weights.items() if abs(w) >= 0.005}

        # Update sleeve final_weight
        for s in sleeves:
            s.final_weight = weights.get(s.sector, 0.0)

        log.info(
            "Layer 10 (Final): %d positions, gross=%.3f, net=%.3f",
            len(weights),
            sum(abs(w) for w in weights.values()),
            sum(weights.values()),
        )
        return weights

    def build_portfolio_diagnostics(
        self, sleeves: List[AllocationSleeve], weights: Dict[str, float],
        current_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Build comprehensive diagnostics for every sleeve and portfolio summary.

        Per sleeve: why it got its weight, exclusion reason, role, haircuts.
        Summary: HHI, top positions, turnover, risk budget usage, exclusion counts.
        """
        sleeve_details: List[Dict[str, Any]] = []
        for s in sleeves:
            detail: Dict[str, Any] = {
                "sleeve_id": s.sleeve_id,
                "sector": s.sector,
                "role": s.allocation_role,
                "approval": s.approval_status,
                "health": s.health_state,
                "conviction": round(s.conviction, 4),
                "raw_score": round(s.raw_allocation_score, 6),
                "final_weight": round(s.final_weight, 4),
                "allocable": s.allocable,
            }
            if not s.allocable:
                detail["exclusion_reason"] = s.exclusion_reason
            if s.confidence_haircut > 0:
                detail["confidence_haircut"] = f"{s.confidence_haircut:.0%}"
            if s.decay_haircut > 0:
                detail["decay_haircut"] = f"{s.decay_haircut:.0%}"
            if s.regime_haircut > 0:
                detail["regime_haircut"] = f"{s.regime_haircut:.0%}"
            sleeve_details.append(detail)

        # Portfolio summary
        active_weights = {t: w for t, w in weights.items() if abs(w) > 0.005}
        gross = sum(abs(w) for w in active_weights.values())
        net = sum(active_weights.values())
        long_wt = sum(w for w in active_weights.values() if w > 0)
        short_wt = sum(abs(w) for w in active_weights.values() if w < 0)

        # HHI concentration
        hhi = sum((w / gross) ** 2 for w in active_weights.values()) if gross > 0 else 0

        # Top 3
        sorted_wts = sorted(active_weights.items(), key=lambda x: abs(x[1]), reverse=True)
        top3 = [{"sector": t, "weight": round(w, 4)} for t, w in sorted_wts[:3]]

        # Turnover
        all_t = set(list(current_weights.keys()) + list(active_weights.keys()))
        turnover = sum(abs(active_weights.get(t, 0) - current_weights.get(t, 0)) for t in all_t) / 2.0

        # Risk budget usage by role
        role_usage: Dict[str, float] = {}
        for s in sleeves:
            if s.allocable:
                role_usage.setdefault(s.allocation_role, 0.0)
                role_usage[s.allocation_role] += abs(s.final_weight)

        diagnostics = {
            "sleeve_details": sleeve_details,
            "summary": {
                "concentration_hhi": round(hhi, 4),
                "top_3_positions": top3,
                "long_exposure": round(long_wt, 4),
                "short_exposure": round(short_wt, 4),
                "gross_exposure": round(gross, 4),
                "net_exposure": round(net, 4),
                "turnover_from_previous": round(turnover, 4),
                "risk_budget_by_role": {k: round(v, 4) for k, v in role_usage.items()},
                "governance_exclusion_count": sum(
                    1 for s in sleeves if s.exclusion_reason.startswith("GOVERNANCE")
                ),
                "decay_haircut_count": sum(1 for s in sleeves if s.decay_haircut > 0),
            },
        }
        return diagnostics

    def _build_enhanced_machine_summary(
        self,
        sleeves: List[AllocationSleeve],
        weights: Dict[str, float],
        regime: str,
        regime_forecast: Dict[str, Any],
        turnover: float,
        target_vol: float,
    ) -> Dict[str, Any]:
        """Build enhanced institutional machine_summary for downstream consumption."""
        active = {t: w for t, w in weights.items() if abs(w) > 0.005}
        gross = sum(abs(w) for w in active.values())
        net = sum(active.values())

        transition_prob = regime_forecast.get("transition_probability", 0.0)
        transition_state = (
            "CONFIRMED" if transition_prob > 0.6
            else "EARLY_WARNING" if transition_prob > 0.35
            else "STABLE"
        )

        # Role counts
        role_counts: Dict[str, int] = {}
        for s in sleeves:
            if s.allocable and abs(s.final_weight) > 0.005:
                role_counts[s.allocation_role] = role_counts.get(s.allocation_role, 0) + 1

        # Top allocations with roles
        sorted_wts = sorted(active.items(), key=lambda x: abs(x[1]), reverse=True)
        top_alloc = []
        for t, w in sorted_wts[:5]:
            role = "UNKNOWN"
            for s in sleeves:
                if s.sector == t:
                    role = s.allocation_role
                    break
            top_alloc.append({t: round(w, 4), "role": role})

        # Largest haircuts
        largest_haircuts = []
        for s in sleeves:
            total_hc = s.confidence_haircut + s.decay_haircut + s.regime_haircut
            if total_hc > 0:
                parts = []
                if s.decay_haircut > 0:
                    parts.append(f"{s.decay_haircut:.0%} decay haircut")
                if s.regime_haircut > 0:
                    parts.append(f"{s.regime_haircut:.0%} regime haircut")
                if s.confidence_haircut > 0:
                    parts.append(f"{s.confidence_haircut:.0%} confidence haircut")
                largest_haircuts.append({s.sector: ", ".join(parts)})

        # Excluded by governance vs decay
        excluded_governance = [
            s.sleeve_name for s in sleeves
            if s.exclusion_reason.startswith("GOVERNANCE")
        ]
        excluded_decay = [
            s.sleeve_name for s in sleeves
            if s.exclusion_reason.startswith("HEALTH")
        ]

        # Execution instructions
        exec_instructions = {}
        for t, w in active.items():
            if w > 0.005:
                action = "BUY"
            elif w < -0.005:
                action = "SELL"
            else:
                continue
            urgency = "HIGH" if abs(w) > 0.10 else "MEDIUM" if abs(w) > 0.05 else "LOW"
            exec_instructions[t] = {
                "action": action,
                "weight": round(w, 4),
                "urgency": urgency,
            }

        # Rebalance urgency
        if turnover > 0.15:
            rebalance_urgency = "HIGH"
        elif turnover > 0.05:
            rebalance_urgency = "MEDIUM"
        else:
            rebalance_urgency = "LOW"

        return {
            "total_positions": len(active),
            "gross_exposure": round(gross, 4),
            "net_exposure": round(net, 4),
            "active_regime": regime,
            "transition_state": transition_state,
            "core_count": role_counts.get("CORE", 0),
            "tactical_count": role_counts.get("TACTICAL", 0),
            "diversifier_count": role_counts.get("DIVERSIFIER", 0),
            "shadow_count": role_counts.get("SHADOW", 0),
            "excluded_count": sum(1 for s in sleeves if not s.allocable),
            "turnover": round(turnover, 4),
            "target_vol": target_vol,
            "top_allocations": top_alloc,
            "largest_haircuts": largest_haircuts,
            "excluded_for_governance": excluded_governance,
            "excluded_for_decay": excluded_decay,
            "execution_instructions": exec_instructions,
            "rebalance_urgency": rebalance_urgency,
        }

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
        """
        Full cycle: Capital Allocation Committee pipeline.

        1. Load governance inputs
        2. Build allocation universe
        3. Assign roles
        4. Compute allocation scores
        5. Apply haircuts
        6. Run multi-layer allocation
        7. Build diagnostics
        8. Build machine_summary
        9. Save + publish
        """
        log.info("=" * 60)
        log.info("CAPITAL ALLOCATION ENGINE — %s", datetime.now(timezone.utc).isoformat()[:19])
        log.info("=" * 60)

        regime = self._get_regime()
        log.info("Regime: %s", regime)

        # ── 1. Governance inputs ──────────────────────────────────────
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

        # ── Risk budget allocation (legacy, still computed) ───────────
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

        # 3 legacy optimization methods (kept for method_weights output)
        mv_weights = self.mean_variance_optimize(er, cov)
        log.info("MV weights: %s", {k: v for k, v in mv_weights.items() if abs(v) > 0.01})

        rp_weights = self.risk_parity_optimize(cov)
        log.info("RP weights: %s", {k: v for k, v in rp_weights.items() if abs(v) > 0.01})

        signals = self.load_current_signals()
        if signals is not None:
            conv_weights = self.conviction_weighted(signals)
        else:
            conv_weights = {t: 0.0 for t in avail}
        log.info("Conv weights: %s", {k: v for k, v in conv_weights.items() if abs(v) > 0.01})

        # ── 2. Build allocation universe ──────────────────────────────
        decay_data = self._load_decay_status()
        regime_forecast = self._load_regime_forecast()
        optimizer_history = self._load_optimizer_history()

        sleeves = self.build_allocation_universe(
            governance, decay_data, regime_forecast, signals, cov,
        )

        # ── 3. Assign roles ───────────────────────────────────────────
        sleeves = self.assign_roles(sleeves, cov, optimizer_history)

        # ── 4. Compute allocation scores ──────────────────────────────
        sleeves = self.compute_allocation_scores(sleeves, regime)

        # ── 5. Apply haircuts ─────────────────────────────────────────
        sleeves = self.apply_haircuts(sleeves, regime, regime_forecast)

        # ── 6. Multi-layer allocation ─────────────────────────────────
        prev_path = OUTPUT_DIR / "portfolio_weights.json"
        current_weights: Dict[str, float] = {}
        if prev_path.exists():
            try:
                prev = json.loads(prev_path.read_text(encoding="utf-8"))
                current_weights = prev.get("weights", {})
            except (json.JSONDecodeError, OSError):
                pass

        final = self.multi_layer_allocate(sleeves, cov, regime, current_weights)

        # Also compute legacy blended path for backward compat
        blended = self.blend_methods(mv_weights, rp_weights, conv_weights, regime)
        for strat in strategies_excluded:
            if strat in blended:
                blended[strat] = 0.0
        blended = self.regime_adjust_weights(blended, regime)
        legacy_final = self.apply_constraints(blended)

        # Turnover (computed against previous)
        all_tickers = set(list(current_weights.keys()) + list(final.keys()))
        turnover = sum(abs(final.get(t, 0) - current_weights.get(t, 0)) for t in all_tickers) / 2.0
        turnover = round(turnover, 4)

        gross = sum(abs(w) for w in final.values())
        net = sum(final.values())

        # Metrics
        final_arr = np.array([final.get(t, 0) for t in avail])
        exp_ret = float(final_arr @ np.array([er.get(t, 0) for t in avail]))
        exp_vol = float(np.sqrt(final_arr @ cov.loc[avail, avail].values @ final_arr))
        sharpe_est = exp_ret / exp_vol if exp_vol > 0 else 0

        # Risk budget usage
        port_vol = exp_vol
        target_vol = 0.10
        risk_budget_used = port_vol / target_vol if target_vol > 0 else 0.0
        risk_budget_used = min(risk_budget_used, 1.0)

        # ── 7. Diagnostics ────────────────────────────────────────────
        diagnostics = self.build_portfolio_diagnostics(sleeves, final, current_weights)

        # ── 8. Enhanced machine summary ───────────────────────────────
        machine_summary = self._build_enhanced_machine_summary(
            sleeves, final, regime, regime_forecast, turnover, target_vol,
        )

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "weights": final,
            "method_weights": {"mv": mv_weights, "rp": rp_weights, "conv": conv_weights},
            "legacy_blend_weights": legacy_final,
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
            "allocation_sleeves": [
                {
                    "sleeve_id": s.sleeve_id,
                    "sector": s.sector,
                    "role": s.allocation_role,
                    "score": round(s.raw_allocation_score, 6),
                    "weight": round(s.final_weight, 4),
                    "allocable": s.allocable,
                    "exclusion_reason": s.exclusion_reason,
                }
                for s in sleeves
            ],
            "diagnostics": diagnostics,
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
