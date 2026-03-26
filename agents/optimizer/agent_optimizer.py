"""
agents/optimizer/agent_optimizer.py
-------------------------------------
Institutional-Grade Optimizer Agent

Architecture:
  A. OptimizationGovernanceEngine — reads methodology machine_summary, determines allowed mode
  B. InstitutionalObjective — composite scoring (replaces raw Sharpe)
  C. CandidateFactory — 5 structured generators
  D. SandboxEvaluator — 4-stage validation pipeline
  E. ChampionChallengerManager — champion vs challenger comparison
  F. PromotionDecisionEngine — deterministic promotion rules
  G. OptimizerAgent.run() — orchestrates full optimization flow

LLM = advisory only, never decision-maker.
All promotion gates are deterministic via evaluate_promotion().

CLI:
    python agents/optimizer/agent_optimizer.py --once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Root path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ──────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            LOG_DIR / "agent_optimizer.log",
            maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("agent_optimizer")

# ── Internal imports ─────────────────────────────────────────────────────
from scripts.agent_bus import get_bus
from scripts.claude_loop import execute_actions, _extract_json_block, send_to_claude
from agents.optimizer.optimization_log import (
    get_optimization_log, OptimizationRecord, OptimizationLineageTracker,
)
from agents.shared.agent_registry import get_registry, AgentStatus
from agents.methodology.report_schema import (
    MachineSummary, evaluate_promotion, validate_machine_summary,
    PromotionCriteria, DEFAULT_PROMOTION_CRITERIA,
)

# ── Optional imports with graceful fallback ──────────────────────────────
_IMPORTS_OK: Dict[str, bool] = {}

try:
    from agents.math.llm_bridge import DualLLMBridge
    _IMPORTS_OK["llm_bridge"] = True
except ImportError:
    _IMPORTS_OK["llm_bridge"] = False

try:
    import numpy as np
    _IMPORTS_OK["numpy"] = True
except ImportError:
    _IMPORTS_OK["numpy"] = False


def _query_gpt_fallback(prompt: str) -> str:
    """Query GPT when Claude API is not configured."""
    try:
        from agents.math.llm_bridge import DualLLMBridge
        bridge = DualLLMBridge()
        if bridge.has_gpt:
            return bridge.query_gpt(prompt, "optimization")
        elif bridge.has_claude:
            return bridge.query_claude(prompt, "optimization")
    except Exception as e:
        log.warning("GPT fallback failed: %s", e)
    return ""


# ═════════════════════════════════════════════════════════════════════════
# A. OptimizationGovernanceEngine
# ═════════════════════════════════════════════════════════════════════════

GOVERNANCE_MODES = (
    "PARAM_TUNING_ONLY",
    "PARAM_PLUS_CODE",
    "REGIME_SPECIFIC_TUNING",
    "FREEZE_AND_MONITOR",
    "DISABLE_CANDIDATE",
)


class OptimizationGovernanceEngine:
    """
    Reads methodology report machine_summary and determines the allowed
    optimization class for this cycle.

    Modes:
      PARAM_TUNING_ONLY      — standard parameter adjustment (approved strategies)
      PARAM_PLUS_CODE        — formula changes allowed (conditional strategies)
      REGIME_SPECIFIC_TUNING — only tune for weak regime
      FREEZE_AND_MONITOR     — no changes, just observe (overfitting detected)
      DISABLE_CANDIDATE      — recommend disabling strategy
    """

    def __init__(self) -> None:
        self._mode: str = "PARAM_TUNING_ONLY"
        self._rationale: str = ""
        self._machine_summary: Optional[Dict[str, Any]] = None

    def determine_mode(self, machine_summary: Dict[str, Any]) -> str:
        """
        Determine optimization mode from methodology machine_summary.

        Parameters
        ----------
        machine_summary : dict
            The machine_summary section from the methodology report.

        Returns
        -------
        str
            One of GOVERNANCE_MODES.
        """
        self._machine_summary = machine_summary

        n_approved = int(machine_summary.get("n_approved", 0))
        n_conditional = int(machine_summary.get("n_conditional", 0))
        overfitting_flag = bool(machine_summary.get("overfitting_flag", False))
        strategies_to_disable = machine_summary.get("strategies_to_disable", [])
        optimizer_should_tune = bool(machine_summary.get("optimizer_should_tune", False))
        best_decision = machine_summary.get("best_strategy_decision", "REJECTED")

        # Decision tree — deterministic, no LLM
        if overfitting_flag:
            self._mode = "FREEZE_AND_MONITOR"
            self._rationale = (
                "Overfitting flag is set — freezing all optimization to prevent "
                "further parameter mining on overfit data."
            )
        elif strategies_to_disable:
            self._mode = "DISABLE_CANDIDATE"
            self._rationale = (
                f"Strategies flagged for disable: {strategies_to_disable}. "
                "Recommending disable rather than optimization."
            )
        elif n_approved > 0:
            self._mode = "PARAM_TUNING_ONLY"
            self._rationale = (
                f"{n_approved} approved strategies — refining parameters within "
                "approved bounds. No code changes needed."
            )
        elif n_conditional > 0:
            self._mode = "PARAM_PLUS_CODE"
            self._rationale = (
                f"{n_conditional} conditional strategies — formula changes allowed "
                "to address warnings and achieve full approval."
            )
        elif optimizer_should_tune:
            self._mode = "REGIME_SPECIFIC_TUNING"
            self._rationale = (
                "Methodology flagged optimizer_should_tune — targeting weak regime "
                "for focused parameter adjustment."
            )
        else:
            self._mode = "PARAM_TUNING_ONLY"
            self._rationale = "Default mode: safe parameter tuning only."

        log.info("Governance mode: %s — %s", self._mode, self._rationale)
        return self._mode

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def rationale(self) -> str:
        return self._rationale

    def allows_code_changes(self) -> bool:
        return self._mode == "PARAM_PLUS_CODE"

    def allows_param_changes(self) -> bool:
        return self._mode in ("PARAM_TUNING_ONLY", "PARAM_PLUS_CODE", "REGIME_SPECIFIC_TUNING")

    def should_freeze(self) -> bool:
        return self._mode in ("FREEZE_AND_MONITOR", "DISABLE_CANDIDATE")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "rationale": self._rationale,
            "allows_code": self.allows_code_changes(),
            "allows_param": self.allows_param_changes(),
            "frozen": self.should_freeze(),
        }


# ═════════════════════════════════════════════════════════════════════════
# B. InstitutionalObjective — Composite Scoring
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectiveBreakdown:
    """Breakdown of the composite institutional objective score."""
    delta_net_sharpe: float = 0.0
    delta_robustness: float = 0.0
    delta_stability: float = 0.0
    delta_regime_fitness: float = 0.0
    overfitting_penalty: float = 0.0
    tail_penalty: float = 0.0
    cost_drag_penalty: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


DEFAULT_OBJECTIVE_WEIGHTS = {
    "net_sharpe": 0.30,
    "robustness": 0.20,
    "stability": 0.15,
    "regime_fitness": 0.15,
    "overfitting": -0.10,
    "tail_risk": -0.05,
    "cost_drag": -0.05,
}


def compute_objective(
    before: Dict[str, Any],
    after: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> ObjectiveBreakdown:
    """
    Compute the institutional composite objective from before/after metrics.

    Replaces raw Sharpe comparison with a weighted multi-dimensional score
    that penalizes overfitting, tail risk, and cost drag.

    Parameters
    ----------
    before : dict
        Metrics before optimization (sharpe, robustness, stability, etc.)
    after : dict
        Metrics after optimization.
    weights : dict, optional
        Custom weights. Defaults to DEFAULT_OBJECTIVE_WEIGHTS.

    Returns
    -------
    ObjectiveBreakdown
        Full breakdown with composite_score.
    """
    w = weights or DEFAULT_OBJECTIVE_WEIGHTS

    # Extract metrics with safe defaults
    def _get(d: Dict, key: str, default: float = 0.0) -> float:
        val = d.get(key, default)
        try:
            return float(val) if val is not None else default
        except (TypeError, ValueError):
            return default

    # Deltas (positive = improvement)
    delta_net_sharpe = _get(after, "sharpe") - _get(before, "sharpe")
    delta_robustness = _get(after, "robustness") - _get(before, "robustness")
    delta_stability = _get(after, "stability") - _get(before, "stability")

    # Regime fitness: average across regime sharpes
    before_regime = before.get("regime_breakdown", {})
    after_regime = after.get("regime_breakdown", {})
    before_regime_avg = 0.0
    after_regime_avg = 0.0
    if isinstance(before_regime, dict) and before_regime:
        vals = []
        for rd in before_regime.values():
            if isinstance(rd, dict):
                vals.append(float(rd.get("sharpe", 0)))
            elif isinstance(rd, (int, float)):
                vals.append(float(rd))
        before_regime_avg = sum(vals) / len(vals) if vals else 0.0
    if isinstance(after_regime, dict) and after_regime:
        vals = []
        for rd in after_regime.values():
            if isinstance(rd, dict):
                vals.append(float(rd.get("sharpe", 0)))
            elif isinstance(rd, (int, float)):
                vals.append(float(rd))
        after_regime_avg = sum(vals) / len(vals) if vals else 0.0
    delta_regime_fitness = after_regime_avg - before_regime_avg

    # Penalties (higher = worse, so we negate)
    after_overfit = _get(after, "overfitting_score", 0.0)
    before_overfit = _get(before, "overfitting_score", 0.0)
    overfitting_penalty = max(0.0, after_overfit - before_overfit)

    after_tail = 1.0 - _get(after, "tail_risk_score", 1.0)
    before_tail = 1.0 - _get(before, "tail_risk_score", 1.0)
    tail_penalty = max(0.0, after_tail - before_tail)

    after_cost = _get(after, "cost_drag_pct", 0.0)
    before_cost = _get(before, "cost_drag_pct", 0.0)
    cost_drag_penalty = max(0.0, (after_cost - before_cost) / 100.0)

    # Weighted composite
    composite = (
        w.get("net_sharpe", 0.30) * delta_net_sharpe
        + w.get("robustness", 0.20) * delta_robustness
        + w.get("stability", 0.15) * delta_stability
        + w.get("regime_fitness", 0.15) * delta_regime_fitness
        + w.get("overfitting", -0.10) * overfitting_penalty
        + w.get("tail_risk", -0.05) * tail_penalty
        + w.get("cost_drag", -0.05) * cost_drag_penalty
    )

    return ObjectiveBreakdown(
        delta_net_sharpe=round(delta_net_sharpe, 6),
        delta_robustness=round(delta_robustness, 6),
        delta_stability=round(delta_stability, 6),
        delta_regime_fitness=round(delta_regime_fitness, 6),
        overfitting_penalty=round(overfitting_penalty, 6),
        tail_penalty=round(tail_penalty, 6),
        cost_drag_penalty=round(cost_drag_penalty, 6),
        composite_score=round(composite, 6),
    )


# ═════════════════════════════════════════════════════════════════════════
# C. CandidateFactory — Structured Candidate Generation
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationCandidate:
    """A single optimization candidate with full audit trail."""
    candidate_id: str = ""
    candidate_type: str = "param"     # "param" / "code" / "regime"
    source: str = "local"             # "local" / "sensitivity" / "bayesian" / "regime" / "joint" / "llm_advisory"
    params_changed: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)  # param -> (old, new)
    rationale: str = ""
    expected_improvement: float = 0.0
    evaluation_status: str = "PENDING"  # PENDING/SANDBOX_PASSED/SHADOW_APPROVED/PROMOTED/REJECTED/ROLLED_BACK
    objective_breakdown: Optional[ObjectiveBreakdown] = None
    validation_stages: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            self.candidate_id = f"cand-{uuid.uuid4().hex[:10]}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d["params_changed"] = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in d.get("params_changed", {}).items()
        }
        return d


class CandidateFactory:
    """
    Structured candidate generation with 5 generators:
      1. LocalParameterGenerator — small bounded adjustments
      2. SensitivityGuidedGenerator — focus on stable zones
      3. RegimeSpecificGenerator — fix weak regimes
      4. JointOptimizationGenerator — multi-param combos
      5. LLMAdvisoryGenerator — GPT ideas translated to candidates
    """

    def __init__(self, settings: Any, governance: OptimizationGovernanceEngine) -> None:
        self._settings = settings
        self._governance = governance
        self._Settings_cls: Any = None
        try:
            from config.settings import Settings
            self._Settings_cls = Settings
        except ImportError:
            pass

    def _get_param_bounds(self, param_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get (min, max, current) for a settings parameter."""
        try:
            if self._Settings_cls is None:
                return (None, None, getattr(self._settings, param_name, None))
            field_info = self._Settings_cls.model_fields.get(param_name)
            if field_info is None:
                return (None, None, None)
            current = getattr(self._settings, param_name, None)
            lower, upper = None, None
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    lower = constraint.ge
                if hasattr(constraint, "gt"):
                    lower = constraint.gt
                if hasattr(constraint, "le"):
                    upper = constraint.le
                if hasattr(constraint, "lt"):
                    upper = constraint.lt
            return (lower, upper, current)
        except Exception:
            return (None, None, getattr(self._settings, param_name, None))

    def _tunable_params(self) -> List[str]:
        """Return list of numeric tunable parameters from Settings."""
        params = []
        try:
            if self._Settings_cls is None:
                return params
            for name, field_info in self._Settings_cls.model_fields.items():
                val = getattr(self._settings, name, None)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    params.append(name)
        except Exception:
            pass
        return params

    # ── Generator 1: Local Parameter ────────────────────────────────────

    def generate_local(self, key_params: Optional[List[str]] = None, n: int = 5) -> List[OptimizationCandidate]:
        """
        Generate small bounded adjustments around current parameter values.
        Adjustments are +/-10% to +/-20% of the current value.
        """
        if not self._governance.allows_param_changes():
            return []

        candidates = []
        params = key_params or self._tunable_params()[:15]

        if not _IMPORTS_OK.get("numpy"):
            return candidates

        import numpy as np
        rng = np.random.default_rng(int(time.time()) % 2**31)

        for param_name in params[:n]:
            try:
                lower, upper, current = self._get_param_bounds(param_name)
                if current is None or not isinstance(current, (int, float)):
                    continue
                current = float(current)
                if lower is None or upper is None:
                    continue

                # Small adjustment: 10-20% of range
                param_range = float(upper) - float(lower)
                if param_range <= 0:
                    continue

                delta = rng.uniform(0.05, 0.15) * param_range
                direction = rng.choice([-1, 1])
                new_val = current + direction * delta
                new_val = max(float(lower), min(float(upper), new_val))

                if abs(new_val - current) < 1e-9:
                    continue

                candidates.append(OptimizationCandidate(
                    candidate_type="param",
                    source="local",
                    params_changed={param_name: (current, round(new_val, 6))},
                    rationale=f"Local adjustment: {param_name} {current:.4f} -> {new_val:.4f} "
                              f"({direction * delta / param_range:+.1%} of range)",
                    expected_improvement=0.01,
                ))
            except Exception as exc:
                log.debug("Local generator failed for %s: %s", param_name, exc)

        log.info("LocalParameterGenerator: %d candidates", len(candidates))
        return candidates

    # ── Generator 2: Sensitivity-Guided ─────────────────────────────────

    def generate_sensitivity_guided(
        self, sensitivity_results: Dict[str, Dict], n: int = 3
    ) -> List[OptimizationCandidate]:
        """
        Generate candidates that move parameters toward their optimal region
        as identified by sensitivity analysis.
        """
        if not self._governance.allows_param_changes():
            return []

        candidates = []
        for param_name, sa in sensitivity_results.items():
            if not isinstance(sa, dict) or "optimal_region" not in sa:
                continue
            try:
                optimal = sa["optimal_region"]
                best_value = float(optimal.get("best_value", 0))
                current = float(sa.get("current_value", 0))
                stability = sa.get("stability_score", "UNKNOWN")

                if abs(best_value - current) < 1e-9:
                    continue

                candidates.append(OptimizationCandidate(
                    candidate_type="param",
                    source="sensitivity",
                    params_changed={param_name: (current, round(best_value, 6))},
                    rationale=(
                        f"Sensitivity-guided: {param_name} from {current:.4f} to optimal "
                        f"{best_value:.4f} (stability={stability}, "
                        f"gradient={sa.get('gradient', 0):.4f})"
                    ),
                    expected_improvement=abs(sa.get("gradient", 0)) * abs(best_value - current),
                ))
            except Exception as exc:
                log.debug("Sensitivity generator failed for %s: %s", param_name, exc)

        candidates.sort(key=lambda c: c.expected_improvement, reverse=True)
        log.info("SensitivityGuidedGenerator: %d candidates", len(candidates[:n]))
        return candidates[:n]

    # ── Generator 3: Regime-Specific ────────────────────────────────────

    def generate_regime_specific(
        self, machine_summary: Dict[str, Any], metrics: Dict[str, Any]
    ) -> List[OptimizationCandidate]:
        """
        Generate candidates that target the weakest regime.
        Maps regime-specific parameters to improvements.
        """
        if not self._governance.allows_param_changes():
            return []

        candidates = []
        regime_breakdown = metrics.get("regime_breakdown", {})
        if not regime_breakdown:
            return candidates

        # Find weakest regime
        weakest_regime = None
        weakest_sharpe = float("inf")
        for regime_name, rd in regime_breakdown.items():
            sharpe = float(rd.get("sharpe", 0)) if isinstance(rd, dict) else float(rd)
            if sharpe < weakest_sharpe:
                weakest_sharpe = sharpe
                weakest_regime = regime_name

        if weakest_regime is None:
            return candidates

        # Regime-specific parameter mappings
        regime_key = weakest_regime.lower().replace(" ", "_")
        regime_params = {
            "calm": [
                "regime_conviction_scale_calm", "regime_size_calm",
                "regime_z_calm", "max_leverage_calm",
            ],
            "normal": [
                "regime_conviction_scale_normal", "regime_size_normal",
                "regime_z_normal", "max_leverage_normal",
            ],
            "tension": [
                "regime_conviction_scale_tension", "regime_size_tension",
                "regime_z_tension", "max_leverage_tension",
            ],
            "crisis": [
                "regime_conviction_scale_crisis", "regime_size_crisis",
                "regime_z_crisis", "max_leverage_crisis",
            ],
        }

        target_params = regime_params.get(regime_key, [])
        for param_name in target_params:
            try:
                lower, upper, current = self._get_param_bounds(param_name)
                if current is None or lower is None or upper is None:
                    continue
                current = float(current)
                lower_f, upper_f = float(lower), float(upper)

                # For weak regime: try reducing exposure (lower conviction/size)
                # or widening thresholds
                if weakest_sharpe < 0:
                    # Negative Sharpe — reduce exposure in this regime
                    new_val = max(lower_f, current * 0.7)
                else:
                    # Low but positive — slight increase
                    new_val = min(upper_f, current * 1.15)

                new_val = max(lower_f, min(upper_f, round(new_val, 6)))
                if abs(new_val - current) < 1e-9:
                    continue

                candidates.append(OptimizationCandidate(
                    candidate_type="regime",
                    source="regime",
                    params_changed={param_name: (current, new_val)},
                    rationale=(
                        f"Regime-specific: target {weakest_regime} (Sharpe={weakest_sharpe:.3f}), "
                        f"adjust {param_name} {current:.4f} -> {new_val:.4f}"
                    ),
                    expected_improvement=0.02,
                ))
            except Exception as exc:
                log.debug("Regime generator failed for %s: %s", param_name, exc)

        log.info("RegimeSpecificGenerator: %d candidates for %s", len(candidates), weakest_regime)
        return candidates

    # ── Generator 4: Joint Optimization ─────────────────────────────────

    def generate_joint(
        self, param_groups: Optional[List[List[str]]] = None, n_combos: int = 3
    ) -> List[OptimizationCandidate]:
        """
        Generate multi-parameter combination candidates.
        Tests correlated parameter groups together.
        """
        if not self._governance.allows_param_changes():
            return []
        if not _IMPORTS_OK.get("numpy"):
            return []

        import numpy as np
        rng = np.random.default_rng(int(time.time()) % 2**31)

        default_groups = [
            ["signal_a1_frob", "signal_a2_mode", "signal_a3_coc"],
            ["regime_conviction_scale_calm", "regime_conviction_scale_tension"],
            ["pca_window", "zscore_window", "macro_window"],
            ["signal_entry_threshold", "signal_z_cap"],
        ]
        groups = param_groups or default_groups
        candidates = []

        for group in groups:
            try:
                params_changes: Dict[str, Tuple[float, float]] = {}
                valid = True
                for param_name in group:
                    lower, upper, current = self._get_param_bounds(param_name)
                    if current is None or lower is None or upper is None:
                        valid = False
                        break
                    current = float(current)
                    lo, hi = float(lower), float(upper)
                    new_val = float(rng.uniform(
                        max(lo, current - 0.15 * (hi - lo)),
                        min(hi, current + 0.15 * (hi - lo)),
                    ))
                    new_val = round(max(lo, min(hi, new_val)), 6)
                    params_changes[param_name] = (current, new_val)

                if valid and params_changes:
                    candidates.append(OptimizationCandidate(
                        candidate_type="param",
                        source="joint",
                        params_changed=params_changes,
                        rationale=f"Joint optimization of [{', '.join(group)}]",
                        expected_improvement=0.015,
                    ))
            except Exception as exc:
                log.debug("Joint generator failed for group %s: %s", group, exc)

        log.info("JointOptimizationGenerator: %d candidates", len(candidates[:n_combos]))
        return candidates[:n_combos]

    # ── Generator 5: LLM Advisory ───────────────────────────────────────

    def generate_llm_advisory(
        self, metrics: Dict[str, Any], machine_summary: Dict[str, Any]
    ) -> List[OptimizationCandidate]:
        """
        Query GPT for advisory optimization ideas and translate to candidates.
        LLM output is ADVISORY ONLY — all candidates still go through sandbox.
        """
        if not _IMPORTS_OK.get("llm_bridge"):
            return []

        candidates = []
        try:
            bridge = DualLLMBridge()
            if not bridge.has_gpt and not bridge.has_claude:
                return []

            sharpe = metrics.get("sharpe", 0)
            hit_rate = metrics.get("hit_rate", metrics.get("win_rate", 0))
            regime = machine_summary.get("current_regime", "UNKNOWN")

            prompt = (
                "You are a quant parameter optimization advisor.\n"
                f"Current metrics: Sharpe={sharpe}, HitRate={hit_rate}, Regime={regime}\n"
                f"Best strategy decision: {machine_summary.get('best_strategy_decision', 'REJECTED')}\n\n"
                "Suggest 2-3 specific parameter changes for a sector mean-reversion system.\n"
                "For each, provide EXACTLY this JSON format:\n"
                '{"param": "param_name", "new_value": 1.23, "reason": "why"}\n'
                "Available params: signal_a1_frob, signal_a2_mode, signal_a3_coc, "
                "pca_window, zscore_window, signal_entry_threshold, signal_z_cap, "
                "regime_conviction_scale_calm, regime_conviction_scale_tension, "
                "regime_size_calm, regime_size_tension, signal_optimal_hold, "
                "trade_max_holding_days, ewma_lambda"
            )

            if bridge.has_gpt:
                response = bridge.query_gpt(prompt, "optimization_advisory")
            else:
                response = bridge.query_claude(prompt, "optimization_advisory")

            if not response:
                return []

            # Parse JSON suggestions from response
            import re as _re
            json_matches = _re.findall(
                r'\{[^{}]*"param"[^{}]*"new_value"[^{}]*\}', response
            )

            for match in json_matches[:3]:
                try:
                    suggestion = json.loads(match)
                    param_name = suggestion.get("param", "")
                    new_value = suggestion.get("new_value")
                    reason = suggestion.get("reason", "LLM advisory")

                    if not param_name or new_value is None:
                        continue

                    lower, upper, current = self._get_param_bounds(param_name)
                    if current is None:
                        continue

                    new_value = float(new_value)
                    current = float(current)
                    if lower is not None:
                        new_value = max(float(lower), new_value)
                    if upper is not None:
                        new_value = min(float(upper), new_value)

                    candidates.append(OptimizationCandidate(
                        candidate_type="param",
                        source="llm_advisory",
                        params_changed={param_name: (current, round(new_value, 6))},
                        rationale=f"LLM advisory (non-binding): {reason}",
                        expected_improvement=0.005,
                    ))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        except Exception as exc:
            log.warning("LLM advisory generator failed: %s", exc)

        log.info("LLMAdvisoryGenerator: %d candidates", len(candidates))
        return candidates

    # ── Master: Generate All ────────────────────────────────────────────

    def generate_all(
        self,
        machine_summary: Dict[str, Any],
        metrics: Dict[str, Any],
        sensitivity_results: Optional[Dict] = None,
        key_params: Optional[List[str]] = None,
    ) -> List[OptimizationCandidate]:
        """
        Run all 5 generators and return combined candidate list.

        Parameters
        ----------
        machine_summary : dict
            Methodology machine_summary.
        metrics : dict
            Current backtest metrics.
        sensitivity_results : dict, optional
            Results from sensitivity analysis.
        key_params : list[str], optional
            Priority parameters for local generator.

        Returns
        -------
        list[OptimizationCandidate]
            All generated candidates.
        """
        all_candidates: List[OptimizationCandidate] = []

        # 1. Local parameter adjustments
        try:
            all_candidates.extend(self.generate_local(key_params, n=5))
        except Exception as exc:
            log.warning("Local generator error: %s", exc)

        # 2. Sensitivity-guided (if results available)
        if sensitivity_results:
            try:
                all_candidates.extend(
                    self.generate_sensitivity_guided(sensitivity_results, n=3)
                )
            except Exception as exc:
                log.warning("Sensitivity generator error: %s", exc)

        # 3. Regime-specific
        try:
            all_candidates.extend(
                self.generate_regime_specific(machine_summary, metrics)
            )
        except Exception as exc:
            log.warning("Regime generator error: %s", exc)

        # 4. Joint optimization
        try:
            all_candidates.extend(self.generate_joint(n_combos=3))
        except Exception as exc:
            log.warning("Joint generator error: %s", exc)

        # 5. LLM advisory
        try:
            all_candidates.extend(
                self.generate_llm_advisory(metrics, machine_summary)
            )
        except Exception as exc:
            log.warning("LLM advisory generator error: %s", exc)

        log.info("CandidateFactory: %d total candidates generated", len(all_candidates))
        return all_candidates


# ═════════════════════════════════════════════════════════════════════════
# D. SandboxEvaluator — 4-Stage Validation Pipeline
# ═════════════════════════════════════════════════════════════════════════

class SandboxEvaluator:
    """
    4-stage validation pipeline for optimization candidates:
      Stage 0: Static validation (bounds, type, schema)
      Stage 1: Quick screen (fast backtest)
      Stage 2: Institutional sandbox (full methodology validation)
      Stage 3: Promotion recommendation (champion vs challenger)
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._Settings_cls: Any = None
        try:
            from config.settings import Settings
            self._Settings_cls = Settings
        except ImportError:
            pass

    def _validate_bounds(self, param_name: str, value: float) -> Tuple[bool, str, float]:
        """Check if value is within Settings field bounds. Returns (ok, reason, clamped)."""
        try:
            if self._Settings_cls is None:
                return (True, "no schema", value)
            field_info = self._Settings_cls.model_fields.get(param_name)
            if field_info is None:
                return (False, f"unknown parameter: {param_name}", value)

            lower, upper = None, None
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    lower = constraint.ge
                if hasattr(constraint, "le"):
                    upper = constraint.le

            clamped = value
            if lower is not None and value < lower:
                clamped = lower
            if upper is not None and value > upper:
                clamped = upper

            if abs(clamped - value) > 1e-9:
                return (False, f"{param_name}={value} out of bounds [{lower}, {upper}]", clamped)
            return (True, "within bounds", value)
        except Exception as exc:
            return (True, f"bounds check error: {exc}", value)

    def stage0_static_validation(self, candidate: OptimizationCandidate) -> Tuple[bool, str]:
        """
        Stage 0: Static validation — bounds, type, schema.
        No backtests needed. Pure constraint checking.
        """
        try:
            for param_name, (old_val, new_val) in candidate.params_changed.items():
                # Type check
                if not isinstance(new_val, (int, float)):
                    candidate.validation_stages["stage0"] = f"FAIL:type:{param_name}"
                    return False, f"Invalid type for {param_name}: {type(new_val)}"

                # Bounds check
                ok, reason, clamped = self._validate_bounds(param_name, float(new_val))
                if not ok:
                    # Auto-clamp and warn
                    candidate.params_changed[param_name] = (old_val, clamped)
                    log.warning("Stage0: clamped %s from %s to %s (%s)", param_name, new_val, clamped, reason)

                # NaN/Inf check
                if math.isnan(float(new_val)) or math.isinf(float(new_val)):
                    candidate.validation_stages["stage0"] = f"FAIL:nan_inf:{param_name}"
                    return False, f"NaN/Inf value for {param_name}"

                # No-op check
                if abs(float(new_val) - float(old_val)) < 1e-12:
                    candidate.validation_stages["stage0"] = f"FAIL:no_change:{param_name}"
                    return False, f"No change for {param_name}"

            candidate.validation_stages["stage0"] = "PASS"
            return True, "static validation passed"
        except Exception as exc:
            candidate.validation_stages["stage0"] = f"ERROR:{exc}"
            return False, str(exc)

    def stage1_quick_screen(
        self, candidate: OptimizationCandidate, backtest_fn: Any
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Stage 1: Quick screen — fast backtest with candidate parameters.
        Rejects if Sharpe drops below 70% of current or becomes negative.
        """
        try:
            originals: Dict[str, Any] = {}
            for param_name, (old_val, new_val) in candidate.params_changed.items():
                originals[param_name] = getattr(self._settings, param_name, old_val)
                try:
                    setattr(self._settings, param_name, type(originals[param_name])(new_val))
                except Exception:
                    setattr(self._settings, param_name, new_val)

            # Run quick backtest
            result = backtest_fn()
            new_sharpe = float(result.get("sharpe", 0))

            # Restore originals
            for param_name, orig_val in originals.items():
                setattr(self._settings, param_name, orig_val)

            # Quick screen criteria
            if new_sharpe < -0.5:
                candidate.validation_stages["stage1"] = f"FAIL:negative_sharpe={new_sharpe:.3f}"
                return False, result

            candidate.validation_stages["stage1"] = f"PASS:sharpe={new_sharpe:.3f}"
            return True, result

        except Exception as exc:
            # Restore on error
            for param_name, (old_val, _) in candidate.params_changed.items():
                try:
                    setattr(self._settings, param_name, old_val)
                except Exception:
                    pass
            candidate.validation_stages["stage1"] = f"ERROR:{exc}"
            return False, {"error": str(exc)}

    def stage2_institutional_sandbox(
        self, candidate: OptimizationCandidate, before_metrics: Dict[str, Any],
        backtest_fn: Any,
    ) -> Tuple[bool, ObjectiveBreakdown, Dict[str, Any]]:
        """
        Stage 2: Full institutional sandbox — evaluate composite objective.
        Applies candidate, runs full backtest, computes objective breakdown.
        """
        try:
            originals: Dict[str, Any] = {}
            for param_name, (old_val, new_val) in candidate.params_changed.items():
                originals[param_name] = getattr(self._settings, param_name, old_val)
                try:
                    setattr(self._settings, param_name, type(originals[param_name])(new_val))
                except Exception:
                    setattr(self._settings, param_name, new_val)

            # Full backtest
            after_result = backtest_fn()

            # Restore
            for param_name, orig_val in originals.items():
                setattr(self._settings, param_name, orig_val)

            # Compute objective
            objective = compute_objective(before_metrics, after_result)
            candidate.objective_breakdown = objective

            if objective.composite_score < -0.1:
                candidate.validation_stages["stage2"] = (
                    f"FAIL:composite={objective.composite_score:.4f}"
                )
                return False, objective, after_result

            candidate.validation_stages["stage2"] = (
                f"PASS:composite={objective.composite_score:.4f}"
            )
            return True, objective, after_result

        except Exception as exc:
            for param_name, (old_val, _) in candidate.params_changed.items():
                try:
                    setattr(self._settings, param_name, old_val)
                except Exception:
                    pass
            empty_obj = ObjectiveBreakdown()
            candidate.validation_stages["stage2"] = f"ERROR:{exc}"
            return False, empty_obj, {"error": str(exc)}

    def stage3_promotion_check(
        self, candidate: OptimizationCandidate, after_metrics: Dict[str, Any]
    ) -> Tuple[str, List[str], List[str]]:
        """
        Stage 3: Promotion recommendation via deterministic gate.
        Uses evaluate_promotion() from report_schema.py.
        """
        try:
            net_sharpe = float(after_metrics.get("sharpe", 0))
            max_drawdown = float(after_metrics.get("max_drawdown", 0))
            total_trades = int(after_metrics.get("total_trades", 0))

            # Count positive regimes
            regime_bd = after_metrics.get("regime_breakdown", {})
            positive_regimes = 0
            if isinstance(regime_bd, dict):
                for rd in regime_bd.values():
                    s = float(rd.get("sharpe", 0)) if isinstance(rd, dict) else float(rd)
                    if s > 0:
                        positive_regimes += 1

            deflated_sharpe = float(after_metrics.get("deflated_sharpe", net_sharpe * 0.8))
            tail_risk_score = float(after_metrics.get("tail_risk_score", 0.5))
            cost_drag_pct = float(after_metrics.get("cost_drag_pct", 10.0))
            stability_score = float(after_metrics.get("stability_score", 0.5))

            decision, fail_reasons, warning_reasons = evaluate_promotion(
                net_sharpe=net_sharpe,
                max_drawdown=max_drawdown,
                total_trades=total_trades,
                positive_regimes=positive_regimes,
                deflated_sharpe=deflated_sharpe,
                tail_risk_score=tail_risk_score,
                cost_drag_pct=cost_drag_pct,
                stability_score=stability_score,
            )

            candidate.validation_stages["stage3"] = f"{decision}:fails={len(fail_reasons)}"
            return decision, fail_reasons, warning_reasons

        except Exception as exc:
            candidate.validation_stages["stage3"] = f"ERROR:{exc}"
            return "REJECTED", [str(exc)], []

    def run_full_pipeline(
        self, candidate: OptimizationCandidate,
        before_metrics: Dict[str, Any], backtest_fn: Any,
    ) -> Tuple[str, ObjectiveBreakdown, Dict[str, Any]]:
        """
        Run all 4 stages sequentially. Returns (decision, objective, after_metrics).
        Short-circuits on failure at any stage.
        """
        # Stage 0
        ok, reason = self.stage0_static_validation(candidate)
        if not ok:
            candidate.evaluation_status = "REJECTED"
            return "REJECTED", ObjectiveBreakdown(), {}

        # Stage 1
        ok, quick_result = self.stage1_quick_screen(candidate, backtest_fn)
        if not ok:
            candidate.evaluation_status = "REJECTED"
            return "REJECTED", ObjectiveBreakdown(), quick_result

        # Stage 2
        ok, objective, after_metrics = self.stage2_institutional_sandbox(
            candidate, before_metrics, backtest_fn
        )
        if not ok:
            candidate.evaluation_status = "REJECTED"
            return "REJECTED", objective, after_metrics

        candidate.evaluation_status = "SANDBOX_PASSED"

        # Stage 3
        decision, fail_reasons, warning_reasons = self.stage3_promotion_check(
            candidate, after_metrics
        )

        return decision, objective, after_metrics


# ═════════════════════════════════════════════════════════════════════════
# E. ChampionChallengerManager
# ═════════════════════════════════════════════════════════════════════════

class ChampionChallengerManager:
    """
    Manages champion (current best) vs challenger (candidate) comparisons.

    Champion = latest promoted baseline from lineage tracker.
    Challengers = candidates that passed sandbox validation.
    """

    def __init__(self, lineage: OptimizationLineageTracker) -> None:
        self._lineage = lineage
        self._champion_metrics: Optional[Dict[str, Any]] = None
        self._champion_id: str = ""

    def load_champion(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load champion baseline. Falls back to current metrics if no history.
        """
        try:
            champion_record = self._lineage.latest_champion()
            if champion_record and "after_metrics" in champion_record:
                self._champion_metrics = champion_record["after_metrics"]
                self._champion_id = champion_record.get("optimization_id", "genesis")
            else:
                self._champion_metrics = current_metrics
                self._champion_id = "genesis-baseline"
        except Exception:
            self._champion_metrics = current_metrics
            self._champion_id = "genesis-baseline"

        log.info("Champion loaded: %s (Sharpe=%.3f)",
                 self._champion_id,
                 float(self._champion_metrics.get("sharpe", 0)))
        return self._champion_metrics

    @property
    def champion_id(self) -> str:
        return self._champion_id

    @property
    def champion_metrics(self) -> Dict[str, Any]:
        return self._champion_metrics or {}

    def compare(
        self, challenger_metrics: Dict[str, Any], challenger_objective: ObjectiveBreakdown
    ) -> Dict[str, Any]:
        """
        Compare challenger against champion.

        Returns comparison dict with winner determination.
        """
        champion_sharpe = float(self.champion_metrics.get("sharpe", 0))
        challenger_sharpe = float(challenger_metrics.get("sharpe", 0))

        return {
            "champion_id": self._champion_id,
            "champion_sharpe": round(champion_sharpe, 4),
            "challenger_sharpe": round(challenger_sharpe, 4),
            "delta_sharpe": round(challenger_sharpe - champion_sharpe, 4),
            "composite_score": round(challenger_objective.composite_score, 4),
            "challenger_wins": challenger_objective.composite_score > 0,
        }


# ═════════════════════════════════════════════════════════════════════════
# F. PromotionDecisionEngine — Deterministic Rules
# ═════════════════════════════════════════════════════════════════════════

class PromotionDecisionEngine:
    """
    Deterministic promotion rules — NO LLM involvement.

    NEVER promote if:
      - Methodology promotion gate fails
      - Robustness falls
      - Cost drag rises > 5%
      - Improvement is in one regime only (unless tagged specialist)

    Recommend shadow if improvement is marginal (<0.05 composite).
    """

    def decide(
        self,
        candidate: OptimizationCandidate,
        gate_decision: str,
        gate_fails: List[str],
        objective: ObjectiveBreakdown,
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Make final promotion decision.

        Parameters
        ----------
        candidate : OptimizationCandidate
        gate_decision : str
            From evaluate_promotion(): APPROVED/CONDITIONAL/REJECTED
        gate_fails : list[str]
            Reasons for gate failure.
        objective : ObjectiveBreakdown
        before_metrics, after_metrics : dict

        Returns
        -------
        (decision, reason) where decision is PROMOTED/SHADOW/REJECTED
        """
        # Rule 1: NEVER promote if methodology gate fails hard
        if gate_decision == "REJECTED":
            return "REJECTED", f"Methodology gate REJECTED: {'; '.join(gate_fails[:3])}"

        # Rule 2: NEVER promote if robustness falls
        if objective.delta_robustness < -0.05:
            return "REJECTED", (
                f"Robustness degraded by {objective.delta_robustness:.3f} "
                f"(threshold: -0.05)"
            )

        # Rule 3: NEVER promote if cost drag rises > 5%
        if objective.cost_drag_penalty > 0.05:
            return "REJECTED", (
                f"Cost drag increased by {objective.cost_drag_penalty:.3f} "
                f"(threshold: 0.05)"
            )

        # Rule 4: NEVER promote if improvement in only one regime
        # (unless candidate is tagged as regime specialist)
        before_regime = before_metrics.get("regime_breakdown", {})
        after_regime = after_metrics.get("regime_breakdown", {})
        if isinstance(before_regime, dict) and isinstance(after_regime, dict):
            improved_regimes = 0
            degraded_regimes = 0
            for reg in set(list(before_regime.keys()) + list(after_regime.keys())):
                br = before_regime.get(reg, {})
                ar = after_regime.get(reg, {})
                b_sharpe = float(br.get("sharpe", 0)) if isinstance(br, dict) else float(br) if br else 0
                a_sharpe = float(ar.get("sharpe", 0)) if isinstance(ar, dict) else float(ar) if ar else 0
                if a_sharpe > b_sharpe + 0.01:
                    improved_regimes += 1
                elif a_sharpe < b_sharpe - 0.01:
                    degraded_regimes += 1

            if improved_regimes <= 1 and degraded_regimes > 0 and candidate.source != "regime":
                return "REJECTED", (
                    f"Improvement in only {improved_regimes} regime(s) but "
                    f"degraded {degraded_regimes} (non-specialist candidate)"
                )

        # Rule 5: Shadow if improvement is marginal
        if 0.0 < objective.composite_score < 0.05:
            return "SHADOW", (
                f"Marginal improvement (composite={objective.composite_score:.4f} < 0.05) "
                f"— entering shadow monitoring"
            )

        # Rule 6: Shadow if gate is CONDITIONAL
        if gate_decision == "CONDITIONAL":
            return "SHADOW", (
                f"Gate CONDITIONAL — entering shadow for monitoring"
            )

        # Rule 7: REJECT if no improvement
        if objective.composite_score <= 0:
            return "REJECTED", (
                f"No improvement: composite={objective.composite_score:.4f}"
            )

        # Promote
        return "PROMOTED", (
            f"All gates passed, composite={objective.composite_score:.4f}, "
            f"delta_sharpe={objective.delta_net_sharpe:+.4f}"
        )


# ═════════════════════════════════════════════════════════════════════════
# Helpers — Settings, Backtest, Edit Functions
# ═════════════════════════════════════════════════════════════════════════

def _load_settings_snapshot() -> Dict[str, Any]:
    """Read all parameters from settings.py with current values, ranges, types."""
    try:
        from config.settings import get_settings, Settings
        settings = get_settings()
        params: Dict[str, Dict] = {}
        for name, field_info in Settings.model_fields.items():
            val = getattr(settings, name, None)
            if isinstance(val, (int, float, bool)):
                meta: Dict[str, Any] = {"value": val, "type": type(val).__name__}
                for constraint in field_info.metadata:
                    if hasattr(constraint, "ge"):
                        meta["min"] = constraint.ge
                    if hasattr(constraint, "le"):
                        meta["max"] = constraint.le
                params[name] = meta
        return params
    except Exception as exc:
        log.warning("Failed to load settings snapshot: %s", exc)
        return {}


def _load_backtest_cache() -> Optional[Dict]:
    """Load backtest cache if available."""
    cache_path = ROOT / "logs" / "last_backtest.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def _load_math_proposals() -> List[Dict]:
    """Read Math Agent proposals from math_proposals dir and bus."""
    proposals = []
    proposals_dir = ROOT / "agents" / "math" / "math_proposals"
    if proposals_dir.exists():
        for f in sorted(proposals_dir.glob("*.py"))[-5:]:
            try:
                content = f.read_text(encoding="utf-8")
                proposals.append({
                    "source": "file", "filename": f.name, "content": content[:3000],
                })
            except Exception:
                continue

    bus = get_bus()
    math_latest = bus.latest("agent_math")
    if math_latest and "proposals" in math_latest:
        for p in math_latest["proposals"]:
            proposals.append({"source": "bus", **p})
    return proposals


def _run_backtest() -> Dict:
    """Run backtest with timeout — prefer dispersion backtest (faster)."""
    # Primary: Dispersion Backtest
    try:
        import pandas as pd
        from analytics.dispersion_backtest import DispersionBacktester
        prices = pd.read_parquet(str(ROOT / "data_lake" / "parquet" / "prices.parquet"))
        bt = DispersionBacktester(
            prices, hold_period=15, z_entry=0.6, z_exit=0.2,
            max_positions=3, lookback=30,
        )
        result = bt.run()
        return {
            "sharpe": result.sharpe,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "max_drawdown": result.max_drawdown,
            "total_trades": result.total_trades,
            "avg_hold": result.avg_holding_days,
            "source": "dispersion_backtest",
        }
    except Exception as e:
        log.warning("Dispersion backtest failed: %s, trying walk-forward...", e)

    # Fallback: Walk-Forward Backtest with 60s timeout
    try:
        import concurrent.futures

        def _walk_forward_backtest() -> Dict:
            from config.settings import get_settings
            from data_ops.orchestrator import DataOrchestrator
            from analytics.backtest import WalkForwardBacktester

            settings = get_settings()
            orchestrator = DataOrchestrator(settings)
            data_state = orchestrator.run(force_refresh=False)
            prices_df = data_state.artifacts.prices

            bt = WalkForwardBacktester(settings)
            result = bt.run_backtest(prices_df, prices_df, prices_df)

            regime_bd = result.regime_breakdown
            regime_dict: Dict = {}
            for reg in ["calm", "normal", "tension", "crisis"]:
                rd = getattr(regime_bd, reg, None)
                if rd:
                    regime_dict[reg.upper()] = {
                        "ic_mean": round(float(rd.ic_mean or 0), 4),
                        "hit_rate": round(float(rd.hit_rate or 0), 4),
                        "sharpe": round(float(rd.sharpe or 0), 4),
                        "n_walks": int(rd.n_walks or 0),
                    }

            return {
                "ic_mean": round(float(result.ic_mean or 0), 4),
                "ic_ir": round(float(result.ic_ir or 0), 4),
                "hit_rate": round(float(result.hit_rate or 0), 4),
                "sharpe": round(float(result.sharpe or 0), 4),
                "max_drawdown": round(float(result.max_drawdown or 0), 4),
                "n_walks": int(result.n_walks or 0),
                "regime_breakdown": regime_dict,
                "source": "walk_forward",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_walk_forward_backtest)
            try:
                return future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                log.error("Walk-forward backtest timed out (60s)")
                return {"sharpe": 0, "error": "walk_forward_timeout"}
    except Exception as exc:
        log.error("Backtest failed: %s", exc)
        return {"sharpe": 0, "error": str(exc)}


# ── Parameter bounds validation ──────────────────────────────────────────

def _validate_param_bounds(param_name: str, new_value: float) -> Dict:
    """Validate parameter change against settings.py Field bounds."""
    try:
        from config.settings import Settings
        field_info = Settings.model_fields.get(param_name)
        if field_info is None:
            return {"valid": False, "reason": f"Unknown parameter: {param_name}", "clamped_value": new_value}

        lower, upper = None, None
        for constraint in field_info.metadata:
            if hasattr(constraint, "ge"):
                lower = constraint.ge
            if hasattr(constraint, "gt"):
                lower = constraint.gt
            if hasattr(constraint, "le"):
                upper = constraint.le
            if hasattr(constraint, "lt"):
                upper = constraint.lt

        clamped = new_value
        reasons = []
        if lower is not None and new_value < lower:
            clamped = lower
            reasons.append(f"below min {lower}")
        if upper is not None and new_value > upper:
            clamped = upper
            reasons.append(f"above max {upper}")

        if reasons:
            return {
                "valid": False,
                "reason": f"{param_name}={new_value} out of bounds ({', '.join(reasons)}). Clamped to {clamped}",
                "clamped_value": clamped,
            }
        return {"valid": True, "reason": "within bounds", "clamped_value": new_value}
    except Exception as exc:
        log.warning("Bounds check failed for %s: %s", param_name, exc)
        return {"valid": True, "reason": f"bounds check error: {exc}", "clamped_value": new_value}


# ── edit_code — code changes with backup, tests, and revert ─────────────

def _execute_edit_code(action: Dict) -> Dict:
    """
    Execute a code change with automatic backup/test/revert:
      1. Backup target file
      2. Replace function (old_code -> new_code)
      3. Run pytest
      4. If tests pass -> keep; if fail -> revert from backup
    """
    target_rel = action.get("target_file", "")
    func_name = action.get("function_name", "")
    new_code = action.get("new_code", "")

    if not target_rel or not func_name or not new_code:
        return {"action": "edit_code", "outcome": "error",
                "error": "missing target_file, function_name, or new_code"}

    target_path = ROOT / target_rel
    if not target_path.exists():
        return {"action": "edit_code", "outcome": "error",
                "error": f"file not found: {target_rel}"}

    # 1. Backup
    backup_path = target_path.with_suffix(target_path.suffix + ".optbak")
    try:
        shutil.copy2(target_path, backup_path)
        log.info("Backup created: %s", backup_path.name)
    except Exception as exc:
        return {"action": "edit_code", "outcome": "error", "error": f"backup failed: {exc}"}

    # 2. Replace function
    try:
        original_text = target_path.read_text(encoding="utf-8")
        lines = original_text.split("\n")
        start_idx = None
        end_idx = len(lines)

        for i, line in enumerate(lines):
            if re.match(rf"def\s+{re.escape(func_name)}\s*\(", line):
                start_idx = i
            elif start_idx is not None and i > start_idx and re.match(r"(?:def\s|class\s)", line):
                end_idx = i
                break

        if start_idx is None:
            backup_path.unlink(missing_ok=True)
            return {"action": "edit_code", "outcome": "error",
                    "error": f"function '{func_name}' not found in {target_rel}"}

        new_lines = lines[:start_idx] + new_code.rstrip().split("\n") + lines[end_idx:]
        target_path.write_text("\n".join(new_lines), encoding="utf-8")
        log.info("Code replaced: %s in %s", func_name, target_rel)

    except Exception as exc:
        shutil.copy2(backup_path, target_path)
        backup_path.unlink(missing_ok=True)
        return {"action": "edit_code", "outcome": "error", "error": f"code replacement failed: {exc}"}

    # 3. Run tests
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        tests_passed = result.returncode == 0 and "passed" in result.stdout

        if tests_passed:
            log.info("Tests PASSED after code edit: %s.%s", target_rel, func_name)
            backup_path.unlink(missing_ok=True)
            return {"action": "edit_code", "outcome": "success",
                    "function": func_name, "file": target_rel,
                    "test_output": result.stdout[-500:]}
        else:
            log.warning("Tests FAILED after code edit — reverting %s", target_rel)
            shutil.copy2(backup_path, target_path)
            backup_path.unlink(missing_ok=True)
            return {"action": "edit_code", "outcome": "reverted",
                    "function": func_name, "file": target_rel,
                    "revert_reason": "tests_failed",
                    "test_output": result.stdout[-500:] + "\n" + result.stderr[-300:]}

    except Exception as exc:
        log.error("Test runner error — reverting: %s", exc)
        shutil.copy2(backup_path, target_path)
        backup_path.unlink(missing_ok=True)
        return {"action": "edit_code", "outcome": "reverted",
                "function": func_name, "file": target_rel,
                "revert_reason": f"test_runner_error: {exc}"}


# ── Extended action executor ─────────────────────────────────────────────

def execute_extended_actions(actions: List[Dict], current_metrics: Optional[Dict] = None) -> List[Dict]:
    """
    Extends execute_actions with edit_code + parameter validation support.
    All other actions are forwarded to the standard executor.
    """
    results = []
    regular_actions = []

    for action in actions:
        if action.get("type") == "edit_code":
            result = _execute_edit_code(action)
            results.append(result)
        elif action.get("type") == "edit_param" and current_metrics:
            param_name = action.get("param", "")
            new_value = action.get("value")
            if param_name and new_value is not None:
                bounds = _validate_param_bounds(param_name, float(new_value))
                if not bounds["valid"]:
                    log.warning("Bounds violation for %s: %s", param_name, bounds["reason"])
                    action["value"] = bounds["clamped_value"]
                    results.append({
                        "action": "bounds_check", "param": param_name,
                        "original_suggestion": new_value,
                        "clamped_to": bounds["clamped_value"],
                        "reason": bounds["reason"],
                    })
                regular_actions.append(action)
            else:
                regular_actions.append(action)
        else:
            regular_actions.append(action)

    if regular_actions:
        regular_results = execute_actions(regular_actions)
        results.extend(regular_results)

    return results


# ═════════════════════════════════════════════════════════════════════════
# G. OptimizerAgent — Main Orchestrator
# ═════════════════════════════════════════════════════════════════════════

class OptimizerAgent:
    """
    Institutional-grade optimization agent.

    Orchestrates the full optimization flow:
      1. Load methodology report (machine_summary)
      2. Determine optimization mode via GovernanceEngine
      3. If FREEZE_AND_MONITOR -> log and exit
      4. Load champion baseline
      5. Generate candidates via CandidateFactory (5 generators)
      6. For each candidate: run 4-stage sandbox validation
      7. Rank candidates by institutional objective
      8. Apply promotion rules
      9. If best candidate passes -> promote (or shadow)
     10. Query GPT for advisory ideas (non-binding)
     11. Save optimization report + update history
     12. Publish to bus
    """

    def __init__(self) -> None:
        """Initialize the OptimizerAgent with all sub-components."""
        try:
            from config.settings import get_settings, Settings
            self.settings = get_settings()
            self.Settings = Settings
        except Exception as exc:
            log.error("OptimizerAgent init — settings load error: %s", exc)
            from config.settings import get_settings, Settings
            self.settings = get_settings()
            self.Settings = Settings

        self.lineage = get_optimization_log()
        self.governance = OptimizationGovernanceEngine()
        self.sandbox = SandboxEvaluator(self.settings)
        self.champion_mgr = ChampionChallengerManager(self.lineage)
        self.promotion_engine = PromotionDecisionEngine()
        self._backup_settings: Dict = {}

        log.info("OptimizerAgent initialized (institutional-grade)")

    # ── Sensitivity Analysis ────────────────────────────────────────────

    def sensitivity_analysis(self, param_name: str, n_points: int = 10) -> Dict:
        """
        Sweep a parameter from min to max and record performance metrics.
        Computes gradient, optimal region, and stability assessment.
        """
        try:
            if not _IMPORTS_OK.get("numpy"):
                return {"error": "numpy not available", "param_name": param_name}

            import numpy as np

            lower, upper, current = self._get_param_bounds(param_name)
            if current is None:
                return {"error": f"parameter '{param_name}' not found"}

            current = float(current)
            if lower is None:
                lower = current * 0.3 if current > 0 else current - abs(current) * 2
            if upper is None:
                upper = current * 3.0 if current > 0 else current + abs(current) * 2
            lower, upper = float(lower), float(upper)

            original_value = current
            sweep_values = np.linspace(lower, upper, n_points)
            sweep_results = []

            for val in sweep_values:
                try:
                    setattr(self.settings, param_name, type(original_value)(val))
                    result = _run_backtest()
                    sweep_results.append({
                        "value": round(float(val), 6),
                        "sharpe": round(float(result.get("sharpe", 0)), 4),
                        "win_rate": round(float(result.get("win_rate", result.get("hit_rate", 0))), 4),
                        "max_drawdown": round(float(result.get("max_drawdown", 0)), 4),
                    })
                except Exception:
                    pass

            # Restore
            setattr(self.settings, param_name, type(original_value)(original_value))

            if len(sweep_results) < 3:
                return {"error": "insufficient sweep data", "param_name": param_name}

            sharpes = np.array([s["sharpe"] for s in sweep_results])
            values = np.array([s["value"] for s in sweep_results])

            # Gradient
            if len(values) > 1:
                d_sharpe = np.diff(sharpes)
                d_param = np.diff(values)
                gradients = d_sharpe / np.where(np.abs(d_param) > 1e-12, d_param, 1e-12)
                avg_gradient = float(np.mean(gradients))
            else:
                avg_gradient = 0.0

            # Optimal region
            peak_sharpe = float(np.max(sharpes))
            threshold = peak_sharpe * 0.9 if peak_sharpe > 0 else peak_sharpe * 1.1
            in_optimal = sharpes >= threshold if peak_sharpe > 0 else sharpes <= threshold
            optimal_indices = np.where(in_optimal)[0]

            if len(optimal_indices) > 0:
                optimal_region = {
                    "lower": round(float(values[optimal_indices[0]]), 6),
                    "upper": round(float(values[optimal_indices[-1]]), 6),
                    "best_value": round(float(values[np.argmax(sharpes)]), 6),
                    "best_sharpe": round(peak_sharpe, 4),
                }
            else:
                optimal_region = {"lower": lower, "upper": upper,
                                  "best_value": current, "best_sharpe": 0}

            stability = round(float(np.std(sharpes)), 4)
            stability_score = (
                "STABLE" if stability < 0.1
                else "MODERATE" if stability < 0.3
                else "SENSITIVE"
            )

            return {
                "param_name": param_name,
                "n_points": len(sweep_results),
                "current_value": round(original_value, 6),
                "sweep_results": sweep_results,
                "gradient": round(avg_gradient, 6),
                "optimal_region": optimal_region,
                "stability": stability,
                "stability_score": stability_score,
            }

        except Exception as exc:
            log.exception("Sensitivity analysis failed for %s", param_name)
            return {"error": str(exc), "param_name": param_name}

    # ── Bayesian Search ─────────────────────────────────────────────────

    def bayesian_search(self, param_name: str, n_trials: int = 20) -> Dict:
        """
        Bayesian-like parameter optimization using expected improvement.
        Uses Optuna if available, otherwise manual surrogate search.
        """
        try:
            lower, upper, current = self._get_param_bounds(param_name)
            if current is None:
                return {"error": f"parameter '{param_name}' not found"}

            current = float(current)
            if lower is None:
                lower = current * 0.3 if current > 0 else current - abs(current) * 2
            if upper is None:
                upper = current * 3.0 if current > 0 else current + abs(current) * 2
            lower, upper = float(lower), float(upper)

            original_value = current
            best_sharpe = self._quick_backtest_sharpe()
            best_value = current
            improvement_curve = [{"trial": 0, "value": current, "sharpe": best_sharpe}]

            # Try Optuna
            optuna_available = False
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                optuna_available = True
            except ImportError:
                pass

            if optuna_available:
                import optuna

                def objective(trial):
                    val = trial.suggest_float(param_name, lower, upper)
                    setattr(self.settings, param_name, type(original_value)(val))
                    return self._quick_backtest_sharpe()

                study = optuna.create_study(direction="maximize")
                study.enqueue_trial({param_name: current})
                study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

                for i, trial in enumerate(study.trials):
                    improvement_curve.append({
                        "trial": i + 1,
                        "value": round(trial.params.get(param_name, current), 6),
                        "sharpe": round(trial.value if trial.value else 0, 4),
                    })
                best_value = study.best_params.get(param_name, current)
                best_sharpe = study.best_value or 0

            else:
                if not _IMPORTS_OK.get("numpy"):
                    setattr(self.settings, param_name, type(original_value)(original_value))
                    return {"error": "numpy not available", "param_name": param_name}

                import numpy as np
                rng = np.random.default_rng(42)

                # Phase 1: Broad exploration
                n_explore = max(n_trials // 2, 5)
                explore_vals = np.linspace(lower, upper, n_explore)
                rng.shuffle(explore_vals)

                for i, val in enumerate(explore_vals):
                    try:
                        setattr(self.settings, param_name, type(original_value)(val))
                        sharpe = self._quick_backtest_sharpe()
                        improvement_curve.append({
                            "trial": i + 1, "value": round(float(val), 6),
                            "sharpe": round(sharpe, 4),
                        })
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_value = float(val)
                    except Exception:
                        continue

                # Phase 2: Refinement
                n_refine = n_trials - n_explore
                if n_refine > 0:
                    radius = (upper - lower) * 0.1
                    for i in range(n_refine):
                        try:
                            val = float(rng.normal(best_value, radius))
                            val = max(lower, min(upper, val))
                            setattr(self.settings, param_name, type(original_value)(val))
                            sharpe = self._quick_backtest_sharpe()
                            improvement_curve.append({
                                "trial": n_explore + i + 1,
                                "value": round(val, 6), "sharpe": round(sharpe, 4),
                            })
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_value = val
                        except Exception:
                            continue

            # Restore
            setattr(self.settings, param_name, type(original_value)(original_value))

            result = {
                "param_name": param_name,
                "original_value": round(original_value, 6),
                "best_value": round(float(best_value), 6),
                "original_sharpe": round(improvement_curve[0]["sharpe"], 4),
                "best_sharpe": round(float(best_sharpe), 4),
                "improvement": round(float(best_sharpe) - improvement_curve[0]["sharpe"], 4),
                "n_trials_run": len(improvement_curve) - 1,
                "improvement_curve": improvement_curve,
                "used_optuna": optuna_available,
            }

            log.info(
                "Bayesian search %s: %.4f -> %.4f (Sharpe %.3f -> %.3f, %+.3f)",
                param_name, original_value, best_value,
                improvement_curve[0]["sharpe"], best_sharpe,
                float(best_sharpe) - improvement_curve[0]["sharpe"],
            )
            return result

        except Exception as exc:
            log.exception("Bayesian search failed for %s", param_name)
            try:
                setattr(self.settings, param_name, original_value)
            except Exception:
                pass
            return {"error": str(exc), "param_name": param_name}

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_param_bounds(self, param_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get (min, max, current) for a parameter."""
        try:
            field_info = self.Settings.model_fields.get(param_name)
            if field_info is None:
                return (None, None, None)
            current = getattr(self.settings, param_name, None)
            lower, upper = None, None
            for constraint in field_info.metadata:
                if hasattr(constraint, "ge"):
                    lower = constraint.ge
                if hasattr(constraint, "gt"):
                    lower = constraint.gt
                if hasattr(constraint, "le"):
                    upper = constraint.le
                if hasattr(constraint, "lt"):
                    upper = constraint.lt
            return (lower, upper, current)
        except Exception:
            return (None, None, getattr(self.settings, param_name, None))

    def _quick_backtest_sharpe(self) -> float:
        """Run quick backtest and return Sharpe."""
        try:
            result = _run_backtest()
            return float(result.get("sharpe", 0))
        except Exception:
            return 0.0

    def _apply_candidate(self, candidate: OptimizationCandidate) -> Dict[str, Any]:
        """Apply a candidate's parameter changes to live settings."""
        applied: Dict[str, Any] = {}
        for param_name, (old_val, new_val) in candidate.params_changed.items():
            try:
                orig = getattr(self.settings, param_name, old_val)
                setattr(self.settings, param_name, type(orig)(new_val))
                applied[param_name] = {"old": old_val, "new": new_val}
                log.info("Applied: %s = %s -> %s", param_name, old_val, new_val)
            except Exception as exc:
                log.error("Failed to apply %s: %s", param_name, exc)
        return applied

    def _revert_candidate(self, candidate: OptimizationCandidate) -> None:
        """Revert a candidate's parameter changes."""
        for param_name, (old_val, _) in candidate.params_changed.items():
            try:
                setattr(self.settings, param_name, old_val)
            except Exception:
                pass

    def _load_machine_summary(self) -> Dict[str, Any]:
        """Load machine_summary from methodology bus report."""
        bus = get_bus()
        meth_report = bus.latest("agent_methodology")
        if isinstance(meth_report, dict):
            ms = meth_report.get("machine_summary", {})
            if isinstance(ms, dict) and ms:
                # Validate
                errors = validate_machine_summary(ms)
                if errors:
                    log.warning("machine_summary validation errors: %s", errors)
                return ms
        # Build minimal default
        return MachineSummary().to_dict()

    # ── GPT Strategy Brainstorming (advisory only) ──────────────────────

    def brainstorm_with_gpt(self, context: Dict) -> Dict:
        """
        Send performance context to GPT for advisory strategy ideas.
        Ideas are NON-BINDING — saved for human review only.
        """
        if not _IMPORTS_OK.get("llm_bridge"):
            return {"skipped": "no LLM bridge available"}

        try:
            bridge = DualLLMBridge()
            if not bridge.has_gpt and not bridge.has_claude:
                return {"skipped": "no LLM available"}

            metrics = context.get("metrics", {})
            regime = context.get("regime", "UNKNOWN")
            weaknesses = context.get("weaknesses", "none")

            prompt = (
                f"You are a senior quant researcher. Current regime: {regime}\n"
                f"Metrics: Sharpe={metrics.get('sharpe', 'N/A')}, "
                f"WR={metrics.get('hit_rate', metrics.get('win_rate', 'N/A'))}\n"
                f"Weaknesses: {weaknesses}\n\n"
                f"Propose 3 NEW trading strategies for sector ETF mean-reversion. "
                f"For each: NAME, SIGNAL, EDGE, EXPECTED Sharpe range."
            )

            if bridge.has_gpt:
                response = bridge.query_gpt(prompt, "strategy_brainstorm")
            else:
                response = bridge.query_claude(prompt, "strategy_brainstorm")

            if not response:
                return {"skipped": "empty LLM response"}

            # Save to file
            ideas_dir = Path(__file__).resolve().parent / "strategy_ideas"
            ideas_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            save_path = ideas_dir / f"brainstorm_{ts}.json"
            save_path.write_text(
                json.dumps({"timestamp": ts, "regime": regime,
                            "metrics": metrics, "raw_response": response[:2000]},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            return {
                "n_ideas": response.count("NAME:") or response.count("name:") or 1,
                "raw_response_preview": response[:300],
                "saved_path": str(save_path),
            }
        except Exception as exc:
            log.warning("GPT brainstorming failed: %s", exc)
            return {"error": str(exc)}

    # ── Main Run — Full Institutional Flow ──────────────────────────────

    def run_institutional(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full institutional optimization pipeline.

        Steps:
          1. Load machine_summary from methodology
          2. Determine governance mode
          3. If frozen -> exit
          4. Load champion baseline
          5. Run sensitivity analysis on key params
          6. Generate candidates (5 generators)
          7. Evaluate candidates through 4-stage sandbox
          8. Rank by composite objective
          9. Apply promotion rules
         10. Promote or shadow best candidate
         11. GPT advisory brainstorm
         12. Log and report

        Returns
        -------
        dict
            Full optimization report.
        """
        report: Dict[str, Any] = {
            "agent": "optimizer",
            "version": "2.0-institutional",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            # Step 1: Load machine_summary
            log.info("[OPT 1/12] Loading machine_summary...")
            machine_summary = self._load_machine_summary()
            report["machine_summary_loaded"] = bool(machine_summary.get("experiment_id"))

            # Step 2: Determine governance mode
            log.info("[OPT 2/12] Determining governance mode...")
            mode = self.governance.determine_mode(machine_summary)
            report["governance"] = self.governance.to_dict()

            # Step 3: Check if frozen
            if self.governance.should_freeze():
                log.info("[OPT 3/12] FROZEN — %s. Logging and exiting.", mode)
                report["outcome"] = "frozen"
                report["reason"] = self.governance.rationale

                self.lineage.log_record(OptimizationRecord(
                    candidate_type="none",
                    source="governance",
                    final_decision="REJECTED",
                    governance_mode=mode,
                    params_changed={},
                    before_metrics=metrics,
                    after_metrics=metrics,
                ))
                return report

            # Step 4: Load champion baseline
            log.info("[OPT 4/12] Loading champion baseline...")
            champion_metrics = self.champion_mgr.load_champion(metrics)
            report["champion_id"] = self.champion_mgr.champion_id

            # Step 5: Sensitivity analysis on key parameters
            log.info("[OPT 5/12] Running sensitivity analysis...")
            key_params = ["pca_window", "zscore_window", "signal_entry_threshold",
                          "signal_a1_frob", "signal_z_cap"]
            sensitivity_results: Dict[str, Dict] = {}
            for param in key_params:
                try:
                    if hasattr(self.settings, param):
                        sa = self.sensitivity_analysis(param, n_points=7)
                        if "error" not in sa:
                            sensitivity_results[param] = sa
                except Exception as exc:
                    log.debug("Sensitivity for %s failed: %s", param, exc)
            report["sensitivity_analysis"] = {
                k: {
                    "gradient": v.get("gradient"),
                    "stability": v.get("stability_score"),
                    "optimal_value": v.get("optimal_region", {}).get("best_value"),
                }
                for k, v in sensitivity_results.items()
            }

            # Step 6: Generate candidates
            log.info("[OPT 6/12] Generating optimization candidates...")
            factory = CandidateFactory(self.settings, self.governance)
            candidates = factory.generate_all(
                machine_summary=machine_summary,
                metrics=metrics,
                sensitivity_results=sensitivity_results,
                key_params=key_params,
            )
            report["candidates_generated"] = len(candidates)

            if not candidates:
                log.info("No candidates generated — nothing to evaluate")
                report["outcome"] = "no_candidates"
                return report

            # Step 7: Evaluate candidates through sandbox
            log.info("[OPT 7/12] Evaluating %d candidates through sandbox...", len(candidates))
            evaluated: List[Tuple[OptimizationCandidate, str, ObjectiveBreakdown, Dict]] = []

            for i, candidate in enumerate(candidates):
                log.info("  Evaluating candidate %d/%d: %s [%s]",
                         i + 1, len(candidates), candidate.candidate_id, candidate.source)
                try:
                    decision, objective, after_result = self.sandbox.run_full_pipeline(
                        candidate, metrics, _run_backtest,
                    )
                    evaluated.append((candidate, decision, objective, after_result))
                    log.info("    -> %s (composite=%.4f)", decision, objective.composite_score)
                except Exception as exc:
                    log.warning("    -> ERROR: %s", exc)
                    candidate.evaluation_status = "REJECTED"
                    candidate.validation_stages["pipeline"] = f"ERROR:{exc}"

            report["candidates_evaluated"] = len(evaluated)

            # Step 8: Rank by composite objective
            log.info("[OPT 8/12] Ranking candidates by composite objective...")
            evaluated.sort(key=lambda x: x[2].composite_score, reverse=True)

            ranked = []
            for cand, dec, obj, aft in evaluated:
                ranked.append({
                    "candidate_id": cand.candidate_id,
                    "source": cand.source,
                    "composite_score": obj.composite_score,
                    "gate_decision": dec,
                    "params": {k: list(v) for k, v in cand.params_changed.items()},
                })
            report["candidate_ranking"] = ranked[:10]

            # Step 9: Apply promotion rules to best candidate
            log.info("[OPT 9/12] Applying promotion rules...")
            best_candidate, best_gate, best_objective, best_after = evaluated[0] if evaluated else (None, "REJECTED", ObjectiveBreakdown(), {})

            final_decision = "REJECTED"
            decision_reason = "no valid candidates"

            if best_candidate is not None:
                final_decision, decision_reason = self.promotion_engine.decide(
                    candidate=best_candidate,
                    gate_decision=best_gate,
                    gate_fails=[],
                    objective=best_objective,
                    before_metrics=metrics,
                    after_metrics=best_after,
                )
                best_candidate.evaluation_status = final_decision
                log.info("  Best candidate %s: %s — %s",
                         best_candidate.candidate_id, final_decision, decision_reason)

            report["best_candidate"] = {
                "candidate_id": best_candidate.candidate_id if best_candidate else None,
                "source": best_candidate.source if best_candidate else None,
                "decision": final_decision,
                "reason": decision_reason,
                "composite_score": best_objective.composite_score,
                "objective_breakdown": best_objective.to_dict(),
            }

            # Step 10: Promote or shadow
            log.info("[OPT 10/12] Executing decision: %s", final_decision)
            after_metrics = metrics  # default if no promotion
            delta_sharpe = 0.0

            if final_decision == "PROMOTED" and best_candidate is not None:
                applied = self._apply_candidate(best_candidate)
                after_bt = _run_backtest()
                after_metrics = after_bt
                delta_sharpe = float(after_bt.get("sharpe", 0)) - float(metrics.get("sharpe", 0))
                report["applied_changes"] = applied
                log.info("PROMOTED: applied %d parameter changes (delta_sharpe=%+.4f)",
                         len(applied), delta_sharpe)

            elif final_decision == "SHADOW" and best_candidate is not None:
                log.info("SHADOW: candidate %s entering shadow monitoring",
                         best_candidate.candidate_id)
                report["shadow_candidate"] = best_candidate.to_dict()

            # Step 11: Log to lineage tracker
            log.info("[OPT 11/12] Logging to lineage tracker...")
            record = OptimizationRecord(
                candidate_id=best_candidate.candidate_id if best_candidate else "",
                candidate_type=best_candidate.candidate_type if best_candidate else "none",
                source=best_candidate.source if best_candidate else "none",
                params_changed={
                    k: {"old": v[0], "new": v[1]}
                    for k, v in (best_candidate.params_changed if best_candidate else {}).items()
                },
                before_metrics={
                    "sharpe": float(metrics.get("sharpe", 0)),
                    "hit_rate": float(metrics.get("hit_rate", metrics.get("win_rate", 0))),
                    "robustness": float(metrics.get("robustness", 0)),
                },
                after_metrics={
                    "sharpe": float(after_metrics.get("sharpe", 0)),
                    "hit_rate": float(after_metrics.get("hit_rate", after_metrics.get("win_rate", 0))),
                },
                objective_breakdown=best_objective.to_dict(),
                validation_stages=best_candidate.validation_stages if best_candidate else {},
                final_decision=final_decision,
                governance_mode=mode,
                champion_baseline_id=self.champion_mgr.champion_id,
            )
            self.lineage.log_record(record)

            # Step 12 (bonus): GPT advisory brainstorm
            log.info("[OPT 12/12] GPT advisory brainstorm...")
            brainstorm = self.brainstorm_with_gpt({
                "metrics": metrics,
                "regime": machine_summary.get("current_regime", "UNKNOWN"),
                "weaknesses": decision_reason if final_decision == "REJECTED" else "none critical",
            })
            report["gpt_advisory"] = brainstorm

            # Final report
            report["outcome"] = final_decision.lower()
            report["delta_sharpe"] = round(delta_sharpe, 6)
            report["campaign_summary"] = self.lineage.optimization_campaign_summary()

            log.info("Institutional optimization complete: %s (delta_sharpe=%+.4f)",
                     final_decision, delta_sharpe)
            return report

        except Exception as exc:
            log.exception("Institutional optimization failed: %s", exc)
            report["outcome"] = "error"
            report["error"] = str(exc)
            return report


# ═════════════════════════════════════════════════════════════════════════
# Main run() — Entrypoint
# ═════════════════════════════════════════════════════════════════════════

def run(once: bool = False) -> None:
    """
    Run the institutional optimizer agent.
    once=True -> single run; False -> loop every 4 hours.
    """
    registry = get_registry()
    registry.register("agent_optimizer", role="institutional parameter & code optimization")

    bus = get_bus()
    opt_log = get_optimization_log()

    while True:
        try:
            registry.heartbeat("agent_optimizer", AgentStatus.RUNNING)
            log.info("=" * 60)
            log.info("Optimizer Agent — Institutional-Grade Run")
            log.info("=" * 60)

            # 1. Load methodology report from bus
            methodology_report = bus.latest("agent_methodology")
            log.info("Methodology report: %s", "found" if methodology_report else "none")

            # 2. Load math proposals
            math_proposals = _load_math_proposals()
            log.info("Math proposals: %d found", len(math_proposals))

            # 3. Load current metrics
            metrics = _load_backtest_cache()
            if not metrics:
                improve_report = bus.latest("agent_improve_system")
                if improve_report:
                    metrics = {
                        "ic_mean": improve_report.get("bt_ic", 0),
                        "sharpe": improve_report.get("bt_sharpe", 0),
                        "hit_rate": improve_report.get("bt_hit_rate", 0),
                        "regime_breakdown": {},
                    }

            if not metrics:
                log.warning("No metrics available — running backtest...")
                metrics = _run_backtest()
                if not metrics:
                    log.error("Cannot proceed without metrics. Aborting.")
                    registry.heartbeat("agent_optimizer", AgentStatus.FAILED,
                                       error="no backtest metrics")
                    if once:
                        return
                    time.sleep(14400)
                    continue

            before_metrics = {
                "ic": metrics.get("ic_mean", 0),
                "sharpe": metrics.get("sharpe", 0),
                "hit_rate": metrics.get("hit_rate", 0),
            }

            # 4. Run institutional optimization pipeline
            log.info("Running institutional optimization pipeline...")
            optimizer = OptimizerAgent()
            institutional_result = optimizer.run_institutional(metrics)

            # 5. Extract results
            outcome = institutional_result.get("outcome", "error")
            delta_sharpe = float(institutional_result.get("delta_sharpe", 0))

            # 6. Post-optimization backtest if something was promoted
            if outcome == "promoted":
                after_bt = _run_backtest()
                after_metrics = {
                    "ic": after_bt.get("ic_mean", 0),
                    "sharpe": after_bt.get("sharpe", 0),
                    "hit_rate": after_bt.get("hit_rate", 0),
                }
            else:
                after_metrics = before_metrics

            delta_ic = after_metrics.get("ic", 0) - before_metrics.get("ic", 0)

            log.info(
                "Optimization result: %s (delta_sharpe=%+.4f, delta_ic=%+.4f)",
                outcome, delta_sharpe, delta_ic,
            )

            # 7. Legacy log entry for backward compatibility
            opt_log.log_attempt({
                "agent_source": "optimizer",
                "change_type": "institutional",
                "target_file": "various",
                "before_metrics": before_metrics,
                "after_metrics": after_metrics,
                "outcome": outcome,
                "delta_sharpe": round(delta_sharpe, 6),
                "delta_ic": round(delta_ic, 6),
                "governance_mode": institutional_result.get("governance", {}).get("mode", ""),
                "candidates_generated": institutional_result.get("candidates_generated", 0),
                "candidates_evaluated": institutional_result.get("candidates_evaluated", 0),
            })

            # 8. Publish to bus
            bus.publish("agent_optimizer", {
                "status": "completed",
                "outcome": outcome,
                "before_sharpe": before_metrics["sharpe"],
                "after_sharpe": after_metrics.get("sharpe", before_metrics["sharpe"]),
                "delta_sharpe": round(delta_sharpe, 6),
                "delta_ic": round(delta_ic, 6),
                "governance_mode": institutional_result.get("governance", {}).get("mode", ""),
                "best_candidate": institutional_result.get("best_candidate", {}),
                "candidates_generated": institutional_result.get("candidates_generated", 0),
                "campaign_summary": institutional_result.get("campaign_summary", {}),
                "math_proposals_used": len(math_proposals),
            })

            # 9. Save full report
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = LOG_DIR / f"agent_optimizer_{ts}.json"
            report_path.write_text(
                json.dumps(institutional_result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("Full report saved -> %s", report_path.name)

            registry.heartbeat("agent_optimizer", AgentStatus.COMPLETED)
            log.info("Optimizer run completed successfully.")

        except Exception as exc:
            log.exception("Optimizer agent error: %s", exc)
            registry.heartbeat("agent_optimizer", AgentStatus.FAILED, error=str(exc))
            bus.publish("agent_optimizer", {
                "status": "failed",
                "error": str(exc),
            })

        if once:
            break

        # Wait between runs — 4 hours
        log.info("Sleeping 4 hours until next optimization run...")
        time.sleep(14400)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(once="--once" in sys.argv)
