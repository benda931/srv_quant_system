"""
agents/methodology/report_schema.py
====================================
Locked JSON report schema for the Methodology Agent.

This module defines the CANONICAL structure of methodology reports.
All downstream agents (Optimizer, Auto-Improve, Execution) consume
reports via this schema. Changes here require version bumps.

Schema version: 2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

REPORT_SCHEMA_VERSION = "2.0"

# ─────────────────────────────────────────────────────────────────────
# Governance record
# ─────────────────────────────────────────────────────────────────────
GOVERNANCE_REQUIRED_KEYS = frozenset({
    "experiment_id",
    "data_fingerprint",
    "settings_fingerprint",
    "methodology_version",
    "run_mode",
    "validation_status",
    "promotion_readiness",
    "fail_reasons",
    "timestamp",
})

# ─────────────────────────────────────────────────────────────────────
# Promotion decisions — the ONLY valid values
# ─────────────────────────────────────────────────────────────────────
VALID_PROMOTION_DECISIONS = frozenset({"APPROVED", "CONDITIONAL", "REJECTED"})

# ─────────────────────────────────────────────────────────────────────
# Strategy classifications
# ─────────────────────────────────────────────────────────────────────
VALID_CLASSIFICATIONS = frozenset({
    "CORE", "DIVERSIFIER", "EXPERIMENTAL", "REDUNDANT", "DISABLE",
})

# ─────────────────────────────────────────────────────────────────────
# Promotion gate criteria (deterministic, no GPT)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PromotionCriteria:
    """Immutable promotion gate thresholds."""
    min_net_sharpe: float = 0.3
    max_drawdown_pct: float = 0.15
    min_trades: int = 100
    min_positive_regimes: int = 2
    min_deflated_sharpe: float = 0.0
    min_tail_risk_score: float = 0.3    # warning, not hard fail
    max_cost_drag_pct: float = 30.0
    min_stability_score: float = 0.5    # warning, not hard fail

DEFAULT_PROMOTION_CRITERIA = PromotionCriteria()


def evaluate_promotion(
    net_sharpe: float,
    max_drawdown: float,
    total_trades: int,
    positive_regimes: int,
    deflated_sharpe: float,
    tail_risk_score: float,
    cost_drag_pct: float,
    stability_score: float,
    criteria: PromotionCriteria = DEFAULT_PROMOTION_CRITERIA,
) -> tuple:
    """
    Pure-function deterministic promotion gate.

    Returns
    -------
    (decision, fail_reasons, warning_reasons)
        decision:        "APPROVED" | "CONDITIONAL" | "REJECTED"
        fail_reasons:    list[str] — hard fails
        warning_reasons: list[str] — soft warnings
    """
    fail_reasons: List[str] = []
    warning_reasons: List[str] = []

    # Hard fails → REJECTED
    if net_sharpe < criteria.min_net_sharpe:
        fail_reasons.append(
            f"net_sharpe={net_sharpe:.3f} < {criteria.min_net_sharpe} min"
        )
    if abs(max_drawdown) > criteria.max_drawdown_pct:
        fail_reasons.append(
            f"max_drawdown={max_drawdown:.3f} > {criteria.max_drawdown_pct:.0%} ceiling"
        )
    if total_trades < criteria.min_trades:
        fail_reasons.append(
            f"total_trades={total_trades} < {criteria.min_trades} min"
        )
    if positive_regimes < criteria.min_positive_regimes:
        fail_reasons.append(
            f"positive_regimes={positive_regimes} < {criteria.min_positive_regimes} min"
        )
    if deflated_sharpe < criteria.min_deflated_sharpe:
        fail_reasons.append(
            f"deflated_sharpe={deflated_sharpe:.3f} < {criteria.min_deflated_sharpe} (overfit)"
        )
    if cost_drag_pct > criteria.max_cost_drag_pct:
        fail_reasons.append(
            f"cost_drag={cost_drag_pct:.1f}% > {criteria.max_cost_drag_pct}% ceiling"
        )

    # Soft warnings → CONDITIONAL
    if tail_risk_score < criteria.min_tail_risk_score:
        warning_reasons.append(
            f"tail_risk_score={tail_risk_score:.2f} < {criteria.min_tail_risk_score} (poor tail)"
        )
    if stability_score < criteria.min_stability_score:
        warning_reasons.append(
            f"stability_score={stability_score:.2f} < {criteria.min_stability_score}"
        )

    # Decision
    if fail_reasons:
        decision = "REJECTED"
    elif warning_reasons:
        decision = "CONDITIONAL"
    else:
        decision = "APPROVED"

    assert decision in VALID_PROMOTION_DECISIONS
    return decision, fail_reasons, warning_reasons


# ─────────────────────────────────────────────────────────────────────
# Machine summary — stable contract for downstream agents
# ─────────────────────────────────────────────────────────────────────
@dataclass
class MachineSummary:
    """
    Stable, versioned contract that downstream agents consume.

    Downstream agents MUST NOT parse raw report JSON.
    They read machine_summary only.
    """
    # Version
    schema_version: str = REPORT_SCHEMA_VERSION
    methodology_version: str = ""
    experiment_id: str = ""
    timestamp: str = ""

    # Counts
    n_strategies: int = 0
    n_approved: int = 0
    n_conditional: int = 0
    n_rejected: int = 0

    # Best strategy
    best_strategy_name: str = ""
    best_strategy_decision: str = "REJECTED"
    best_net_sharpe: float = 0.0
    best_gross_sharpe: float = 0.0
    best_deflated_sharpe: float = 0.0
    best_hit_rate: float = 0.0
    best_max_drawdown: float = 0.0
    best_total_trades: int = 0
    best_classification: str = "DISABLE"

    # Regime
    current_regime: str = "UNKNOWN"
    best_per_regime: Dict[str, str] = field(default_factory=dict)

    # Validation flags
    validation_complete: bool = False
    governance_complete: bool = False
    overfitting_flag: bool = False
    cost_drag_flag: bool = False

    # Robustness
    mean_robustness: float = 0.0
    mean_stability: float = 0.0

    # Rankings (strategy_name → rank)
    final_rankings: Dict[str, int] = field(default_factory=dict)

    # Action signals for downstream agents
    optimizer_should_tune: bool = False  # True if best is CONDITIONAL
    strategies_to_disable: List[str] = field(default_factory=list)
    strategies_to_observe: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# Report section keys — canonical ordering
# ─────────────────────────────────────────────────────────────────────
REPORT_SECTIONS = [
    "governance",
    "machine_summary",
    "pm_summary",
    "methodology_scorecards",
    "approval_matrix",
    "regime_fitness_map",
    "validation_suite",
    "overfitting_control",
    "cost_analysis",
    "attribution",
    "ensemble_analysis",
    "robustness_summary",
    "promotion_gate",
    "degradation_alerts",
    "advisory_gpt",
    "raw_strategy_results",
]


def validate_report(report: Dict[str, Any]) -> List[str]:
    """
    Validate a methodology report against the locked schema.

    Returns list of validation errors (empty = valid).
    """
    errors: List[str] = []

    # 1. Check governance
    gov = report.get("governance")
    if gov is None:
        errors.append("MISSING: governance section")
    elif isinstance(gov, dict):
        missing = GOVERNANCE_REQUIRED_KEYS - set(gov.keys())
        if missing:
            errors.append(f"governance missing keys: {missing}")
    else:
        errors.append(f"governance is not a dict: {type(gov)}")

    # 2. Check machine_summary
    ms = report.get("machine_summary")
    if ms is None:
        errors.append("MISSING: machine_summary section")
    elif isinstance(ms, dict):
        required = {"schema_version", "experiment_id", "n_strategies",
                     "best_strategy_name", "best_strategy_decision",
                     "best_net_sharpe", "validation_complete"}
        missing = required - set(ms.keys())
        if missing:
            errors.append(f"machine_summary missing keys: {missing}")
        # Validate decision values
        decision = ms.get("best_strategy_decision", "")
        if decision and decision not in VALID_PROMOTION_DECISIONS:
            errors.append(f"invalid best_strategy_decision: {decision}")
    else:
        errors.append(f"machine_summary is not a dict: {type(ms)}")

    # 3. Check scorecards
    sc = report.get("methodology_scorecards")
    if sc is not None and isinstance(sc, list):
        for i, card in enumerate(sc):
            if not isinstance(card, dict):
                errors.append(f"scorecard[{i}] is not a dict")
                continue
            if "promotion_decision" in card:
                if card["promotion_decision"] not in VALID_PROMOTION_DECISIONS:
                    errors.append(
                        f"scorecard[{i}].promotion_decision invalid: "
                        f"{card['promotion_decision']}"
                    )
            if "classification" in card:
                if card["classification"] not in VALID_CLASSIFICATIONS:
                    errors.append(
                        f"scorecard[{i}].classification invalid: "
                        f"{card['classification']}"
                    )

    # 4. Check approval_matrix
    am = report.get("approval_matrix")
    if am is not None and isinstance(am, dict):
        for name, entry in am.items():
            if isinstance(entry, dict):
                decision = entry.get("decision", "")
                if decision not in VALID_PROMOTION_DECISIONS:
                    errors.append(
                        f"approval_matrix[{name}].decision invalid: {decision}"
                    )

    return errors


def validate_machine_summary(ms: Dict[str, Any]) -> List[str]:
    """
    Strict validation of machine_summary for downstream consumers.
    Returns list of errors (empty = valid).
    """
    errors: List[str] = []
    if not isinstance(ms, dict):
        return ["machine_summary is not a dict"]

    required_fields = {
        "schema_version": str,
        "experiment_id": str,
        "n_strategies": (int, float),
        "n_approved": (int, float),
        "n_rejected": (int, float),
        "best_strategy_name": str,
        "best_strategy_decision": str,
        "best_net_sharpe": (int, float),
        "validation_complete": bool,
        "governance_complete": bool,
    }

    for field_name, expected_type in required_fields.items():
        if field_name not in ms:
            errors.append(f"MISSING: {field_name}")
        elif not isinstance(ms[field_name], expected_type):
            errors.append(
                f"TYPE: {field_name} expected {expected_type}, got {type(ms[field_name])}"
            )

    decision = ms.get("best_strategy_decision", "")
    if decision and decision not in VALID_PROMOTION_DECISIONS:
        errors.append(f"INVALID: best_strategy_decision={decision}")

    return errors
