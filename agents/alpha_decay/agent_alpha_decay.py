"""
Alpha Decay Monitor -- Institutional Strategy Health, Failure & Retirement
Governance Engine.

This is the STRATEGY RETIREMENT AUTHORITY for the institutional quant system.
It assembles evidence from ALL 8 upstream agents, computes 7 integrity
dimensions, classifies strategies into 12 health states with strict rules,
builds proper hazard models, diagnoses root causes with 16 failure tags,
determines lifecycle actions, and produces downstream contracts for every agent.

Health Dimensions (7):
  alpha_integrity, robustness_integrity, regime_integrity,
  implementation_integrity, tail_integrity, diversification_integrity,
  lifecycle_integrity

Health States (12):
  HEALTHY, WATCH, EARLY_DECAY, REGIME_SUPPRESSED, IMPLEMENTATION_DECAY,
  ROBUSTNESS_BREAKDOWN, STRUCTURAL_DECAY, SHADOW_FAILURE,
  POST_PROMOTION_FAILURE, RETIREMENT_WATCH, DEAD, INSUFFICIENT_EVIDENCE

Lifecycle Stages:
  CHAMPION_LIVE, SHADOW_CANDIDATE, RECENTLY_PROMOTED, RETIREMENT_WATCH, RETIRED

Legacy backward-compatible taxonomy:
  HEALTHY / EARLY_DECAY / DECAYING / DEAD
  (via SAFE_LEGACY_MAP -- INSUFFICIENT_EVIDENCE NEVER maps to HEALTHY)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "agents" / "alpha_decay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evidence source paths -- all 8 agents
REPORTS_DIR = ROOT / "agents" / "methodology" / "reports"
OPTIMIZER_HISTORY_PATH = ROOT / "agents" / "optimizer" / "optimization_history.json"
REGIME_FORECAST_PATH = ROOT / "agents" / "regime_forecaster" / "regime_forecast.json"
PORTFOLIO_WEIGHTS_PATH = ROOT / "agents" / "portfolio_construction" / "portfolio_weights.json"
RISK_STATUS_PATH = ROOT / "agents" / "risk_guardian" / "risk_status.json"
EXECUTION_STATE_DIR = ROOT / "agents" / "execution"
SCOUT_REPORT_PATH = ROOT / "agents" / "data_scout" / "scout_report.json"
MATH_PROPOSALS_DIR = ROOT / "agents" / "math" / "math_proposals"
PRICES_PATH = ROOT / "data_lake" / "parquet" / "prices.parquet"

DECAY_STATUS_PATH = OUTPUT_DIR / "decay_status.json"
DECAY_HISTORY_PATH = OUTPUT_DIR / "decay_history.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_DIR / "alpha_decay.log", maxBytes=10_000_000, backupCount=3),
    ],
)
log = logging.getLogger("alpha_decay")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
MR_WHITELIST = {"XLC", "XLF", "XLI", "XLU"}

MAX_HISTORY_ENTRIES = 365
EVIDENCE_SOURCE_COUNT = 8  # methodology, optimizer, regime, portfolio, risk, execution, scout, math
MIN_SOURCES_WITHOUT_PRICE = 3
CHANGEPOINT_LOOKBACK = 20
CHANGEPOINT_Z_THRESHOLD = 2.0
CONSECUTIVE_DECLINE_THRESHOLD = 3
CONSECUTIVE_IMPROVE_THRESHOLD = 3
HAZARD_DECAY_FACTOR = 1.35
PROMOTION_SURVEILLANCE_WINDOW = 5

# Dimension weights for composite health score
DIMENSION_WEIGHTS = {
    "alpha_integrity": 0.25,
    "robustness_integrity": 0.20,
    "regime_integrity": 0.15,
    "implementation_integrity": 0.10,
    "tail_integrity": 0.12,
    "diversification_integrity": 0.08,
    "lifecycle_integrity": 0.10,
}

# Evidence source weights for quality scoring
EVIDENCE_SOURCE_WEIGHTS = {
    "methodology": 0.20,
    "optimizer": 0.12,
    "regime": 0.15,
    "portfolio": 0.10,
    "risk": 0.13,
    "execution": 0.10,
    "scout": 0.10,
    "math": 0.10,
}

# Health states ordered by severity (ascending)
HEALTH_STATES = [
    "HEALTHY",
    "WATCH",
    "EARLY_DECAY",
    "REGIME_SUPPRESSED",
    "IMPLEMENTATION_DECAY",
    "ROBUSTNESS_BREAKDOWN",
    "STRUCTURAL_DECAY",
    "SHADOW_FAILURE",
    "POST_PROMOTION_FAILURE",
    "RETIREMENT_WATCH",
    "DEAD",
    "INSUFFICIENT_EVIDENCE",
]

HEALTH_STATE_SEVERITY = {s: i for i, s in enumerate(HEALTH_STATES)}

# Lifecycle stages
LIFECYCLE_STAGES = [
    "CHAMPION_LIVE",
    "SHADOW_CANDIDATE",
    "RECENTLY_PROMOTED",
    "RETIREMENT_WATCH",
    "RETIRED",
]

# Safe legacy mapping -- INSUFFICIENT_EVIDENCE NEVER maps to HEALTHY
SAFE_LEGACY_MAP = {
    "HEALTHY": "HEALTHY",
    "WATCH": "EARLY_DECAY",
    "EARLY_DECAY": "EARLY_DECAY",
    "REGIME_SUPPRESSED": "EARLY_DECAY",
    "IMPLEMENTATION_DECAY": "DECAYING",
    "ROBUSTNESS_BREAKDOWN": "DECAYING",
    "STRUCTURAL_DECAY": "DECAYING",
    "SHADOW_FAILURE": "DECAYING",
    "POST_PROMOTION_FAILURE": "DECAYING",
    "RETIREMENT_WATCH": "DECAYING",
    "DEAD": "DEAD",
    "INSUFFICIENT_EVIDENCE": "EARLY_DECAY",
}

# Lifecycle actions
LIFECYCLE_ACTIONS = [
    "KEEP_LIVE",
    "KEEP_LIVE_WITH_WATCH",
    "REDUCE_EXPOSURE",
    "REGIME_DISABLE",
    "SHADOW_ONLY",
    "FREEZE_PROMOTION",
    "SEND_TO_OPTIMIZER",
    "SEND_TO_METHOD_REVIEW",
    "SEND_TO_MATH_REVIEW",
    "REQUIRE_EXECUTION_REVIEW",
    "REQUIRE_RISK_REVIEW",
    "RETIRE_STRATEGY",
    "KEEP_IN_RETIREMENT_WATCH",
    "NO_ACTION_INSUFFICIENT_EVIDENCE",
]

# Root cause tags
ROOT_CAUSE_TAGS = [
    "predictive_edge_erosion",
    "gross_to_net_collapse",
    "optimizer_repair_failure",
    "post_promotion_instability",
    "regime_transition_breakdown",
    "narrow_regime_dependency",
    "robustness_non_generalization",
    "threshold_pathology",
    "turnover_pathology",
    "execution_drag_linked",
    "diversification_role_loss",
    "repeated_shadow_failure",
    "repeated_rollback_pattern",
    "insufficient_real_evidence",
    "proxy_only_monitoring",
    "structural_nonviability",
]

# Evidence quality tiers
EVIDENCE_QUALITY_FULL = "FULL"
EVIDENCE_QUALITY_PARTIAL = "PARTIAL"
EVIDENCE_QUALITY_DEGRADED = "DEGRADED"
EVIDENCE_QUALITY_INSUFFICIENT = "INSUFFICIENT"

# Regime-specific decay state constants (legacy compat)
REGIME_DECAY_STATES = [
    "CALM_DECAY", "NORMAL_DECAY", "TENSION_BREAKDOWN",
    "CRISIS_FAILURE", "TRANSITION_FRAGILITY", "HEALTHY",
]


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class StrategyEvidenceProfile:
    """Complete evidence assembled from all 8 upstream agents plus price data."""
    methodology_evidence: Dict[str, Any] = field(default_factory=dict)
    optimizer_evidence: Dict[str, Any] = field(default_factory=dict)
    regime_evidence: Dict[str, Any] = field(default_factory=dict)
    portfolio_evidence: Dict[str, Any] = field(default_factory=dict)
    risk_evidence: Dict[str, Any] = field(default_factory=dict)
    execution_evidence: Dict[str, Any] = field(default_factory=dict)
    scout_evidence: Dict[str, Any] = field(default_factory=dict)
    math_evidence: Dict[str, Any] = field(default_factory=dict)
    price_supplemental: Dict[str, Any] = field(default_factory=dict)
    evidence_quality: str = EVIDENCE_QUALITY_INSUFFICIENT
    evidence_quality_score: float = 0.0
    evidence_gaps: List[str] = field(default_factory=list)
    degraded_mode: bool = False

    def available_source_count(self) -> int:
        """Count how many of the 8 agent sources provided data."""
        count = 0
        if self.methodology_evidence:
            count += 1
        if self.optimizer_evidence:
            count += 1
        if self.regime_evidence:
            count += 1
        if self.portfolio_evidence:
            count += 1
        if self.risk_evidence:
            count += 1
        if self.execution_evidence:
            count += 1
        if self.scout_evidence:
            count += 1
        if self.math_evidence:
            count += 1
        return count

    def has_price_data(self) -> bool:
        return bool(self.price_supplemental)

    def source_names(self) -> List[str]:
        """Return list of available source names."""
        sources = []
        if self.methodology_evidence:
            sources.append("methodology")
        if self.optimizer_evidence:
            sources.append("optimizer")
        if self.regime_evidence:
            sources.append("regime")
        if self.portfolio_evidence:
            sources.append("portfolio")
        if self.risk_evidence:
            sources.append("risk")
        if self.execution_evidence:
            sources.append("execution")
        if self.scout_evidence:
            sources.append("scout")
        if self.math_evidence:
            sources.append("math")
        return sources


@dataclass
class ChangePointResult:
    """Results from change-point detection on a strategy's health timeline."""
    abrupt_degradation: bool = False
    persistent_weakening: bool = False
    re_stabilization: bool = False
    z_score_latest: float = 0.0
    deterioration_persistence_score: float = 0.0
    recovery_persistence_score: float = 0.0
    consecutive_declining: int = 0
    consecutive_improving: int = 0
    trough_index: int = -1
    trend_direction: str = "STABLE"  # DECLINING / IMPROVING / STABLE


@dataclass
class StrategyHealthScorecard:
    """Complete health assessment for a single strategy -- the atomic output."""
    name: str = ""
    lifecycle_stage: str = "SHADOW_CANDIDATE"
    health_state: str = "INSUFFICIENT_EVIDENCE"
    health_score: float = 0.5
    hazard_30d: float = 0.5
    hazard_90d: float = 0.5
    retirement_hazard: float = 0.0
    confidence: float = 0.0
    evidence_quality_score: float = 0.0
    degraded_mode: bool = False

    # 7 integrity dimensions
    alpha_integrity: float = 0.5
    robustness_integrity: float = 0.5
    regime_integrity: float = 0.5
    implementation_integrity: float = 0.5
    tail_integrity: float = 0.5
    diversification_integrity: float = 0.5
    lifecycle_integrity: float = 0.5

    # Root cause diagnostics
    primary_root_cause: str = ""
    failure_tags: List[str] = field(default_factory=list)

    # Lifecycle actions
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Evidence gaps
    evidence_gaps: List[str] = field(default_factory=list)

    # Legacy compatibility
    legacy_decay_level: str = "HEALTHY"

    # Change-point
    change_point: Optional[Dict[str, Any]] = field(default=None)

    # Downstream contracts
    downstream_contracts: Dict[str, Any] = field(default_factory=dict)

    # Regime-level detail
    regime_decay_states: Dict[str, str] = field(default_factory=dict)
    strongest_regimes: List[str] = field(default_factory=list)
    weakest_regimes: List[str] = field(default_factory=list)

    # Promotion surveillance
    promotion_instability: float = 0.0
    rollback_count: int = 0
    post_promotion_decline: bool = False

    # Metadata
    timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def dimension_dict(self) -> Dict[str, float]:
        return {
            "alpha_integrity": self.alpha_integrity,
            "robustness_integrity": self.robustness_integrity,
            "regime_integrity": self.regime_integrity,
            "implementation_integrity": self.implementation_integrity,
            "tail_integrity": self.tail_integrity,
            "diversification_integrity": self.diversification_integrity,
            "lifecycle_integrity": self.lifecycle_integrity,
        }

    def dimension_values(self) -> List[float]:
        return [
            self.alpha_integrity,
            self.robustness_integrity,
            self.regime_integrity,
            self.implementation_integrity,
            self.tail_integrity,
            self.diversification_integrity,
            self.lifecycle_integrity,
        ]

    # Backward-compat shims
    @property
    def hazard_score_30d(self) -> float:
        return self.hazard_30d

    @property
    def hazard_score_90d(self) -> float:
        return self.hazard_90d


# Legacy compat alias
StrategyHealthCard = StrategyHealthScorecard


@dataclass
class DecayDimensions:
    """Legacy 6-dimension wrapper for backward compatibility."""
    alpha_quality: float = 0.5
    robustness: float = 0.5
    regime_fitness: float = 0.5
    implementation: float = 0.5
    tail_risk: float = 0.5
    diversification: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def values(self) -> List[float]:
        return [
            self.alpha_quality, self.robustness, self.regime_fitness,
            self.implementation, self.tail_risk, self.diversification,
        ]


# ============================================================================
# 1. StrategyEvidenceAssembler
# ============================================================================
class StrategyEvidenceAssembler:
    """
    Load evidence from ALL 8 upstream agents for every strategy.
    Each source is loaded with try/except and JSON parsing.
    Also loads supplemental price data for real-time validation.
    """

    def __init__(self) -> None:
        self.prices: Optional[pd.DataFrame] = None
        self.methodology_reports: Dict[str, Any] = {}
        self.optimizer_history: Dict[str, Any] = {}
        self.regime_forecast: Dict[str, Any] = {}
        self.portfolio_weights: Dict[str, Any] = {}
        self.risk_status: Dict[str, Any] = {}
        self.execution_state: Dict[str, Any] = {}
        self.scout_report: Dict[str, Any] = {}
        self.math_proposals: Dict[str, Any] = {}
        self.previous_status: Optional[Dict] = None
        self.previous_history: List[Dict] = []

    def load_all(self) -> None:
        """Load all evidence sources. Each source is independent -- failures don't cascade."""
        self._load_prices()
        self._load_methodology()
        self._load_optimizer_history()
        self._load_regime_forecast()
        self._load_portfolio_weights()
        self._load_risk_status()
        self._load_execution_state()
        self._load_scout_report()
        self._load_math_proposals()
        self._load_previous_status()
        self._load_previous_history()
        log.info(
            "Evidence assembly complete: methodology=%s optimizer=%s regime=%s portfolio=%s "
            "risk=%s execution=%s scout=%s math=%s prices=%s",
            bool(self.methodology_reports), bool(self.optimizer_history),
            bool(self.regime_forecast), bool(self.portfolio_weights),
            bool(self.risk_status), bool(self.execution_state),
            bool(self.scout_report), bool(self.math_proposals),
            self.prices is not None,
        )

    # -- prices ---------------------------------------------------------------
    def _load_prices(self) -> None:
        try:
            if PRICES_PATH.exists():
                self.prices = pd.read_parquet(PRICES_PATH)
                log.info("Loaded prices: %d rows x %d cols", len(self.prices), len(self.prices.columns))
        except Exception as exc:
            log.warning("Failed to load prices: %s", exc)

    # -- methodology reports --------------------------------------------------
    def _load_methodology(self) -> None:
        try:
            if not REPORTS_DIR.exists():
                return
            files = sorted(REPORTS_DIR.glob("*_methodology_lab.json"), reverse=True)
            if files:
                raw = json.loads(files[0].read_text(encoding="utf-8"))
                self.methodology_reports = raw if isinstance(raw, dict) else {}
                log.info("Loaded methodology report: %s", files[0].name)
        except Exception as exc:
            log.warning("Failed to load methodology report: %s", exc)

    # -- optimizer history ----------------------------------------------------
    def _load_optimizer_history(self) -> None:
        try:
            if OPTIMIZER_HISTORY_PATH.exists():
                raw = json.loads(OPTIMIZER_HISTORY_PATH.read_text(encoding="utf-8"))
                self.optimizer_history = raw if isinstance(raw, dict) else {}
                log.info("Loaded optimizer history")
        except Exception as exc:
            log.warning("Failed to load optimizer history: %s", exc)

    # -- regime forecast ------------------------------------------------------
    def _load_regime_forecast(self) -> None:
        try:
            if REGIME_FORECAST_PATH.exists():
                raw = json.loads(REGIME_FORECAST_PATH.read_text(encoding="utf-8"))
                self.regime_forecast = raw if isinstance(raw, dict) else {}
                log.info("Loaded regime forecast")
        except Exception as exc:
            log.warning("Failed to load regime forecast: %s", exc)

    # -- portfolio weights ----------------------------------------------------
    def _load_portfolio_weights(self) -> None:
        try:
            if PORTFOLIO_WEIGHTS_PATH.exists():
                raw = json.loads(PORTFOLIO_WEIGHTS_PATH.read_text(encoding="utf-8"))
                self.portfolio_weights = raw if isinstance(raw, dict) else {}
                log.info("Loaded portfolio weights")
        except Exception as exc:
            log.warning("Failed to load portfolio weights: %s", exc)

    # -- risk status ----------------------------------------------------------
    def _load_risk_status(self) -> None:
        try:
            if RISK_STATUS_PATH.exists():
                raw = json.loads(RISK_STATUS_PATH.read_text(encoding="utf-8"))
                self.risk_status = raw if isinstance(raw, dict) else {}
                log.info("Loaded risk status")
        except Exception as exc:
            log.warning("Failed to load risk status: %s", exc)

    # -- execution state ------------------------------------------------------
    def _load_execution_state(self) -> None:
        try:
            if EXECUTION_STATE_DIR.exists():
                # Look for execution output files
                candidates = list(EXECUTION_STATE_DIR.glob("execution_*.json"))
                if not candidates:
                    candidates = list(EXECUTION_STATE_DIR.glob("*_state.json"))
                if candidates:
                    latest = sorted(candidates, reverse=True)[0]
                    raw = json.loads(latest.read_text(encoding="utf-8"))
                    self.execution_state = raw if isinstance(raw, dict) else {}
                    log.info("Loaded execution state: %s", latest.name)
        except Exception as exc:
            log.warning("Failed to load execution state: %s", exc)

    # -- scout report ---------------------------------------------------------
    def _load_scout_report(self) -> None:
        try:
            if SCOUT_REPORT_PATH.exists():
                raw = json.loads(SCOUT_REPORT_PATH.read_text(encoding="utf-8"))
                self.scout_report = raw if isinstance(raw, dict) else {}
                log.info("Loaded scout report")
        except Exception as exc:
            log.warning("Failed to load scout report: %s", exc)

    # -- math proposals -------------------------------------------------------
    def _load_math_proposals(self) -> None:
        try:
            if MATH_PROPOSALS_DIR.exists():
                proposal_files = sorted(MATH_PROPOSALS_DIR.glob("*.json"), reverse=True)
                if proposal_files:
                    raw = json.loads(proposal_files[0].read_text(encoding="utf-8"))
                    self.math_proposals = raw if isinstance(raw, dict) else {}
                    log.info("Loaded math proposals: %s", proposal_files[0].name)
        except Exception as exc:
            log.warning("Failed to load math proposals: %s", exc)

    # -- previous decay status ------------------------------------------------
    def _load_previous_status(self) -> None:
        try:
            if DECAY_STATUS_PATH.exists():
                self.previous_status = json.loads(DECAY_STATUS_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to load previous status: %s", exc)

    # -- previous decay history -----------------------------------------------
    def _load_previous_history(self) -> None:
        try:
            if DECAY_HISTORY_PATH.exists():
                raw = json.loads(DECAY_HISTORY_PATH.read_text(encoding="utf-8"))
                self.previous_history = raw if isinstance(raw, list) else []
        except Exception as exc:
            log.warning("Failed to load previous history: %s", exc)

    # -- assemble per-strategy evidence profiles ------------------------------
    def assemble_evidence(self, strategy_name: str, strategy_data: Dict) -> StrategyEvidenceProfile:
        """
        Build a StrategyEvidenceProfile for a single strategy by gathering
        relevant data from all 8 sources.
        """
        profile = StrategyEvidenceProfile()

        # 1. Methodology evidence
        try:
            meth = self._extract_methodology_evidence(strategy_name, strategy_data)
            if meth:
                profile.methodology_evidence = meth
        except Exception as exc:
            log.debug("Methodology evidence extraction failed for %s: %s", strategy_name, exc)

        # 2. Optimizer evidence
        try:
            opt = self._extract_optimizer_evidence(strategy_name)
            if opt:
                profile.optimizer_evidence = opt
        except Exception as exc:
            log.debug("Optimizer evidence extraction failed for %s: %s", strategy_name, exc)

        # 3. Regime evidence
        try:
            reg = self._extract_regime_evidence(strategy_name, strategy_data)
            if reg:
                profile.regime_evidence = reg
        except Exception as exc:
            log.debug("Regime evidence extraction failed for %s: %s", strategy_name, exc)

        # 4. Portfolio evidence
        try:
            port = self._extract_portfolio_evidence(strategy_name)
            if port:
                profile.portfolio_evidence = port
        except Exception as exc:
            log.debug("Portfolio evidence extraction failed for %s: %s", strategy_name, exc)

        # 5. Risk evidence
        try:
            risk = self._extract_risk_evidence(strategy_name)
            if risk:
                profile.risk_evidence = risk
        except Exception as exc:
            log.debug("Risk evidence extraction failed for %s: %s", strategy_name, exc)

        # 6. Execution evidence
        try:
            exe = self._extract_execution_evidence(strategy_name)
            if exe:
                profile.execution_evidence = exe
        except Exception as exc:
            log.debug("Execution evidence extraction failed for %s: %s", strategy_name, exc)

        # 7. Scout evidence
        try:
            scout = self._extract_scout_evidence(strategy_name)
            if scout:
                profile.scout_evidence = scout
        except Exception as exc:
            log.debug("Scout evidence extraction failed for %s: %s", strategy_name, exc)

        # 8. Math evidence
        try:
            math_ev = self._extract_math_evidence(strategy_name)
            if math_ev:
                profile.math_evidence = math_ev
        except Exception as exc:
            log.debug("Math evidence extraction failed for %s: %s", strategy_name, exc)

        # 9. Price supplemental
        try:
            price_data = self._compute_price_supplemental(strategy_name)
            if price_data:
                profile.price_supplemental = price_data
        except Exception as exc:
            log.debug("Price supplemental failed for %s: %s", strategy_name, exc)

        # 10. Compute evidence gaps
        profile.evidence_gaps = self._compute_evidence_gaps(profile)

        return profile

    # -- per-source extraction helpers ----------------------------------------

    def _extract_methodology_evidence(self, name: str, data: Dict) -> Dict:
        """Extract methodology signals for a strategy."""
        evidence = {}
        # From the strategy data itself (which comes from methodology report)
        for key in ("sharpe", "net_sharpe", "ic_mean", "mean_ic", "hit_rate",
                     "win_rate", "walk_forward", "wf_stats", "n_folds", "fold_count",
                     "stability_score", "stability", "total_trades", "regime_performance",
                     "regimes", "gross_sharpe", "sharpe_gross", "max_drawdown", "max_dd",
                     "skewness", "return_skew", "crisis_performance", "crisis_sharpe",
                     "avg_correlation", "correlation_to_portfolio", "incremental_alpha",
                     "marginal_alpha", "turnover", "annual_turnover", "cost_drag", "cost_bps"):
            val = data.get(key)
            if val is not None:
                evidence[key] = val

        # From methodology report for this specific strategy
        if self.methodology_reports:
            strats = self.methodology_reports.get("strategies", {})
            if isinstance(strats, dict) and name in strats:
                strat_data = strats[name]
                if isinstance(strat_data, dict):
                    for k, v in strat_data.items():
                        if k not in evidence:
                            evidence[k] = v
            results = self.methodology_reports.get("results", {})
            if isinstance(results, dict) and name in results:
                res_data = results[name]
                if isinstance(res_data, dict):
                    for k, v in res_data.items():
                        if k not in evidence:
                            evidence[k] = v
        return evidence

    def _extract_optimizer_evidence(self, name: str) -> Dict:
        """Extract optimizer history for a strategy."""
        if not self.optimizer_history:
            return {}
        evidence = {}
        # Check direct strategy key
        strat_hist = self.optimizer_history.get(name, {})
        if isinstance(strat_hist, dict) and strat_hist:
            evidence = dict(strat_hist)
        # Check optimization_runs list
        runs = self.optimizer_history.get("optimization_runs", [])
        if isinstance(runs, list):
            strat_runs = [r for r in runs if isinstance(r, dict) and r.get("strategy") == name]
            if strat_runs:
                evidence["optimization_runs"] = strat_runs
                latest = strat_runs[-1]
                evidence["last_optimization"] = latest
                evidence["optimization_count"] = len(strat_runs)
                if latest.get("improvement"):
                    evidence["last_improvement"] = latest["improvement"]
        # Check if recently optimized
        history_entries = self.optimizer_history.get("history", [])
        if isinstance(history_entries, list):
            for entry in reversed(history_entries[-10:]):
                if isinstance(entry, dict) and entry.get("strategy") == name:
                    evidence["optimized_recently"] = True
                    evidence["last_optimizer_result"] = entry.get("result", "unknown")
                    break
        return evidence

    def _extract_regime_evidence(self, name: str, data: Dict) -> Dict:
        """Extract regime forecast and fitness data."""
        evidence = {}
        # From strategy data
        regimes = data.get("regime_performance", data.get("regimes", {}))
        if isinstance(regimes, dict) and regimes:
            evidence["regime_performance"] = regimes

        # From regime forecast agent
        if self.regime_forecast:
            current_regime = self.regime_forecast.get("current_regime",
                                                       self.regime_forecast.get("regime"))
            if current_regime:
                evidence["current_regime"] = current_regime
            probs = self.regime_forecast.get("regime_probabilities",
                                              self.regime_forecast.get("probabilities", {}))
            if probs:
                evidence["regime_probabilities"] = probs
            transitions = self.regime_forecast.get("transition_matrix",
                                                    self.regime_forecast.get("transitions", {}))
            if transitions:
                evidence["transition_matrix"] = transitions
            forecast = self.regime_forecast.get("forecast", self.regime_forecast.get("prediction", {}))
            if forecast:
                evidence["regime_forecast"] = forecast
            # Strategy-specific regime fitness
            fitness_map = self.regime_forecast.get("strategy_fitness", {})
            if isinstance(fitness_map, dict) and name in fitness_map:
                evidence["regime_fitness_map"] = fitness_map[name]
        return evidence

    def _extract_portfolio_evidence(self, name: str) -> Dict:
        """Extract portfolio construction evidence for a strategy."""
        if not self.portfolio_weights:
            return {}
        evidence = {}
        weights = self.portfolio_weights.get("weights", self.portfolio_weights.get("allocations", {}))
        if isinstance(weights, dict):
            if name in weights:
                evidence["current_weight"] = weights[name]
            evidence["total_strategies"] = len(weights)
            evidence["weight_distribution"] = {k: v for k, v in weights.items()
                                                 if isinstance(v, (int, float))}
        # Correlation data
        corr = self.portfolio_weights.get("correlation_matrix", {})
        if isinstance(corr, dict) and name in corr:
            evidence["correlation_to_portfolio"] = corr[name]
        # Marginal contribution
        marginal = self.portfolio_weights.get("marginal_contribution", {})
        if isinstance(marginal, dict) and name in marginal:
            evidence["marginal_contribution"] = marginal[name]
        return evidence

    def _extract_risk_evidence(self, name: str) -> Dict:
        """Extract risk guardian evidence for a strategy."""
        if not self.risk_status:
            return {}
        evidence = {}
        # Strategy-level risk
        strat_risk = self.risk_status.get("strategy_risk", {})
        if isinstance(strat_risk, dict) and name in strat_risk:
            evidence.update(strat_risk[name] if isinstance(strat_risk[name], dict) else {})
        # Global risk state
        for key in ("overall_risk_state", "vix_level", "drawdown_state",
                     "stress_test_results", "risk_budget_utilization",
                     "tail_risk_metrics", "concentration_risk"):
            val = self.risk_status.get(key)
            if val is not None:
                evidence[key] = val
        # Per-strategy risk metrics
        risk_metrics = self.risk_status.get("risk_metrics", {})
        if isinstance(risk_metrics, dict) and name in risk_metrics:
            evidence["risk_metrics"] = risk_metrics[name]
        return evidence

    def _extract_execution_evidence(self, name: str) -> Dict:
        """Extract execution agent evidence."""
        if not self.execution_state:
            return {}
        evidence = {}
        # Cost analysis
        costs = self.execution_state.get("cost_analysis", {})
        if isinstance(costs, dict):
            if name in costs:
                evidence["cost_analysis"] = costs[name]
            else:
                evidence["cost_analysis"] = costs
        # Slippage
        slippage = self.execution_state.get("slippage", {})
        if isinstance(slippage, dict):
            if name in slippage:
                evidence["slippage"] = slippage[name]
        # Execution quality
        quality = self.execution_state.get("execution_quality", {})
        if isinstance(quality, dict):
            if name in quality:
                evidence["execution_quality"] = quality[name]
        # Drag metrics
        drag = self.execution_state.get("execution_drag", self.execution_state.get("drag", {}))
        if isinstance(drag, dict):
            evidence["execution_drag"] = drag.get(name, drag)
        return evidence

    def _extract_scout_evidence(self, name: str) -> Dict:
        """Extract data scout evidence."""
        if not self.scout_report:
            return {}
        evidence = {}
        # Data quality
        quality = self.scout_report.get("data_quality", {})
        if isinstance(quality, dict):
            evidence["data_quality"] = quality
        # Anomalies
        anomalies = self.scout_report.get("anomalies", [])
        if isinstance(anomalies, list):
            strat_anomalies = [a for a in anomalies if isinstance(a, dict)
                               and a.get("strategy") == name]
            if strat_anomalies:
                evidence["anomalies"] = strat_anomalies
        # Market condition
        for key in ("market_condition", "volatility_state", "data_freshness",
                     "coverage_score", "missing_data"):
            val = self.scout_report.get(key)
            if val is not None:
                evidence[key] = val
        return evidence

    def _extract_math_evidence(self, name: str) -> Dict:
        """Extract math agent proposals relevant to a strategy."""
        if not self.math_proposals:
            return {}
        evidence = {}
        proposals = self.math_proposals.get("proposals", [])
        if isinstance(proposals, list):
            strat_proposals = [p for p in proposals if isinstance(p, dict)
                               and p.get("strategy") == name]
            if strat_proposals:
                evidence["proposals"] = strat_proposals
                evidence["proposal_count"] = len(strat_proposals)
        # Parameter suggestions
        params = self.math_proposals.get("parameter_suggestions", {})
        if isinstance(params, dict) and name in params:
            evidence["parameter_suggestions"] = params[name]
        return evidence

    def _compute_price_supplemental(self, name: str) -> Dict:
        """Compute rolling IC/Sharpe from raw prices for MR-like strategies."""
        prices = self.prices
        if prices is None or len(prices) < 123:
            return {}
        if "MR" not in name.upper() and "WHITELIST" not in name.upper():
            return {}
        try:
            spy = prices.get("SPY", pd.Series(dtype=float)).dropna()
            sectors = [s for s in MR_WHITELIST if s in prices.columns]
            if not sectors or spy.empty:
                return {}
            lookback = 63
            z_lookback = 60
            log_rel = np.log(prices[sectors].div(spy, axis=0))
            rets = log_rel.diff().dropna()
            z_scores = {}
            for s in sectors:
                z = (log_rel[s] - log_rel[s].rolling(z_lookback).mean()) / log_rel[s].rolling(z_lookback).std()
                z_scores[s] = z
            fwd_5d = rets.shift(-5).rolling(5).mean()
            n = len(rets)
            ic_list, sharpe_list = [], []
            step = 5
            for end in range(lookback + z_lookback, n - 5, step):
                start = end - lookback
                period_ics, period_rets = [], []
                for s in sectors:
                    z = z_scores[s].iloc[start:end].dropna()
                    fwd = fwd_5d[s].iloc[start:end].dropna()
                    common = z.index.intersection(fwd.index)
                    if len(common) >= 20:
                        ic_val = float(z.loc[common].corr(fwd.loc[common]))
                        if math.isfinite(ic_val):
                            period_ics.append(ic_val)
                    for i in range(max(0, len(z) - 5), len(z)):
                        idx = z.index[i] if i < len(z.index) else None
                        if idx is None:
                            continue
                        z_val = float(z.iloc[i]) if math.isfinite(float(z.iloc[i])) else 0
                        if abs(z_val) > 0.7 and idx in fwd.index:
                            direction = -1 if z_val > 0 else 1
                            ret = direction * float(fwd.loc[idx])
                            period_rets.append(ret)
                ic_list.append(float(np.mean(period_ics)) if period_ics else 0)
                if period_rets:
                    arr = np.array(period_rets)
                    sh = float(arr.mean() / arr.std() * np.sqrt(252 / 5)) if arr.std() > 1e-10 else 0
                else:
                    sh = 0
                sharpe_list.append(sh)
            return {
                "ic_current": ic_list[-1] if ic_list else 0,
                "ic_mean": float(np.mean(ic_list)) if ic_list else 0,
                "ic_std": float(np.std(ic_list)) if ic_list else 0,
                "sharpe_current": sharpe_list[-1] if sharpe_list else 0,
                "sharpe_mean": float(np.mean(sharpe_list)) if sharpe_list else 0,
                "n_periods": len(ic_list),
                "ic_trend": self._compute_trend(ic_list[-10:]) if len(ic_list) >= 5 else 0,
                "sharpe_trend": self._compute_trend(sharpe_list[-10:]) if len(sharpe_list) >= 5 else 0,
            }
        except Exception as exc:
            log.debug("Price supplemental failed for %s: %s", name, exc)
            return {}

    @staticmethod
    def _compute_trend(values: List[float]) -> float:
        """Simple linear trend: positive = improving, negative = declining."""
        if len(values) < 2:
            return 0.0
        try:
            x = np.arange(len(values), dtype=float)
            y = np.array(values, dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < 2:
                return 0.0
            slope = float(np.polyfit(x[mask], y[mask], 1)[0])
            return slope
        except Exception:
            return 0.0

    def _compute_evidence_gaps(self, profile: StrategyEvidenceProfile) -> List[str]:
        """Identify which evidence sources are missing."""
        gaps = []
        if not profile.methodology_evidence:
            gaps.append("methodology_report_missing")
        if not profile.optimizer_evidence:
            gaps.append("optimizer_history_missing")
        if not profile.regime_evidence:
            gaps.append("regime_forecast_missing")
        if not profile.portfolio_evidence:
            gaps.append("portfolio_weights_missing")
        if not profile.risk_evidence:
            gaps.append("risk_status_missing")
        if not profile.execution_evidence:
            gaps.append("execution_state_missing")
        if not profile.scout_evidence:
            gaps.append("scout_report_missing")
        if not profile.math_evidence:
            gaps.append("math_proposals_missing")
        if not profile.price_supplemental:
            gaps.append("price_supplemental_missing")
        return gaps

    # -- strategy extraction --------------------------------------------------
    def extract_strategies(self) -> Dict[str, Dict]:
        """
        Extract strategy list from methodology report.
        Falls back to a single MR whitelist strategy from prices.
        """
        strategies: Dict[str, Dict] = {}

        report = self.methodology_reports
        if report:
            strats = report.get("strategies", {})
            if isinstance(strats, dict):
                strategies.update(strats)
            results = report.get("results", {})
            if isinstance(results, dict):
                for key, val in results.items():
                    if isinstance(val, dict) and key not in strategies:
                        strategies[key] = val
            rankings = report.get("strategy_rankings", report.get("rankings", []))
            if isinstance(rankings, list):
                for item in rankings:
                    if isinstance(item, dict) and "name" in item:
                        name = item["name"]
                        if name not in strategies:
                            strategies[name] = item

        if not strategies and self.prices is not None:
            strategies["ALPHA_WHITELIST_MR"] = {
                "sharpe": 0,
                "hit_rate": 0.5,
                "_source": "price_fallback",
            }

        return strategies

    # Backward compat aliases
    @property
    def methodology_report(self):
        return self.methodology_reports


# Legacy alias
StrategyEvidenceLoader = StrategyEvidenceAssembler


# ============================================================================
# 2. EvidenceQualityEngine
# ============================================================================
class EvidenceQualityEngine:
    """
    Score and classify evidence quality for a strategy.
    Determines whether the system has enough information for reliable assessment.
    """

    def score(self, profile: StrategyEvidenceProfile) -> StrategyEvidenceProfile:
        """Compute evidence quality score and classification, mutate profile."""
        # Weighted source availability
        availability_score = 0.0
        source_map = {
            "methodology": bool(profile.methodology_evidence),
            "optimizer": bool(profile.optimizer_evidence),
            "regime": bool(profile.regime_evidence),
            "portfolio": bool(profile.portfolio_evidence),
            "risk": bool(profile.risk_evidence),
            "execution": bool(profile.execution_evidence),
            "scout": bool(profile.scout_evidence),
            "math": bool(profile.math_evidence),
        }
        for source, available in source_map.items():
            if available:
                availability_score += EVIDENCE_SOURCE_WEIGHTS.get(source, 0.125)

        # Evidence completeness: check key fields within available sources
        completeness = self._score_completeness(profile)

        # Evidence recency: check timestamps
        recency = self._score_recency(profile)

        # Evidence conflicts: check contradictory signals
        conflicts = self._score_conflicts(profile)

        # Composite quality score
        quality_score = (
            availability_score * 0.40
            + completeness * 0.25
            + recency * 0.20
            + (1.0 - conflicts) * 0.15
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        # Classify tier
        n_sources = profile.available_source_count()
        if n_sources >= 7:
            tier = EVIDENCE_QUALITY_FULL
        elif n_sources >= 4:
            tier = EVIDENCE_QUALITY_PARTIAL
        elif n_sources >= 2:
            tier = EVIDENCE_QUALITY_DEGRADED
        else:
            tier = EVIDENCE_QUALITY_INSUFFICIENT

        # INSUFFICIENT if < 3 sources AND no price data
        if n_sources < MIN_SOURCES_WITHOUT_PRICE and not profile.has_price_data():
            tier = EVIDENCE_QUALITY_INSUFFICIENT

        profile.evidence_quality = tier
        profile.evidence_quality_score = quality_score
        profile.degraded_mode = tier in (EVIDENCE_QUALITY_DEGRADED, EVIDENCE_QUALITY_INSUFFICIENT)

        return profile

    def _score_completeness(self, profile: StrategyEvidenceProfile) -> float:
        """Score how complete the available evidence is (key fields present)."""
        expected_fields = {
            "methodology": ["sharpe", "net_sharpe", "ic_mean", "mean_ic", "hit_rate",
                            "win_rate", "walk_forward", "wf_stats"],
            "regime": ["regime_performance", "current_regime"],
            "risk": ["overall_risk_state", "tail_risk_metrics"],
            "optimizer": ["optimization_count", "last_improvement"],
        }
        found = 0
        total = 0
        for source, fields in expected_fields.items():
            source_data = getattr(profile, f"{source}_evidence", {})
            for f in fields:
                total += 1
                if f in source_data:
                    found += 1
        return found / max(total, 1)

    def _score_recency(self, profile: StrategyEvidenceProfile) -> float:
        """Score evidence freshness based on timestamps."""
        now = datetime.now(timezone.utc)
        timestamps = []
        for source_name in ("methodology_evidence", "optimizer_evidence", "regime_evidence",
                            "risk_evidence", "execution_evidence", "scout_evidence"):
            source_data = getattr(profile, source_name, {})
            ts = source_data.get("timestamp", source_data.get("updated_at", source_data.get("ts")))
            if ts:
                try:
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        age_hours = (now - dt).total_seconds() / 3600
                        timestamps.append(age_hours)
                except Exception:
                    pass
        if not timestamps:
            return 0.5  # Unknown recency -- neutral
        # Average age in hours, map to 0-1 (0h=1.0, 72h=0.0)
        avg_age = statistics.mean(timestamps)
        return float(np.clip(1.0 - avg_age / 72.0, 0.0, 1.0))

    def _score_conflicts(self, profile: StrategyEvidenceProfile) -> float:
        """Detect contradictory signals across sources."""
        conflict_score = 0.0
        conflict_count = 0

        # Check methodology vs risk: if methodology says good Sharpe but risk says high drawdown
        meth = profile.methodology_evidence
        risk = profile.risk_evidence
        if meth and risk:
            meth_sharpe = meth.get("sharpe", meth.get("net_sharpe"))
            risk_state = risk.get("overall_risk_state", "")
            if isinstance(meth_sharpe, (int, float)) and meth_sharpe > 1.0:
                if isinstance(risk_state, str) and risk_state.upper() in ("CRITICAL", "HIGH_RISK", "STRESSED"):
                    conflict_score += 0.3
                    conflict_count += 1

        # Check optimizer says improved but methodology shows decline
        opt = profile.optimizer_evidence
        if opt and meth:
            last_improvement = opt.get("last_improvement")
            meth_sharpe = meth.get("sharpe", meth.get("net_sharpe", 0))
            if isinstance(last_improvement, (int, float)) and last_improvement > 0:
                if isinstance(meth_sharpe, (int, float)) and meth_sharpe < 0:
                    conflict_score += 0.3
                    conflict_count += 1

        # Check regime says favorable but strategy performance is poor
        reg = profile.regime_evidence
        if reg and meth:
            regime_perf = reg.get("regime_performance", {})
            if isinstance(regime_perf, dict):
                sharpes = [v.get("sharpe", 0) if isinstance(v, dict) else v
                           for v in regime_perf.values()
                           if isinstance(v, (dict, int, float))]
                sharpes = [s for s in sharpes if isinstance(s, (int, float)) and math.isfinite(s)]
                if sharpes and statistics.mean(sharpes) > 0.5:
                    overall_sharpe = meth.get("sharpe", meth.get("net_sharpe", 0))
                    if isinstance(overall_sharpe, (int, float)) and overall_sharpe < -0.5:
                        conflict_score += 0.2
                        conflict_count += 1

        return float(np.clip(conflict_score, 0.0, 1.0))


# ============================================================================
# 3. HealthDimensionEngine -- 7 dimensions, each 0-1
# ============================================================================
class HealthDimensionEngine:
    """
    Compute 7 institutional health integrity dimensions, each in [0, 1].
    Higher = healthier. Uses full evidence profile from all agents.
    """

    def compute_all(self, profile: StrategyEvidenceProfile,
                    promotion_data: Optional[Dict] = None) -> Dict[str, float]:
        """Return dict of all 7 dimensions."""
        promotion_data = promotion_data or {}
        return {
            "alpha_integrity": self.alpha_integrity(profile),
            "robustness_integrity": self.robustness_integrity(profile),
            "regime_integrity": self.regime_integrity(profile),
            "implementation_integrity": self.implementation_integrity(profile),
            "tail_integrity": self.tail_integrity(profile),
            "diversification_integrity": self.diversification_integrity(profile),
            "lifecycle_integrity": self.lifecycle_integrity(profile, promotion_data),
        }

    def alpha_integrity(self, profile: StrategyEvidenceProfile) -> float:
        """Score from methodology Sharpe/IC/WR and decay trends."""
        scores: List[float] = []
        meth = profile.methodology_evidence

        # Sharpe contribution: 0-2 range mapped to 0-1
        sharpe = meth.get("sharpe", meth.get("net_sharpe"))
        if isinstance(sharpe, (int, float)) and math.isfinite(sharpe):
            scores.append(float(np.clip(sharpe / 2.0, 0.0, 1.0)))

        # IC contribution: 0-0.1 mapped to 0-1
        ic = meth.get("ic_mean", meth.get("mean_ic"))
        if isinstance(ic, (int, float)) and math.isfinite(ic):
            scores.append(float(np.clip(ic / 0.1, 0.0, 1.0)))

        # Hit rate: 0.5 -> 0.5 score, 0.7 -> 1.0
        hr = meth.get("hit_rate", meth.get("win_rate"))
        if isinstance(hr, (int, float)) and math.isfinite(hr):
            scores.append(float(np.clip((hr - 0.3) / 0.4, 0.0, 1.0)))

        # Price supplemental IC and Sharpe
        price = profile.price_supplemental
        if price:
            ic_cur = price.get("ic_current", 0)
            sharpe_cur = price.get("sharpe_current", 0)
            if isinstance(ic_cur, (int, float)) and math.isfinite(ic_cur):
                scores.append(float(np.clip(ic_cur / 0.05 * 0.5 + 0.5, 0.0, 1.0)))
            if isinstance(sharpe_cur, (int, float)) and math.isfinite(sharpe_cur):
                scores.append(float(np.clip(sharpe_cur / 2.0, 0.0, 1.0)))
            # IC trend penalty/boost
            ic_trend = price.get("ic_trend", 0)
            if isinstance(ic_trend, (int, float)) and math.isfinite(ic_trend):
                # Negative trend penalizes, positive boosts
                trend_factor = float(np.clip(0.5 + ic_trend * 5.0, 0.2, 0.8))
                scores.append(trend_factor)

        # Optimizer repair signal: if recently optimized and improved, slight boost
        opt = profile.optimizer_evidence
        if opt:
            improvement = opt.get("last_improvement")
            if isinstance(improvement, (int, float)) and improvement > 0:
                scores.append(min(0.7, 0.5 + improvement))

        return float(np.mean(scores)) if scores else 0.5

    def robustness_integrity(self, profile: StrategyEvidenceProfile) -> float:
        """Score from methodology walk-forward, fold consistency, stability."""
        scores: List[float] = []
        meth = profile.methodology_evidence

        # Walk-forward stats
        wf = meth.get("walk_forward", meth.get("wf_stats", {}))
        if isinstance(wf, dict):
            wf_sharpe = wf.get("mean_sharpe", wf.get("sharpe", 0))
            wf_std = wf.get("std_sharpe", wf.get("sharpe_std", 1))
            if isinstance(wf_sharpe, (int, float)) and math.isfinite(wf_sharpe):
                scores.append(float(np.clip(wf_sharpe / 1.5, 0.0, 1.0)))
            if isinstance(wf_std, (int, float)) and wf_std > 0 and math.isfinite(wf_std):
                scores.append(float(np.clip(1.0 - wf_std / 1.0, 0.0, 1.0)))

        # Fold count
        n_folds = meth.get("n_folds", meth.get("fold_count", 0))
        if isinstance(n_folds, (int, float)):
            scores.append(float(np.clip(n_folds / 5.0, 0.0, 1.0)))

        # Stability metric
        stability = meth.get("stability_score", meth.get("stability"))
        if isinstance(stability, (int, float)) and math.isfinite(stability):
            scores.append(float(np.clip(stability, 0.0, 1.0)))

        # Total trades sufficiency
        trades = meth.get("total_trades", 0)
        if isinstance(trades, (int, float)):
            scores.append(float(np.clip(trades / 100.0, 0.0, 1.0)))

        # Math proposals: if math agent suggested parameter changes, slight concern
        math_ev = profile.math_evidence
        if math_ev and math_ev.get("proposal_count", 0) > 2:
            scores.append(0.4)  # Penalty for too many math suggestions

        return float(np.mean(scores)) if scores else 0.5

    def regime_integrity(self, profile: StrategyEvidenceProfile) -> float:
        """Score from regime fitness map, per-regime Sharpe, regime breadth."""
        scores: List[float] = []
        reg = profile.regime_evidence
        meth = profile.methodology_evidence

        # Per-regime Sharpe from methodology or regime evidence
        regimes = reg.get("regime_performance", meth.get("regime_performance",
                          meth.get("regimes", {})))
        if isinstance(regimes, dict) and regimes:
            regime_sharpes = []
            for rname, rdata in regimes.items():
                rs = rdata if isinstance(rdata, (int, float)) else (
                    rdata.get("sharpe", 0) if isinstance(rdata, dict) else 0
                )
                if isinstance(rs, (int, float)) and math.isfinite(rs):
                    regime_sharpes.append(rs)
            if regime_sharpes:
                mean_rs = float(np.mean(regime_sharpes))
                scores.append(float(np.clip(mean_rs / 1.5, 0.0, 1.0)))
                # Breadth: fraction of regimes with positive sharpe
                pos_frac = sum(1 for s in regime_sharpes if s > 0) / len(regime_sharpes)
                scores.append(pos_frac)
                # Worst regime penalty
                worst = min(regime_sharpes)
                scores.append(float(np.clip((worst + 0.5) / 1.0, 0.0, 1.0)))

        # Regime fitness map from regime forecaster
        fitness_map = reg.get("regime_fitness_map", {})
        if isinstance(fitness_map, dict):
            fitness_vals = [v for v in fitness_map.values()
                           if isinstance(v, (int, float)) and math.isfinite(v)]
            if fitness_vals:
                scores.append(float(np.clip(np.mean(fitness_vals), 0.0, 1.0)))

        # Current regime alignment
        current_regime = reg.get("current_regime")
        if current_regime and isinstance(regimes, dict):
            current_data = regimes.get(current_regime, regimes.get(str(current_regime).upper(), {}))
            if isinstance(current_data, dict):
                cur_sharpe = current_data.get("sharpe", 0)
                if isinstance(cur_sharpe, (int, float)) and math.isfinite(cur_sharpe):
                    scores.append(float(np.clip(cur_sharpe / 1.0, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    def implementation_integrity(self, profile: StrategyEvidenceProfile) -> float:
        """Score from cost analysis, execution drag, net vs gross."""
        scores: List[float] = []
        meth = profile.methodology_evidence
        exe = profile.execution_evidence

        # Gross vs net Sharpe gap
        gross_sharpe = meth.get("gross_sharpe", meth.get("sharpe_gross"))
        net_sharpe = meth.get("net_sharpe", meth.get("sharpe"))
        if (isinstance(gross_sharpe, (int, float)) and isinstance(net_sharpe, (int, float))
                and math.isfinite(gross_sharpe) and math.isfinite(net_sharpe)):
            gap = gross_sharpe - net_sharpe
            scores.append(float(np.clip(1.0 - gap / 0.5, 0.0, 1.0)))

        # Turnover
        turnover = meth.get("turnover", meth.get("annual_turnover"))
        if isinstance(turnover, (int, float)) and math.isfinite(turnover):
            scores.append(float(np.clip(1.0 - turnover / 200.0, 0.0, 1.0)))

        # Cost drag
        cost_drag = meth.get("cost_drag", meth.get("cost_bps"))
        if isinstance(cost_drag, (int, float)) and math.isfinite(cost_drag):
            scores.append(float(np.clip(1.0 - cost_drag / 100.0, 0.0, 1.0)))

        # Execution evidence: slippage and quality
        if exe:
            slippage = exe.get("slippage", {})
            if isinstance(slippage, dict):
                avg_slippage = slippage.get("avg_slippage_bps", slippage.get("mean"))
                if isinstance(avg_slippage, (int, float)) and math.isfinite(avg_slippage):
                    scores.append(float(np.clip(1.0 - abs(avg_slippage) / 50.0, 0.0, 1.0)))

            eq = exe.get("execution_quality", {})
            if isinstance(eq, dict):
                fill_rate = eq.get("fill_rate", eq.get("execution_rate"))
                if isinstance(fill_rate, (int, float)) and math.isfinite(fill_rate):
                    scores.append(float(np.clip(fill_rate, 0.0, 1.0)))

            drag = exe.get("execution_drag", {})
            if isinstance(drag, dict):
                drag_bps = drag.get("drag_bps", drag.get("total_drag"))
                if isinstance(drag_bps, (int, float)) and math.isfinite(drag_bps):
                    scores.append(float(np.clip(1.0 - abs(drag_bps) / 80.0, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.7  # Default generous

    def tail_integrity(self, profile: StrategyEvidenceProfile) -> float:
        """Score from drawdown, stress test, tail metrics."""
        scores: List[float] = []
        meth = profile.methodology_evidence
        risk = profile.risk_evidence

        # Max drawdown
        max_dd = meth.get("max_drawdown", meth.get("max_dd"))
        if isinstance(max_dd, (int, float)) and math.isfinite(max_dd):
            dd_abs = abs(max_dd)
            scores.append(float(np.clip(1.0 - dd_abs / 0.30, 0.0, 1.0)))

        # Skewness
        skew = meth.get("skewness", meth.get("return_skew"))
        if isinstance(skew, (int, float)) and math.isfinite(skew):
            scores.append(float(np.clip((skew + 1.0) / 2.0, 0.0, 1.0)))

        # Crisis performance
        crisis = meth.get("crisis_performance", meth.get("crisis_sharpe"))
        if isinstance(crisis, (int, float)) and math.isfinite(crisis):
            scores.append(float(np.clip((crisis + 0.5) / 1.5, 0.0, 1.0)))

        # Risk guardian: tail metrics
        if risk:
            tail_metrics = risk.get("tail_risk_metrics", {})
            if isinstance(tail_metrics, dict):
                cvar = tail_metrics.get("cvar_95", tail_metrics.get("expected_shortfall"))
                if isinstance(cvar, (int, float)) and math.isfinite(cvar):
                    scores.append(float(np.clip(1.0 - abs(cvar) / 0.10, 0.0, 1.0)))
                var_val = tail_metrics.get("var_99", tail_metrics.get("value_at_risk"))
                if isinstance(var_val, (int, float)) and math.isfinite(var_val):
                    scores.append(float(np.clip(1.0 - abs(var_val) / 0.15, 0.0, 1.0)))

            # Stress test results
            stress = risk.get("stress_test_results", {})
            if isinstance(stress, dict):
                worst_scenario = min(
                    (v for v in stress.values() if isinstance(v, (int, float)) and math.isfinite(v)),
                    default=None
                )
                if worst_scenario is not None:
                    scores.append(float(np.clip((worst_scenario + 0.2) / 0.4, 0.0, 1.0)))

            # Drawdown state
            dd_state = risk.get("drawdown_state", "")
            if isinstance(dd_state, str):
                dd_state_upper = dd_state.upper()
                if "CRITICAL" in dd_state_upper or "SEVERE" in dd_state_upper:
                    scores.append(0.1)
                elif "WARNING" in dd_state_upper or "ELEVATED" in dd_state_upper:
                    scores.append(0.4)
                elif "NORMAL" in dd_state_upper or "LOW" in dd_state_upper:
                    scores.append(0.8)

        return float(np.mean(scores)) if scores else 0.5

    def diversification_integrity(self, profile: StrategyEvidenceProfile) -> float:
        """Score from correlation to peers, incremental alpha, portfolio role."""
        scores: List[float] = []
        meth = profile.methodology_evidence
        port = profile.portfolio_evidence

        # Correlation from methodology
        corr = meth.get("avg_correlation", meth.get("correlation_to_portfolio"))
        if isinstance(corr, (int, float)) and math.isfinite(corr):
            scores.append(float(np.clip(1.0 - abs(corr), 0.0, 1.0)))

        # Incremental alpha
        incr = meth.get("incremental_alpha", meth.get("marginal_alpha"))
        if isinstance(incr, (int, float)) and math.isfinite(incr):
            scores.append(float(np.clip(incr / 0.05, 0.0, 1.0)))

        # Portfolio evidence: correlation and marginal contribution
        if port:
            port_corr = port.get("correlation_to_portfolio")
            if isinstance(port_corr, (int, float)) and math.isfinite(port_corr):
                scores.append(float(np.clip(1.0 - abs(port_corr), 0.0, 1.0)))
            marginal = port.get("marginal_contribution")
            if isinstance(marginal, (int, float)) and math.isfinite(marginal):
                scores.append(float(np.clip(marginal / 0.05 + 0.5, 0.0, 1.0)))
            # Weight relative to total
            weight = port.get("current_weight")
            total = port.get("total_strategies", 1)
            if isinstance(weight, (int, float)) and isinstance(total, int) and total > 0:
                expected_weight = 1.0 / total
                if expected_weight > 0:
                    weight_ratio = weight / expected_weight
                    # Extreme over/under-weighting is bad for diversification
                    scores.append(float(np.clip(1.0 - abs(1.0 - weight_ratio) * 0.5, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    def lifecycle_integrity(self, profile: StrategyEvidenceProfile,
                            promotion_data: Dict) -> float:
        """Score from promotion history, rollback count, repair attempts."""
        scores: List[float] = []

        # Rollback count: 0 = 1.0, 1 = 0.7, 2 = 0.4, 3+ = 0.1
        rollbacks = promotion_data.get("rollback_count", 0)
        if isinstance(rollbacks, (int, float)):
            scores.append(float(np.clip(1.0 - rollbacks * 0.3, 0.1, 1.0)))

        # Repair attempts: too many indicates chronic issues
        repairs = promotion_data.get("repair_attempts", 0)
        if isinstance(repairs, (int, float)):
            scores.append(float(np.clip(1.0 - repairs * 0.15, 0.1, 1.0)))

        # Post-promotion stability
        post_promo_decline = promotion_data.get("post_promotion_decline", False)
        if post_promo_decline:
            scores.append(0.3)
        else:
            scores.append(0.8)

        # Shadow test success rate
        shadow_successes = promotion_data.get("shadow_successes", 0)
        shadow_failures = promotion_data.get("shadow_failures", 0)
        total_shadow = shadow_successes + shadow_failures
        if total_shadow > 0:
            scores.append(shadow_successes / total_shadow)

        # Optimizer repair success
        opt = profile.optimizer_evidence
        if opt:
            opt_runs = opt.get("optimization_runs", [])
            if isinstance(opt_runs, list) and opt_runs:
                successful = sum(1 for r in opt_runs if isinstance(r, dict)
                                 and r.get("result") in ("improved", "success"))
                scores.append(float(np.clip(successful / len(opt_runs), 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.6


# ============================================================================
# 4. ChangePointEngine
# ============================================================================
class ChangePointEngine:
    """
    Detect abrupt degradation, persistent weakening, and re-stabilization
    in strategy health timelines.
    """

    def detect(self, health_timeline: List[float]) -> ChangePointResult:
        """
        Analyze a strategy's health score timeline for change points.
        health_timeline: list of health scores (oldest first).
        """
        result = ChangePointResult()

        if len(health_timeline) < 3:
            return result

        latest = health_timeline[-1]
        n = len(health_timeline)

        # 1. Abrupt degradation: z-score of latest vs lookback
        lookback = min(CHANGEPOINT_LOOKBACK, n - 1)
        history = health_timeline[-lookback - 1:-1]
        if len(history) >= 3:
            mean_h = statistics.mean(history)
            std_h = statistics.stdev(history) if len(history) > 1 else 0.01
            std_h = max(std_h, 0.01)
            z = (latest - mean_h) / std_h
            result.z_score_latest = z
            if z < -CHANGEPOINT_Z_THRESHOLD:
                result.abrupt_degradation = True

        # 2. Persistent weakening: 3+ consecutive declining
        consecutive_decline = 0
        for i in range(n - 1, 0, -1):
            if health_timeline[i] < health_timeline[i - 1] - 0.005:
                consecutive_decline += 1
            else:
                break
        result.consecutive_declining = consecutive_decline
        if consecutive_decline >= CONSECUTIVE_DECLINE_THRESHOLD:
            result.persistent_weakening = True

        # 3. Re-stabilization: 3+ consecutive improving after decline
        # Find trough (minimum in recent history)
        trough_idx = -1
        min_val = float("inf")
        for i in range(max(0, n - 20), n):
            if health_timeline[i] < min_val:
                min_val = health_timeline[i]
                trough_idx = i
        result.trough_index = trough_idx

        consecutive_improve = 0
        if trough_idx >= 0 and trough_idx < n - 1:
            for i in range(trough_idx + 1, n):
                if health_timeline[i] > health_timeline[i - 1] + 0.005:
                    consecutive_improve += 1
                else:
                    consecutive_improve = 0
        result.consecutive_improving = consecutive_improve
        if consecutive_improve >= CONSECUTIVE_IMPROVE_THRESHOLD:
            result.re_stabilization = True

        # 4. Deterioration persistence score
        total_periods = max(n - 1, 1)
        declining_periods = sum(
            1 for i in range(1, n)
            if health_timeline[i] < health_timeline[i - 1] - 0.005
        )
        result.deterioration_persistence_score = declining_periods / total_periods

        # 5. Recovery persistence score (improving periods after trough)
        if trough_idx >= 0 and trough_idx < n - 1:
            post_trough_periods = n - trough_idx - 1
            improving_after_trough = sum(
                1 for i in range(trough_idx + 1, n)
                if health_timeline[i] > health_timeline[i - 1] + 0.005
            )
            result.recovery_persistence_score = (
                improving_after_trough / max(post_trough_periods, 1)
            )

        # 6. Trend direction
        if consecutive_decline >= 2:
            result.trend_direction = "DECLINING"
        elif consecutive_improve >= 2:
            result.trend_direction = "IMPROVING"
        else:
            result.trend_direction = "STABLE"

        return result


# ============================================================================
# 5. LifecycleClassifier -- 12 states with STRICT rules
# ============================================================================
class LifecycleClassifier:
    """
    Classify strategy into one of 12 health states using strict,
    deterministic rules. Order matters -- first match wins.
    """

    def classify(
        self,
        health_score: float,
        dimensions: Dict[str, float],
        evidence_quality: str,
        evidence_quality_score: float,
        change_point: ChangePointResult,
        lifecycle_stage: str,
        consecutive_below_threshold: int,
        consecutive_declining: int,
    ) -> str:
        """
        Classify into one of 12 health states.
        CRITICAL: INSUFFICIENT_EVIDENCE NEVER maps to HEALTHY.
        """
        # Rule 0: Insufficient evidence
        if evidence_quality == EVIDENCE_QUALITY_INSUFFICIENT:
            return "INSUFFICIENT_EVIDENCE"

        # Rule 1: DEAD -- health < 0.15 for 5+ periods OR explicit retirement
        if health_score < 0.15 and consecutive_below_threshold >= 5:
            return "DEAD"

        # Rule 2: RETIREMENT_WATCH -- health < 0.25 for 3+ periods
        if health_score < 0.25 and consecutive_below_threshold >= 3:
            return "RETIREMENT_WATCH"

        # Rule 3: POST_PROMOTION_FAILURE -- recently promoted and declining 3+ periods
        if lifecycle_stage == "RECENTLY_PROMOTED" and consecutive_declining >= 3:
            return "POST_PROMOTION_FAILURE"

        # Rule 4: SHADOW_FAILURE -- shadow candidate with health < 0.40
        if lifecycle_stage == "SHADOW_CANDIDATE" and health_score < 0.40:
            return "SHADOW_FAILURE"

        # Rule 5: STRUCTURAL_DECAY -- 3+ dimensions < 0.35
        dim_vals = list(dimensions.values())
        failing_count = sum(1 for v in dim_vals if v < 0.35)
        if failing_count >= 3:
            return "STRUCTURAL_DECAY"

        # Rule 6: ROBUSTNESS_BREAKDOWN -- robustness < 0.30 and not broadly structural
        robustness = dimensions.get("robustness_integrity", 0.5)
        alpha = dimensions.get("alpha_integrity", 0.5)
        if robustness < 0.30 and failing_count < 3:
            return "ROBUSTNESS_BREAKDOWN"

        # Rule 7: IMPLEMENTATION_DECAY -- implementation < 0.30 and alpha > 0.45
        implementation = dimensions.get("implementation_integrity", 0.5)
        if implementation < 0.30 and alpha > 0.45:
            return "IMPLEMENTATION_DECAY"

        # Rule 8: REGIME_SUPPRESSED -- regime < 0.35 and alpha > 0.50 and evidence shows other regimes OK
        regime = dimensions.get("regime_integrity", 0.5)
        if regime < 0.35 and alpha > 0.50:
            # Check if at least some non-regime dimensions are acceptable
            non_regime_ok = sum(1 for k, v in dimensions.items()
                                if k != "regime_integrity" and v > 0.45)
            if non_regime_ok >= 3:
                return "REGIME_SUPPRESSED"

        # Rule 9: EARLY_DECAY -- health 0.35-0.50 and persistence declining
        if 0.35 <= health_score < 0.50 and change_point.persistent_weakening:
            return "EARLY_DECAY"

        # Rule 10: EARLY_DECAY (alternative) -- health < 0.50 with declining trend
        if health_score < 0.50 and change_point.trend_direction == "DECLINING":
            return "EARLY_DECAY"

        # Rule 11: WATCH -- health 0.50-0.65 OR 1-2 dimensions declining
        declining_dims = sum(1 for v in dim_vals if v < 0.50)
        if 0.50 <= health_score < 0.65:
            return "WATCH"
        if declining_dims >= 1 and declining_dims <= 2 and health_score < 0.70:
            return "WATCH"

        # Rule 12: HEALTHY -- health > 0.65 AND evidence >= PARTIAL AND no dimension < 0.4
        no_dim_below_threshold = all(v >= 0.4 for v in dim_vals)
        if (health_score > 0.65
                and evidence_quality in (EVIDENCE_QUALITY_FULL, EVIDENCE_QUALITY_PARTIAL)
                and no_dim_below_threshold):
            return "HEALTHY"

        # Default fallback: if nothing else matched, WATCH
        if health_score >= 0.50:
            return "WATCH"
        return "EARLY_DECAY"


# ============================================================================
# 6. HazardModel -- NOT just 1-health
# ============================================================================
class HazardModel:
    """
    Compute proper hazard scores: 30d, 90d, retirement, and promotion instability.
    These are probability-like measures of failure within the time horizon.
    """

    def compute(
        self,
        health_score: float,
        dimensions: Dict[str, float],
        change_point: ChangePointResult,
        evidence_quality_score: float,
        lifecycle_stage: str,
        consecutive_below_threshold: int,
        rollback_count: int,
    ) -> Dict[str, float]:
        """Compute all hazard measures."""
        return {
            "hazard_30d": self._hazard_30d(
                health_score, dimensions, change_point, evidence_quality_score, lifecycle_stage
            ),
            "hazard_90d": self._hazard_90d(
                health_score, dimensions, change_point, evidence_quality_score, lifecycle_stage
            ),
            "retirement_hazard": self._retirement_hazard(
                health_score, consecutive_below_threshold, lifecycle_stage
            ),
            "promotion_instability": self._promotion_instability(
                health_score, lifecycle_stage, change_point, rollback_count
            ),
        }

    def _hazard_30d(
        self,
        health_score: float,
        dimensions: Dict[str, float],
        change_point: ChangePointResult,
        evidence_quality_score: float,
        lifecycle_stage: str,
    ) -> float:
        """
        hazard_30d = f(deterioration_breadth, persistence, evidence_strength, lifecycle_risk)
        """
        # Base hazard from health score inversion
        base = 1.0 - health_score

        # Deterioration breadth: how many dimensions are weak
        dim_vals = list(dimensions.values())
        weak_count = sum(1 for v in dim_vals if v < 0.40)
        breadth_factor = weak_count / max(len(dim_vals), 1)

        # Persistence: from change-point engine
        persistence_factor = change_point.deterioration_persistence_score

        # Evidence strength: lower quality = higher uncertainty = higher hazard
        evidence_factor = 1.0 - evidence_quality_score

        # Lifecycle risk: recently promoted = higher risk
        lifecycle_factor = 0.0
        if lifecycle_stage == "RECENTLY_PROMOTED":
            lifecycle_factor = 0.15
        elif lifecycle_stage == "SHADOW_CANDIDATE":
            lifecycle_factor = 0.05
        elif lifecycle_stage == "RETIREMENT_WATCH":
            lifecycle_factor = 0.25

        # Abrupt degradation spike
        abrupt_factor = 0.20 if change_point.abrupt_degradation else 0.0

        hazard = (
            base * 0.35
            + breadth_factor * 0.20
            + persistence_factor * 0.15
            + evidence_factor * 0.10
            + lifecycle_factor
            + abrupt_factor
        )
        return float(np.clip(hazard, 0.0, 1.0))

    def _hazard_90d(
        self,
        health_score: float,
        dimensions: Dict[str, float],
        change_point: ChangePointResult,
        evidence_quality_score: float,
        lifecycle_stage: str,
    ) -> float:
        """hazard_90d = hazard_30d * decay_factor + structural_risk"""
        h30 = self._hazard_30d(health_score, dimensions, change_point,
                                evidence_quality_score, lifecycle_stage)

        # Structural risk: multiple weak dimensions compound over time
        dim_vals = list(dimensions.values())
        structural_risk = 0.0
        very_weak = sum(1 for v in dim_vals if v < 0.30)
        if very_weak >= 2:
            structural_risk = 0.15
        elif very_weak >= 1:
            structural_risk = 0.05

        hazard_90 = h30 * HAZARD_DECAY_FACTOR + structural_risk
        return float(np.clip(hazard_90, 0.0, 1.0))

    def _retirement_hazard(
        self,
        health_score: float,
        consecutive_below_threshold: int,
        lifecycle_stage: str,
    ) -> float:
        """retirement_hazard = f(consecutive_below_threshold, lifecycle_stage)"""
        # Base from consecutive periods below threshold
        if consecutive_below_threshold >= 5:
            base = 0.95
        elif consecutive_below_threshold >= 3:
            base = 0.70
        elif consecutive_below_threshold >= 2:
            base = 0.40
        elif consecutive_below_threshold >= 1:
            base = 0.20
        else:
            base = 0.05

        # Lifecycle stage adjustment
        stage_factor = 0.0
        if lifecycle_stage == "RETIREMENT_WATCH":
            stage_factor = 0.15
        elif lifecycle_stage == "RETIRED":
            stage_factor = 0.50

        # Health score: very low health increases retirement probability
        health_factor = max(0, (0.25 - health_score) * 2.0)

        hazard = base + stage_factor + health_factor
        return float(np.clip(hazard, 0.0, 1.0))

    def _promotion_instability(
        self,
        health_score: float,
        lifecycle_stage: str,
        change_point: ChangePointResult,
        rollback_count: int,
    ) -> float:
        """promotion_instability = f(post_promotion_decline, rollback_count)"""
        instability = 0.0

        if lifecycle_stage == "RECENTLY_PROMOTED":
            # Declining after promotion
            if change_point.persistent_weakening:
                instability += 0.30
            if change_point.abrupt_degradation:
                instability += 0.25

        # Rollback history
        if rollback_count >= 3:
            instability += 0.35
        elif rollback_count >= 2:
            instability += 0.20
        elif rollback_count >= 1:
            instability += 0.10

        # Health penalty
        if health_score < 0.40:
            instability += 0.15

        return float(np.clip(instability, 0.0, 1.0))


# ============================================================================
# 7. RootCauseEngine -- deeper tags
# ============================================================================
class RootCauseEngine:
    """
    Diagnose root causes of strategy health issues with 16 institutional
    failure tags. Returns primary root cause and full tag list.
    """

    def diagnose(
        self,
        dimensions: Dict[str, float],
        health_state: str,
        health_score: float,
        evidence: StrategyEvidenceProfile,
        change_point: ChangePointResult,
        promotion_data: Dict,
    ) -> Tuple[str, List[str]]:
        """Return (primary_root_cause, failure_tags)."""
        if health_state == "HEALTHY":
            return ("", [])

        tags: List[str] = []
        alpha = dimensions.get("alpha_integrity", 0.5)
        robustness = dimensions.get("robustness_integrity", 0.5)
        regime = dimensions.get("regime_integrity", 0.5)
        implementation = dimensions.get("implementation_integrity", 0.5)
        tail = dimensions.get("tail_integrity", 0.5)
        diversification = dimensions.get("diversification_integrity", 0.5)
        lifecycle = dimensions.get("lifecycle_integrity", 0.5)

        # predictive_edge_erosion: alpha declining and IC trend negative
        if alpha < 0.40:
            tags.append("predictive_edge_erosion")
        price = evidence.price_supplemental
        if price and price.get("ic_trend", 0) < -0.005:
            if "predictive_edge_erosion" not in tags:
                tags.append("predictive_edge_erosion")

        # gross_to_net_collapse: big gap between gross and net
        meth = evidence.methodology_evidence
        gross = meth.get("gross_sharpe", meth.get("sharpe_gross"))
        net = meth.get("net_sharpe", meth.get("sharpe"))
        if (isinstance(gross, (int, float)) and isinstance(net, (int, float))
                and gross - net > 0.4):
            tags.append("gross_to_net_collapse")

        # optimizer_repair_failure: recently optimized but still declining
        opt = evidence.optimizer_evidence
        if opt and opt.get("optimized_recently") and change_point.persistent_weakening:
            tags.append("optimizer_repair_failure")

        # post_promotion_instability
        if promotion_data.get("post_promotion_decline"):
            tags.append("post_promotion_instability")

        # regime_transition_breakdown: abrupt degradation + regime change
        reg = evidence.regime_evidence
        if change_point.abrupt_degradation and reg and reg.get("regime_forecast"):
            tags.append("regime_transition_breakdown")

        # narrow_regime_dependency: only 1 regime is profitable
        regime_perf = reg.get("regime_performance", meth.get("regime_performance", {})) if reg else {}
        if isinstance(regime_perf, dict) and len(regime_perf) >= 2:
            sharpes = []
            for v in regime_perf.values():
                s = v.get("sharpe", v) if isinstance(v, dict) else v
                if isinstance(s, (int, float)) and math.isfinite(s):
                    sharpes.append(s)
            if sharpes and sum(1 for s in sharpes if s > 0) <= 1:
                tags.append("narrow_regime_dependency")

        # robustness_non_generalization
        if robustness < 0.35:
            tags.append("robustness_non_generalization")

        # threshold_pathology: unstable around decision boundaries
        wf = meth.get("walk_forward", meth.get("wf_stats", {}))
        if isinstance(wf, dict):
            wf_std = wf.get("std_sharpe", wf.get("sharpe_std", 0))
            if isinstance(wf_std, (int, float)) and wf_std > 0.8:
                tags.append("threshold_pathology")

        # turnover_pathology: excessive turnover destroying returns
        turnover = meth.get("turnover", meth.get("annual_turnover"))
        if isinstance(turnover, (int, float)) and turnover > 150 and implementation < 0.40:
            tags.append("turnover_pathology")

        # execution_drag_linked: execution costs are the primary issue
        if implementation < 0.35 and alpha > 0.50:
            tags.append("execution_drag_linked")

        # diversification_role_loss
        if diversification < 0.30:
            tags.append("diversification_role_loss")

        # repeated_shadow_failure
        if promotion_data.get("shadow_failures", 0) >= 2:
            tags.append("repeated_shadow_failure")

        # repeated_rollback_pattern
        if promotion_data.get("rollback_count", 0) >= 2:
            tags.append("repeated_rollback_pattern")

        # insufficient_real_evidence
        if evidence.evidence_quality in (EVIDENCE_QUALITY_DEGRADED, EVIDENCE_QUALITY_INSUFFICIENT):
            tags.append("insufficient_real_evidence")

        # proxy_only_monitoring: only price data, no agent evidence
        if evidence.available_source_count() == 0 and evidence.has_price_data():
            tags.append("proxy_only_monitoring")

        # structural_nonviability: too many dimensions failing
        failing = sum(1 for v in dimensions.values() if v < 0.30)
        if failing >= 4:
            tags.append("structural_nonviability")

        tags = sorted(set(tags))

        # Determine primary root cause (first by priority)
        priority_order = [
            "structural_nonviability",
            "predictive_edge_erosion",
            "gross_to_net_collapse",
            "optimizer_repair_failure",
            "post_promotion_instability",
            "regime_transition_breakdown",
            "robustness_non_generalization",
            "execution_drag_linked",
            "turnover_pathology",
            "narrow_regime_dependency",
            "diversification_role_loss",
            "threshold_pathology",
            "repeated_rollback_pattern",
            "repeated_shadow_failure",
            "insufficient_real_evidence",
            "proxy_only_monitoring",
        ]
        primary = ""
        for candidate in priority_order:
            if candidate in tags:
                primary = candidate
                break
        if not primary and tags:
            primary = tags[0]

        return primary, tags

    # Backward compat shim for old-style diagnose signature
    def diagnose_legacy(
        self,
        dims: Any,
        health_state: str,
        confidence: float,
        strategy_data: Dict,
    ) -> List[str]:
        """Legacy diagnose returning just tags."""
        if health_state == "HEALTHY":
            return []
        tags = []
        if isinstance(dims, DecayDimensions):
            dim_dict = dims.to_dict()
        elif isinstance(dims, dict):
            dim_dict = dims
        else:
            dim_dict = {}

        tag_rules = [
            ("weak_predictive_power", "alpha_quality", 0.4),
            ("unstable_ic", "alpha_quality", 0.3),
            ("robustness_breakdown", "robustness", 0.35),
            ("overfit_risk_rising", "robustness", 0.3),
            ("regime_mismatch", "regime_fitness", 0.35),
            ("cost_drag_excessive", "implementation", 0.35),
            ("turnover_too_high", "implementation", 0.25),
            ("tail_fragility", "tail_risk", 0.35),
            ("crisis_failure", "tail_risk", 0.25),
            ("diversification_loss", "diversification", 0.3),
        ]
        for tag, dim_key, threshold in tag_rules:
            val = dim_dict.get(dim_key, 1.0)
            if val < threshold:
                tags.append(tag)
        if confidence < 0.3:
            tags.append("insufficient_sample")
        return sorted(set(tags))


# ============================================================================
# 8. LifecycleActionEngine -- 14 actions
# ============================================================================
class LifecycleActionEngine:
    """
    Determine governance actions based on health state, root causes,
    and lifecycle context. Returns ordered list of actions with urgency.
    """

    # State -> default actions
    STATE_ACTIONS: Dict[str, List[Tuple[str, str]]] = {
        "HEALTHY": [
            ("KEEP_LIVE", "LOW"),
        ],
        "WATCH": [
            ("KEEP_LIVE_WITH_WATCH", "LOW"),
        ],
        "EARLY_DECAY": [
            ("REDUCE_EXPOSURE", "MEDIUM"),
            ("SEND_TO_OPTIMIZER", "MEDIUM"),
        ],
        "REGIME_SUPPRESSED": [
            ("REGIME_DISABLE", "MEDIUM"),
            ("KEEP_LIVE_WITH_WATCH", "MEDIUM"),
        ],
        "IMPLEMENTATION_DECAY": [
            ("REQUIRE_EXECUTION_REVIEW", "HIGH"),
            ("SEND_TO_OPTIMIZER", "HIGH"),
        ],
        "ROBUSTNESS_BREAKDOWN": [
            ("SEND_TO_METHOD_REVIEW", "HIGH"),
            ("FREEZE_PROMOTION", "HIGH"),
        ],
        "STRUCTURAL_DECAY": [
            ("FREEZE_PROMOTION", "CRITICAL"),
            ("SEND_TO_METHOD_REVIEW", "CRITICAL"),
            ("REQUIRE_RISK_REVIEW", "HIGH"),
        ],
        "SHADOW_FAILURE": [
            ("SHADOW_ONLY", "MEDIUM"),
            ("SEND_TO_METHOD_REVIEW", "MEDIUM"),
        ],
        "POST_PROMOTION_FAILURE": [
            ("REDUCE_EXPOSURE", "HIGH"),
            ("SEND_TO_OPTIMIZER", "HIGH"),
            ("FREEZE_PROMOTION", "HIGH"),
        ],
        "RETIREMENT_WATCH": [
            ("KEEP_IN_RETIREMENT_WATCH", "HIGH"),
            ("REDUCE_EXPOSURE", "HIGH"),
            ("REQUIRE_RISK_REVIEW", "HIGH"),
        ],
        "DEAD": [
            ("RETIRE_STRATEGY", "CRITICAL"),
        ],
        "INSUFFICIENT_EVIDENCE": [
            ("NO_ACTION_INSUFFICIENT_EVIDENCE", "LOW"),
        ],
    }

    def determine_actions(
        self,
        health_state: str,
        primary_root_cause: str,
        failure_tags: List[str],
        lifecycle_stage: str,
        hazard_30d: float,
    ) -> List[Dict[str, Any]]:
        """Determine lifecycle actions with urgency and reasoning."""
        base_actions = self.STATE_ACTIONS.get(health_state, [("KEEP_LIVE_WITH_WATCH", "LOW")])

        actions = []
        for entry in base_actions:
            action_dict: Dict[str, Any] = {
                "action": entry[0],
                "urgency": entry[1],
                "reason": f"state={health_state}",
            }
            actions.append(action_dict)

        # Augment with root-cause-specific actions
        if "optimizer_repair_failure" in failure_tags and health_state != "DEAD":
            if not any(a["action"] == "SEND_TO_METHOD_REVIEW" for a in actions):
                actions.append({
                    "action": "SEND_TO_METHOD_REVIEW",
                    "urgency": "HIGH",
                    "reason": "optimizer_repair_failure -- escalate to methodology",
                })

        if "gross_to_net_collapse" in failure_tags:
            if not any(a["action"] == "REQUIRE_EXECUTION_REVIEW" for a in actions):
                actions.append({
                    "action": "REQUIRE_EXECUTION_REVIEW",
                    "urgency": "HIGH",
                    "reason": "gross_to_net_collapse -- execution review needed",
                })

        if "threshold_pathology" in failure_tags or "turnover_pathology" in failure_tags:
            if not any(a["action"] == "SEND_TO_MATH_REVIEW" for a in actions):
                actions.append({
                    "action": "SEND_TO_MATH_REVIEW",
                    "urgency": "MEDIUM",
                    "reason": "parameter pathology -- math review needed",
                })

        if "repeated_rollback_pattern" in failure_tags:
            if not any(a["action"] == "FREEZE_PROMOTION" for a in actions):
                actions.append({
                    "action": "FREEZE_PROMOTION",
                    "urgency": "HIGH",
                    "reason": "repeated_rollback_pattern -- promotion frozen",
                })

        # High hazard escalation
        if hazard_30d > 0.80 and health_state not in ("DEAD", "RETIREMENT_WATCH"):
            if not any(a["action"] == "REQUIRE_RISK_REVIEW" for a in actions):
                actions.append({
                    "action": "REQUIRE_RISK_REVIEW",
                    "urgency": "CRITICAL",
                    "reason": f"hazard_30d={hazard_30d:.2f} -- risk review required",
                })

        return actions

    # Backward compat
    def get_actions(self, health_state: str) -> List[Dict[str, str]]:
        """Legacy action getter."""
        base = self.STATE_ACTIONS.get(health_state, [("KEEP_LIVE_WITH_WATCH", "LOW")])
        return [{"action": a[0], "urgency": a[1]} for a in base]


# Legacy alias
ActionPolicyEngine = LifecycleActionEngine


# ============================================================================
# 9. PromotionSurveillanceEngine
# ============================================================================
class PromotionSurveillanceEngine:
    """
    Track recently promoted strategies, shadow candidates, and rolled-back
    strategies. Detect post-promotion decline, sandbox-to-live failure,
    and repeated rollback patterns.
    """

    def __init__(self) -> None:
        self._promotion_log: List[Dict] = []
        self._rollback_log: List[Dict] = []
        self._shadow_log: List[Dict] = []

    def load_from_history(self, history: List[Dict]) -> None:
        """Load promotion/rollback data from decay history."""
        for record in history:
            table = record.get("strategy_health_table", [])
            ts = record.get("timestamp", "")
            for strat in table:
                if not isinstance(strat, dict):
                    continue
                name = strat.get("name", "")

                # Track lifecycle stage transitions
                lifecycle = strat.get("lifecycle_stage", "")
                if lifecycle == "RECENTLY_PROMOTED":
                    self._promotion_log.append({"name": name, "timestamp": ts})
                elif lifecycle == "RETIRED":
                    pass  # Track separately if needed

                # Track rollbacks
                health_state = strat.get("health_state", "")
                if health_state == "POST_PROMOTION_FAILURE":
                    self._rollback_log.append({"name": name, "timestamp": ts})

                # Track shadow failures
                if health_state == "SHADOW_FAILURE":
                    self._shadow_log.append({"name": name, "timestamp": ts})

    def get_promotion_data(self, strategy_name: str, current_health_score: float,
                           current_health_timeline: List[float]) -> Dict:
        """Get promotion surveillance data for a strategy."""
        recent_promotions = [p for p in self._promotion_log if p["name"] == strategy_name]
        rollbacks = [r for r in self._rollback_log if r["name"] == strategy_name]
        shadow_failures = [s for s in self._shadow_log if s["name"] == strategy_name]

        # Detect post-promotion decline
        post_promo_decline = False
        if recent_promotions and len(current_health_timeline) >= 3:
            # Check if declining since promotion
            recent = current_health_timeline[-3:]
            if len(recent) >= 3 and all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                post_promo_decline = True

        return {
            "promotion_count": len(recent_promotions),
            "rollback_count": len(rollbacks),
            "shadow_failures": len(shadow_failures),
            "shadow_successes": max(0, len(recent_promotions) - len(rollbacks)),
            "post_promotion_decline": post_promo_decline,
            "repair_attempts": len([r for r in self._rollback_log if r["name"] == strategy_name]),
            "recently_promoted": len(recent_promotions) > 0,
        }

    def determine_lifecycle_stage(self, strategy_name: str, health_state: str,
                                  health_score: float, consecutive_below: int) -> str:
        """Determine the lifecycle stage for a strategy."""
        recent_promotions = [p for p in self._promotion_log if p["name"] == strategy_name]
        rollbacks = [r for r in self._rollback_log if r["name"] == strategy_name]

        if health_state == "DEAD":
            return "RETIRED"
        if health_state == "RETIREMENT_WATCH":
            return "RETIREMENT_WATCH"
        if recent_promotions and len(recent_promotions) <= PROMOTION_SURVEILLANCE_WINDOW:
            return "RECENTLY_PROMOTED"
        if health_state in ("SHADOW_FAILURE",):
            return "SHADOW_CANDIDATE"
        if health_score >= 0.50 and health_state in ("HEALTHY", "WATCH"):
            return "CHAMPION_LIVE"
        return "SHADOW_CANDIDATE"


# ============================================================================
# 10. DecayHistoryStore
# ============================================================================
class DecayHistoryStore:
    """
    Append-only JSON history for decay observations.
    Capped at MAX_HISTORY_ENTRIES (365).
    Provides timeline queries per strategy.
    """

    def __init__(self, history: Optional[List[Dict]] = None) -> None:
        self._history: List[Dict] = list(history) if history else []

    def append(self, entry: Dict) -> None:
        self._history.append(entry)
        if len(self._history) > MAX_HISTORY_ENTRIES:
            self._history = self._history[-MAX_HISTORY_ENTRIES:]

    def save(self) -> None:
        try:
            DECAY_HISTORY_PATH.write_text(
                json.dumps(self._history, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to save decay history: %s", exc)

    @property
    def history(self) -> List[Dict]:
        return self._history

    def get_strategy_timeline(self, name: str, n: int = 30) -> List[Dict]:
        """Get last n history entries for a strategy."""
        entries = []
        for record in self._history:
            table = record.get("strategy_health_table", [])
            for strat in table:
                if isinstance(strat, dict) and strat.get("name") == name:
                    entries.append({
                        "timestamp": record.get("timestamp"),
                        "health_score": strat.get("health_score"),
                        "health_state": strat.get("health_state"),
                        "hazard_30d": strat.get("hazard_30d", strat.get("hazard_score_30d")),
                        "hazard_90d": strat.get("hazard_90d", strat.get("hazard_score_90d")),
                        "lifecycle_stage": strat.get("lifecycle_stage", ""),
                    })
                    break
        return entries[-n:]

    def get_health_scores(self, name: str, n: int = 30) -> List[float]:
        """Return last n health scores for a strategy."""
        timeline = self.get_strategy_timeline(name, n)
        return [h.get("health_score", 0.5) for h in timeline
                if isinstance(h.get("health_score"), (int, float))]

    def get_hazard_trend(self, name: str, n: int = 10) -> List[float]:
        """Return last n hazard scores for a strategy."""
        timeline = self.get_strategy_timeline(name, n)
        return [h.get("hazard_30d", h.get("hazard_score_30d", 0.5)) for h in timeline]

    def get_retirement_watchlist(self) -> List[str]:
        """Strategies with DEAD state or high hazard in most recent entry."""
        if not self._history:
            return []
        latest = self._history[-1].get("strategy_health_table", [])
        return [
            s["name"] for s in latest
            if isinstance(s, dict) and (
                s.get("health_state") == "DEAD"
                or s.get("health_state") == "RETIREMENT_WATCH"
                or s.get("hazard_30d", s.get("hazard_score_30d", 0)) > 0.8
            )
        ]

    def get_post_promotion_failures(self) -> List[str]:
        """Strategies that experienced POST_PROMOTION_FAILURE."""
        failures = set()
        for record in self._history:
            table = record.get("strategy_health_table", [])
            for strat in table:
                if isinstance(strat, dict) and strat.get("health_state") == "POST_PROMOTION_FAILURE":
                    failures.add(strat.get("name", ""))
        failures.discard("")
        return sorted(failures)

    def consecutive_below_threshold(self, name: str, threshold: float = 0.25) -> int:
        """Count consecutive most recent periods where health < threshold."""
        count = 0
        for record in reversed(self._history):
            table = record.get("strategy_health_table", [])
            found = False
            for strat in table:
                if isinstance(strat, dict) and strat.get("name") == name:
                    if isinstance(strat.get("health_score"), (int, float)) and strat["health_score"] < threshold:
                        count += 1
                        found = True
                    else:
                        return count
                    break
            if not found:
                break
        return count

    def consecutive_dead_count(self, name: str) -> int:
        """Count consecutive observations where health_score < 0.2 (legacy compat)."""
        return self.consecutive_below_threshold(name, 0.2)

    def state_transitions(self, name: str) -> List[Dict]:
        """Get state transitions for a strategy."""
        transitions = []
        prev_state = None
        for record in self._history:
            table = record.get("strategy_health_table", [])
            for strat in table:
                if isinstance(strat, dict) and strat.get("name") == name:
                    state = strat.get("health_state")
                    if prev_state and state != prev_state:
                        transitions.append({
                            "timestamp": record.get("timestamp"),
                            "from": prev_state,
                            "to": state,
                        })
                    prev_state = state
                    break
        return transitions

    def latest_decay_events(self) -> List[Dict]:
        """Strategies that transitioned to a worse state in the last entry."""
        if len(self._history) < 2:
            return []
        prev_table = {s["name"]: s for s in self._history[-2].get("strategy_health_table", [])
                      if isinstance(s, dict)}
        curr_table = self._history[-1].get("strategy_health_table", [])
        events = []
        for strat in curr_table:
            if not isinstance(strat, dict):
                continue
            prev = prev_table.get(strat.get("name"))
            if prev:
                curr_sev = HEALTH_STATE_SEVERITY.get(strat.get("health_state"), 0)
                prev_sev = HEALTH_STATE_SEVERITY.get(prev.get("health_state"), 0)
                if curr_sev > prev_sev:
                    events.append({
                        "name": strat["name"],
                        "from": prev["health_state"],
                        "to": strat["health_state"],
                    })
        return events

    def strategies_near_retirement(self) -> List[str]:
        """Legacy compat: strategies near retirement."""
        return self.get_retirement_watchlist()


# Legacy aliases
DecayHistoryTracker = DecayHistoryStore


# ============================================================================
# 11. DownstreamContractBuilder
# ============================================================================
class DownstreamContractBuilder:
    """
    Build per-agent downstream contracts based on strategy health state
    and lifecycle actions. Each agent gets specific instructions.
    """

    def build(
        self,
        scorecard: StrategyHealthScorecard,
        actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build downstream contracts for all agents."""
        state = scorecard.health_state
        name = scorecard.name
        tags = scorecard.failure_tags

        contracts: Dict[str, Any] = {
            "methodology": self._methodology_contract(name, state, tags, scorecard),
            "optimizer": self._optimizer_contract(name, state, tags, actions, scorecard),
            "regime_forecaster": self._regime_contract(name, state, tags, scorecard),
            "portfolio_construction": self._portfolio_contract(name, state, actions, scorecard),
            "risk_guardian": self._risk_contract(name, state, tags, scorecard),
            "execution": self._execution_contract(name, state, tags, scorecard),
            "auto_improve": self._auto_improve_contract(name, state, tags, actions, scorecard),
            "math": self._math_contract(name, state, tags, scorecard),
        }
        return contracts

    def _methodology_contract(self, name: str, state: str, tags: List[str],
                               sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        if state in ("STRUCTURAL_DECAY", "ROBUSTNESS_BREAKDOWN"):
            contract["instructions"].append("FULL_REVALIDATION_REQUIRED")
            contract["instructions"].append("walk_forward_retest_mandatory")
            contract["priority"] = "CRITICAL"
        elif state == "EARLY_DECAY":
            contract["instructions"].append("revalidate_ic_and_sharpe")
            contract["instructions"].append("check_walk_forward_stability")
            contract["priority"] = "HIGH"
        elif state == "POST_PROMOTION_FAILURE":
            contract["instructions"].append("post_promotion_revalidation")
            contract["instructions"].append("compare_shadow_vs_live_metrics")
            contract["priority"] = "HIGH"
        elif state in ("DEAD", "RETIREMENT_WATCH"):
            contract["instructions"].append("archive_strategy_report")
            contract["instructions"].append("document_failure_mode")
            contract["priority"] = "LOW"
        else:
            contract["instructions"].append("routine_monitoring")
            contract["priority"] = "LOW"

        if "robustness_non_generalization" in tags:
            contract["instructions"].append("investigate_overfitting")
        if "predictive_edge_erosion" in tags:
            contract["instructions"].append("investigate_alpha_source")

        return contract

    def _optimizer_contract(self, name: str, state: str, tags: List[str],
                             actions: List[Dict], sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        send_to_opt = any(a.get("action") == "SEND_TO_OPTIMIZER" for a in actions)

        if send_to_opt:
            contract["instructions"].append("OPTIMIZATION_REQUESTED")
            contract["priority"] = "HIGH"
            if "gross_to_net_collapse" in tags:
                contract["instructions"].append("focus_cost_reduction")
                contract["focus"] = "cost_reduction"
            if "turnover_pathology" in tags:
                contract["instructions"].append("reduce_turnover")
                contract["focus"] = "turnover_reduction"
            if "predictive_edge_erosion" in tags:
                contract["instructions"].append("re_optimize_alpha_parameters")
        elif state == "IMPLEMENTATION_DECAY":
            contract["instructions"].append("cost_optimization_needed")
            contract["priority"] = "MEDIUM"
        else:
            contract["instructions"].append("no_optimization_needed")
            contract["priority"] = "LOW"

        if "optimizer_repair_failure" in tags:
            contract["instructions"].append("WARNING_previous_optimization_failed")

        return contract

    def _regime_contract(self, name: str, state: str, tags: List[str],
                          sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        if state == "REGIME_SUPPRESSED":
            contract["instructions"].append("provide_regime_fitness_map")
            contract["instructions"].append("identify_favorable_regimes")
            contract["priority"] = "HIGH"
        elif "regime_transition_breakdown" in tags:
            contract["instructions"].append("analyze_transition_impact")
            contract["priority"] = "MEDIUM"
        elif "narrow_regime_dependency" in tags:
            contract["instructions"].append("assess_regime_breadth")
            contract["priority"] = "MEDIUM"
        else:
            contract["instructions"].append("routine_regime_monitoring")
            contract["priority"] = "LOW"
        return contract

    def _portfolio_contract(self, name: str, state: str, actions: List[Dict],
                             sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        reduce = any(a.get("action") == "REDUCE_EXPOSURE" for a in actions)
        retire = any(a.get("action") == "RETIRE_STRATEGY" for a in actions)

        if retire:
            contract["instructions"].append("REMOVE_FROM_PORTFOLIO")
            contract["priority"] = "CRITICAL"
        elif reduce:
            contract["instructions"].append("REDUCE_WEIGHT")
            contract["reduction_factor"] = 0.5 if sc.hazard_30d > 0.7 else 0.75
            contract["priority"] = "HIGH"
        elif state == "REGIME_SUPPRESSED":
            contract["instructions"].append("CONDITIONAL_WEIGHT_BY_REGIME")
            contract["priority"] = "MEDIUM"
        elif state in ("WATCH", "EARLY_DECAY"):
            contract["instructions"].append("MONITOR_WEIGHT_STABILITY")
            contract["priority"] = "LOW"
        else:
            contract["instructions"].append("maintain_current_weight")
            contract["priority"] = "LOW"
        return contract

    def _risk_contract(self, name: str, state: str, tags: List[str],
                        sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        if state in ("STRUCTURAL_DECAY", "RETIREMENT_WATCH", "DEAD"):
            contract["instructions"].append("ELEVATED_RISK_MONITORING")
            contract["instructions"].append("tighten_risk_limits")
            contract["priority"] = "CRITICAL"
        elif sc.tail_integrity < 0.35:
            contract["instructions"].append("tail_risk_elevated")
            contract["instructions"].append("stress_test_required")
            contract["priority"] = "HIGH"
        elif sc.hazard_30d > 0.6:
            contract["instructions"].append("heightened_risk_watch")
            contract["priority"] = "MEDIUM"
        else:
            contract["instructions"].append("routine_risk_monitoring")
            contract["priority"] = "LOW"
        return contract

    def _execution_contract(self, name: str, state: str, tags: List[str],
                             sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        if state == "IMPLEMENTATION_DECAY" or "execution_drag_linked" in tags:
            contract["instructions"].append("EXECUTION_REVIEW_REQUIRED")
            contract["instructions"].append("analyze_slippage_and_costs")
            contract["priority"] = "HIGH"
        elif "gross_to_net_collapse" in tags:
            contract["instructions"].append("cost_analysis_requested")
            contract["priority"] = "MEDIUM"
        elif state == "DEAD":
            contract["instructions"].append("CEASE_EXECUTION")
            contract["priority"] = "CRITICAL"
        else:
            contract["instructions"].append("routine_execution")
            contract["priority"] = "LOW"
        return contract

    def _auto_improve_contract(self, name: str, state: str, tags: List[str],
                                actions: List[Dict], sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        if state in ("EARLY_DECAY", "ROBUSTNESS_BREAKDOWN", "STRUCTURAL_DECAY"):
            contract["instructions"].append("IMPROVEMENT_CYCLE_RECOMMENDED")
            contract["focus_areas"] = list(tags)
            contract["priority"] = "HIGH"
        elif state == "POST_PROMOTION_FAILURE":
            contract["instructions"].append("post_promotion_repair")
            contract["priority"] = "HIGH"
        elif state == "DEAD":
            contract["instructions"].append("strategy_archived_no_improvement")
            contract["priority"] = "NONE"
        else:
            contract["instructions"].append("routine_improvement_scan")
            contract["priority"] = "LOW"
        return contract

    def _math_contract(self, name: str, state: str, tags: List[str],
                        sc: StrategyHealthScorecard) -> Dict:
        contract: Dict[str, Any] = {"strategy": name, "instructions": []}
        send_to_math = any(t in tags for t in ("threshold_pathology", "turnover_pathology"))
        if send_to_math:
            contract["instructions"].append("MATH_REVIEW_REQUESTED")
            contract["instructions"].append("review_parameter_boundaries")
            contract["priority"] = "HIGH"
        elif state == "ROBUSTNESS_BREAKDOWN":
            contract["instructions"].append("review_model_assumptions")
            contract["priority"] = "MEDIUM"
        else:
            contract["instructions"].append("routine_math_check")
            contract["priority"] = "LOW"
        return contract


# ============================================================================
# Legacy compatibility: HealthScorer wrapper
# ============================================================================
class HealthScorer:
    """Backward-compatible scorer wrapping the new engines."""

    def __init__(self, evidence: StrategyEvidenceAssembler) -> None:
        self.ev = evidence
        self._dim_engine = HealthDimensionEngine()

    def score_strategy(self, strategy_name: str, strategy_data: Dict) -> Tuple[DecayDimensions, float, float]:
        """Return (dimensions, health_score, confidence) -- legacy interface."""
        profile = self.ev.assemble_evidence(strategy_name, strategy_data)
        dims_dict = self._dim_engine.compute_all(profile)

        # Map 7 dims to legacy 6 dims
        legacy_dims = DecayDimensions(
            alpha_quality=dims_dict.get("alpha_integrity", 0.5),
            robustness=dims_dict.get("robustness_integrity", 0.5),
            regime_fitness=dims_dict.get("regime_integrity", 0.5),
            implementation=dims_dict.get("implementation_integrity", 0.5),
            tail_risk=dims_dict.get("tail_integrity", 0.5),
            diversification=dims_dict.get("diversification_integrity", 0.5),
        )

        health = sum(
            dims_dict.get(k, 0.5) * w for k, w in DIMENSION_WEIGHTS.items()
        )
        health = float(np.clip(health, 0.0, 1.0))
        confidence = self._estimate_confidence(strategy_name, strategy_data, profile)
        return legacy_dims, health, confidence

    def _estimate_confidence(self, name: str, data: Dict,
                              profile: StrategyEvidenceProfile) -> float:
        n_sources = profile.available_source_count()
        has_price = profile.has_price_data()
        return float(np.clip((n_sources + (1 if has_price else 0)) / 9.0, 0.0, 1.0))

    def estimate_evidence_quality(self, name: str, data: Dict) -> float:
        profile = self.ev.assemble_evidence(name, data)
        eq_engine = EvidenceQualityEngine()
        eq_engine.score(profile)
        return profile.evidence_quality_score

    def classify_regime_decay(self, name: str, data: Dict) -> Dict[str, str]:
        """Classify per-regime decay states (legacy compat)."""
        regime_states: Dict[str, str] = {}
        regimes = data.get("regime_performance", data.get("regimes", {}))
        for regime_name, metrics in regimes.items():
            if isinstance(metrics, dict):
                wr = metrics.get("win_rate", metrics.get("wr", 0.5))
                sh = metrics.get("sharpe", 0)
            else:
                continue
            regime_upper = str(regime_name).upper()
            if sh < -0.5 or wr < 0.40:
                if "CALM" in regime_upper:
                    regime_states[regime_upper] = "CALM_DECAY"
                elif "NORMAL" in regime_upper:
                    regime_states[regime_upper] = "NORMAL_DECAY"
                elif "TENSION" in regime_upper:
                    regime_states[regime_upper] = "TENSION_BREAKDOWN"
                elif "CRISIS" in regime_upper:
                    regime_states[regime_upper] = "CRISIS_FAILURE"
                else:
                    regime_states[regime_upper] = "REGIME_DECAY"
            elif sh < 0 or wr < 0.48:
                regime_states[regime_upper] = "TRANSITION_FRAGILITY"
            else:
                regime_states[regime_upper] = "HEALTHY"
        return regime_states

    def identify_regime_strengths(self, name: str, data: Dict) -> Tuple[List[str], List[str]]:
        """Identify strongest and weakest regimes (legacy compat)."""
        regimes = data.get("regime_performance", data.get("regimes", {}))
        scored = []
        for rname, metrics in regimes.items():
            if isinstance(metrics, dict):
                sh = metrics.get("sharpe", 0)
                wr = metrics.get("win_rate", metrics.get("wr", 0.5))
                score = sh * 0.6 + (wr - 0.5) * 2.0 * 0.4
                scored.append((str(rname).upper(), score))
        scored.sort(key=lambda x: x[1], reverse=True)
        strongest = [r for r, s in scored if s > 0][:3]
        weakest = [r for r, s in scored if s <= 0][:3]
        return strongest, weakest

    def _compute_price_metrics(self, name: str) -> Optional[Dict]:
        """Legacy compat: compute price metrics."""
        return self.ev._compute_price_supplemental(name) or None


# ============================================================================
# Legacy compatibility: HealthStateClassifier wrapper
# ============================================================================
class HealthStateClassifier:
    """Backward-compatible classifier wrapping LifecycleClassifier."""

    def __init__(self) -> None:
        self._classifier = LifecycleClassifier()

    def classify(
        self,
        dims: Any,
        health_score: float,
        confidence: float,
        consecutive_dead: int = 0,
    ) -> str:
        """Legacy classify returning old-style states."""
        if isinstance(dims, DecayDimensions):
            dim_dict = {
                "alpha_integrity": dims.alpha_quality,
                "robustness_integrity": dims.robustness,
                "regime_integrity": dims.regime_fitness,
                "implementation_integrity": dims.implementation,
                "tail_integrity": dims.tail_risk,
                "diversification_integrity": dims.diversification,
                "lifecycle_integrity": 0.6,
            }
        elif isinstance(dims, dict):
            dim_dict = dims
        else:
            dim_dict = {}

        if confidence < 0.2:
            return "INSUFFICIENT_EVIDENCE"

        cp = ChangePointResult()
        state = self._classifier.classify(
            health_score=health_score,
            dimensions=dim_dict,
            evidence_quality=EVIDENCE_QUALITY_PARTIAL if confidence >= 0.4 else EVIDENCE_QUALITY_DEGRADED,
            evidence_quality_score=confidence,
            change_point=cp,
            lifecycle_stage="CHAMPION_LIVE",
            consecutive_below_threshold=consecutive_dead,
            consecutive_declining=0,
        )
        return state


# ============================================================================
# 12. AlphaDecayMonitor (Main Orchestrator)
# ============================================================================
class AlphaDecayMonitor:
    """
    Institutional Strategy Health, Failure & Retirement Governance Engine.

    This is the STRATEGY RETIREMENT AUTHORITY. It orchestrates the complete
    health assessment pipeline:
      1. Assemble evidence from all 8 agents
      2. Score evidence quality per strategy
      3. Compute 7 health dimensions per strategy
      4. Run change-point detection
      5. Classify lifecycle/health state (strict rules)
      6. Compute hazard (not just 1-health)
      7. Diagnose root causes
      8. Determine lifecycle actions
      9. Build downstream contracts
      10. Build scorecards
      11. Save decay_status.json + history
      12. Publish to bus
      13. Registry heartbeat

    Backward-compatible: AlphaDecayMonitor class name, run() method, CLI,
    bus, registry, decay_level output.
    """

    SECTORS = SECTORS
    MR_WHITELIST = MR_WHITELIST

    def __init__(self, settings=None) -> None:
        self._settings = settings

        # Core engines
        self._evidence = StrategyEvidenceAssembler()
        self._eq_engine = EvidenceQualityEngine()
        self._dim_engine = HealthDimensionEngine()
        self._cp_engine = ChangePointEngine()
        self._classifier = LifecycleClassifier()
        self._hazard_model = HazardModel()
        self._root_cause_engine = RootCauseEngine()
        self._action_engine = LifecycleActionEngine()
        self._promo_engine = PromotionSurveillanceEngine()
        self._contract_builder = DownstreamContractBuilder()

        # State
        self._history_store: Optional[DecayHistoryStore] = None
        self._scorer: Optional[HealthScorer] = None

    @property
    def settings(self):
        if self._settings is None:
            try:
                from config.settings import get_settings
                self._settings = get_settings()
            except Exception:
                pass
        return self._settings

    @property
    def prices(self) -> Optional[pd.DataFrame]:
        return self._evidence.prices

    @property
    def methodology_results(self) -> Optional[Dict]:
        return self._evidence.methodology_reports or None

    # -- backward-compatible legacy methods ------------------------------------

    def classify_decay(self, decay_signals: Dict) -> str:
        """Legacy classify method for backward compatibility."""
        neg_ic = decay_signals.get("consecutive_negative_ic", 0)
        neg_sharpe = decay_signals.get("consecutive_negative_sharpe", 0)
        declining_count = sum([
            decay_signals.get("ic_declining", False),
            decay_signals.get("sharpe_declining", False),
            decay_signals.get("wr_declining", False),
        ])
        if neg_sharpe >= 6 or neg_ic >= 6:
            return "DEAD"
        if neg_sharpe >= 3 or neg_ic >= 3 or declining_count >= 2:
            return "DECAYING"
        if declining_count >= 1 or neg_sharpe >= 1:
            return "EARLY_DECAY"
        return "HEALTHY"

    def generate_recommendations(self, decay_level: str, metrics: Dict) -> List[str]:
        """Legacy recommendations for backward compatibility."""
        recs = []
        if decay_level == "HEALTHY":
            recs.append("Continue current strategy -- all metrics within normal range")
        elif decay_level == "EARLY_DECAY":
            recs.append("Tighten z-entry threshold from 0.7 to 0.9")
            recs.append("Reduce position sizes by 20%")
            recs.append("Monitor daily for further deterioration")
        elif decay_level == "DECAYING":
            recs.append("ALERT: Strategy is decaying -- switch to defensive params")
            recs.append("Reduce max positions from 4 to 2")
            recs.append("Increase VIX kill threshold from 32 to 28")
            recs.append("Notify Optimizer agent for parameter re-calibration")
        elif decay_level == "DEAD":
            recs.append("CRITICAL: Strategy appears dead -- recommend DISABLE")
            recs.append("Close all open positions from this strategy")
            recs.append("Notify all agents to switch to alternative strategy")
            recs.append("Run full re-calibration cycle")
        return recs

    # -- strategy extraction ---------------------------------------------------

    def _extract_strategies(self) -> Dict[str, Dict]:
        """Extract strategy list from evidence assembler."""
        return self._evidence.extract_strategies()

    # -- overall status --------------------------------------------------------

    @staticmethod
    def _compute_overall_status(scorecards: List[StrategyHealthScorecard]) -> str:
        """Determine overall portfolio health status from scorecards."""
        states = [c.health_state for c in scorecards]
        if any(s == "DEAD" for s in states):
            return "STRUCTURAL_DECAY"
        if any(s in ("STRUCTURAL_DECAY", "RETIREMENT_WATCH") for s in states):
            return "STRUCTURAL_DECAY"
        if any(s in ("IMPLEMENTATION_DECAY", "ROBUSTNESS_BREAKDOWN",
                      "POST_PROMOTION_FAILURE", "SHADOW_FAILURE") for s in states):
            return "EARLY_DECAY"
        if any(s in ("EARLY_DECAY", "REGIME_SUPPRESSED") for s in states):
            return "WATCH"
        if any(s == "WATCH" for s in states):
            return "WATCH"
        return "HEALTHY"

    @staticmethod
    def _compute_legacy_level(scorecards: List[StrategyHealthScorecard]) -> str:
        """Worst legacy level across all strategies."""
        legacy_order = {"HEALTHY": 0, "EARLY_DECAY": 1, "DECAYING": 2, "DEAD": 3}
        worst = "HEALTHY"
        for c in scorecards:
            leg = c.legacy_decay_level
            if legacy_order.get(leg, 0) > legacy_order.get(worst, 0):
                worst = leg
        return worst

    # -- machine summary -------------------------------------------------------

    def _build_machine_summary(
        self,
        scorecards: List[StrategyHealthScorecard],
        overall_status: str,
    ) -> Dict[str, Any]:
        """Build machine-readable summary for downstream consumers."""
        at_risk = [c for c in scorecards
                   if c.health_state not in ("HEALTHY", "INSUFFICIENT_EVIDENCE")]
        most_urgent = None
        if at_risk:
            worst = max(at_risk, key=lambda c: c.hazard_30d)
            if worst.actions:
                most_urgent = f"{worst.actions[0]['action']} on {worst.name}"

        regime_warnings: List[str] = []
        regime_weak = [c for c in scorecards if c.health_state == "REGIME_SUPPRESSED"]
        if regime_weak:
            regime_warnings.append(
                f"Regime-suppressed strategies: {', '.join(c.name for c in regime_weak)}"
            )

        retirement_warnings: List[str] = []
        retirement_watch = [c for c in scorecards
                            if c.health_state in ("RETIREMENT_WATCH", "DEAD")]
        if retirement_watch:
            retirement_warnings.append(
                f"Retirement candidates: {', '.join(c.name for c in retirement_watch)}"
            )

        promotion_warnings: List[str] = []
        promo_failures = [c for c in scorecards
                          if c.health_state == "POST_PROMOTION_FAILURE"]
        if promo_failures:
            promotion_warnings.append(
                f"Post-promotion failures: {', '.join(c.name for c in promo_failures)}"
            )

        optimizer_instructions: Dict[str, List[str]] = {
            "cost_reduction": [],
            "regime_repair": [],
            "alpha_repair": [],
            "robustness_repair": [],
        }
        for c in scorecards:
            if c.health_state == "IMPLEMENTATION_DECAY":
                optimizer_instructions["cost_reduction"].append(c.name)
            if c.health_state == "REGIME_SUPPRESSED":
                optimizer_instructions["regime_repair"].append(c.name)
            if "predictive_edge_erosion" in c.failure_tags:
                optimizer_instructions["alpha_repair"].append(c.name)
            if c.health_state == "ROBUSTNESS_BREAKDOWN":
                optimizer_instructions["robustness_repair"].append(c.name)

        # Downstream contracts aggregated
        all_contracts: Dict[str, List[Dict]] = {}
        for c in scorecards:
            for agent, contract in c.downstream_contracts.items():
                if agent not in all_contracts:
                    all_contracts[agent] = []
                all_contracts[agent].append(contract)

        return {
            "overall_health": overall_status,
            "strategies_assessed": len(scorecards),
            "strategies_at_risk": len(at_risk),
            "most_urgent_action": most_urgent,
            "regime_warnings": regime_warnings,
            "retirement_warnings": retirement_warnings,
            "promotion_warnings": promotion_warnings,
            "optimizer_instructions": optimizer_instructions,
            "downstream_contracts": all_contracts,
            "retirement_watchlist": [c.name for c in retirement_watch],
            "post_promotion_failures": [c.name for c in promo_failures],
        }

    # -- regime decay map ------------------------------------------------------

    @staticmethod
    def _build_regime_decay_map(scorecards: List[StrategyHealthScorecard]) -> Dict[str, List[str]]:
        """Map regimes to strategies that are weak in them."""
        regime_map: Dict[str, List[str]] = {}
        for c in scorecards:
            if c.regime_integrity < 0.4:
                regime_map.setdefault("WEAK_REGIME", []).append(c.name)
            for regime_name, regime_state in c.regime_decay_states.items():
                if regime_state != "HEALTHY":
                    regime_map.setdefault(regime_name, []).append(c.name)
        return regime_map

    # =========================================================================
    # run() -- main entry point
    # =========================================================================
    def run(self) -> Dict:
        """
        Full institutional health assessment cycle.

        Pipeline:
          1. Assemble evidence from all agents
          2. Score evidence quality per strategy
          3. Compute 7 health dimensions per strategy
          4. Run change-point detection
          5. Classify lifecycle/health state (strict rules)
          6. Compute hazard (not just 1-health)
          7. Diagnose root causes
          8. Determine lifecycle actions
          9. Build downstream contracts
          10. Build scorecards
          11. Save decay_status.json + history
          12. Publish to bus
          13. Registry heartbeat
        """
        ts = datetime.now(timezone.utc)
        ts_iso = ts.isoformat()

        log.info("=" * 78)
        log.info("ALPHA DECAY GOVERNANCE ENGINE -- Strategy Retirement Authority -- %s", ts_iso[:19])
        log.info("=" * 78)

        # ── Step 1: Assemble evidence from all agents ───────────────────────
        self._evidence.load_all()
        self._history_store = DecayHistoryStore(list(self._evidence.previous_history))
        self._scorer = HealthScorer(self._evidence)

        # Load promotion surveillance from history
        self._promo_engine.load_from_history(self._history_store.history)

        # ── Step 2: Extract strategies ──────────────────────────────────────
        strategies = self._extract_strategies()
        if not strategies:
            log.warning("No strategies found -- nothing to assess")
            result = {
                "timestamp": ts_iso,
                "status": "ERROR",
                "error": "No strategies found",
                "decay_level": "HEALTHY",
            }
            DECAY_STATUS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        log.info("Assessing %d strategies across %d evidence sources",
                 len(strategies), self._count_available_sources())

        # ── Step 3-10: Process each strategy ────────────────────────────────
        scorecards: List[StrategyHealthScorecard] = []

        for name, data in strategies.items():
            try:
                scorecard = self._assess_strategy(name, data, ts_iso)
                scorecards.append(scorecard)
            except Exception as exc:
                log.error("Failed to assess strategy %s: %s", name, exc, exc_info=True)
                # Create minimal error scorecard
                scorecards.append(StrategyHealthScorecard(
                    name=name,
                    health_state="INSUFFICIENT_EVIDENCE",
                    health_score=0.0,
                    legacy_decay_level="EARLY_DECAY",
                    evidence_gaps=["assessment_error"],
                    notes=f"Assessment failed: {exc}",
                    timestamp=ts_iso,
                ))

        # Sort by hazard (worst first)
        scorecards.sort(key=lambda c: c.hazard_30d, reverse=True)

        # ── Step 11: Aggregate and save ─────────────────────────────────────
        overall_status = self._compute_overall_status(scorecards)
        legacy_level = self._compute_legacy_level(scorecards)

        healthy_count = sum(1 for c in scorecards
                           if c.health_state in ("HEALTHY",))
        insufficient_count = sum(1 for c in scorecards
                                 if c.health_state == "INSUFFICIENT_EVIDENCE")
        dead_count = sum(1 for c in scorecards if c.health_state == "DEAD")
        at_risk_count = len(scorecards) - healthy_count - insufficient_count - dead_count

        machine_summary = self._build_machine_summary(scorecards, overall_status)
        regime_decay_map = self._build_regime_decay_map(scorecards)

        near_retirement = [c.name for c in scorecards
                           if c.health_state in ("DEAD", "RETIREMENT_WATCH") or c.hazard_30d > 0.8]
        under_watch = [
            c.name for c in scorecards
            if c.health_state in ("WATCH", "EARLY_DECAY", "REGIME_SUPPRESSED",
                                   "IMPLEMENTATION_DECAY", "ROBUSTNESS_BREAKDOWN",
                                   "SHADOW_FAILURE", "POST_PROMOTION_FAILURE")
        ]

        result = {
            "timestamp": ts_iso,
            "overall_status": overall_status,
            "legacy_decay_level": legacy_level,
            "decay_level": legacy_level,
            "strategy_count": len(scorecards),
            "healthy_count": healthy_count,
            "at_risk_count": at_risk_count,
            "dead_count": dead_count,
            "insufficient_count": insufficient_count,
            "strategy_health_table": [c.to_dict() for c in scorecards],
            "hazard_ranking": [c.name for c in scorecards],
            "regime_decay_map": regime_decay_map,
            "near_retirement": near_retirement,
            "under_watch": under_watch,
            "machine_summary": machine_summary,
            "retirement_watchlist": machine_summary.get("retirement_watchlist", []),
            "post_promotion_failures": machine_summary.get("post_promotion_failures", []),
            # Legacy compat fields
            "metrics": {
                "ic_current": 0,
                "sharpe_current": 0,
                "wr_current": 0.5,
                "n_periods": 0,
            },
            "signals": {},
            "recommendations": [],
        }

        # Fill legacy metrics from first MR strategy
        for c in scorecards:
            if "MR" in c.name.upper() or "WHITELIST" in c.name.upper():
                pm = self._evidence._compute_price_supplemental(c.name)
                if pm:
                    result["metrics"] = {
                        "ic_current": pm.get("ic_current", 0),
                        "sharpe_current": pm.get("sharpe_current", 0),
                        "wr_current": 0.5,
                        "n_periods": pm.get("n_periods", 0),
                    }
                break

        # Legacy recommendations
        recs = []
        for c in scorecards:
            for a in c.actions:
                recs.append(f"{a.get('action', 'UNKNOWN')} ({a.get('urgency', 'LOW')}): {c.name}")
        result["recommendations"] = recs

        # Log summary
        log.info("-" * 78)
        log.info("Overall: %s  (legacy: %s)", overall_status, legacy_level)
        log.info("Healthy=%d  AtRisk=%d  Dead=%d  Insufficient=%d",
                 healthy_count, at_risk_count, dead_count, insufficient_count)
        for c in scorecards:
            log.info(
                "  %-35s  state=%-25s  health=%.3f  h30=%.3f  h90=%.3f  ret=%.3f  conf=%.2f  tags=%s",
                c.name, c.health_state, c.health_score, c.hazard_30d, c.hazard_90d,
                c.retirement_hazard, c.confidence,
                ",".join(c.failure_tags[:3]) if c.failure_tags else "-",
            )
        if machine_summary.get("most_urgent_action"):
            log.info("Most urgent: %s", machine_summary["most_urgent_action"])
        if near_retirement:
            log.info("Near retirement: %s", ", ".join(near_retirement))

        # Save status
        DECAY_STATUS_PATH.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

        # ── Step 11b: Append to history ─────────────────────────────────────
        history_entry = {
            "timestamp": ts_iso,
            "overall_status": overall_status,
            "strategy_health_table": [c.to_dict() for c in scorecards],
        }
        self._history_store.append(history_entry)
        self._history_store.save()

        # ── Step 12: Publish to bus ─────────────────────────────────────────
        try:
            from agents.shared.agent_bus import get_bus
            bus = get_bus()
            bus.publish("alpha_decay", result)
        except Exception:
            pass

        # ── Step 13: Registry heartbeat ─────────────────────────────────────
        try:
            from agents.shared.agent_registry import get_registry, AgentStatus
            reg = get_registry()
            reg.heartbeat("alpha_decay", AgentStatus.COMPLETED)
        except Exception:
            pass

        return result

    # =========================================================================
    # Per-strategy assessment pipeline
    # =========================================================================
    def _assess_strategy(self, name: str, data: Dict, ts_iso: str) -> StrategyHealthScorecard:
        """Run the full assessment pipeline for a single strategy."""
        # Step 2: Assemble evidence
        profile = self._evidence.assemble_evidence(name, data)

        # Step 3: Score evidence quality
        self._eq_engine.score(profile)

        # Step 4: Compute 7 health dimensions
        health_timeline = self._history_store.get_health_scores(name)
        promo_data = self._promo_engine.get_promotion_data(name, 0.5, health_timeline)
        dimensions = self._dim_engine.compute_all(profile, promo_data)

        # Compute composite health score
        health_score = sum(
            dimensions.get(k, 0.5) * w for k, w in DIMENSION_WEIGHTS.items()
        )
        health_score = float(np.clip(health_score, 0.0, 1.0))

        # Update promo data with actual health
        promo_data = self._promo_engine.get_promotion_data(name, health_score, health_timeline)

        # Step 5: Run change-point detection
        full_timeline = health_timeline + [health_score]
        change_point = self._cp_engine.detect(full_timeline)

        # Step 6: Classify health state
        consecutive_below_25 = self._history_store.consecutive_below_threshold(name, 0.25)
        consecutive_below_15 = self._history_store.consecutive_below_threshold(name, 0.15)
        consecutive_below = max(consecutive_below_25, consecutive_below_15)

        lifecycle_stage = self._promo_engine.determine_lifecycle_stage(
            name, "HEALTHY", health_score, consecutive_below
        )

        health_state = self._classifier.classify(
            health_score=health_score,
            dimensions=dimensions,
            evidence_quality=profile.evidence_quality,
            evidence_quality_score=profile.evidence_quality_score,
            change_point=change_point,
            lifecycle_stage=lifecycle_stage,
            consecutive_below_threshold=consecutive_below,
            consecutive_declining=change_point.consecutive_declining,
        )

        # Re-determine lifecycle stage with correct health state
        lifecycle_stage = self._promo_engine.determine_lifecycle_stage(
            name, health_state, health_score, consecutive_below
        )

        # Step 7: Compute hazard
        hazards = self._hazard_model.compute(
            health_score=health_score,
            dimensions=dimensions,
            change_point=change_point,
            evidence_quality_score=profile.evidence_quality_score,
            lifecycle_stage=lifecycle_stage,
            consecutive_below_threshold=consecutive_below,
            rollback_count=promo_data.get("rollback_count", 0),
        )

        # Step 8: Diagnose root causes
        primary_cause, failure_tags = self._root_cause_engine.diagnose(
            dimensions=dimensions,
            health_state=health_state,
            health_score=health_score,
            evidence=profile,
            change_point=change_point,
            promotion_data=promo_data,
        )

        # Step 9: Determine lifecycle actions
        actions = self._action_engine.determine_actions(
            health_state=health_state,
            primary_root_cause=primary_cause,
            failure_tags=failure_tags,
            lifecycle_stage=lifecycle_stage,
            hazard_30d=hazards["hazard_30d"],
        )

        # Step 10: Build scorecard
        legacy_level = SAFE_LEGACY_MAP.get(health_state, "EARLY_DECAY")

        # Confidence
        confidence = float(np.clip(
            (profile.available_source_count() + (1 if profile.has_price_data() else 0)) / 9.0,
            0.0, 1.0
        ))

        # Regime analysis
        regime_decay_states = self._scorer.classify_regime_decay(name, data) if self._scorer else {}
        strongest, weakest = (self._scorer.identify_regime_strengths(name, data)
                              if self._scorer else ([], []))

        scorecard = StrategyHealthScorecard(
            name=name,
            lifecycle_stage=lifecycle_stage,
            health_state=health_state,
            health_score=round(health_score, 4),
            hazard_30d=round(hazards["hazard_30d"], 4),
            hazard_90d=round(hazards["hazard_90d"], 4),
            retirement_hazard=round(hazards["retirement_hazard"], 4),
            confidence=round(confidence, 2),
            evidence_quality_score=round(profile.evidence_quality_score, 4),
            degraded_mode=profile.degraded_mode,
            alpha_integrity=round(dimensions.get("alpha_integrity", 0.5), 4),
            robustness_integrity=round(dimensions.get("robustness_integrity", 0.5), 4),
            regime_integrity=round(dimensions.get("regime_integrity", 0.5), 4),
            implementation_integrity=round(dimensions.get("implementation_integrity", 0.5), 4),
            tail_integrity=round(dimensions.get("tail_integrity", 0.5), 4),
            diversification_integrity=round(dimensions.get("diversification_integrity", 0.5), 4),
            lifecycle_integrity=round(dimensions.get("lifecycle_integrity", 0.5), 4),
            primary_root_cause=primary_cause,
            failure_tags=failure_tags,
            actions=actions,
            evidence_gaps=profile.evidence_gaps,
            legacy_decay_level=legacy_level,
            change_point=asdict(change_point),
            regime_decay_states=regime_decay_states,
            strongest_regimes=strongest,
            weakest_regimes=weakest,
            promotion_instability=round(hazards["promotion_instability"], 4),
            rollback_count=promo_data.get("rollback_count", 0),
            post_promotion_decline=promo_data.get("post_promotion_decline", False),
            timestamp=ts_iso,
        )

        # Step 9b: Build downstream contracts
        contracts = self._contract_builder.build(scorecard, actions)
        scorecard.downstream_contracts = contracts

        return scorecard

    def _count_available_sources(self) -> int:
        """Count how many evidence sources are loaded."""
        count = 0
        if self._evidence.methodology_reports:
            count += 1
        if self._evidence.optimizer_history:
            count += 1
        if self._evidence.regime_forecast:
            count += 1
        if self._evidence.portfolio_weights:
            count += 1
        if self._evidence.risk_status:
            count += 1
        if self._evidence.execution_state:
            count += 1
        if self._evidence.scout_report:
            count += 1
        if self._evidence.math_proposals:
            count += 1
        if self._evidence.prices is not None:
            count += 1
        return count


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Alpha Decay Governance Engine -- Strategy Retirement Authority"
    )
    parser.add_argument("--once", action="store_true",
                        help="Run one health assessment cycle")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("alpha_decay").setLevel(logging.DEBUG)

    if args.once:
        monitor = AlphaDecayMonitor()
        result = monitor.run()
        level = result.get("overall_status", result.get("decay_level", "UNKNOWN"))
        print(f"\nOverall Status: {level}")
        print(f"Legacy Level:   {result.get('decay_level', 'UNKNOWN')}")
        print(f"Strategies:     {result.get('strategy_count', 0)}  "
              f"Healthy: {result.get('healthy_count', 0)}  "
              f"At-Risk: {result.get('at_risk_count', 0)}  "
              f"Dead: {result.get('dead_count', 0)}  "
              f"Insufficient: {result.get('insufficient_count', 0)}")
        if result.get("near_retirement"):
            print(f"Near Retirement: {', '.join(result['near_retirement'])}")
        if result.get("under_watch"):
            print(f"Under Watch:     {', '.join(result['under_watch'])}")
        ms = result.get("machine_summary", {})
        if ms.get("most_urgent_action"):
            print(f"Most Urgent:     {ms['most_urgent_action']}")
    else:
        print("Usage: python -m agents.alpha_decay --once")
        print("       python -m agents.alpha_decay --once --verbose")


if __name__ == "__main__":
    main()
