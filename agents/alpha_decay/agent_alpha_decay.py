"""
Alpha Decay Monitor -- Institutional Strategy Health & Death Engine.

Multi-dimensional health scoring across 6 axes:
  alpha_quality, robustness, regime_fitness, implementation, tail_risk, diversification

Health states (7):
  HEALTHY, UNDER_OBSERVATION, EARLY_DECAY, REGIME_SUPPRESSED,
  IMPLEMENTATION_DECAY, STRUCTURAL_DECAY, DEAD, INSUFFICIENT_EVIDENCE

Backward-compatible with the legacy HEALTHY / EARLY_DECAY / DECAYING / DEAD taxonomy.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
REPORTS_DIR = ROOT / "agents" / "methodology" / "reports"
OPTIMIZER_HISTORY = ROOT / "agents" / "optimizer" / "optimization_history.json"
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
DIMENSION_WEIGHTS = {
    "alpha_quality": 0.30,
    "robustness": 0.25,
    "regime_fitness": 0.20,
    "implementation": 0.10,
    "tail_risk": 0.10,
    "diversification": 0.05,
}

HEALTH_STATES = [
    "HEALTHY",
    "UNDER_OBSERVATION",
    "EARLY_DECAY",
    "REGIME_SUPPRESSED",
    "IMPLEMENTATION_DECAY",
    "STRUCTURAL_DECAY",
    "DEAD",
    "INSUFFICIENT_EVIDENCE",
]

LEGACY_MAP = {
    "HEALTHY": "HEALTHY",
    "UNDER_OBSERVATION": "HEALTHY",
    "EARLY_DECAY": "EARLY_DECAY",
    "REGIME_SUPPRESSED": "EARLY_DECAY",
    "IMPLEMENTATION_DECAY": "DECAYING",
    "ROBUSTNESS_DECAY": "DECAYING",
    "STRUCTURAL_DECAY": "DECAYING",
    "DEAD": "DEAD",
    "INSUFFICIENT_EVIDENCE": "HEALTHY",
}

# Regime-specific decay state constants
REGIME_DECAY_STATES = [
    "CALM_DECAY", "NORMAL_DECAY", "TENSION_BREAKDOWN",
    "CRISIS_FAILURE", "TRANSITION_FRAGILITY", "HEALTHY",
]

SECTORS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
MR_WHITELIST = {"XLC", "XLF", "XLI", "XLU"}

MAX_HISTORY_ENTRIES = 365

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecayDimensions:
    """Six health dimensions, each in [0.0, 1.0] (higher = healthier)."""
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


@dataclass
class StrategyHealthCard:
    """Complete health assessment for a single strategy."""
    name: str = ""
    health_state: str = "INSUFFICIENT_EVIDENCE"
    health_score: float = 0.5
    hazard_score_30d: float = 0.5
    hazard_score_90d: float = 0.5
    dimensions: DecayDimensions = field(default_factory=DecayDimensions)
    failure_tags: List[str] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    evidence_quality_score: float = 0.0
    legacy_decay_level: str = "HEALTHY"
    timestamp: str = ""
    # Regime-specific decay states
    regime_decay_states: Dict[str, str] = field(default_factory=dict)  # e.g. {"CALM": "CALM_DECAY", "TENSION": "TENSION_BREAKDOWN"}
    strongest_regimes: List[str] = field(default_factory=list)
    weakest_regimes: List[str] = field(default_factory=list)
    # Lifecycle tracking
    champion_watch: bool = False
    shadow_watch: bool = False
    post_promotion_watch: bool = False
    under_watch: bool = False
    near_retirement: bool = False
    root_causes: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ============================================================================
# 1. StrategyEvidenceLoader
# ============================================================================
class StrategyEvidenceLoader:
    """Loads all evidence sources for health scoring."""

    def __init__(self) -> None:
        self.prices: Optional[pd.DataFrame] = None
        self.methodology_report: Optional[Dict] = None
        self.optimizer_history: Optional[Dict] = None
        self.previous_status: Optional[Dict] = None
        self.previous_history: List[Dict] = []

    def load_all(self) -> None:
        self._load_prices()
        self._load_methodology()
        self._load_optimizer_history()
        self._load_previous_status()
        self._load_previous_history()

    # -- prices ---------------------------------------------------------------
    def _load_prices(self) -> None:
        try:
            if PRICES_PATH.exists():
                self.prices = pd.read_parquet(PRICES_PATH)
                log.info("Loaded prices: %d rows x %d cols", len(self.prices), len(self.prices.columns))
        except Exception as exc:
            log.warning("Failed to load prices: %s", exc)

    # -- methodology report ---------------------------------------------------
    def _load_methodology(self) -> None:
        try:
            if not REPORTS_DIR.exists():
                return
            files = sorted(REPORTS_DIR.glob("*_methodology_lab.json"), reverse=True)
            if files:
                self.methodology_report = json.loads(files[0].read_text(encoding="utf-8"))
                log.info("Loaded methodology report: %s", files[0].name)
        except Exception as exc:
            log.warning("Failed to load methodology report: %s", exc)

    # -- optimizer history ----------------------------------------------------
    def _load_optimizer_history(self) -> None:
        try:
            if OPTIMIZER_HISTORY.exists():
                self.optimizer_history = json.loads(OPTIMIZER_HISTORY.read_text(encoding="utf-8"))
                log.info("Loaded optimizer history")
        except Exception as exc:
            log.warning("Failed to load optimizer history: %s", exc)

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
                self.previous_history = json.loads(DECAY_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to load previous history: %s", exc)


# ============================================================================
# 2. HealthScorer
# ============================================================================
class HealthScorer:
    """Compute 6-dimensional health scores per strategy."""

    def __init__(self, evidence: StrategyEvidenceLoader) -> None:
        self.ev = evidence

    # -- public ---------------------------------------------------------------
    def score_strategy(self, strategy_name: str, strategy_data: Dict) -> Tuple[DecayDimensions, float, float]:
        """Return (dimensions, health_score, confidence)."""
        dims = DecayDimensions(
            alpha_quality=self._score_alpha_quality(strategy_name, strategy_data),
            robustness=self._score_robustness(strategy_name, strategy_data),
            regime_fitness=self._score_regime_fitness(strategy_name, strategy_data),
            implementation=self._score_implementation(strategy_name, strategy_data),
            tail_risk=self._score_tail_risk(strategy_name, strategy_data),
            diversification=self._score_diversification(strategy_name, strategy_data),
        )
        health = sum(
            getattr(dims, k) * w for k, w in DIMENSION_WEIGHTS.items()
        )
        health = float(np.clip(health, 0.0, 1.0))
        confidence = self._estimate_confidence(strategy_name, strategy_data)
        return dims, health, confidence

    # -- dimension scorers (all return 0.0-1.0) --------------------------------

    def _score_alpha_quality(self, name: str, data: Dict) -> float:
        """IC trend, net Sharpe trend, hit rate trend."""
        scores: List[float] = []
        # From methodology report
        sharpe = data.get("sharpe", data.get("net_sharpe", 0))
        ic = data.get("ic_mean", data.get("mean_ic", 0))
        hit_rate = data.get("hit_rate", data.get("win_rate", 0.5))

        # Sharpe contribution: 0-2 range mapped to 0-1
        if isinstance(sharpe, (int, float)) and math.isfinite(sharpe):
            scores.append(float(np.clip(sharpe / 2.0, 0.0, 1.0)))
        # IC contribution: 0-0.1 mapped to 0-1
        if isinstance(ic, (int, float)) and math.isfinite(ic):
            scores.append(float(np.clip(ic / 0.1, 0.0, 1.0)))
        # Hit rate: 0.5 -> 0.5 score, 0.7 -> 1.0
        if isinstance(hit_rate, (int, float)) and math.isfinite(hit_rate):
            scores.append(float(np.clip((hit_rate - 0.3) / 0.4, 0.0, 1.0)))

        # Rolling metrics from prices if available
        price_metrics = self._compute_price_metrics(name)
        if price_metrics:
            ic_cur = price_metrics.get("ic_current", 0)
            sharpe_cur = price_metrics.get("sharpe_current", 0)
            if isinstance(ic_cur, (int, float)) and math.isfinite(ic_cur):
                scores.append(float(np.clip(ic_cur / 0.05 * 0.5 + 0.5, 0.0, 1.0)))
            if isinstance(sharpe_cur, (int, float)) and math.isfinite(sharpe_cur):
                scores.append(float(np.clip(sharpe_cur / 2.0, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    def _score_robustness(self, name: str, data: Dict) -> float:
        """Walk-forward consistency, fold variance, stability."""
        scores: List[float] = []
        # Walk-forward stats from methodology
        wf = data.get("walk_forward", data.get("wf_stats", {}))
        if isinstance(wf, dict):
            wf_sharpe = wf.get("mean_sharpe", wf.get("sharpe", 0))
            wf_std = wf.get("std_sharpe", wf.get("sharpe_std", 1))
            if isinstance(wf_sharpe, (int, float)) and math.isfinite(wf_sharpe):
                scores.append(float(np.clip(wf_sharpe / 1.5, 0.0, 1.0)))
            if isinstance(wf_std, (int, float)) and wf_std > 0 and math.isfinite(wf_std):
                # Lower variance = better. std < 0.3 => 1.0, std > 1.0 => 0.0
                scores.append(float(np.clip(1.0 - wf_std / 1.0, 0.0, 1.0)))

        # Fold count / sufficient data
        n_folds = data.get("n_folds", data.get("fold_count", 0))
        if isinstance(n_folds, (int, float)):
            scores.append(float(np.clip(n_folds / 5.0, 0.0, 1.0)))

        # Stability metrics
        stability = data.get("stability_score", data.get("stability", None))
        if isinstance(stability, (int, float)) and math.isfinite(stability):
            scores.append(float(np.clip(stability, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    def _score_regime_fitness(self, name: str, data: Dict) -> float:
        """Performance across regimes, regime breadth."""
        scores: List[float] = []
        regimes = data.get("regime_performance", data.get("regimes", {}))
        if isinstance(regimes, dict) and regimes:
            regime_sharpes = []
            for regime_name, rdata in regimes.items():
                rs = rdata if isinstance(rdata, (int, float)) else (
                    rdata.get("sharpe", 0) if isinstance(rdata, dict) else 0
                )
                if isinstance(rs, (int, float)) and math.isfinite(rs):
                    regime_sharpes.append(rs)
            if regime_sharpes:
                # Mean regime sharpe
                mean_rs = float(np.mean(regime_sharpes))
                scores.append(float(np.clip(mean_rs / 1.5, 0.0, 1.0)))
                # Breadth: fraction of regimes with positive sharpe
                pos_frac = sum(1 for s in regime_sharpes if s > 0) / len(regime_sharpes)
                scores.append(pos_frac)
                # Worst regime penalty
                worst = min(regime_sharpes)
                scores.append(float(np.clip((worst + 0.5) / 1.0, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    def _score_implementation(self, name: str, data: Dict) -> float:
        """Cost drag, turnover, net vs gross gap."""
        scores: List[float] = []
        gross_sharpe = data.get("gross_sharpe", data.get("sharpe_gross", None))
        net_sharpe = data.get("net_sharpe", data.get("sharpe", None))

        if (isinstance(gross_sharpe, (int, float)) and isinstance(net_sharpe, (int, float))
                and math.isfinite(gross_sharpe) and math.isfinite(net_sharpe)):
            # Gap: smaller = better
            gap = gross_sharpe - net_sharpe
            scores.append(float(np.clip(1.0 - gap / 0.5, 0.0, 1.0)))

        turnover = data.get("turnover", data.get("annual_turnover", None))
        if isinstance(turnover, (int, float)) and math.isfinite(turnover):
            # Lower turnover better: <50 => 1.0, >200 => 0.0
            scores.append(float(np.clip(1.0 - turnover / 200.0, 0.0, 1.0)))

        cost_drag = data.get("cost_drag", data.get("cost_bps", None))
        if isinstance(cost_drag, (int, float)) and math.isfinite(cost_drag):
            scores.append(float(np.clip(1.0 - cost_drag / 100.0, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.7  # default generous

    def _score_tail_risk(self, name: str, data: Dict) -> float:
        """Drawdown profile, skewness, crisis survival."""
        scores: List[float] = []
        max_dd = data.get("max_drawdown", data.get("max_dd", None))
        if isinstance(max_dd, (int, float)) and math.isfinite(max_dd):
            dd_abs = abs(max_dd)
            scores.append(float(np.clip(1.0 - dd_abs / 0.30, 0.0, 1.0)))

        skew = data.get("skewness", data.get("return_skew", None))
        if isinstance(skew, (int, float)) and math.isfinite(skew):
            # Positive skew good, negative bad. Range roughly -2 to +2
            scores.append(float(np.clip((skew + 1.0) / 2.0, 0.0, 1.0)))

        crisis = data.get("crisis_performance", data.get("crisis_sharpe", None))
        if isinstance(crisis, (int, float)) and math.isfinite(crisis):
            scores.append(float(np.clip((crisis + 0.5) / 1.5, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    def _score_diversification(self, name: str, data: Dict) -> float:
        """Correlation to other strategies, incremental alpha."""
        scores: List[float] = []
        corr = data.get("avg_correlation", data.get("correlation_to_portfolio", None))
        if isinstance(corr, (int, float)) and math.isfinite(corr):
            # Lower correlation = better diversification
            scores.append(float(np.clip(1.0 - abs(corr), 0.0, 1.0)))

        incr_alpha = data.get("incremental_alpha", data.get("marginal_alpha", None))
        if isinstance(incr_alpha, (int, float)) and math.isfinite(incr_alpha):
            scores.append(float(np.clip(incr_alpha / 0.05, 0.0, 1.0)))

        return float(np.mean(scores)) if scores else 0.5

    # -- helpers ---------------------------------------------------------------

    def _compute_price_metrics(self, strategy_name: str) -> Optional[Dict]:
        """Compute rolling IC/Sharpe from raw prices for the MR whitelist."""
        prices = self.ev.prices
        if prices is None or len(prices) < 123:
            return None
        # Only for MR-like strategies
        if "MR" not in strategy_name.upper() and "WHITELIST" not in strategy_name.upper():
            return None
        try:
            spy = prices.get("SPY", pd.Series(dtype=float)).dropna()
            sectors = [s for s in MR_WHITELIST if s in prices.columns]
            if not sectors or spy.empty:
                return None
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
                "sharpe_current": sharpe_list[-1] if sharpe_list else 0,
                "n_periods": len(ic_list),
            }
        except Exception as exc:
            log.debug("Price metrics failed for %s: %s", strategy_name, exc)
            return None

    def _estimate_confidence(self, name: str, data: Dict) -> float:
        """Confidence in the health score based on evidence availability."""
        evidence_count = 0
        if data.get("sharpe") is not None or data.get("net_sharpe") is not None:
            evidence_count += 1
        if data.get("walk_forward") or data.get("wf_stats"):
            evidence_count += 1
        if data.get("regime_performance") or data.get("regimes"):
            evidence_count += 1
        if data.get("max_drawdown") is not None or data.get("max_dd") is not None:
            evidence_count += 1
        if self.ev.prices is not None:
            evidence_count += 1
        return float(np.clip(evidence_count / 5.0, 0.0, 1.0))

    def estimate_evidence_quality(self, name: str, data: Dict) -> float:
        """Evidence quality score: how much data do we actually have?"""
        score = 0.0
        # Methodology report coverage
        if self.ev.methodology_results and name in self.ev.methodology_results:
            score += 0.25
        # Trade count sufficiency
        trades = data.get("total_trades", 0)
        if trades >= 100:
            score += 0.25
        elif trades >= 50:
            score += 0.15
        elif trades >= 20:
            score += 0.05
        # Regime coverage
        regimes = data.get("regime_performance") or data.get("regimes") or {}
        if len(regimes) >= 3:
            score += 0.25
        elif len(regimes) >= 2:
            score += 0.15
        # Price history
        if self.ev.prices is not None and len(self.ev.prices) >= 500:
            score += 0.25
        elif self.ev.prices is not None:
            score += 0.10
        return float(np.clip(score, 0.0, 1.0))

    def classify_regime_decay(self, name: str, data: Dict) -> Dict[str, str]:
        """Classify per-regime decay states: CALM_DECAY, TENSION_BREAKDOWN, etc."""
        regime_states: Dict[str, str] = {}
        regimes = data.get("regime_performance") or data.get("regimes") or {}
        for regime_name, metrics in regimes.items():
            if isinstance(metrics, dict):
                wr = metrics.get("win_rate", metrics.get("wr", 0.5))
                sh = metrics.get("sharpe", 0)
            else:
                continue
            # Classify
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

    def identify_regime_strengths(self, name: str, data: Dict) -> tuple:
        """Identify strongest and weakest regimes for a strategy."""
        regimes = data.get("regime_performance") or data.get("regimes") or {}
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


# ============================================================================
# 3. Health State Classifier
# ============================================================================
class HealthStateClassifier:
    """Classify strategy health state from dimensions and scores."""

    def classify(
        self,
        dims: DecayDimensions,
        health_score: float,
        confidence: float,
        consecutive_dead: int = 0,
    ) -> str:
        if confidence < 0.2:
            return "INSUFFICIENT_EVIDENCE"

        # DEAD: health < 0.2 for 3+ consecutive observations
        if health_score < 0.2 and consecutive_dead >= 2:
            return "DEAD"

        # STRUCTURAL_DECAY: multiple dimensions failing
        failing = sum(1 for v in dims.values() if v < 0.4)
        if failing >= 3:
            return "STRUCTURAL_DECAY"

        # ROBUSTNESS_DECAY: robustness collapsing while alpha still OK
        if dims.robustness < 0.30 and dims.alpha_quality >= 0.45:
            return "ROBUSTNESS_DECAY"

        # IMPLEMENTATION_DECAY: gross OK but net collapsing
        if dims.implementation < 0.35 and dims.alpha_quality >= 0.5:
            return "IMPLEMENTATION_DECAY"

        # REGIME_SUPPRESSED: healthy overall but regime_fitness weak
        if dims.regime_fitness < 0.35 and dims.alpha_quality >= 0.5 and dims.robustness >= 0.5:
            return "REGIME_SUPPRESSED"

        # EARLY_DECAY: health dropping, alpha or robustness weak
        if health_score < 0.45 or dims.alpha_quality < 0.4 or dims.robustness < 0.4:
            return "EARLY_DECAY"

        # UNDER_OBSERVATION: 1-2 dimensions declining
        declining = sum(1 for v in dims.values() if v < 0.5)
        if declining >= 1 and declining <= 2:
            return "UNDER_OBSERVATION"

        # HEALTHY: all dimensions > 0.6 preferred, but acceptable if score is good
        if all(v >= 0.5 for v in dims.values()) and health_score >= 0.5:
            return "HEALTHY"

        return "UNDER_OBSERVATION"


# ============================================================================
# 4. RootCauseEngine
# ============================================================================
class RootCauseEngine:
    """Generate failure tags for non-healthy strategies."""

    TAG_RULES: List[Tuple[str, str, float]] = [
        # (tag, dimension_or_field, threshold_below_which_tag_fires)
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

    def diagnose(
        self,
        dims: DecayDimensions,
        health_state: str,
        confidence: float,
        strategy_data: Dict,
    ) -> List[str]:
        if health_state == "HEALTHY":
            return []

        tags: List[str] = []
        dim_dict = dims.to_dict()

        for tag, dim_key, threshold in self.TAG_RULES:
            val = dim_dict.get(dim_key, 1.0)
            if val < threshold:
                tags.append(tag)

        if confidence < 0.3:
            tags.append("insufficient_sample")

        # Check if degraded after optimization
        if strategy_data.get("optimized_recently", False) and health_state in (
            "EARLY_DECAY", "STRUCTURAL_DECAY", "IMPLEMENTATION_DECAY"
        ):
            tags.append("degraded_after_optimization")

        return sorted(set(tags))


# ============================================================================
# 5. ActionPolicyEngine
# ============================================================================
class ActionPolicyEngine:
    """Deterministic action policies per health state."""

    ACTIONS: Dict[str, List[Tuple]] = {
        "HEALTHY": [("KEEP_LIVE", "LOW")],
        "UNDER_OBSERVATION": [("KEEP_UNDER_OBSERVATION", "LOW"), ("REDUCE_EXPOSURE", "LOW")],
        "EARLY_DECAY": [("REDUCE_EXPOSURE", "MEDIUM"), ("SEND_TO_OPTIMIZER", "MEDIUM")],
        "REGIME_SUPPRESSED": [("REGIME_DISABLE", "MEDIUM")],
        "IMPLEMENTATION_DECAY": [("SEND_TO_OPTIMIZER", "HIGH", "cost_reduction")],
        "STRUCTURAL_DECAY": [("FREEZE_PROMOTION", "HIGH"), ("REQUIRE_METHOD_REVIEW", "HIGH")],
        "DEAD": [("RETIRE_STRATEGY", "CRITICAL")],
        "INSUFFICIENT_EVIDENCE": [("KEEP_UNDER_OBSERVATION", "LOW")],
    }

    def get_actions(self, health_state: str) -> List[Dict[str, str]]:
        raw = self.ACTIONS.get(health_state, [("KEEP_UNDER_OBSERVATION", "LOW")])
        actions = []
        for entry in raw:
            action_dict: Dict[str, str] = {"action": entry[0], "urgency": entry[1]}
            if len(entry) > 2:
                action_dict["focus"] = entry[2]
            actions.append(action_dict)
        return actions


# ============================================================================
# 6. DecayHistoryTracker
# ============================================================================
class DecayHistoryTracker:
    """Append-only JSON history for decay observations."""

    def __init__(self, history: Optional[List[Dict]] = None) -> None:
        self._history: List[Dict] = history if history is not None else []

    def append(self, entry: Dict) -> None:
        self._history.append(entry)
        # Cap at MAX_HISTORY_ENTRIES
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

    def get_strategy_history(self, name: str, n: int = 30) -> List[Dict]:
        """Get last n history entries for a strategy."""
        entries = []
        for record in self._history:
            table = record.get("strategy_health_table", [])
            for strat in table:
                if strat.get("name") == name:
                    entries.append({
                        "timestamp": record.get("timestamp"),
                        "health_score": strat.get("health_score"),
                        "health_state": strat.get("health_state"),
                        "hazard_score_30d": strat.get("hazard_score_30d"),
                    })
                    break
        return entries[-n:]

    def get_hazard_trend(self, name: str, n: int = 10) -> List[float]:
        """Return last n hazard scores for a strategy."""
        hist = self.get_strategy_history(name, n)
        return [h.get("hazard_score_30d", 0.5) for h in hist]

    def latest_decay_events(self) -> List[Dict]:
        """Strategies that transitioned to a worse state in the last entry."""
        if len(self._history) < 2:
            return []
        prev_table = {s["name"]: s for s in self._history[-2].get("strategy_health_table", [])}
        curr_table = self._history[-1].get("strategy_health_table", [])
        events = []
        worse_order = {s: i for i, s in enumerate(HEALTH_STATES)}
        for strat in curr_table:
            prev = prev_table.get(strat["name"])
            if prev and worse_order.get(strat["health_state"], 0) > worse_order.get(prev["health_state"], 0):
                events.append({
                    "name": strat["name"],
                    "from": prev["health_state"],
                    "to": strat["health_state"],
                })
        return events

    def strategies_near_retirement(self) -> List[str]:
        """Strategies with DEAD state or hazard > 0.8 in recent history."""
        if not self._history:
            return []
        latest = self._history[-1].get("strategy_health_table", [])
        return [
            s["name"] for s in latest
            if s.get("health_state") == "DEAD" or s.get("hazard_score_30d", 0) > 0.8
        ]

    def consecutive_dead_count(self, name: str) -> int:
        """Count consecutive observations where health_score < 0.2."""
        count = 0
        for record in reversed(self._history):
            table = record.get("strategy_health_table", [])
            found = False
            for strat in table:
                if strat.get("name") == name:
                    if strat.get("health_score", 1.0) < 0.2:
                        count += 1
                        found = True
                    else:
                        return count
                    break
            if not found:
                break
        return count


# ============================================================================
# 7. AlphaDecayMonitor (main orchestrator)
# ============================================================================
class AlphaDecayMonitor:
    """
    Institutional Strategy Health & Death Engine.

    Backward-compatible: keeps `run()` returning a dict with `decay_level`.
    """

    SECTORS = SECTORS
    MR_WHITELIST = MR_WHITELIST

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._evidence = StrategyEvidenceLoader()
        self._scorer = HealthScorer(self._evidence)
        self._classifier = HealthStateClassifier()
        self._root_cause = RootCauseEngine()
        self._action_policy = ActionPolicyEngine()
        self._history_tracker: Optional[DecayHistoryTracker] = None

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
        return self._evidence.methodology_report

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
        """
        Extract strategy list from methodology report.
        Falls back to a single MR whitelist strategy from prices.
        """
        strategies: Dict[str, Dict] = {}

        report = self._evidence.methodology_report
        if report:
            # Try 'strategies' key first
            strats = report.get("strategies", {})
            if isinstance(strats, dict):
                strategies.update(strats)
            # Try 'results' key
            results = report.get("results", {})
            if isinstance(results, dict):
                for key, val in results.items():
                    if isinstance(val, dict) and key not in strategies:
                        strategies[key] = val
            # Try 'strategy_rankings' or 'rankings'
            rankings = report.get("strategy_rankings", report.get("rankings", []))
            if isinstance(rankings, list):
                for item in rankings:
                    if isinstance(item, dict) and "name" in item:
                        name = item["name"]
                        if name not in strategies:
                            strategies[name] = item

        # Fallback: create a synthetic MR whitelist entry from prices
        if not strategies and self._evidence.prices is not None:
            strategies["ALPHA_WHITELIST_MR"] = {
                "sharpe": 0,
                "hit_rate": 0.5,
                "_source": "price_fallback",
            }

        return strategies

    # -- hazard score with windowing -------------------------------------------

    def _compute_hazard_scores(
        self, strategy_name: str, health_score: float
    ) -> Tuple[float, float]:
        """Compute 30d and 90d hazard scores using history."""
        hazard_30d = 1.0 - health_score
        hazard_90d = 1.0 - health_score

        if self._history_tracker:
            trend = self._history_tracker.get_hazard_trend(strategy_name, 6)  # ~30d
            if trend:
                hazard_30d = float(np.mean(trend + [1.0 - health_score]))
            trend_90 = self._history_tracker.get_hazard_trend(strategy_name, 18)  # ~90d
            if trend_90:
                hazard_90d = float(np.mean(trend_90 + [1.0 - health_score]))

        return round(hazard_30d, 4), round(hazard_90d, 4)

    # -- overall status --------------------------------------------------------

    @staticmethod
    def _compute_overall_status(cards: List[StrategyHealthCard]) -> str:
        """Determine overall portfolio health status."""
        states = [c.health_state for c in cards]
        if any(s == "DEAD" for s in states):
            return "STRUCTURAL_DECAY"
        if any(s in ("STRUCTURAL_DECAY", "IMPLEMENTATION_DECAY") for s in states):
            return "EARLY_DECAY"
        if any(s in ("EARLY_DECAY", "REGIME_SUPPRESSED") for s in states):
            return "UNDER_OBSERVATION"
        if any(s == "UNDER_OBSERVATION" for s in states):
            return "UNDER_OBSERVATION"
        return "HEALTHY"

    @staticmethod
    def _compute_legacy_level(cards: List[StrategyHealthCard]) -> str:
        """Worst legacy level across all strategies."""
        legacy_order = {"HEALTHY": 0, "EARLY_DECAY": 1, "DECAYING": 2, "DEAD": 3}
        worst = "HEALTHY"
        for c in cards:
            leg = c.legacy_decay_level
            if legacy_order.get(leg, 0) > legacy_order.get(worst, 0):
                worst = leg
        return worst

    # -- machine summary -------------------------------------------------------

    def _build_machine_summary(
        self, cards: List[StrategyHealthCard], overall_status: str
    ) -> Dict[str, Any]:
        at_risk = [c for c in cards if c.health_state not in ("HEALTHY", "INSUFFICIENT_EVIDENCE")]
        most_urgent = None
        if at_risk:
            worst = max(at_risk, key=lambda c: c.hazard_score_30d)
            if worst.actions:
                most_urgent = f"{worst.actions[0]['action']} on {worst.name}"

        regime_warnings: List[str] = []
        regime_weak = [c for c in cards if c.health_state == "REGIME_SUPPRESSED"]
        if regime_weak:
            regime_warnings.append(
                f"Regime-suppressed strategies: {', '.join(c.name for c in regime_weak)}"
            )

        optimizer_instructions: Dict[str, List[str]] = {
            "cost_reduction": [],
            "regime_repair": [],
        }
        for c in cards:
            if c.health_state == "IMPLEMENTATION_DECAY":
                optimizer_instructions["cost_reduction"].append(c.name)
            if c.health_state == "REGIME_SUPPRESSED":
                optimizer_instructions["regime_repair"].append(c.name)

        return {
            "overall_health": overall_status,
            "strategies_at_risk": len(at_risk),
            "most_urgent_action": most_urgent,
            "regime_warnings": regime_warnings,
            "optimizer_instructions": optimizer_instructions,
        }

    # -- regime decay map ------------------------------------------------------

    @staticmethod
    def _build_regime_decay_map(cards: List[StrategyHealthCard]) -> Dict[str, List[str]]:
        """Map regimes to strategies that are weak in them."""
        regime_map: Dict[str, List[str]] = {}
        for c in cards:
            if c.dimensions.regime_fitness < 0.4:
                regime_map.setdefault("WEAK_REGIME", []).append(c.name)
        return regime_map

    # =========================================================================
    # run() -- main entry point
    # =========================================================================
    def run(self) -> Dict:
        """Full health assessment cycle."""
        ts = datetime.now(timezone.utc)
        ts_iso = ts.isoformat()

        log.info("=" * 70)
        log.info("ALPHA DECAY MONITOR -- Strategy Health Engine -- %s", ts_iso[:19])
        log.info("=" * 70)

        # 1. Load evidence
        self._evidence.load_all()
        self._history_tracker = DecayHistoryTracker(list(self._evidence.previous_history))

        # 2. Extract strategies
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

        log.info("Assessing %d strategies", len(strategies))

        # 3. Score each strategy
        cards: List[StrategyHealthCard] = []
        for name, data in strategies.items():
            dims, health, confidence = self._scorer.score_strategy(name, data)
            consecutive_dead = self._history_tracker.consecutive_dead_count(name)
            hazard_30d, hazard_90d = self._compute_hazard_scores(name, health)

            state = self._classifier.classify(dims, health, confidence, consecutive_dead)
            failure_tags = self._root_cause.diagnose(dims, state, confidence, data)
            actions = self._action_policy.get_actions(state)
            legacy = LEGACY_MAP.get(state, "HEALTHY")

            # Evidence quality and regime analysis
            ev_quality = self._scorer.estimate_evidence_quality(name, data)
            regime_decay_states = self._scorer.classify_regime_decay(name, data)
            strongest, weakest = self._scorer.identify_regime_strengths(name, data)

            card = StrategyHealthCard(
                name=name,
                health_state=state,
                health_score=round(health, 4),
                hazard_score_30d=hazard_30d,
                hazard_score_90d=hazard_90d,
                dimensions=dims,
                failure_tags=failure_tags,
                actions=actions,
                confidence=round(confidence, 2),
                evidence_quality_score=round(ev_quality, 2),
                legacy_decay_level=legacy,
                timestamp=ts_iso,
                regime_decay_states=regime_decay_states,
                strongest_regimes=strongest,
                weakest_regimes=weakest,
                champion_watch=state in ("HEALTHY", "UNDER_OBSERVATION"),
                shadow_watch=state in ("EARLY_DECAY", "REGIME_SUPPRESSED"),
                post_promotion_watch=state in ("IMPLEMENTATION_DECAY", "ROBUSTNESS_DECAY"),
                under_watch=state not in ("HEALTHY", "DEAD", "INSUFFICIENT_EVIDENCE"),
                near_retirement=state in ("STRUCTURAL_DECAY", "DEAD"),
                root_causes=failure_tags,
            )
            cards.append(card)

            log.info(
                "  %-35s  state=%-22s  health=%.3f  hazard30=%.3f  conf=%.2f  tags=%s",
                name, state, health, hazard_30d, confidence,
                ",".join(failure_tags) if failure_tags else "-",
            )

        # 4. Sort by hazard (worst first)
        cards.sort(key=lambda c: c.hazard_score_30d, reverse=True)

        # 5. Aggregate
        overall_status = self._compute_overall_status(cards)
        legacy_level = self._compute_legacy_level(cards)

        healthy_count = sum(1 for c in cards if c.health_state in ("HEALTHY", "INSUFFICIENT_EVIDENCE"))
        dead_count = sum(1 for c in cards if c.health_state == "DEAD")
        at_risk_count = len(cards) - healthy_count - dead_count

        machine_summary = self._build_machine_summary(cards, overall_status)
        regime_decay_map = self._build_regime_decay_map(cards)

        near_retirement = [c.name for c in cards if c.health_state == "DEAD" or c.hazard_score_30d > 0.8]
        under_watch = [
            c.name for c in cards
            if c.health_state in ("UNDER_OBSERVATION", "EARLY_DECAY", "REGIME_SUPPRESSED")
        ]

        # 6. Build result
        result = {
            "timestamp": ts_iso,
            "overall_status": overall_status,
            "legacy_decay_level": legacy_level,
            # Backward compat field
            "decay_level": legacy_level,
            "strategy_count": len(cards),
            "healthy_count": healthy_count,
            "at_risk_count": at_risk_count,
            "dead_count": dead_count,
            "strategy_health_table": [c.to_dict() for c in cards],
            "hazard_ranking": [c.name for c in cards],
            "regime_decay_map": regime_decay_map,
            "near_retirement": near_retirement,
            "under_watch": under_watch,
            "machine_summary": machine_summary,
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

        # Fill legacy metrics from first MR strategy card
        for c in cards:
            if "MR" in c.name.upper() or "WHITELIST" in c.name.upper():
                pm = self._scorer._compute_price_metrics(c.name)
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
        for c in cards:
            for a in c.actions:
                recs.append(f"{a['action']} ({a['urgency']}): {c.name}")
        result["recommendations"] = recs

        log.info("-" * 70)
        log.info("Overall: %s  (legacy: %s)", overall_status, legacy_level)
        log.info("Healthy=%d  AtRisk=%d  Dead=%d", healthy_count, at_risk_count, dead_count)
        if machine_summary.get("most_urgent_action"):
            log.info("Most urgent: %s", machine_summary["most_urgent_action"])

        # 7. Save status
        DECAY_STATUS_PATH.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

        # 8. Append to history
        history_entry = {
            "timestamp": ts_iso,
            "overall_status": overall_status,
            "strategy_health_table": [c.to_dict() for c in cards],
        }
        self._history_tracker.append(history_entry)
        self._history_tracker.save()

        # 9. Publish to agent bus
        try:
            from agents.shared.agent_bus import get_bus
            bus = get_bus()
            bus.publish("alpha_decay", result)
        except Exception:
            pass

        # 10. Registry heartbeat
        try:
            from agents.shared.agent_registry import get_registry, AgentStatus
            reg = get_registry()
            reg.heartbeat("alpha_decay", AgentStatus.COMPLETED)
        except Exception:
            pass

        return result


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Alpha Decay Monitor -- Strategy Health Engine")
    parser.add_argument("--once", action="store_true", help="Run one health assessment cycle")
    args = parser.parse_args()

    if args.once:
        monitor = AlphaDecayMonitor()
        result = monitor.run()
        level = result.get("overall_status", result.get("decay_level", "UNKNOWN"))
        print(f"\nOverall Status: {level}")
        print(f"Strategies: {result.get('strategy_count', 0)}  "
              f"Healthy: {result.get('healthy_count', 0)}  "
              f"At-Risk: {result.get('at_risk_count', 0)}  "
              f"Dead: {result.get('dead_count', 0)}")
    else:
        print("Usage: python -m agents.alpha_decay --once")


if __name__ == "__main__":
    main()
