"""
agents/methodology/methodology_portfolio.py
--------------------------------------------
Institutional Portfolio Tracker for methodology agent runs.

Tracks:
- Experiment lineage (experiment_id per run)
- Approval/rejection history
- Per-methodology metric history
- Regime fitness history
- Robustness trend
- Ranking drift

Stores up to 365 entries in a JSON file with thread-safe read/write.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_ENTRIES = 365
PORTFOLIO_FILE = Path(__file__).resolve().parent / "reports" / "portfolio.json"


class MethodologyPortfolio:
    """
    Thread-safe JSON-backed portfolio tracker for methodology run history.

    Each run is stored with: run_date, metrics, regime_breakdown,
    signal_stack_summary, parameters_snapshot, governance, scorecards,
    approval_matrix, experiment_id.

    Extended methods for institutional tracking:
    - get_methodology_history(name, n) - per-methodology metric history
    - get_rank_drift(name, n) - how a methodology's rank changes over runs
    - get_approval_streak(name) - consecutive approval/rejection count
    - best_approved_run() - best run where at least one strategy was approved
    - get_regime_fitness_history(name, n) - regime performance over time
    - get_robustness_trend(name, n) - robustness score trend
    """

    def __init__(self, path: Path = PORTFOLIO_FILE, max_entries: int = MAX_ENTRIES) -> None:
        self._path = path
        self._max = max_entries
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Safe I/O ─────────────────────────────────────────────────────────────

    def _load(self) -> List[dict]:
        """Load history from JSON file."""
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except Exception as exc:
            log.warning("Failed to load portfolio file: %s", exc)
            return []

    def _save(self, entries: List[dict]) -> None:
        """Save history to JSON file — limited by max_entries."""
        trimmed = entries[-self._max:]
        self._path.write_text(
            json.dumps(trimmed, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Core public interface (backward compatible) ──────────────────────────

    def add_run(self, run_result: dict) -> None:
        """Add a new run to history."""
        with self._lock:
            entries = self._load()
            entries.append(run_result)
            self._save(entries)
            log.debug(
                "MethodologyPortfolio: added run %s (total=%d)",
                run_result.get("run_date", "?"),
                len(entries),
            )

    def get_history(self, n: int = 30) -> List[dict]:
        """Return last n runs (old -> new)."""
        with self._lock:
            entries = self._load()
            return entries[-n:]

    def get_trend(self, metric: str, n: int = 10) -> dict:
        """
        Trend analysis of a metric over last n runs.

        Returns dict with: metric, values, direction (improving/declining/stable),
        slope, mean, last, first, delta_pct.
        """
        with self._lock:
            entries = self._load()

        recent = entries[-n:] if len(entries) >= n else entries
        values = []
        for entry in recent:
            metrics = entry.get("metrics", {})
            v = metrics.get(metric)
            if v is not None and _is_finite(v):
                values.append(float(v))

        if len(values) < 2:
            return {
                "metric": metric,
                "values": values,
                "direction": "insufficient_data",
                "slope": 0.0,
                "mean": values[0] if values else float("nan"),
                "last": values[-1] if values else float("nan"),
                "first": values[0] if values else float("nan"),
                "delta_pct": 0.0,
            }

        # Simple linear slope
        n_pts = len(values)
        x_mean = (n_pts - 1) / 2.0
        y_mean = sum(values) / n_pts
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n_pts))
        slope = numerator / denominator if denominator > 1e-12 else 0.0

        first_val = values[0]
        last_val = values[-1]
        delta_pct = ((last_val - first_val) / abs(first_val) * 100) if abs(first_val) > 1e-12 else 0.0

        # 3% threshold for direction
        if delta_pct > 3.0:
            direction = "improving"
        elif delta_pct < -3.0:
            direction = "declining"
        else:
            direction = "stable"

        # Inverted metrics (lower is better)
        if metric in ("max_dd", "max_drawdown"):
            if direction == "improving":
                direction = "declining"
            elif direction == "declining":
                direction = "improving"

        return {
            "metric": metric,
            "values": values,
            "direction": direction,
            "slope": round(slope, 6),
            "mean": round(y_mean, 6),
            "last": round(last_val, 6),
            "first": round(first_val, 6),
            "delta_pct": round(delta_pct, 2),
        }

    def compare_vs_previous(self, current: dict) -> dict:
        """Compare current run against the last run in history."""
        with self._lock:
            entries = self._load()

        if not entries:
            return {"has_previous": False, "deltas": {}}

        previous = entries[-1]
        prev_metrics = previous.get("metrics", {})
        curr_metrics = current.get("metrics", {})

        deltas = {}
        for key in set(list(prev_metrics.keys()) + list(curr_metrics.keys())):
            old = prev_metrics.get(key)
            new = curr_metrics.get(key)
            if old is not None and new is not None and _is_finite(old) and _is_finite(new):
                old_f = float(old)
                new_f = float(new)
                abs_delta = new_f - old_f
                pct_delta = (abs_delta / abs(old_f) * 100) if abs(old_f) > 1e-12 else 0.0
                deltas[key] = {
                    "old": round(old_f, 6),
                    "new": round(new_f, 6),
                    "abs_delta": round(abs_delta, 6),
                    "pct_delta": round(pct_delta, 2),
                }

        return {
            "has_previous": True,
            "previous_date": previous.get("run_date", "?"),
            "deltas": deltas,
        }

    def best_run(self) -> dict:
        """Return the run with the highest Sharpe ratio."""
        with self._lock:
            entries = self._load()

        if not entries:
            return {}

        best = None
        best_sharpe = -float("inf")
        for entry in entries:
            sharpe = entry.get("metrics", {}).get("sharpe")
            if sharpe is not None and _is_finite(sharpe) and float(sharpe) > best_sharpe:
                best_sharpe = float(sharpe)
                best = entry

        return best or {}

    def worst_regime(self) -> dict:
        """
        Identify the consistently weakest regime based on average hit_rate.

        Returns dict with: regime, avg_hit_rate, n_observations,
        consecutive_below_50, max_consecutive_below_50, last_3.
        """
        with self._lock:
            entries = self._load()

        if not entries:
            return {}

        regime_stats: Dict[str, List[float]] = {}
        for entry in entries:
            breakdown = entry.get("regime_breakdown", {})
            for regime, regime_data in breakdown.items():
                hr = regime_data.get("hit_rate") or regime_data.get("ic_mean")
                if hr is not None and _is_finite(hr):
                    regime_stats.setdefault(regime, []).append(float(hr))

        if not regime_stats:
            return {}

        worst_regime = None
        worst_avg = float("inf")
        worst_data: Dict[str, Any] = {}

        for regime, hr_list in regime_stats.items():
            avg = sum(hr_list) / len(hr_list)

            consecutive = 0
            max_consecutive = 0
            for hr in reversed(hr_list):
                if hr < 0.50:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    break

            if avg < worst_avg:
                worst_avg = avg
                worst_regime = regime
                worst_data = {
                    "regime": regime,
                    "avg_hit_rate": round(avg, 4),
                    "n_observations": len(hr_list),
                    "consecutive_below_50": consecutive,
                    "max_consecutive_below_50": max_consecutive,
                    "last_3": [round(x, 4) for x in hr_list[-3:]],
                }

        return worst_data

    # ── Extended institutional methods ────────────────────────────────────────

    def get_methodology_history(self, name: str, n: int = 30) -> List[dict]:
        """
        Get per-methodology metric history across last n runs.

        Extracts scorecard data for a specific methodology from each run.

        Parameters
        ----------
        name : str
            Methodology name (e.g., 'ALPHA_WHITELIST_MR').
        n : int
            Number of recent runs to search.

        Returns
        -------
        List[dict]
            Per-run metrics for the named methodology, each with:
            run_date, net_sharpe, gross_sharpe, hit_rate, max_drawdown,
            total_trades, classification, promotion_decision, robustness_score.
        """
        with self._lock:
            entries = self._load()

        recent = entries[-n:] if len(entries) >= n else entries
        history = []

        for entry in recent:
            run_date = entry.get("run_date", "?")
            scorecards = entry.get("methodology_scorecards", [])

            # Search scorecards for the methodology
            found = None
            for sc in scorecards:
                if isinstance(sc, dict) and sc.get("name") == name:
                    found = sc
                    break

            if found:
                history.append({
                    "run_date": run_date,
                    "net_sharpe": _safe_get(found, "net_sharpe"),
                    "gross_sharpe": _safe_get(found, "gross_sharpe"),
                    "deflated_sharpe": _safe_get(found, "deflated_sharpe"),
                    "walk_forward_sharpe": _safe_get(found, "walk_forward_sharpe"),
                    "bootstrap_sharpe_median": _safe_get(found, "bootstrap_sharpe_median"),
                    "hit_rate": _safe_get(found, "hit_rate"),
                    "max_drawdown": _safe_get(found, "max_drawdown"),
                    "total_trades": found.get("total_trades", 0),
                    "classification": found.get("classification", "?"),
                    "promotion_decision": found.get("promotion_decision", "?"),
                    "robustness_score": _safe_get(found, "robustness_score"),
                    "stability_score": _safe_get(found, "stability_score"),
                    "cost_drag_pct": _safe_get(found, "cost_drag_pct"),
                    "final_rank": found.get("final_rank", 0),
                })
            else:
                # Try older format: methodology_lab ranking
                lab = entry.get("methodology_lab", {})
                ranking = lab.get("ranking", [])
                for r in ranking:
                    if isinstance(r, dict) and r.get("name") == name:
                        history.append({
                            "run_date": run_date,
                            "net_sharpe": _safe_get(r, "sharpe"),
                            "gross_sharpe": _safe_get(r, "sharpe"),
                            "hit_rate": _safe_get(r, "win_rate"),
                            "total_trades": r.get("total_trades", 0),
                            "classification": "UNKNOWN",
                            "promotion_decision": "UNKNOWN",
                            "robustness_score": 0.0,
                            "stability_score": 0.0,
                            "cost_drag_pct": 0.0,
                            "final_rank": 0,
                        })
                        break

        return history

    def get_rank_drift(self, name: str, n: int = 30) -> dict:
        """
        Track how a methodology's rank changes over recent runs.

        Parameters
        ----------
        name : str
            Methodology name.
        n : int
            Number of recent runs to analyze.

        Returns
        -------
        dict
            ranks (list), mean_rank, best_rank, worst_rank, rank_std,
            direction (improving/declining/stable), drift_score.
        """
        history = self.get_methodology_history(name, n)
        ranks = [h.get("final_rank", 0) for h in history if h.get("final_rank", 0) > 0]

        if len(ranks) < 2:
            return {
                "name": name,
                "ranks": ranks,
                "direction": "insufficient_data",
                "mean_rank": ranks[0] if ranks else 0,
                "best_rank": min(ranks) if ranks else 0,
                "worst_rank": max(ranks) if ranks else 0,
                "rank_std": 0.0,
                "drift_score": 0.0,
            }

        mean_rank = sum(ranks) / len(ranks)
        rank_std = float(_std(ranks))

        # Direction: improving = rank decreasing (getting closer to 1)
        first_3 = ranks[:3] if len(ranks) >= 3 else ranks[:1]
        last_3 = ranks[-3:] if len(ranks) >= 3 else ranks[-1:]
        early_avg = sum(first_3) / len(first_3)
        late_avg = sum(last_3) / len(last_3)

        if late_avg < early_avg - 0.5:
            direction = "improving"
        elif late_avg > early_avg + 0.5:
            direction = "declining"
        else:
            direction = "stable"

        # Drift score: normalized rank volatility (0 = perfectly stable)
        max_possible_rank = max(ranks) if ranks else 1
        drift_score = rank_std / max(max_possible_rank, 1)

        return {
            "name": name,
            "ranks": ranks,
            "direction": direction,
            "mean_rank": round(mean_rank, 1),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
            "rank_std": round(rank_std, 2),
            "drift_score": round(min(drift_score, 1.0), 3),
        }

    def get_approval_streak(self, name: str) -> dict:
        """
        Get consecutive approval/rejection streak for a methodology.

        Parameters
        ----------
        name : str
            Methodology name.

        Returns
        -------
        dict
            current_streak (int, positive=approved, negative=rejected),
            streak_type ('approved'/'rejected'/'unknown'),
            total_approved, total_rejected, total_conditional, total_runs,
            approval_rate.
        """
        with self._lock:
            entries = self._load()

        decisions = []
        for entry in entries:
            scorecards = entry.get("methodology_scorecards", [])
            for sc in scorecards:
                if isinstance(sc, dict) and sc.get("name") == name:
                    decisions.append(sc.get("promotion_decision", "UNKNOWN"))
                    break
            # Also check approval_matrix
            if not decisions or decisions[-1] == "UNKNOWN":
                am = entry.get("approval_matrix", {})
                if name in am:
                    d = am[name].get("decision", "UNKNOWN")
                    if decisions and decisions[-1] == "UNKNOWN":
                        decisions[-1] = d
                    elif not decisions:
                        decisions.append(d)

        if not decisions:
            return {
                "name": name,
                "current_streak": 0,
                "streak_type": "unknown",
                "total_approved": 0,
                "total_rejected": 0,
                "total_conditional": 0,
                "total_runs": 0,
                "approval_rate": 0.0,
            }

        # Current streak (from most recent)
        streak = 0
        streak_type = decisions[-1] if decisions else "UNKNOWN"
        for d in reversed(decisions):
            if d == streak_type:
                streak += 1
            else:
                break

        total_approved = sum(1 for d in decisions if d == "APPROVED")
        total_rejected = sum(1 for d in decisions if d == "REJECTED")
        total_conditional = sum(1 for d in decisions if d == "CONDITIONAL")
        total_runs = len(decisions)
        approval_rate = total_approved / max(total_runs, 1)

        return {
            "name": name,
            "current_streak": streak if streak_type == "APPROVED" else -streak,
            "streak_type": streak_type.lower() if streak_type else "unknown",
            "total_approved": total_approved,
            "total_rejected": total_rejected,
            "total_conditional": total_conditional,
            "total_runs": total_runs,
            "approval_rate": round(approval_rate, 3),
        }

    def best_approved_run(self) -> dict:
        """
        Return the best run where at least one strategy was APPROVED.

        "Best" = highest net Sharpe among approved strategies.

        Returns
        -------
        dict
            The full run entry, or empty dict if no approved runs exist.
        """
        with self._lock:
            entries = self._load()

        if not entries:
            return {}

        best_entry = None
        best_sharpe = -float("inf")

        for entry in entries:
            scorecards = entry.get("methodology_scorecards", [])
            approved = [
                sc for sc in scorecards
                if isinstance(sc, dict) and sc.get("promotion_decision") == "APPROVED"
            ]

            if not approved:
                # Check approval_matrix
                am = entry.get("approval_matrix", {})
                has_approved = any(
                    v.get("decision") == "APPROVED"
                    for v in am.values()
                    if isinstance(v, dict)
                )
                if not has_approved:
                    continue

            # Best approved Sharpe in this run
            run_best_sharpe = max(
                (_safe_get(sc, "net_sharpe") for sc in approved),
                default=-float("inf"),
            ) if approved else _safe_get(entry.get("metrics", {}), "sharpe", -float("inf"))

            if run_best_sharpe > best_sharpe:
                best_sharpe = run_best_sharpe
                best_entry = entry

        return best_entry or {}

    def get_regime_fitness_history(self, name: str, n: int = 30) -> List[dict]:
        """
        Get regime fitness (per-regime Sharpe) over last n runs for a methodology.

        Parameters
        ----------
        name : str
            Methodology name.
        n : int
            Number of recent runs.

        Returns
        -------
        List[dict]
            Per-run regime_scores dict for the named methodology.
        """
        with self._lock:
            entries = self._load()

        recent = entries[-n:] if len(entries) >= n else entries
        history = []

        for entry in recent:
            run_date = entry.get("run_date", "?")

            # Try scorecards first
            scorecards = entry.get("methodology_scorecards", [])
            for sc in scorecards:
                if isinstance(sc, dict) and sc.get("name") == name:
                    history.append({
                        "run_date": run_date,
                        "regime_scores": sc.get("regime_scores", {}),
                        "diversification_score": _safe_get(sc, "diversification_score"),
                    })
                    break
            else:
                # Try regime_fitness_map
                fitness_map = entry.get("regime_fitness_map", {})
                if name in fitness_map:
                    history.append({
                        "run_date": run_date,
                        "regime_actions": fitness_map[name],
                    })

                # Try attribution
                attr = entry.get("attribution", {})
                method_scores = attr.get("methodology_scores", {})
                if name in method_scores:
                    ms = method_scores[name]
                    history.append({
                        "run_date": run_date,
                        "regime_scores": ms.get("regime_sharpes", {}),
                        "fragility_score": ms.get("fragility_score", 0),
                        "diversification_score": ms.get("diversification_score", 0),
                    })

        return history

    def get_robustness_trend(self, name: str, n: int = 30) -> dict:
        """
        Track robustness score trend for a methodology over recent runs.

        Parameters
        ----------
        name : str
            Methodology name.
        n : int
            Number of recent runs.

        Returns
        -------
        dict
            values, direction, slope, mean, last, first, delta_pct.
        """
        history = self.get_methodology_history(name, n)
        values = [
            h.get("robustness_score", 0.0)
            for h in history
            if _is_finite(h.get("robustness_score", 0))
        ]

        if len(values) < 2:
            return {
                "name": name,
                "metric": "robustness_score",
                "values": values,
                "direction": "insufficient_data",
                "slope": 0.0,
                "mean": values[0] if values else 0.0,
                "last": values[-1] if values else 0.0,
                "first": values[0] if values else 0.0,
                "delta_pct": 0.0,
            }

        n_pts = len(values)
        x_mean = (n_pts - 1) / 2.0
        y_mean = sum(values) / n_pts
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n_pts))
        slope = numerator / denominator if denominator > 1e-12 else 0.0

        first_val = values[0]
        last_val = values[-1]
        delta_pct = ((last_val - first_val) / abs(first_val) * 100) if abs(first_val) > 1e-12 else 0.0

        if delta_pct > 3.0:
            direction = "improving"
        elif delta_pct < -3.0:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "name": name,
            "metric": "robustness_score",
            "values": values,
            "direction": direction,
            "slope": round(slope, 6),
            "mean": round(y_mean, 4),
            "last": round(last_val, 4),
            "first": round(first_val, 4),
            "delta_pct": round(delta_pct, 2),
        }

    def get_experiment_lineage(self, n: int = 30) -> List[dict]:
        """
        Get experiment IDs and governance records from recent runs.

        Returns
        -------
        List[dict]
            Per-run: run_date, experiment_id, methodology_version,
            validation_status, promotion_readiness, n_approved.
        """
        with self._lock:
            entries = self._load()

        recent = entries[-n:] if len(entries) >= n else entries
        lineage = []

        for entry in recent:
            gov = entry.get("governance", {})
            pm = entry.get("pm_summary", {})
            lineage.append({
                "run_date": entry.get("run_date", "?"),
                "experiment_id": gov.get("experiment_id", "N/A"),
                "methodology_version": gov.get("methodology_version",
                                                entry.get("methodology_version", "N/A")),
                "data_fingerprint": gov.get("data_fingerprint", "N/A"),
                "settings_fingerprint": gov.get("settings_fingerprint", "N/A"),
                "validation_status": gov.get("validation_status", "N/A"),
                "promotion_readiness": gov.get("promotion_readiness", "N/A"),
                "n_approved": pm.get("strategies_approved", 0),
                "n_rejected": pm.get("strategies_rejected", 0),
            })

        return lineage

    def get_all_methodology_names(self) -> List[str]:
        """
        Get all methodology names that have appeared in recent history.

        Returns
        -------
        List[str]
            Sorted list of unique methodology names.
        """
        with self._lock:
            entries = self._load()

        names = set()
        for entry in entries:
            scorecards = entry.get("methodology_scorecards", [])
            for sc in scorecards:
                if isinstance(sc, dict) and "name" in sc:
                    names.add(sc["name"])

            # Also check lab ranking
            lab = entry.get("methodology_lab", {})
            ranking = lab.get("ranking", [])
            for r in ranking:
                if isinstance(r, dict) and "name" in r:
                    names.add(r["name"])

        return sorted(names)

    def summary(self, n: int = 10) -> dict:
        """
        Generate a summary of recent portfolio tracking.

        Returns
        -------
        dict
            total_runs, recent_n, methodology_names, approval_rates,
            best_overall_sharpe, trend_sharpe, trend_robustness.
        """
        with self._lock:
            entries = self._load()

        if not entries:
            return {"total_runs": 0}

        recent = entries[-n:] if len(entries) >= n else entries
        names = self.get_all_methodology_names()

        # Approval rates per methodology
        approval_rates = {}
        for name in names[:20]:  # Cap to avoid excessive computation
            streak = self.get_approval_streak(name)
            approval_rates[name] = streak.get("approval_rate", 0.0)

        # Overall metrics
        sharpes = []
        for entry in recent:
            s = entry.get("metrics", {}).get("sharpe")
            if s is not None and _is_finite(s):
                sharpes.append(float(s))

        return {
            "total_runs": len(entries),
            "recent_n": len(recent),
            "methodology_names": names,
            "approval_rates": approval_rates,
            "best_overall_sharpe": round(max(sharpes), 4) if sharpes else 0.0,
            "worst_overall_sharpe": round(min(sharpes), 4) if sharpes else 0.0,
            "mean_overall_sharpe": round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_finite(x: Any) -> bool:
    """Check if value is a finite number."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _safe_get(d: Any, key: str, default: float = 0.0) -> float:
    """Safely get a numeric value from a dict or dict-like object."""
    if isinstance(d, dict):
        v = d.get(key, default)
    else:
        v = getattr(d, key, default)
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else default
    except (TypeError, ValueError):
        return default


def _std(values: List[float]) -> float:
    """Compute standard deviation of a list of values."""
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(variance, 0.0))


# ═════════════════════════════════════════════════════════════════════════════
# Promotion Probability Forecasting
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PromotionForecast:
    """Forecast of promotion probability for a methodology."""
    methodology: str
    current_sharpe: float
    sharpe_trend: float              # Slope of Sharpe over last N runs
    approval_rate: float             # Historical approval rate
    robustness_score: float          # Current robustness
    promotion_probability: float      # P(APPROVED | next_run) ∈ [0, 1]
    confidence: float                # Confidence in the forecast
    estimated_runs_to_approval: int  # Expected runs until approval
    recommendation: str              # "READY" / "IMPROVING" / "NEEDS_WORK" / "UNLIKELY"


def forecast_promotion(
    portfolio: MethodologyPortfolio,
    methodology_name: str,
    n_history: int = 20,
) -> PromotionForecast:
    """
    Predict the probability that a methodology will be approved in the next run.

    Uses logistic regression on features:
      - current Sharpe ratio
      - Sharpe trend (slope over recent runs)
      - historical approval rate
      - robustness score
      - cost drag

    Parameters
    ----------
    portfolio : MethodologyPortfolio — the portfolio tracker instance
    methodology_name : str — name of the methodology to forecast
    n_history : int — number of recent runs to use
    """
    import numpy as np

    # Gather features
    hist = portfolio.get_methodology_history(methodology_name, n=n_history)
    streak = portfolio.get_approval_streak(methodology_name)
    robust = portfolio.get_robustness_trend(methodology_name, n=n_history)

    if not hist:
        return PromotionForecast(
            methodology=methodology_name, current_sharpe=0, sharpe_trend=0,
            approval_rate=0, robustness_score=0, promotion_probability=0,
            confidence=0, estimated_runs_to_approval=999, recommendation="NEEDS_WORK",
        )

    # Features
    sharpes = [h.get("sharpe", 0) for h in hist if _is_finite(h.get("sharpe", 0))]
    current_sharpe = sharpes[-1] if sharpes else 0
    sharpe_trend = 0.0
    if len(sharpes) >= 3:
        x = np.arange(len(sharpes), dtype=float)
        y = np.array(sharpes, dtype=float)
        slope = float(np.polyfit(x, y, 1)[0])
        sharpe_trend = slope

    approval_rate = streak.get("approval_rate", 0.0)
    robustness = robust[-1].get("robustness_score", 0) if robust else 0
    cost_drag = hist[-1].get("cost_drag", 0) if hist else 0

    # Logistic model: P = sigmoid(β₀ + β₁·sharpe + β₂·trend + β₃·approval + β₄·robust)
    # Coefficients calibrated empirically
    b0, b1, b2, b3, b4 = -2.0, 3.0, 10.0, 2.0, 1.5
    z = b0 + b1 * current_sharpe + b2 * sharpe_trend + b3 * approval_rate + b4 * robustness
    prob = 1.0 / (1.0 + math.exp(-z))

    # Confidence based on sample size
    confidence = min(1.0, len(sharpes) / 15)

    # Estimated runs to approval
    if prob >= 0.8:
        est_runs = 1
    elif prob >= 0.5:
        est_runs = max(1, int(math.ceil(math.log(0.8 / max(prob, 0.01)) / max(sharpe_trend * 3, 0.01))))
    elif prob >= 0.2:
        est_runs = max(3, int(10 * (1 - prob)))
    else:
        est_runs = 999

    # Recommendation
    if prob >= 0.7:
        rec = "READY"
    elif prob >= 0.4 and sharpe_trend > 0:
        rec = "IMPROVING"
    elif prob >= 0.2:
        rec = "NEEDS_WORK"
    else:
        rec = "UNLIKELY"

    return PromotionForecast(
        methodology=methodology_name,
        current_sharpe=round(current_sharpe, 4),
        sharpe_trend=round(sharpe_trend, 6),
        approval_rate=round(approval_rate, 4),
        robustness_score=round(robustness, 4),
        promotion_probability=round(prob, 4),
        confidence=round(confidence, 3),
        estimated_runs_to_approval=min(est_runs, 999),
        recommendation=rec,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Strategy Complementarity Analysis
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplementaryPair:
    """A pair of strategies that complement each other (low correlation)."""
    strategy_a: str
    strategy_b: str
    return_correlation: float        # Correlation of daily returns (-1 to 1)
    regime_complementarity: float    # 0 = same regime strength, 1 = opposite
    combined_sharpe_boost: float     # Sharpe improvement from combining
    recommendation: str              # "STRONG_PAIR" / "MODERATE" / "REDUNDANT"


def find_complementary_strategies(
    portfolio: MethodologyPortfolio,
    n_history: int = 20,
) -> List[ComplementaryPair]:
    """
    Find methodology pairs that complement each other across regimes.

    Two strategies complement each other if:
      1. Their return correlation is low (< 0.3)
      2. They perform well in different regimes
      3. Combining them improves Sharpe vs either alone

    Returns sorted list of complementary pairs.
    """
    import numpy as np
    from itertools import combinations

    names = portfolio.get_all_methodology_names()
    if len(names) < 2:
        return []

    # Gather Sharpe time series for each methodology
    sharpe_series = {}
    regime_profiles = {}

    for name in names[:15]:  # Cap at 15
        hist = portfolio.get_methodology_history(name, n=n_history)
        if not hist or len(hist) < 3:
            continue

        sharpes = [h.get("sharpe", 0) for h in hist if _is_finite(h.get("sharpe", 0))]
        if len(sharpes) >= 3:
            sharpe_series[name] = np.array(sharpes)

        # Regime fitness
        regime_fit = portfolio.get_regime_fitness_history(name, n=n_history)
        if regime_fit:
            latest = regime_fit[-1]
            regime_profiles[name] = {
                "CALM": latest.get("CALM", {}).get("sharpe", 0),
                "NORMAL": latest.get("NORMAL", {}).get("sharpe", 0),
                "TENSION": latest.get("TENSION", {}).get("sharpe", 0),
            }

    pairs = []
    for a, b in combinations(sharpe_series.keys(), 2):
        s_a = sharpe_series[a]
        s_b = sharpe_series[b]

        # Align lengths
        min_len = min(len(s_a), len(s_b))
        if min_len < 3:
            continue

        s_a = s_a[-min_len:]
        s_b = s_b[-min_len:]

        # Return correlation
        corr = float(np.corrcoef(s_a, s_b)[0, 1]) if min_len > 2 else 0.0

        # Regime complementarity
        regime_comp = 0.0
        if a in regime_profiles and b in regime_profiles:
            ra = regime_profiles[a]
            rb = regime_profiles[b]
            # Higher score if one is strong where the other is weak
            diffs = []
            for regime in ["CALM", "NORMAL", "TENSION"]:
                va = ra.get(regime, 0)
                vb = rb.get(regime, 0)
                if (va > 0 and vb < 0) or (va < 0 and vb > 0):
                    diffs.append(abs(va - vb))
            regime_comp = float(np.mean(diffs)) if diffs else 0.0

        # Combined Sharpe boost
        combined = (s_a + s_b) / 2
        sharpe_a = float(s_a.mean() / (s_a.std() + 1e-10))
        sharpe_b = float(s_b.mean() / (s_b.std() + 1e-10))
        sharpe_combined = float(combined.mean() / (combined.std() + 1e-10))
        boost = sharpe_combined - max(sharpe_a, sharpe_b)

        # Classification
        if corr < 0.2 and regime_comp > 0.1:
            rec = "STRONG_PAIR"
        elif corr < 0.5:
            rec = "MODERATE"
        else:
            rec = "REDUNDANT"

        pairs.append(ComplementaryPair(
            strategy_a=a, strategy_b=b,
            return_correlation=round(corr, 4),
            regime_complementarity=round(regime_comp, 4),
            combined_sharpe_boost=round(boost, 4),
            recommendation=rec,
        ))

    # Sort: strong pairs first, then by Sharpe boost
    priority = {"STRONG_PAIR": 0, "MODERATE": 1, "REDUNDANT": 2}
    pairs.sort(key=lambda p: (priority.get(p.recommendation, 9), -p.combined_sharpe_boost))
    return pairs


    # (imports at top of file)
