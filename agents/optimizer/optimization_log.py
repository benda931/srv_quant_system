"""
agents/optimizer/optimization_log.py
--------------------------------------
Institutional-Grade Optimization Lineage Tracker

Full lineage tracking for every optimization candidate:
  - Candidate lifecycle: PENDING -> SANDBOX_PASSED -> SHADOW_APPROVED -> PROMOTED / REJECTED / ROLLED_BACK
  - 4-stage validation audit trail
  - Champion/challenger tracking with rollback lineage
  - Campaign-level analytics (success rate, rejection reasons, shadow pipeline)
  - Composite objective breakdown per candidate

Storage: agents/optimizer/optimization_history.json
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_FILE = ROOT / "agents" / "optimizer" / "optimization_history.json"
MAX_HISTORY = 1000  # max records — prevents bloat

log = logging.getLogger("optimization_log")


# ─────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────

VALID_DECISIONS = frozenset({
    "PROMOTED", "SHADOW", "REJECTED", "ROLLED_BACK", "PENDING",
    "SANDBOX_PASSED", "SHADOW_APPROVED",
})

VALID_GOVERNANCE_MODES = frozenset({
    "PARAM_TUNING_ONLY", "PARAM_PLUS_CODE", "REGIME_SPECIFIC_TUNING",
    "FREEZE_AND_MONITOR", "DISABLE_CANDIDATE",
})


@dataclass
class OptimizationRecord:
    """
    Full lineage record for a single optimization candidate evaluation.

    Tracks the complete lifecycle from candidate generation through
    sandbox validation, shadow monitoring, and final promotion/rejection.
    """
    optimization_id: str = ""
    timestamp: str = ""
    candidate_id: str = ""
    candidate_type: str = ""          # "param" / "code" / "regime"
    source: str = ""                  # "local" / "sensitivity" / "bayesian" / "regime" / "joint" / "llm_advisory"
    params_changed: Dict[str, Any] = field(default_factory=dict)
    before_metrics: Dict[str, Any] = field(default_factory=dict)
    after_metrics: Dict[str, Any] = field(default_factory=dict)
    objective_breakdown: Dict[str, float] = field(default_factory=dict)
    validation_stages: Dict[str, str] = field(default_factory=dict)
    final_decision: str = "PENDING"   # PROMOTED / SHADOW / REJECTED / ROLLED_BACK
    governance_mode: str = ""
    champion_baseline_id: str = ""
    revert_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.optimization_id:
            self.optimization_id = f"opt-{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationRecord":
        """Deserialize from dict."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# ─────────────────────────────────────────────────────────────────────────
# OptimizationLineageTracker
# ─────────────────────────────────────────────────────────────────────────

class OptimizationLineageTracker:
    """
    Institutional-grade optimization lineage tracker.

    Provides full audit trail for every optimization candidate:
    - Champion/challenger lifecycle tracking
    - Multi-stage validation audit
    - Rollback lineage and revert history
    - Campaign-level analytics and statistics
    """

    def __init__(self, history_file: Path = HISTORY_FILE) -> None:
        self._path = history_file
        self._path.parent.mkdir(parents=True, exist_ok=True)
        log.info("OptimizationLineageTracker initialized: %s", self._path)

    # ── I/O ─────────────────────────────────────────────────────────────

    def _load(self) -> List[Dict[str, Any]]:
        """Load history from disk."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Error reading optimization history: %s", exc)
        return []

    def _save(self, entries: List[Dict[str, Any]]) -> None:
        """Save history to disk with truncation if needed."""
        if len(entries) > MAX_HISTORY:
            entries = entries[-MAX_HISTORY:]
        try:
            self._path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            log.error("Failed to save optimization history: %s", exc)

    # ── Record Operations ───────────────────────────────────────────────

    def log_record(self, record: OptimizationRecord) -> str:
        """
        Log a complete optimization record.

        Parameters
        ----------
        record : OptimizationRecord
            Full lineage record for one candidate evaluation.

        Returns
        -------
        str
            The optimization_id of the logged record.
        """
        try:
            entries = self._load()
            entry = record.to_dict()
            entries.append(entry)
            self._save(entries)
            log.info(
                "Optimization record logged: %s [%s] decision=%s source=%s",
                record.optimization_id,
                record.candidate_type,
                record.final_decision,
                record.source,
            )
            return record.optimization_id
        except Exception as exc:
            log.error("Failed to log optimization record: %s", exc)
            return record.optimization_id

    def log_attempt(self, entry: Dict[str, Any]) -> None:
        """
        Legacy compatibility: log a raw dict entry.

        Supports the old OptimizationLog.log_attempt() interface so
        existing callers continue to work without changes.
        """
        try:
            entries = self._load()
            if "timestamp" not in entry:
                entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            entries.append(entry)
            self._save(entries)
            log.info(
                "Optimization logged (legacy): %s -> %s [%s] outcome=%s",
                entry.get("agent_source", "?"),
                entry.get("target_file", "?"),
                entry.get("change_type", "?"),
                entry.get("outcome", entry.get("final_decision", "?")),
            )
        except Exception as exc:
            log.error("Failed to log attempt: %s", exc)

    def update_decision(
        self, optimization_id: str, new_decision: str, reason: Optional[str] = None
    ) -> bool:
        """
        Update the final_decision of an existing record (e.g., SHADOW -> PROMOTED).

        Parameters
        ----------
        optimization_id : str
            The ID of the record to update.
        new_decision : str
            New decision value.
        reason : str, optional
            Reason for the update (stored in revert_reason if ROLLED_BACK).

        Returns
        -------
        bool
            True if record was found and updated.
        """
        try:
            entries = self._load()
            for entry in reversed(entries):
                if entry.get("optimization_id") == optimization_id:
                    entry["final_decision"] = new_decision
                    if reason:
                        entry["revert_reason"] = reason
                    entry["decision_updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._save(entries)
                    log.info(
                        "Decision updated: %s -> %s (reason=%s)",
                        optimization_id, new_decision, reason,
                    )
                    return True
            log.warning("Record not found for update: %s", optimization_id)
            return False
        except Exception as exc:
            log.error("Failed to update decision for %s: %s", optimization_id, exc)
            return False

    # ── Query: Champion ─────────────────────────────────────────────────

    def latest_champion(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recently PROMOTED record (the current champion).

        Returns
        -------
        dict or None
            The champion record, or None if no promotions exist.
        """
        try:
            entries = self._load()
            for entry in reversed(entries):
                if entry.get("final_decision") == "PROMOTED":
                    return entry
            # Fallback: look for "improved" outcome (legacy format)
            for entry in reversed(entries):
                if entry.get("outcome") == "improved":
                    return entry
            return None
        except Exception as exc:
            log.error("Failed to get latest champion: %s", exc)
            return None

    # ── Query: Promotion History ────────────────────────────────────────

    def get_promotion_history(self, n: int = 20) -> List[Dict[str, Any]]:
        """
        Return the last N promoted records (chronological, newest last).

        Parameters
        ----------
        n : int
            Maximum number of records to return.

        Returns
        -------
        list[dict]
            Promoted records.
        """
        try:
            entries = self._load()
            promoted = [
                e for e in entries
                if e.get("final_decision") in ("PROMOTED", "SHADOW_APPROVED")
                or e.get("outcome") == "improved"  # legacy
            ]
            return promoted[-n:]
        except Exception as exc:
            log.error("Failed to get promotion history: %s", exc)
            return []

    # ── Query: Rejections by Reason ─────────────────────────────────────

    def get_rejections_by_reason(self) -> Dict[str, int]:
        """
        Count rejection reasons across all history.

        Returns
        -------
        dict[str, int]
            Mapping of reason -> count, sorted by frequency descending.
        """
        try:
            entries = self._load()
            reasons: Dict[str, int] = {}
            for e in entries:
                if e.get("final_decision") == "REJECTED" or e.get("outcome") == "reverted":
                    reason = (
                        e.get("revert_reason")
                        or e.get("reason", "unknown")
                    )
                    # Extract key from validation stages if available
                    stages = e.get("validation_stages", {})
                    for stage_name, stage_result in stages.items():
                        if "FAIL" in str(stage_result).upper():
                            reason = f"{stage_name}:{stage_result}"
                            break
                    reasons[reason] = reasons.get(reason, 0) + 1
            # Sort descending
            return dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
        except Exception as exc:
            log.error("Failed to get rejections: %s", exc)
            return {}

    # ── Query: Shadow Candidates ────────────────────────────────────────

    def current_shadow_candidates(self) -> List[Dict[str, Any]]:
        """
        Return all records currently in SHADOW or SHADOW_APPROVED state.

        These are candidates being monitored before full promotion.

        Returns
        -------
        list[dict]
            Shadow candidate records.
        """
        try:
            entries = self._load()
            return [
                e for e in entries
                if e.get("final_decision") in ("SHADOW", "SHADOW_APPROVED")
            ]
        except Exception as exc:
            log.error("Failed to get shadow candidates: %s", exc)
            return []

    # ── Query: Rollback Lineage ─────────────────────────────────────────

    def rollback_lineage(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return the last N rolled-back records with their revert reasons.

        Useful for understanding what kinds of changes degrade performance.

        Parameters
        ----------
        n : int
            Maximum records to return.

        Returns
        -------
        list[dict]
            Rolled-back records with reasons.
        """
        try:
            entries = self._load()
            rolled_back = [
                {
                    "optimization_id": e.get("optimization_id", ""),
                    "timestamp": e.get("timestamp", ""),
                    "candidate_type": e.get("candidate_type", e.get("change_type", "")),
                    "source": e.get("source", e.get("agent_source", "")),
                    "params_changed": e.get("params_changed", {}),
                    "revert_reason": e.get("revert_reason", e.get("reason", "unknown")),
                    "before_metrics": e.get("before_metrics", {}),
                    "after_metrics": e.get("after_metrics", {}),
                }
                for e in entries
                if e.get("final_decision") == "ROLLED_BACK"
                or e.get("outcome") == "reverted"
            ]
            return rolled_back[-n:]
        except Exception as exc:
            log.error("Failed to get rollback lineage: %s", exc)
            return []

    # ── Query: Campaign Summary ─────────────────────────────────────────

    def optimization_campaign_summary(self) -> Dict[str, Any]:
        """
        Aggregate statistics across all optimization history.

        Returns
        -------
        dict
            Campaign-level analytics: total attempts, success rate,
            promotions, rejections, rollbacks, avg improvement, etc.
        """
        try:
            entries = self._load()
            if not entries:
                return {
                    "total_attempts": 0,
                    "promotions": 0,
                    "rejections": 0,
                    "rollbacks": 0,
                    "shadows": 0,
                    "success_rate": 0.0,
                    "avg_composite_score": 0.0,
                    "avg_delta_sharpe": 0.0,
                    "top_sources": {},
                }

            total = len(entries)
            promotions = sum(
                1 for e in entries
                if e.get("final_decision") == "PROMOTED"
                or e.get("outcome") == "improved"
            )
            rejections = sum(
                1 for e in entries
                if e.get("final_decision") == "REJECTED"
                or e.get("outcome") in ("reverted", "failed")
            )
            rollbacks = sum(
                1 for e in entries
                if e.get("final_decision") == "ROLLED_BACK"
            )
            shadows = sum(
                1 for e in entries
                if e.get("final_decision") in ("SHADOW", "SHADOW_APPROVED")
            )

            # Average composite score (for records that have it)
            composites = [
                e.get("objective_breakdown", {}).get("composite_score", 0.0)
                for e in entries
                if isinstance(e.get("objective_breakdown"), dict)
                and "composite_score" in e.get("objective_breakdown", {})
            ]
            avg_composite = sum(composites) / len(composites) if composites else 0.0

            # Average delta sharpe
            deltas = [
                float(e.get("delta_sharpe", 0))
                for e in entries
                if isinstance(e.get("delta_sharpe"), (int, float))
            ]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

            # Top sources
            source_counts: Dict[str, int] = {}
            for e in entries:
                src = e.get("source", e.get("agent_source", "unknown"))
                source_counts[src] = source_counts.get(src, 0) + 1

            return {
                "total_attempts": total,
                "promotions": promotions,
                "rejections": rejections,
                "rollbacks": rollbacks,
                "shadows": shadows,
                "success_rate": round(promotions / total, 4) if total > 0 else 0.0,
                "avg_composite_score": round(avg_composite, 4),
                "avg_delta_sharpe": round(avg_delta, 6),
                "top_sources": dict(sorted(source_counts.items(), key=lambda kv: -kv[1])),
            }
        except Exception as exc:
            log.error("Failed to compute campaign summary: %s", exc)
            return {"error": str(exc)}

    # ── Legacy Compatibility Methods ────────────────────────────────────

    def get_history(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last N records (newest last). Legacy compat."""
        entries = self._load()
        return entries[-n:]

    def success_rate(self) -> float:
        """
        Percentage of attempts that resulted in improvement.
        Returns 0.0 if no history. Legacy compat.
        """
        entries = self._load()
        if not entries:
            return 0.0
        improved = sum(
            1 for e in entries
            if e.get("final_decision") == "PROMOTED"
            or e.get("outcome") == "improved"
        )
        return improved / len(entries)

    def best_changes(self) -> List[Dict[str, Any]]:
        """Top 5 changes by delta_sharpe or composite_score. Legacy compat."""
        entries = self._load()
        scored = []
        for e in entries:
            score = 0.0
            obj = e.get("objective_breakdown", {})
            if isinstance(obj, dict) and "composite_score" in obj:
                score = float(obj["composite_score"])
            elif isinstance(e.get("delta_sharpe"), (int, float)):
                score = float(e["delta_sharpe"])
            else:
                continue
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:5]]

    def revert_reasons(self) -> Dict[str, int]:
        """Count revert reasons. Legacy compat."""
        return self.get_rejections_by_reason()

    def recent_trend(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Return summary of last N runs for trend analysis. Legacy compat.
        """
        entries = self._load()
        recent = entries[-n:]
        return [
            {
                "timestamp": e.get("timestamp"),
                "outcome": e.get("final_decision", e.get("outcome")),
                "delta_sharpe": e.get("delta_sharpe",
                    e.get("objective_breakdown", {}).get("delta_net_sharpe")),
                "delta_ic": e.get("delta_ic"),
                "param_name": e.get("param_name",
                    list(e.get("params_changed", {}).keys())[:1] or [None])[0]
                    if not isinstance(e.get("param_name"), str)
                    else e.get("param_name"),
                "change_type": e.get("candidate_type", e.get("change_type")),
                "composite_score": e.get("objective_breakdown", {}).get(
                    "composite_score") if isinstance(
                    e.get("objective_breakdown"), dict) else None,
                "source": e.get("source", e.get("agent_source")),
            }
            for e in recent
        ]


# ── Singleton ────────────────────────────────────────────────────────────
_tracker_instance: Optional[OptimizationLineageTracker] = None


def get_optimization_log() -> OptimizationLineageTracker:
    """Return singleton lineage tracker (backward-compatible name)."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = OptimizationLineageTracker()
    return _tracker_instance


# Alias for explicit naming
get_lineage_tracker = get_optimization_log


# ── CLI — Statistics Display ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    tracker = get_optimization_log()
    summary = tracker.optimization_campaign_summary()

    if summary.get("total_attempts", 0) == 0:
        print("No optimization history yet.")
        sys.exit(0)

    print(f"Optimization Campaign Summary ({HISTORY_FILE}):")
    print(f"  Total attempts:    {summary['total_attempts']}")
    print(f"  Promotions:        {summary['promotions']}")
    print(f"  Rejections:        {summary['rejections']}")
    print(f"  Rollbacks:         {summary['rollbacks']}")
    print(f"  Shadow candidates: {summary['shadows']}")
    print(f"  Success rate:      {summary['success_rate']:.1%}")
    print(f"  Avg composite:     {summary['avg_composite_score']:.4f}")
    print(f"  Avg delta Sharpe:  {summary['avg_delta_sharpe']:+.4f}")
    print()

    best = tracker.best_changes()
    if best:
        print("Top 5 changes by score:")
        for i, b in enumerate(best, 1):
            obj = b.get("objective_breakdown", {})
            cs = obj.get("composite_score", b.get("delta_sharpe", 0))
            decision = b.get("final_decision", b.get("outcome", "?"))
            source = b.get("source", b.get("agent_source", "?"))
            print(f"  {i}. score={cs:+.4f} decision={decision} source={source}")
    print()

    reasons = tracker.get_rejections_by_reason()
    if reasons:
        print("Rejection reasons:")
        for reason, count in list(reasons.items())[:10]:
            print(f"  {reason}: {count}")
    print()

    shadows = tracker.current_shadow_candidates()
    if shadows:
        print(f"Active shadow candidates: {len(shadows)}")
        for s in shadows:
            print(f"  {s.get('optimization_id', '?')} source={s.get('source', '?')}")
