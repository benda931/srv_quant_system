"""
agents/architect/improvement_log.py
====================================
Tracks all Architect Agent scan cycles and improvements.
JSON-backed, thread-safe, singleton pattern (mirrors optimization_log.py).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOG_PATH = Path(__file__).resolve().parent / "improvement_history.json"
_MAX_HISTORY = 500
_lock = threading.Lock()
_instance: Optional["ImprovementLog"] = None


class ImprovementLog:
    """Append-only log of Architect Agent improvement cycles."""

    def __init__(self, path: Path = _LOG_PATH):
        self._path = path
        self._data: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._data = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data[-_MAX_HISTORY:], indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Public API ──────────────────────────────────────────────

    def log_cycle(self, entry: Dict[str, Any]) -> None:
        """Log a complete scan cycle."""
        with _lock:
            entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            self._data.append(entry)
            self._save()

    def get_history(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return last n cycles."""
        return self._data[-n:]

    def domain_success_rate(self, domain: str) -> float:
        """Fraction of cycles for this domain where validation passed."""
        relevant = [e for e in self._data if e.get("domain") == domain]
        if not relevant:
            return 0.0
        passed = sum(
            1 for e in relevant
            if e.get("validation_result", {}).get("tests_pass", False)
        )
        return passed / len(relevant)

    def last_scan_date(self, domain: str) -> Optional[str]:
        """ISO date of last scan for this domain, or None."""
        for entry in reversed(self._data):
            if entry.get("domain") == domain:
                return entry.get("timestamp", entry.get("cycle_id", ""))[:10]
        return None

    def get_pending_domains(self, all_domains: List[str], days_threshold: int = 7) -> List[str]:
        """Domains not scanned in last N days."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).isoformat()[:10]
        pending = []
        for d in all_domains:
            last = self.last_scan_date(d)
            if last is None or last < cutoff:
                pending.append(d)
        return pending

    def total_improvements(self) -> int:
        """Total number of successful improvements made."""
        return sum(
            1 for e in self._data
            if e.get("validation_result", {}).get("tests_pass", False)
            and len(e.get("actions_taken", [])) > 0
        )

    def domain_stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-domain statistics."""
        stats: Dict[str, Dict[str, Any]] = {}
        for entry in self._data:
            d = entry.get("domain", "unknown")
            if d not in stats:
                stats[d] = {"cycles": 0, "successes": 0, "last_scan": None}
            stats[d]["cycles"] += 1
            if entry.get("validation_result", {}).get("tests_pass", False):
                stats[d]["successes"] += 1
            stats[d]["last_scan"] = entry.get("timestamp", "")[:10]
        return stats


def get_improvement_log() -> ImprovementLog:
    """Singleton accessor."""
    global _instance
    if _instance is None:
        _instance = ImprovementLog()
    return _instance


# ─────────────────────────────────────────────────────────────────────────────
# Architecture Debt Tracking (module-level functions using singleton)
# ─────────────────────────────────────────────────────────────────────────────

def get_architecture_debt_summary() -> Dict[str, Any]:
    """
    Aggregate architecture debt from improvement history.
    Returns total unresolved items, domain-level debt counts,
    recurring domain names, and failed intervention count.
    """
    log = get_improvement_log()
    history = log.get_history(n=200)

    # Count unresolved: cycles where actions were taken but tests failed
    unresolved = 0
    domain_debt: Dict[str, int] = {}
    recurring_domains: Dict[str, int] = {}

    for entry in history:
        domain = entry.get("domain", "unknown")

        # Track recurring domain appearances
        recurring_domains[domain] = recurring_domains.get(domain, 0) + 1

        # Unresolved = had actions but validation failed
        actions = entry.get("actions_taken", [])
        tests_pass = entry.get("validation_result", {}).get("tests_pass", True)
        if actions and not tests_pass:
            unresolved += 1
            domain_debt[domain] = domain_debt.get(domain, 0) + 1

        # Also count architecture-level unresolved flags
        ms = entry.get("machine_summary", {})
        if isinstance(ms, dict):
            flags = ms.get("unresolved_flags", [])
            if flags:
                unresolved += len(flags)
                domain_debt[domain] = domain_debt.get(domain, 0) + len(flags)

    return {
        "total_unresolved": unresolved,
        "domain_debt": domain_debt,
        "recurring_domains": recurring_domains,
        "total_cycles": len(history),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_recurring_domain_issues(n: int = 5) -> List[Dict[str, Any]]:
    """
    Return the top N domains with the most repeated issues
    (cycles where the same domain had failures multiple times).
    """
    log = get_improvement_log()
    history = log.get_history(n=200)

    domain_failures: Dict[str, int] = {}
    domain_last_issue: Dict[str, str] = {}

    for entry in history:
        domain = entry.get("domain", "unknown")
        tests_pass = entry.get("validation_result", {}).get("tests_pass", True)
        findings = entry.get("scan_findings", [])
        has_problems = (
            not tests_pass
            or any(f.startswith("FAILED") or f.startswith("CRITICAL") for f in findings)
        )
        if has_problems:
            domain_failures[domain] = domain_failures.get(domain, 0) + 1
            domain_last_issue[domain] = entry.get("timestamp", "")

    # Sort by failure count descending
    sorted_domains = sorted(domain_failures.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "domain": d,
            "failure_count": count,
            "last_issue": domain_last_issue.get(d, ""),
        }
        for d, count in sorted_domains[:n]
    ]


def get_failed_interventions() -> List[Dict[str, Any]]:
    """
    Return cycles where the architect took action but validation failed.
    These represent interventions that made things worse or didn't help.
    """
    log = get_improvement_log()
    history = log.get_history(n=200)

    failed = []
    for entry in history:
        actions = entry.get("actions_taken", [])
        tests_pass = entry.get("validation_result", {}).get("tests_pass", True)
        if actions and not tests_pass:
            failed.append({
                "cycle_id": entry.get("cycle_id", "unknown"),
                "domain": entry.get("domain", "unknown"),
                "timestamp": entry.get("timestamp", ""),
                "actions_count": len(actions),
                "diagnosis": entry.get("gpt_diagnosis", "")[:200],
            })

    return failed


def get_domain_health_trend(domain: str, n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the last N cycle results for a specific domain,
    showing health trajectory over time.
    """
    log = get_improvement_log()
    history = log.get_history(n=500)

    domain_entries = [e for e in history if e.get("domain") == domain]
    recent = domain_entries[-n:]

    trend = []
    for entry in recent:
        ms = entry.get("machine_summary", {})
        scores = entry.get("architecture_scores", {})
        trend.append({
            "cycle_id": entry.get("cycle_id", ""),
            "timestamp": entry.get("timestamp", ""),
            "tests_pass": entry.get("validation_result", {}).get("tests_pass", False),
            "actions_count": len(entry.get("actions_taken", [])),
            "findings_count": len(entry.get("scan_findings", [])),
            "architecture_health": (
                ms.get("architecture_health_score")
                or scores.get("architecture_health_score", None)
            ),
            "diagnosis": entry.get("architecture_diagnosis", entry.get("gpt_diagnosis", "")[:100]),
        })

    return trend


def get_unresolved_flags() -> List[str]:
    """
    Collect all unresolved architecture flags from recent history.
    A flag is unresolved if it appeared in a recent cycle's machine_summary
    and has not been marked resolved in a subsequent cycle.
    """
    log = get_improvement_log()
    history = log.get_history(n=50)

    all_flags: List[str] = []
    resolved: set = set()

    for entry in reversed(history):
        ms = entry.get("machine_summary", {})
        if isinstance(ms, dict):
            flags = ms.get("unresolved_flags", [])
            for f in flags:
                if isinstance(f, str) and f not in resolved:
                    all_flags.append(f)

        # Check if actions resolved prior flags
        actions = entry.get("actions_taken", [])
        tests_pass = entry.get("validation_result", {}).get("tests_pass", False)
        if actions and tests_pass:
            # Assume successful actions resolve flags from that domain
            domain = entry.get("domain", "")
            resolved.add(domain)

    # Deduplicate preserving order
    seen: set = set()
    unique: List[str] = []
    for f in all_flags:
        if f not in seen:
            unique.append(f)
            seen.add(f)

    return unique[:20]
