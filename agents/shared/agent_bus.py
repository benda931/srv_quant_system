"""
agents/shared/agent_bus.py
----------------------------
Institutional Event Bus — Runtime Control Plane messaging layer.

Extends the original AgentBus (scripts/agent_bus.py) with:
  - Typed EventEnvelope with full metadata
  - Correlation chains and event lineage
  - Dead letter queue for malformed/expired events
  - Lightweight payload validation
  - Capped event history (1000 events)

Backward compatible:
    from agents.shared.agent_bus import get_bus, AgentBus
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.agent_bus import AgentBus  # noqa: F401 — re-export for backward compat

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent  # srv_quant_system/
EVENT_HISTORY_FILE = ROOT / "logs" / "event_history.json"
DEAD_LETTER_FILE = ROOT / "logs" / "dead_letters.json"

log = logging.getLogger("institutional_bus")

# ── Priority levels ──────────────────────────────────────────────────────────
VALID_PRIORITIES = {"CRITICAL", "HIGH", "NORMAL", "LOW"}

# ── Event Envelope ───────────────────────────────────────────────────────────
@dataclass
class EventEnvelope:
    """Typed event envelope with full institutional metadata."""
    event_id: str = ""                # UUID
    event_type: str = ""              # "machine_summary" / "report" / "alert" / "veto"
    producer: str = ""                # agent name
    timestamp: str = ""               # ISO 8601
    run_id: str = ""                  # UUID for this run
    correlation_id: str = ""          # UUID linking related events
    workflow_id: str = ""             # which workflow cycle
    schema_version: str = "1.0"
    payload: Dict[str, Any] = field(default_factory=dict)
    payload_summary: str = ""         # one-line summary
    priority: str = "NORMAL"          # CRITICAL/HIGH/NORMAL/LOW
    ttl_seconds: int = 0              # 0 = no expiry

    def __post_init__(self) -> None:
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
        if self.priority not in VALID_PRIORITIES:
            self.priority = "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventEnvelope":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        try:
            created = datetime.fromisoformat(self.timestamp)
            return datetime.now(timezone.utc) - created > timedelta(seconds=self.ttl_seconds)
        except (ValueError, TypeError):
            return False


# ── Required payload fields per agent (lightweight schema) ───────────────────
PAYLOAD_SCHEMAS: Dict[str, List[str]] = {
    "methodology": ["status"],
    "optimizer": ["status"],
    "risk_guardian": ["status"],
    "regime_forecaster": ["status"],
    "execution": ["status"],
    "data_scout": ["status"],
    "portfolio_construction": ["status"],
    "alpha_decay": ["status"],
    "math": ["status"],
    "auto_improve": ["status"],
}

MAX_EVENT_HISTORY = 1000


# ── Institutional Bus ────────────────────────────────────────────────────────
class InstitutionalBus(AgentBus):
    """
    Extends AgentBus with envelope-based messaging, correlation lineage,
    dead letter queue, and payload validation.

    All original AgentBus methods (publish, latest, history, all_latest, clear)
    continue to work unchanged.
    """

    def __init__(self, bus_file: Optional[Path] = None,
                 event_history_file: Path = EVENT_HISTORY_FILE,
                 dead_letter_file: Path = DEAD_LETTER_FILE) -> None:
        # Call parent with default if no bus_file given
        if bus_file is not None:
            super().__init__(bus_file)
        else:
            super().__init__()
        self._event_history_path = event_history_file
        self._dead_letter_path = dead_letter_file
        self._event_history_path.parent.mkdir(parents=True, exist_ok=True)
        self._dead_letter_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Event history persistence ────────────────────────────────────────────
    def _load_event_history(self) -> List[Dict[str, Any]]:
        if self._event_history_path.exists():
            try:
                return json.loads(self._event_history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_event_history(self, events: List[Dict[str, Any]]) -> None:
        # Cap at MAX_EVENT_HISTORY
        if len(events) > MAX_EVENT_HISTORY:
            events = events[-MAX_EVENT_HISTORY:]
        self._event_history_path.write_text(
            json.dumps(events, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    # ── Dead letter queue persistence ────────────────────────────────────────
    def _load_dead_letters(self) -> List[Dict[str, Any]]:
        if self._dead_letter_path.exists():
            try:
                return json.loads(self._dead_letter_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_dead_letters(self, letters: List[Dict[str, Any]]) -> None:
        self._dead_letter_path.write_text(
            json.dumps(letters, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _add_dead_letter(self, envelope: EventEnvelope, reason: str) -> None:
        letters = self._load_dead_letters()
        letters.append({
            "envelope": envelope.to_dict(),
            "reason": reason,
            "rejected_at": datetime.now(timezone.utc).isoformat(),
        })
        # Cap dead letters at 500
        if len(letters) > 500:
            letters = letters[-500:]
        self._save_dead_letters(letters)
        log.warning("Dead letter: producer=%s reason=%s event_id=%s",
                     envelope.producer, reason, envelope.event_id)

    # ── Publish envelope ─────────────────────────────────────────────────────
    def publish_envelope(self, envelope: EventEnvelope) -> bool:
        """
        Publish an event envelope to the bus with full metadata.
        Also publishes to the legacy bus format for backward compat.
        Returns True if accepted, False if rejected (dead-lettered).
        """
        # Validate envelope has minimum required fields
        if not envelope.producer:
            self._add_dead_letter(envelope, "missing producer")
            return False
        if not envelope.event_type:
            self._add_dead_letter(envelope, "missing event_type")
            return False

        # Check TTL expiry
        if envelope.is_expired():
            self._add_dead_letter(envelope, "expired (TTL)")
            return False

        # Validate payload if schema exists
        validation = self.validate_payload(envelope.producer, envelope.payload)
        if not validation["valid"]:
            self._add_dead_letter(envelope, f"schema violation: {validation['errors']}")
            return False

        # Store in event history
        events = self._load_event_history()
        events.append(envelope.to_dict())
        self._save_event_history(events)

        # Also publish to legacy bus for backward compat
        self.publish(envelope.producer, envelope.payload)

        log.debug("Envelope published: producer=%s type=%s id=%s",
                   envelope.producer, envelope.event_type, envelope.event_id)
        return True

    # ── Latest envelope ──────────────────────────────────────────────────────
    def latest_envelope(self, agent_name: str) -> Optional[EventEnvelope]:
        """Return the most recent EventEnvelope for this agent, or None."""
        events = self._load_event_history()
        for event in reversed(events):
            if event.get("producer") == agent_name:
                return EventEnvelope.from_dict(event)
        return None

    # ── Event lineage ────────────────────────────────────────────────────────
    def get_event_lineage(self, correlation_id: str) -> List[EventEnvelope]:
        """Return all events sharing a correlation_id, in chronological order."""
        events = self._load_event_history()
        chain = [
            EventEnvelope.from_dict(e)
            for e in events
            if e.get("correlation_id") == correlation_id
        ]
        return sorted(chain, key=lambda e: e.timestamp)

    # ── Dead letter queue ────────────────────────────────────────────────────
    def get_dead_letters(self) -> List[Dict[str, Any]]:
        """Return all dead-lettered events (malformed, expired, rejected)."""
        return self._load_dead_letters()

    # ── Payload validation ───────────────────────────────────────────────────
    def validate_payload(self, agent_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight schema check — ensures required fields are present.
        Returns {"valid": True/False, "errors": [...]}.
        """
        required = PAYLOAD_SCHEMAS.get(agent_name, [])
        if not required:
            return {"valid": True, "errors": []}

        missing = [f for f in required if f not in payload]
        if missing:
            return {"valid": False, "errors": [f"missing fields: {missing}"]}
        return {"valid": True, "errors": []}

    # ── Query helpers ────────────────────────────────────────────────────────
    def get_events_by_type(self, event_type: str, limit: int = 50) -> List[EventEnvelope]:
        """Return recent events of a given type."""
        events = self._load_event_history()
        matched = [
            EventEnvelope.from_dict(e)
            for e in events
            if e.get("event_type") == event_type
        ]
        return matched[-limit:]

    def get_events_by_producer(self, producer: str, limit: int = 50) -> List[EventEnvelope]:
        """Return recent events from a given producer."""
        events = self._load_event_history()
        matched = [
            EventEnvelope.from_dict(e)
            for e in events
            if e.get("producer") == producer
        ]
        return matched[-limit:]

    def get_events_by_workflow(self, workflow_id: str) -> List[EventEnvelope]:
        """Return all events from a specific workflow cycle."""
        events = self._load_event_history()
        return [
            EventEnvelope.from_dict(e)
            for e in events
            if e.get("workflow_id") == workflow_id
        ]

    def get_event_count(self) -> int:
        """Return total event count in history."""
        return len(self._load_event_history())

    def compute_machine_summary(self) -> Dict[str, Any]:
        """Machine-readable summary of bus state."""
        events = self._load_event_history()
        dead_letters = self._load_dead_letters()
        producers = set(e.get("producer", "") for e in events)
        types = {}
        for e in events:
            t = e.get("event_type", "unknown")
            types[t] = types.get(t, 0) + 1

        return {
            "total_events": len(events),
            "total_dead_letters": len(dead_letters),
            "unique_producers": sorted(producers - {""}),
            "event_types": types,
            "max_history": MAX_EVENT_HISTORY,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_institutional_bus: Optional[InstitutionalBus] = None


def get_bus() -> InstitutionalBus:
    """
    Return singleton InstitutionalBus.
    Backward compatible — InstitutionalBus extends AgentBus, so all
    existing .publish() / .latest() / .history() calls work unchanged.
    """
    global _institutional_bus
    if _institutional_bus is None:
        _institutional_bus = InstitutionalBus()
    return _institutional_bus


__all__ = ["AgentBus", "InstitutionalBus", "EventEnvelope", "get_bus"]
