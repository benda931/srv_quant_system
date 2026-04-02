"""
services/run_context.py
========================
RunContext — canonical metadata container that flows through the entire
analytics pipeline. Every engine, service, and persistence call receives
the same RunContext so that lineage is traceable.

Usage:
    ctx = RunContext.create(settings)
    results = engine_service.compute_all(ctx)
    # ctx.run_id, ctx.run_date, ctx.regime available everywhere
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config.settings import Settings


@dataclass
class RunContext:
    """
    Immutable metadata for a single analytics run.

    Created once at pipeline start, threaded through every engine call.
    Persisted to DuckDB analytics.runs for full audit trail.
    """

    # Identity
    run_id: int = -1                         # DB-assigned run ID (set after write_run)
    run_uuid: str = ""                       # UUID for cross-system correlation
    run_date: str = ""                       # ISO date (YYYY-MM-DD)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Config snapshot (frozen at run start)
    settings: Optional[Settings] = field(default=None, repr=False)

    # Data health
    data_health_label: str = "UNKNOWN"
    data_health_score: float = 0.0
    prices_rows: int = 0
    prices_cols: int = 0
    prices_latest_date: str = ""

    # Regime (set after QuantEngine runs)
    regime: str = "UNKNOWN"
    vix_level: float = 0.0
    avg_correlation: float = 0.0
    safety_label: str = "UNKNOWN"
    safety_score: float = 0.0

    # Pipeline progress
    steps_completed: list = field(default_factory=list)
    steps_failed: list = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    # Timing
    finished_at: Optional[datetime] = None
    duration_s: float = 0.0

    @classmethod
    def create(cls, settings: Settings) -> "RunContext":
        """Factory: create a fresh RunContext from settings."""
        return cls(
            run_uuid=str(uuid.uuid4())[:8],
            run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            started_at=datetime.now(timezone.utc),
            settings=settings,
        )

    def mark_step(self, step_name: str, success: bool = True, error: str = ""):
        """Record a pipeline step completion."""
        if success:
            self.steps_completed.append(step_name)
        else:
            self.steps_failed.append(step_name)
            if error:
                self.errors[step_name] = error[:200]

    def finalize(self):
        """Mark run as complete and compute duration."""
        self.finished_at = datetime.now(timezone.utc)
        self.duration_s = (self.finished_at - self.started_at).total_seconds()

    @property
    def is_healthy(self) -> bool:
        return len(self.steps_failed) == 0

    @property
    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging/persistence."""
        return {
            "run_id": self.run_id,
            "run_uuid": self.run_uuid,
            "run_date": self.run_date,
            "regime": self.regime,
            "vix": self.vix_level,
            "safety": self.safety_label,
            "data_health": self.data_health_label,
            "steps_ok": len(self.steps_completed),
            "steps_fail": len(self.steps_failed),
            "duration_s": round(self.duration_s, 1),
        }
