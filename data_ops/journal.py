"""
data_ops/journal.py

PM Decision Journal — SQLite-backed audit trail for the SRV DSS.

Stores every PM decision alongside the model's attribution scores so that
agreement / disagreement between the model and the PM can be tracked and
evaluated over time.

Schema
------
decisions
    id               INTEGER PK AUTOINCREMENT
    timestamp        TEXT    ISO8601 UTC  e.g. "2026-03-21T14:32:01.123456Z"
    sector           TEXT    e.g. "XLK"
    model_direction  TEXT    LONG | SHORT | NEUTRAL
    pm_direction     TEXT    LONG | SHORT | NEUTRAL | OVERRIDE
    conviction_score REAL    0–100 composite conviction from QuantEngine
    notes            TEXT    free-form PM annotation
    sds_score        REAL    Statistical Dislocation Score   (0–1)
    fjs_score        REAL    Fundamental Justification Score (0–1)
    mss_score        REAL    Macro Shift Score               (0–1)
    stf_score        REAL    Structural Trend Filter         (0–1)
    regime           TEXT    CALM | NORMAL | TENSION | CRISIS

overrides
    id           INTEGER PK AUTOINCREMENT
    decision_id  INTEGER FK → decisions.id
    reason       TEXT    required: why the PM deviated from the model
    outcome      TEXT    nullable: set when resolved ("CORRECT" / "INCORRECT" / free text)
    resolved_at  TEXT    nullable: ISO8601 UTC timestamp of resolution

Thread safety
-------------
Connection-per-call pattern: every public method opens a fresh sqlite3.connect(),
executes within a with-block (auto-commit / rollback), and closes on exit.
No shared connection state between calls.  WAL journal mode is set at table
creation time so concurrent readers do not block writers.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

from analytics.attribution import AttributionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed enum values — validated on write, never on read
# ---------------------------------------------------------------------------
_VALID_MODEL_DIRECTIONS = {"LONG", "SHORT", "NEUTRAL"}
_VALID_PM_DIRECTIONS = {"LONG", "SHORT", "NEUTRAL", "OVERRIDE"}


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    sector           TEXT    NOT NULL,
    model_direction  TEXT    NOT NULL,
    pm_direction     TEXT    NOT NULL,
    conviction_score REAL,
    notes            TEXT,
    sds_score        REAL,
    fjs_score        REAL,
    mss_score        REAL,
    stf_score        REAL,
    regime           TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_sector    ON decisions (sector);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions (timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_regime    ON decisions (regime);

CREATE TABLE IF NOT EXISTS overrides (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id  INTEGER NOT NULL REFERENCES decisions (id) ON DELETE CASCADE,
    reason       TEXT    NOT NULL,
    outcome      TEXT,
    resolved_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_overrides_decision_id ON overrides (decision_id);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    """Return the current UTC time as an ISO8601 string with 'Z' suffix."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_real(x: Any) -> Optional[float]:
    """Convert to float for SQLite REAL storage; return None if not finite."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that opens a connection, yields it, commits on success,
    rolls back on exception, and always closes — guaranteeing no file-handle
    leaks even under WAL mode on Windows.
    """
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# PMJournal
# ---------------------------------------------------------------------------
class PMJournal:
    """
    SQLite-backed audit trail for PM decisions vs. model recommendations.

    Parameters
    ----------
    db_path : Path
        Absolute path to the SQLite database file.  Created (with parent
        directories) on first instantiation.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("PMJournal initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_decision(
        self,
        sector: str,
        attribution_result: AttributionResult,
        pm_direction: str,
        notes: str = "",
        *,
        model_direction: str = "NEUTRAL",
        conviction_score: float = float("nan"),
        regime: str = "UNKNOWN",
    ) -> int:
        """
        Persist one PM decision alongside the model's attribution scores.

        Parameters
        ----------
        sector : str
            Sector ETF ticker, e.g. "XLK".
        attribution_result : AttributionResult
            Output from analytics.attribution.compute_attribution_row.
            Provides sds, fjs, mss, stf scores automatically.
        pm_direction : str
            PM's chosen direction: LONG | SHORT | NEUTRAL | OVERRIDE.
        notes : str
            Free-form PM annotation (trade rationale, caveats, etc.).
        model_direction : str
            Direction implied by the model signal: LONG | SHORT | NEUTRAL.
        conviction_score : float
            0–100 composite conviction score from QuantEngine.calculate_conviction_score.
        regime : str
            Market regime label at time of decision: CALM | NORMAL | TENSION | CRISIS.

        Returns
        -------
        int
            Auto-assigned decision_id (primary key).
        """
        pm_dir = pm_direction.strip().upper()
        model_dir = model_direction.strip().upper()

        if pm_dir not in _VALID_PM_DIRECTIONS:
            raise ValueError(
                f"pm_direction must be one of {_VALID_PM_DIRECTIONS}, got {pm_dir!r}."
            )
        if model_dir not in _VALID_MODEL_DIRECTIONS:
            raise ValueError(
                f"model_direction must be one of {_VALID_MODEL_DIRECTIONS}, got {model_dir!r}."
            )

        row = (
            _utcnow(),
            sector.strip().upper(),
            model_dir,
            pm_dir,
            _safe_real(conviction_score),
            notes or None,
            _safe_real(attribution_result.sds),
            _safe_real(attribution_result.fjs),
            _safe_real(attribution_result.mss),
            _safe_real(attribution_result.stf),
            regime.strip().upper() or None,
        )

        sql = """
            INSERT INTO decisions
                (timestamp, sector, model_direction, pm_direction,
                 conviction_score, notes, sds_score, fjs_score,
                 mss_score, stf_score, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with _db(self.db_path) as conn:
            cur = conn.execute(sql, row)
            decision_id = cur.lastrowid

        logger.debug(
            "Logged decision id=%d sector=%s model=%s pm=%s regime=%s",
            decision_id, sector, model_dir, pm_dir, regime,
        )
        return decision_id

    def log_override(self, decision_id: int, reason: str) -> int:
        """
        Record that the PM overrode or deviated from the model recommendation.

        An override is distinct from pm_direction=OVERRIDE: it is an audit
        entry that links a justification narrative to a specific decision.
        Multiple overrides can be attached to one decision.

        Parameters
        ----------
        decision_id : int
            Primary key of the decision being annotated.
        reason : str
            Mandatory explanation of why the PM deviated from the model.

        Returns
        -------
        int
            Auto-assigned override_id.
        """
        if not reason or not reason.strip():
            raise ValueError("reason must not be empty.")

        sql = """
            INSERT INTO overrides (decision_id, reason, outcome, resolved_at)
            VALUES (?, ?, NULL, NULL)
        """
        with _db(self.db_path) as conn:
            cur = conn.execute(sql, (decision_id, reason.strip()))
            override_id = cur.lastrowid

        logger.debug(
            "Logged override id=%d for decision_id=%d", override_id, decision_id
        )
        return override_id

    def resolve_override(self, override_id: int, outcome: str) -> None:
        """
        Mark an override as resolved with a retrospective outcome assessment.

        Parameters
        ----------
        override_id : int
            Primary key of the override row to resolve.
        outcome : str
            Outcome assessment, typically "CORRECT" or "INCORRECT", but any
            free text is accepted.  get_override_accuracy() treats any value
            whose stripped upper-case form starts with "CORRECT" as a PM win.
        """
        if not outcome or not outcome.strip():
            raise ValueError("outcome must not be empty.")

        sql = """
            UPDATE overrides
               SET outcome = ?, resolved_at = ?
             WHERE id = ?
        """
        with _db(self.db_path) as conn:
            conn.execute(sql, (outcome.strip(), _utcnow(), override_id))

        logger.debug("Resolved override id=%d outcome=%r", override_id, outcome)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_recent(self, n: int = 50) -> pd.DataFrame:
        """
        Return the most recent n decisions joined with their override count.

        Columns
        -------
        id, timestamp, sector, model_direction, pm_direction,
        conviction_score, notes, sds_score, fjs_score, mss_score, stf_score,
        regime, override_count, agreement
        """
        sql = """
            SELECT
                d.id,
                d.timestamp,
                d.sector,
                d.model_direction,
                d.pm_direction,
                d.conviction_score,
                d.notes,
                d.sds_score,
                d.fjs_score,
                d.mss_score,
                d.stf_score,
                d.regime,
                COUNT(o.id) AS override_count
            FROM decisions d
            LEFT JOIN overrides o ON o.decision_id = d.id
            GROUP BY d.id
            ORDER BY d.timestamp DESC
            LIMIT ?
        """
        with _db(self.db_path) as conn:
            rows = conn.execute(sql, (max(1, int(n)),)).fetchall()

        df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(
            columns=[
                "id", "timestamp", "sector", "model_direction", "pm_direction",
                "conviction_score", "notes", "sds_score", "fjs_score",
                "mss_score", "stf_score", "regime", "override_count",
            ]
        )
        if not df.empty:
            df["agreement"] = df["model_direction"] == df["pm_direction"]
        return df

    def get_override_accuracy(self) -> Dict[str, Any]:
        """
        Compare PM vs. model hit rate on decisions where they disagreed.

        Resolution logic: an override is counted as a PM win when its outcome
        (stripped, upper-cased) starts with "CORRECT".

        Returns
        -------
        dict with keys:
            n_disagreements  — total overrides logged
            n_resolved       — overrides with a non-null outcome
            n_pm_correct     — resolved overrides where PM was right
            n_model_correct  — resolved overrides where model was right (complementary)
            pm_accuracy      — n_pm_correct / n_resolved  (None if 0 resolved)
            model_accuracy   — n_model_correct / n_resolved (None if 0 resolved)
            regime_breakdown — dict[regime, dict] with per-regime accuracy stats
        """
        sql_all = """
            SELECT
                o.id,
                o.outcome,
                d.regime,
                d.model_direction,
                d.pm_direction
            FROM overrides o
            JOIN decisions d ON d.id = o.decision_id
        """
        with _db(self.db_path) as conn:
            rows = [dict(r) for r in conn.execute(sql_all).fetchall()]

        n_disagreements = len(rows)
        resolved = [r for r in rows if r["outcome"] is not None]
        n_resolved = len(resolved)

        def _is_pm_correct(outcome: str) -> bool:
            return str(outcome).strip().upper().startswith("CORRECT")

        n_pm_correct = sum(1 for r in resolved if _is_pm_correct(r["outcome"]))
        n_model_correct = n_resolved - n_pm_correct

        pm_accuracy = (float(n_pm_correct) / n_resolved) if n_resolved > 0 else None
        model_accuracy = (float(n_model_correct) / n_resolved) if n_resolved > 0 else None

        # Per-regime breakdown
        regime_breakdown: Dict[str, Dict[str, Any]] = {}
        all_regimes = {r["regime"] for r in rows}
        for regime in sorted(all_regimes):
            regime_rows = [r for r in rows if r["regime"] == regime]
            regime_resolved = [r for r in regime_rows if r["outcome"] is not None]
            n_r = len(regime_rows)
            n_rr = len(regime_resolved)
            n_rpc = sum(1 for r in regime_resolved if _is_pm_correct(r["outcome"]))
            regime_breakdown[str(regime)] = {
                "n_overrides": n_r,
                "n_resolved": n_rr,
                "n_pm_correct": n_rpc,
                "pm_accuracy": float(n_rpc) / n_rr if n_rr > 0 else None,
            }

        return {
            "n_disagreements": n_disagreements,
            "n_resolved": n_resolved,
            "n_pm_correct": n_pm_correct,
            "n_model_correct": n_model_correct,
            "pm_accuracy": pm_accuracy,
            "model_accuracy": model_accuracy,
            "regime_breakdown": regime_breakdown,
        }

    def get_sector_history(self, sector: str, days: int = 90) -> pd.DataFrame:
        """
        Return all decisions for a given sector in the past `days` calendar days.

        Columns
        -------
        id, timestamp, model_direction, pm_direction, conviction_score,
        notes, sds_score, fjs_score, mss_score, stf_score, regime,
        override_count, resolved_outcomes
        """
        cutoff = _cutoff_iso(days)
        sql = """
            SELECT
                d.id,
                d.timestamp,
                d.model_direction,
                d.pm_direction,
                d.conviction_score,
                d.notes,
                d.sds_score,
                d.fjs_score,
                d.mss_score,
                d.stf_score,
                d.regime,
                COUNT(o.id)                                    AS override_count,
                GROUP_CONCAT(o.outcome, ' | ')                 AS resolved_outcomes
            FROM decisions d
            LEFT JOIN overrides o ON o.decision_id = d.id
            WHERE d.sector = ?
              AND d.timestamp >= ?
            GROUP BY d.id
            ORDER BY d.timestamp DESC
        """
        with _db(self.db_path) as conn:
            rows = conn.execute(sql, (sector.strip().upper(), cutoff)).fetchall()

        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(
            columns=[
                "id", "timestamp", "model_direction", "pm_direction",
                "conviction_score", "notes", "sds_score", "fjs_score",
                "mss_score", "stf_score", "regime",
                "override_count", "resolved_outcomes",
            ]
        )

    # ------------------------------------------------------------------
    # Maintenance / introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return lightweight summary statistics about the journal contents.

        Useful for dashboard status cards without pulling full DataFrames.

        Returns
        -------
        dict with keys:
            n_decisions, n_overrides, n_resolved_overrides,
            sectors, first_entry, last_entry
        """
        sql = """
            SELECT
                (SELECT COUNT(*) FROM decisions)                         AS n_decisions,
                (SELECT COUNT(*) FROM overrides)                         AS n_overrides,
                (SELECT COUNT(*) FROM overrides WHERE outcome IS NOT NULL) AS n_resolved_overrides,
                (SELECT MIN(timestamp) FROM decisions)                   AS first_entry,
                (SELECT MAX(timestamp) FROM decisions)                   AS last_entry
        """
        sql_sectors = "SELECT DISTINCT sector FROM decisions ORDER BY sector"
        with _db(self.db_path) as conn:
            row = dict(conn.execute(sql).fetchone())
            sectors = [r[0] for r in conn.execute(sql_sectors).fetchall()]

        row["sectors"] = sectors
        return row

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables and indexes if they do not already exist."""
        with _db(self.db_path) as conn:
            conn.executescript(_DDL)
        logger.debug("Database schema verified at %s", self.db_path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _cutoff_iso(days: int) -> str:
    """Return ISO8601 UTC string for `days` calendar days ago."""
    from datetime import timedelta
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=max(1, int(days)))
    return cutoff.isoformat().replace("+00:00", "Z")


def open_journal(db_path: Optional[Path] = None) -> PMJournal:
    """
    Convenience factory that resolves the default journal path from the project root.

    Parameters
    ----------
    db_path : Path, optional
        Override the default path.  Defaults to <project_root>/logs/pm_journal.db.

    Returns
    -------
    PMJournal
    """
    if db_path is None:
        # Resolve relative to this file: data_ops/ → project_root/ → logs/
        project_root = Path(__file__).resolve().parents[1]
        db_path = project_root / "logs" / "pm_journal.db"
    return PMJournal(db_path)
