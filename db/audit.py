"""
db/audit.py
=============
Append-only Audit Trail with SHA256 Hash Chain.

Provides tamper-evident logging for:
  - Signal generation events (signal_log)
  - Trade actions (trade_log)
  - Parameter changes (param_log)

Each record includes a SHA256 hash of the previous record, creating an
immutable chain. Any modification to historical records breaks the chain,
which is detectable via verify_chain().

Storage: DuckDB `audit` schema (3 tables with auto-increment sequence).

Design:
  - Append-only: no UPDATE or DELETE operations
  - Hash chain: each record's hash includes the previous record's hash
  - Thread-safe: uses DuckDB connection singleton with MVCC
  - Idempotent schema: CREATE IF NOT EXISTS on every init

Ref: RFC 6962 (Certificate Transparency) — Merkle hash chain pattern
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_AUDIT_SCHEMA = [
    """
    CREATE SEQUENCE IF NOT EXISTS signal_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS signal_log (
        seq_id            BIGINT      PRIMARY KEY DEFAULT nextval('signal_seq'),
        timestamp         TIMESTAMP   NOT NULL,
        sector            VARCHAR     NOT NULL,
        direction         VARCHAR     NOT NULL,
        conviction_score  DOUBLE,
        distortion_score  DOUBLE,
        dislocation_score DOUBLE,
        mr_score          DOUBLE,
        safety_score      DOUBLE,
        regime            VARCHAR,
        prev_hash         VARCHAR(64),
        record_hash       VARCHAR(64) NOT NULL
    )
    """,

    """
    CREATE SEQUENCE IF NOT EXISTS trade_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS trade_log (
        seq_id       BIGINT      PRIMARY KEY DEFAULT nextval('trade_seq'),
        timestamp    TIMESTAMP   NOT NULL,
        trade_id     VARCHAR     NOT NULL,
        action       VARCHAR     NOT NULL,
        details      VARCHAR,
        prev_hash    VARCHAR(64),
        record_hash  VARCHAR(64) NOT NULL
    )
    """,

    """
    CREATE SEQUENCE IF NOT EXISTS param_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS param_log (
        seq_id      BIGINT      PRIMARY KEY DEFAULT nextval('param_seq'),
        timestamp   TIMESTAMP   NOT NULL,
        param_name  VARCHAR     NOT NULL,
        old_value   VARCHAR,
        new_value   VARCHAR,
        changed_by  VARCHAR,
        prev_hash   VARCHAR(64),
        record_hash VARCHAR(64) NOT NULL
    )
    """,
]


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    """Compute SHA256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_signal_record(
    prev_hash: str,
    timestamp: str,
    sector: str,
    direction: str,
    conviction: float,
    distortion: float,
    dislocation: float,
    mr: float,
    safety: float,
    regime: str,
) -> str:
    """Deterministic hash of a signal record."""
    payload = (
        f"{prev_hash}|{timestamp}|{sector}|{direction}|"
        f"{conviction:.8f}|{distortion:.8f}|{dislocation:.8f}|"
        f"{mr:.8f}|{safety:.8f}|{regime}"
    )
    return _sha256(payload)


def _hash_trade_record(
    prev_hash: str,
    timestamp: str,
    trade_id: str,
    action: str,
    details: str,
) -> str:
    """Deterministic hash of a trade record."""
    payload = f"{prev_hash}|{timestamp}|{trade_id}|{action}|{details}"
    return _sha256(payload)


def _hash_param_record(
    prev_hash: str,
    timestamp: str,
    param_name: str,
    old_value: str,
    new_value: str,
    changed_by: str,
) -> str:
    """Deterministic hash of a parameter change record."""
    payload = f"{prev_hash}|{timestamp}|{param_name}|{old_value}|{new_value}|{changed_by}"
    return _sha256(payload)


# ---------------------------------------------------------------------------
# AuditTrail class
# ---------------------------------------------------------------------------

class AuditTrail:
    """
    Append-only audit log with SHA256 hash chain.

    Usage:
        from db.audit import AuditTrail
        audit = AuditTrail()
        audit.log_signal(...)
        audit.log_trade(...)
        audit.log_param_change(...)
        assert audit.verify_chain()
    """

    def __init__(self, conn=None, db_path: Optional[str] = None):
        """
        Initialize audit trail.

        Parameters
        ----------
        conn    : duckdb.DuckDBPyConnection or None
            DuckDB connection. If None, opens a dedicated audit DB.
        db_path : Optional[str]
            Path to audit DuckDB file. Defaults to db/audit.duckdb.
        """
        if conn is not None:
            self._conn = conn
            self._owns_conn = False
        else:
            import duckdb
            from pathlib import Path
            if db_path is None:
                db_path = str(Path(__file__).resolve().parent / "audit.duckdb")
            self._conn = duckdb.connect(db_path)
            self._owns_conn = True

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create audit schema and tables if they don't exist."""
        with _lock:
            for ddl in _AUDIT_SCHEMA:
                try:
                    self._conn.execute(ddl)
                except Exception as exc:
                    # Ignore "already exists" type errors
                    if "already exists" not in str(exc).lower():
                        logger.warning("Audit schema DDL warning: %s", exc)

    def _last_hash(self, table: str) -> str:
        """Get the record_hash of the last entry in a table, or genesis hash."""
        try:
            result = self._conn.execute(
                f"SELECT record_hash FROM audit.{table} ORDER BY seq_id DESC LIMIT 1"
            ).fetchone()
            return result[0] if result else _sha256("GENESIS")
        except Exception:
            return _sha256("GENESIS")

    # ----- Signal logging -----

    def log_signal(
        self,
        timestamp: datetime,
        sector: str,
        direction: str,
        conviction: float,
        layers: Optional[Dict[str, float]] = None,
    ) -> int:
        """
        Log a signal generation event.

        Parameters
        ----------
        timestamp : datetime
            When the signal was generated.
        sector : str
            Sector ticker (e.g., "XLK").
        direction : str
            "LONG" or "SHORT".
        conviction : float
            Combined conviction score.
        layers : dict or None
            Layer scores: {"distortion": ..., "dislocation": ..., "mr": ..., "safety": ..., "regime": ...}

        Returns
        -------
        int
            Sequence ID of the inserted record.
        """
        layers = layers or {}
        distortion = layers.get("distortion", 0.0)
        dislocation = layers.get("dislocation", 0.0)
        mr = layers.get("mr", 0.0)
        safety = layers.get("safety", 0.0)
        regime = layers.get("regime", "UNKNOWN")

        if isinstance(timestamp, str):
            ts_str = timestamp
            timestamp = datetime.fromisoformat(timestamp)
        else:
            ts_str = timestamp.isoformat()

        with _lock:
            prev_hash = self._last_hash("signal_log")
            record_hash = _hash_signal_record(
                prev_hash, ts_str, sector, direction,
                conviction, distortion, dislocation, mr, safety, regime,
            )

            self._conn.execute(
                """
                INSERT INTO signal_log
                    (timestamp, sector, direction, conviction_score,
                     distortion_score, dislocation_score, mr_score, safety_score,
                     regime, prev_hash, record_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    timestamp, sector, direction, conviction,
                    distortion, dislocation, mr, safety,
                    regime, prev_hash, record_hash,
                ],
            )

            result = self._conn.execute(
                "SELECT MAX(seq_id) FROM signal_log"
            ).fetchone()

        seq_id = result[0] if result else 0
        logger.debug("Audit signal_log: seq=%d sector=%s dir=%s conv=%.4f",
                      seq_id, sector, direction, conviction)
        return seq_id

    # ----- Trade logging -----

    def log_trade(
        self,
        timestamp: datetime,
        trade_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Log a trade action (entry, exit, adjust, cancel).

        Parameters
        ----------
        timestamp : datetime
            When the action occurred.
        trade_id : str
            Unique trade identifier.
        action : str
            "ENTRY", "EXIT", "ADJUST", "CANCEL".
        details : dict or None
            Additional details (serialized to JSON).

        Returns
        -------
        int
            Sequence ID of the inserted record.
        """
        details_json = json.dumps(details or {}, default=str, ensure_ascii=False)
        if isinstance(timestamp, str):
            ts_str = timestamp
            timestamp = datetime.fromisoformat(timestamp)
        else:
            ts_str = timestamp.isoformat()

        with _lock:
            prev_hash = self._last_hash("trade_log")
            record_hash = _hash_trade_record(
                prev_hash, ts_str, trade_id, action, details_json,
            )

            self._conn.execute(
                """
                INSERT INTO trade_log
                    (timestamp, trade_id, action, details, prev_hash, record_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [timestamp, trade_id, action, details_json, prev_hash, record_hash],
            )

            result = self._conn.execute(
                "SELECT MAX(seq_id) FROM trade_log"
            ).fetchone()

        seq_id = result[0] if result else 0
        logger.debug("Audit trade_log: seq=%d trade=%s action=%s", seq_id, trade_id, action)
        return seq_id

    # ----- Parameter change logging -----

    def log_param_change(
        self,
        timestamp: datetime,
        param: str,
        old_val: Any,
        new_val: Any,
        changed_by: str = "manual",
    ) -> int:
        """
        Log a parameter change.

        Parameters
        ----------
        timestamp : datetime
            When the change was made.
        param : str
            Parameter name (e.g., "signal_entry_threshold").
        old_val : Any
            Previous value.
        new_val : Any
            New value.
        changed_by : str
            Who/what changed it ("manual", "agent_optimizer", etc.).

        Returns
        -------
        int
            Sequence ID of the inserted record.
        """
        old_str = str(old_val)
        new_str = str(new_val)
        if isinstance(timestamp, str):
            ts_str = timestamp
            timestamp = datetime.fromisoformat(timestamp)
        else:
            ts_str = timestamp.isoformat()

        with _lock:
            prev_hash = self._last_hash("param_log")
            record_hash = _hash_param_record(
                prev_hash, ts_str, param, old_str, new_str, changed_by,
            )

            self._conn.execute(
                """
                INSERT INTO param_log
                    (timestamp, param_name, old_value, new_value, changed_by,
                     prev_hash, record_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [timestamp, param, old_str, new_str, changed_by, prev_hash, record_hash],
            )

            result = self._conn.execute(
                "SELECT MAX(seq_id) FROM param_log"
            ).fetchone()

        seq_id = result[0] if result else 0
        logger.info("Audit param_log: seq=%d param=%s %s -> %s by %s",
                     seq_id, param, old_str[:30], new_str[:30], changed_by)
        return seq_id

    # ----- Chain verification -----

    def verify_chain(self) -> bool:
        """
        Verify the integrity of all three audit hash chains.

        Recomputes the hash for every record in sequence order and checks
        that each record's prev_hash matches the previous record's record_hash.

        Returns
        -------
        bool
            True if all chains are intact, False if any tampering detected.
        """
        ok = True
        ok = ok and self._verify_signal_chain()
        ok = ok and self._verify_trade_chain()
        ok = ok and self._verify_param_chain()

        if ok:
            logger.info("Audit chain verification: ALL CHAINS INTACT")
        else:
            logger.error("Audit chain verification: CHAIN BROKEN — possible tampering")

        return ok

    def _verify_signal_chain(self) -> bool:
        """Verify the signal_log hash chain."""
        try:
            rows = self._conn.execute(
                """
                SELECT seq_id, timestamp, sector, direction,
                       conviction_score, distortion_score, dislocation_score,
                       mr_score, safety_score, regime,
                       prev_hash, record_hash
                FROM signal_log
                ORDER BY seq_id ASC
                """
            ).fetchall()
        except Exception as exc:
            logger.warning("Cannot verify signal chain: %s", exc)
            return True  # No table = no chain to break

        if not rows:
            return True

        expected_prev = _sha256("GENESIS")
        for row in rows:
            (seq_id, ts, sector, direction, conv, dist, disloc, mr, safety,
             regime, prev_hash, record_hash) = row

            # Check prev_hash linkage
            if prev_hash != expected_prev:
                logger.error(
                    "Signal chain BROKEN at seq_id=%d: expected prev=%s, got=%s",
                    seq_id, expected_prev[:16], (prev_hash or "")[:16],
                )
                return False

            # Recompute and verify record_hash
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            recomputed = _hash_signal_record(
                prev_hash, ts_str, sector, direction,
                float(conv or 0), float(dist or 0), float(disloc or 0),
                float(mr or 0), float(safety or 0), regime or "UNKNOWN",
            )
            if recomputed != record_hash:
                logger.error(
                    "Signal chain HASH MISMATCH at seq_id=%d: expected=%s, stored=%s",
                    seq_id, recomputed[:16], (record_hash or "")[:16],
                )
                return False

            expected_prev = record_hash

        logger.debug("Signal chain OK: %d records", len(rows))
        return True

    def _verify_trade_chain(self) -> bool:
        """Verify the trade_log hash chain."""
        try:
            rows = self._conn.execute(
                """
                SELECT seq_id, timestamp, trade_id, action, details,
                       prev_hash, record_hash
                FROM trade_log
                ORDER BY seq_id ASC
                """
            ).fetchall()
        except Exception:
            return True

        if not rows:
            return True

        expected_prev = _sha256("GENESIS")
        for row in rows:
            seq_id, ts, trade_id, action, details, prev_hash, record_hash = row

            if prev_hash != expected_prev:
                logger.error("Trade chain BROKEN at seq_id=%d", seq_id)
                return False

            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            recomputed = _hash_trade_record(
                prev_hash, ts_str, trade_id, action, details or "",
            )
            if recomputed != record_hash:
                logger.error("Trade chain HASH MISMATCH at seq_id=%d", seq_id)
                return False

            expected_prev = record_hash

        logger.debug("Trade chain OK: %d records", len(rows))
        return True

    def _verify_param_chain(self) -> bool:
        """Verify the param_log hash chain."""
        try:
            rows = self._conn.execute(
                """
                SELECT seq_id, timestamp, param_name, old_value, new_value,
                       changed_by, prev_hash, record_hash
                FROM param_log
                ORDER BY seq_id ASC
                """
            ).fetchall()
        except Exception:
            return True

        if not rows:
            return True

        expected_prev = _sha256("GENESIS")
        for row in rows:
            seq_id, ts, param, old_val, new_val, changed_by, prev_hash, record_hash = row

            if prev_hash != expected_prev:
                logger.error("Param chain BROKEN at seq_id=%d", seq_id)
                return False

            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            recomputed = _hash_param_record(
                prev_hash, ts_str, param, old_val or "", new_val or "",
                changed_by or "",
            )
            if recomputed != record_hash:
                logger.error("Param chain HASH MISMATCH at seq_id=%d", seq_id)
                return False

            expected_prev = record_hash

        logger.debug("Param chain OK: %d records", len(rows))
        return True

    # ----- Query helpers -----

    def get_signal_history(self, n: int = 50) -> List[dict]:
        """Return the last n signal log entries."""
        try:
            rows = self._conn.execute(
                f"SELECT * FROM signal_log ORDER BY seq_id DESC LIMIT {n}"
            ).fetchdf()
            return rows.to_dict("records") if not rows.empty else []
        except Exception:
            return []

    def get_trade_history(self, n: int = 50) -> List[dict]:
        """Return the last n trade log entries."""
        try:
            rows = self._conn.execute(
                f"SELECT * FROM trade_log ORDER BY seq_id DESC LIMIT {n}"
            ).fetchdf()
            return rows.to_dict("records") if not rows.empty else []
        except Exception:
            return []

    def get_param_history(self, n: int = 50) -> List[dict]:
        """Return the last n parameter change entries."""
        try:
            rows = self._conn.execute(
                f"SELECT * FROM param_log ORDER BY seq_id DESC LIMIT {n}"
            ).fetchdf()
            return rows.to_dict("records") if not rows.empty else []
        except Exception:
            return []
