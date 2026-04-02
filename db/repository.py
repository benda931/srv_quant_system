"""
db/repository.py
==================
Repository layer — structured read/write interface for canonical data access.

Replaces ad-hoc direct DuckDB queries with explicit, named access methods
that enforce data quality semantics.

Key concepts:
  - latest_successful_run(): returns run_id where steps_fail == 0
  - latest_run(): returns max(run_id) regardless of success
  - data_freshness(): returns staleness diagnostics for all tables
  - read_signals_for_run(): explicit run-scoped reads
  - write_data_quality(): persists quality checks per pipeline run

Usage:
    from db.repository import Repository
    repo = Repository(settings.db_path)
    run_id = repo.latest_successful_run()
    signals = repo.read_signals(run_id)
    freshness = repo.data_freshness()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("db.repository")


@dataclass
class RunSummary:
    """Compact summary of a pipeline run for diagnostics."""
    run_id: int
    run_date: str
    regime: str
    vix_level: float
    safety_label: str
    duration_s: float
    steps_ok: int
    steps_fail: int
    data_health: str
    is_successful: bool


@dataclass
class DataFreshness:
    """Staleness diagnostics for all canonical tables."""
    prices_latest: Optional[str] = None
    prices_stale_days: int = 0
    prices_rows: int = 0
    fundamentals_latest: Optional[str] = None
    fundamentals_stale_days: int = 0
    holdings_latest: Optional[str] = None
    runs_count: int = 0
    last_run_date: Optional[str] = None
    last_successful_run_id: int = -1
    last_successful_date: Optional[str] = None
    is_fresh: bool = False           # True if prices < 3 days old (weekend-aware)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataQualityCheck:
    """Result of a single data quality check."""
    check_name: str
    table_name: str
    status: str          # PASS / WARN / FAIL
    message: str
    value: float = 0.0   # Numeric metric (e.g., row count, null pct)
    threshold: float = 0.0


class Repository:
    """
    Canonical data access layer for the SRV Quantamental DSS.

    All reads go through this repository to ensure:
      - Explicit "latest successful run" semantics
      - Data quality awareness
      - Consistent error handling
      - Audit trail
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            from db.connection import get_connection
            self._conn = get_connection(self.db_path)
        return self._conn

    # ── Run metadata queries ─────────────────────────────────────────────

    def latest_run(self) -> Optional[int]:
        """Return the most recent run_id (successful or not)."""
        try:
            row = self.conn.execute(
                "SELECT max(run_id) FROM analytics.runs"
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    def latest_successful_run(self) -> Optional[int]:
        """
        Return the most recent run_id where ALL steps passed.
        Falls back to latest_run() if run_contexts table is empty.
        """
        try:
            row = self.conn.execute("""
                SELECT max(run_id) FROM analytics.run_contexts
                WHERE steps_fail = 0
            """).fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
        return self.latest_run()

    def run_summary(self, run_id: Optional[int] = None) -> Optional[RunSummary]:
        """Get summary for a specific run (or latest)."""
        if run_id is None:
            run_id = self.latest_run()
        if run_id is None:
            return None
        try:
            row = self.conn.execute("""
                SELECT run_id, run_date, regime, vix_level, safety_label,
                       duration_s, steps_ok, steps_fail, data_health
                FROM analytics.run_contexts
                WHERE run_id = ?
            """, [run_id]).fetchone()
            if row:
                return RunSummary(
                    run_id=int(row[0]),
                    run_date=str(row[1]),
                    regime=str(row[2] or "UNKNOWN"),
                    vix_level=float(row[3] or 0),
                    safety_label=str(row[4] or "UNKNOWN"),
                    duration_s=float(row[5] or 0),
                    steps_ok=int(row[6] or 0),
                    steps_fail=int(row[7] or 0),
                    data_health=str(row[8] or "UNKNOWN"),
                    is_successful=int(row[7] or 0) == 0,
                )
        except Exception:
            pass
        return None

    def run_history(self, n: int = 30) -> pd.DataFrame:
        """Return last N run summaries as a DataFrame."""
        try:
            return self.conn.execute(f"""
                SELECT run_id, run_date, regime, vix_level, safety_label,
                       duration_s, steps_ok, steps_fail, data_health
                FROM analytics.run_contexts
                ORDER BY run_id DESC
                LIMIT {n}
            """).fetchdf()
        except Exception:
            return pd.DataFrame()

    # ── Data freshness ───────────────────────────────────────────────────

    def data_freshness(self) -> DataFreshness:
        """
        Comprehensive staleness check across all canonical tables.
        Returns DataFreshness with warnings for stale data.
        """
        result = DataFreshness()
        today = date.today()

        # Prices
        try:
            row = self.conn.execute(
                "SELECT max(date), count(*) FROM market_data.prices"
            ).fetchone()
            if row and row[0]:
                latest = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
                result.prices_latest = str(latest)
                result.prices_stale_days = (today - latest).days
                result.prices_rows = int(row[1] or 0)
                # Weekend-aware: 3+ days = stale (accounts for Fri→Mon gap)
                result.is_fresh = result.prices_stale_days <= 3
                if result.prices_stale_days > 5:
                    result.warnings.append(f"Prices {result.prices_stale_days}d stale (latest: {latest})")
        except Exception:
            result.warnings.append("Cannot read prices table")

        # Fundamentals
        try:
            row = self.conn.execute(
                "SELECT max(snapshot_date) FROM fundamentals.quotes"
            ).fetchone()
            if row and row[0]:
                latest = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
                result.fundamentals_latest = str(latest)
                result.fundamentals_stale_days = (today - latest).days
                if result.fundamentals_stale_days > 5:
                    result.warnings.append(f"Fundamentals {result.fundamentals_stale_days}d stale")
        except Exception:
            pass

        # Holdings
        try:
            row = self.conn.execute(
                "SELECT max(snapshot_date) FROM holdings.etf_holdings"
            ).fetchone()
            if row and row[0]:
                result.holdings_latest = str(row[0])
        except Exception:
            pass

        # Run context
        try:
            row = self.conn.execute("""
                SELECT count(*), max(run_date) FROM analytics.run_contexts
            """).fetchone()
            if row:
                result.runs_count = int(row[0] or 0)
                result.last_run_date = str(row[1]) if row[1] else None
        except Exception:
            pass

        # Latest successful
        result.last_successful_run_id = self.latest_successful_run() or -1
        try:
            if result.last_successful_run_id > 0:
                row = self.conn.execute("""
                    SELECT run_date FROM analytics.run_contexts
                    WHERE run_id = ?
                """, [result.last_successful_run_id]).fetchone()
                if row:
                    result.last_successful_date = str(row[0])
        except Exception:
            pass

        return result

    # ── Signal reads (run-scoped) ────────────────────────────────────────

    def read_signals(self, run_id: Optional[int] = None) -> pd.DataFrame:
        """Read sector signals for a specific run (default: latest successful)."""
        if run_id is None:
            run_id = self.latest_successful_run()
        if run_id is None:
            return pd.DataFrame()
        try:
            return self.conn.execute("""
                SELECT * FROM analytics.sector_signals
                WHERE run_id = ?
                ORDER BY conviction_score DESC
            """, [run_id]).fetchdf()
        except Exception:
            return pd.DataFrame()

    def read_trade_book(self, run_id: Optional[int] = None,
                        active_only: bool = False) -> pd.DataFrame:
        """Read trade book for a specific run (default: latest)."""
        if run_id is None:
            run_id = self.latest_run()
        if run_id is None:
            return pd.DataFrame()
        try:
            query = "SELECT * FROM analytics.trade_book WHERE run_id = ?"
            if active_only:
                query += " AND is_active = true"
            query += " ORDER BY conviction_score DESC"
            return self.conn.execute(query, [run_id]).fetchdf()
        except Exception:
            return pd.DataFrame()

    # ── Data quality checks ──────────────────────────────────────────────

    def run_data_quality_checks(self) -> List[DataQualityCheck]:
        """
        Run all data quality checks and return results.
        Called by EngineService at pipeline end.
        """
        checks = []

        # 1. Price coverage: all 11 sector ETFs have recent data
        try:
            row = self.conn.execute("""
                SELECT count(DISTINCT ticker) FROM market_data.prices
                WHERE date >= CURRENT_DATE - INTERVAL '7' DAY
            """).fetchone()
            n_tickers = int(row[0]) if row else 0
            checks.append(DataQualityCheck(
                check_name="price_ticker_coverage",
                table_name="market_data.prices",
                status="PASS" if n_tickers >= 11 else "WARN" if n_tickers >= 5 else "FAIL",
                message=f"{n_tickers} tickers with recent prices (7d)",
                value=n_tickers,
                threshold=11,
            ))
        except Exception as e:
            checks.append(DataQualityCheck(
                check_name="price_ticker_coverage",
                table_name="market_data.prices",
                status="FAIL", message=str(e),
            ))

        # 2. Price null check
        try:
            row = self.conn.execute("""
                SELECT count(*) AS total,
                       sum(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS nulls
                FROM market_data.prices
                WHERE date >= CURRENT_DATE - INTERVAL '30' DAY
            """).fetchone()
            total = int(row[0]) if row else 0
            nulls = int(row[1]) if row else 0
            null_pct = nulls / max(total, 1) * 100
            checks.append(DataQualityCheck(
                check_name="price_null_rate",
                table_name="market_data.prices",
                status="PASS" if null_pct < 1 else "WARN" if null_pct < 5 else "FAIL",
                message=f"{null_pct:.1f}% null closes in last 30d ({nulls}/{total})",
                value=null_pct,
                threshold=1.0,
            ))
        except Exception as e:
            checks.append(DataQualityCheck(
                check_name="price_null_rate",
                table_name="market_data.prices",
                status="FAIL", message=str(e),
            ))

        # 3. Run consistency: signals exist for latest run
        try:
            latest_run = self.latest_run()
            if latest_run:
                row = self.conn.execute("""
                    SELECT count(*) FROM analytics.sector_signals
                    WHERE run_id = ?
                """, [latest_run]).fetchone()
                n_signals = int(row[0]) if row else 0
                checks.append(DataQualityCheck(
                    check_name="signals_for_latest_run",
                    table_name="analytics.sector_signals",
                    status="PASS" if n_signals >= 10 else "WARN" if n_signals > 0 else "FAIL",
                    message=f"{n_signals} signals for run_id={latest_run}",
                    value=n_signals,
                    threshold=10,
                ))
        except Exception as e:
            checks.append(DataQualityCheck(
                check_name="signals_for_latest_run",
                table_name="analytics.sector_signals",
                status="FAIL", message=str(e),
            ))

        # 4. Fundamentals coverage
        try:
            row = self.conn.execute("""
                SELECT count(DISTINCT symbol) FROM fundamentals.quotes
                WHERE snapshot_date = (SELECT max(snapshot_date) FROM fundamentals.quotes)
            """).fetchone()
            n_syms = int(row[0]) if row else 0
            checks.append(DataQualityCheck(
                check_name="fundamentals_symbol_coverage",
                table_name="fundamentals.quotes",
                status="PASS" if n_syms >= 100 else "WARN" if n_syms >= 10 else "FAIL",
                message=f"{n_syms} symbols in latest fundamentals snapshot",
                value=n_syms,
                threshold=100,
            ))
        except Exception as e:
            checks.append(DataQualityCheck(
                check_name="fundamentals_coverage",
                table_name="fundamentals.quotes",
                status="FAIL", message=str(e),
            ))

        # 5. Pipeline success rate (last 10 runs)
        try:
            df = self.conn.execute("""
                SELECT steps_fail FROM analytics.run_contexts
                ORDER BY run_id DESC LIMIT 10
            """).fetchdf()
            if not df.empty:
                n_success = int((df["steps_fail"] == 0).sum())
                checks.append(DataQualityCheck(
                    check_name="pipeline_success_rate",
                    table_name="analytics.run_contexts",
                    status="PASS" if n_success >= 8 else "WARN" if n_success >= 5 else "FAIL",
                    message=f"{n_success}/10 recent runs fully successful",
                    value=n_success,
                    threshold=8,
                ))
        except Exception:
            pass

        return checks

    # ── Write data quality results ───────────────────────────────────────

    def write_data_quality(self, run_id: int, checks: List[DataQualityCheck]) -> None:
        """Persist data quality check results for a run."""
        try:
            for check in checks:
                self.conn.execute("""
                    INSERT OR REPLACE INTO analytics.data_quality (
                        run_id, check_name, table_name, status,
                        message, value, threshold
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    run_id, check.check_name, check.table_name, check.status,
                    check.message[:500], check.value, check.threshold,
                ])
        except Exception as e:
            logger.warning("write_data_quality failed: %s", e)

    # ── Agent snapshots ──────────────────────────────────────────────────

    def write_agent_snapshot(self, run_id: int, agent_name: str,
                             status: str, output_json: str) -> None:
        """Persist agent output snapshot to DuckDB (replaces scattered JSON)."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO analytics.agent_snapshots (
                    run_id, agent_name, snapshot_date, status, output_json
                ) VALUES (?, ?, CURRENT_DATE, ?, ?)
            """, [run_id, agent_name, status, output_json[:50000]])
        except Exception as e:
            logger.debug("write_agent_snapshot failed: %s", e)

    def read_latest_agent_snapshot(self, agent_name: str) -> Optional[Dict]:
        """Read latest agent snapshot from DB."""
        try:
            import json
            row = self.conn.execute("""
                SELECT output_json FROM analytics.agent_snapshots
                WHERE agent_name = ?
                ORDER BY run_id DESC LIMIT 1
            """, [agent_name]).fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception:
            pass
        return None
