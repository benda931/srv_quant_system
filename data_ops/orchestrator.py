"""
data_ops/orchestrator.py

Health-aware ingestion orchestration.

DataOrchestrator is a drop-in replacement for the raw
DataLakeManager.build_snapshot() block in main.py.

Usage:
    orchestrator = DataOrchestrator(settings)
    state        = orchestrator.run()
    artifacts    = state.artifacts   # same ParquetArtifacts as before
    health       = state.health      # DataHealthReport
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, get_settings
from data.pipeline    import DataLakeManager, ParquetArtifacts
from data_ops.status_report import DataHealthReport, build_data_health_report

logger = logging.getLogger(__name__)


@dataclass
class DataState:
    """Result of a single DataOrchestrator.run() cycle."""
    artifacts: ParquetArtifacts
    health: DataHealthReport
    snapshot_refreshed: bool
    cycle_completed_at: datetime


class DataOrchestrator:
    """
    Wraps DataLakeManager with automatic health reporting.

    On each run():
      1. Checks snapshot freshness (via DataLakeManager.is_snapshot_fresh)
      2. Refreshes if stale or force_refresh=True
      3. Builds a DataHealthReport against the resulting artifacts
      4. Returns DataState with both artifacts and health

    Errors during snapshot build are caught and surfaced in the health
    report rather than propagated — the system can still start with
    stale/degraded data and display the health warning in the UI.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._lake    = DataLakeManager(self.settings)
        self._last_state: Optional[DataState] = None

    def run(self, force_refresh: bool = False) -> DataState:
        """
        Run the data pipeline and produce a DataState.

        Args:
            force_refresh: if True, always rebuild the snapshot.

        Returns:
            DataState with artifacts + health report.
        """
        # Initialize DuckDB schema on every startup (idempotent)
        try:
            from db.schema import SchemaManager
            from db.connection import get_connection
            _schema_mgr = SchemaManager(get_connection(self.settings.db_path))
            _schema_mgr.apply_migrations()
        except Exception as _e:
            logger.warning("DuckDB schema init failed (non-fatal): %s", _e)

        _run_start = datetime.now(timezone.utc)
        refreshed = False

        if force_refresh or not self._lake.is_snapshot_fresh():
            try:
                artifacts = self._lake.build_snapshot(force_refresh=force_refresh)
                refreshed = True
            except Exception as exc:
                logger.error(
                    "Snapshot build FAILED: %s — continuing with existing artifacts.",
                    exc,
                )
                artifacts = self._lake.artifacts
        else:
            artifacts = self._lake.artifacts

        try:
            health = build_data_health_report(
                prices_path=artifacts.prices_path,
                fundamentals_path=artifacts.fundamentals_path,
                weights_path=artifacts.weights_path,
                max_age_hours=float(self.settings.cache_max_age_hours),
                sector_tickers=self.settings.sector_list(),
                spy_ticker=self.settings.spy_ticker,
                expected_sectors=list(self.settings.sector_tickers.keys()),
                all_price_tickers=self.settings.all_price_tickers(),
            )
        except Exception as exc:
            logger.error("Health report build failed: %s", exc)
            health = _minimal_degraded_report(str(exc))

        # Log the health state prominently so it appears in srv_system.log
        lbl = health.health_label
        log_fn = (
            logger.error   if lbl == "CRITICAL" else
            logger.warning if lbl == "DEGRADED" else
            logger.info
        )
        log_fn(
            "DATA HEALTH: %s (score=%.0f%%, errors=%d, warnings=%d)",
            lbl,
            health.health_score * 100,
            health.validation.error_count,
            len(health.warnings),
        )

        state = DataState(
            artifacts=artifacts,
            health=health,
            snapshot_refreshed=refreshed,
            cycle_completed_at=datetime.now(),
        )
        self._last_state = state

        return state

    @property
    def last_state(self) -> Optional[DataState]:
        return self._last_state


# =========================================================================
# Minimal degraded fallback (used when health report build itself fails)
# =========================================================================

def _minimal_degraded_report(error_message: str) -> DataHealthReport:
    """
    Build a minimal CRITICAL DataHealthReport when the normal build fails.

    This is a last-resort fallback — it ensures the orchestrator always
    returns a DataHealthReport even if something unexpected goes wrong.
    """
    from data_ops.freshness  import FreshnessSummary
    from data_ops.quality    import CoverageReport, SPYSectorWeightCoverage
    from data_ops.validators import ValidationReport, ValidationIssue, ERROR

    freshness = FreshnessSummary(
        as_of=datetime.now(),
        artifacts={},
        price_detail=None,
        overall_state="UNKNOWN",
        warnings=[error_message],
        degraded=True,
    )
    coverage = CoverageReport(
        price_coverage={},
        fundamentals_coverage={},
        holdings_coverage={},
        spy_sector_weight=SPYSectorWeightCoverage(0, [], [], False, None),
        missing_price_tickers=[],
        sparse_price_tickers=[],
        fallback_etfs=[],
        health_score=0.0,
        degraded=True,
        warnings=[error_message],
    )
    validation = ValidationReport()
    validation.add(ValidationIssue(
        severity=ERROR,
        source="orchestrator",
        code="HEALTH_REPORT_BUILD_FAILED",
        message=f"Health report build failed: {error_message}",
        remediation="Check logs. Verify that Parquet artifacts exist and are readable.",
    ))

    return DataHealthReport(
        as_of=datetime.now(),
        health_score=0.0,
        health_label="CRITICAL",
        degraded=True,
        freshness=freshness,
        coverage=coverage,
        validation=validation,
        warnings=[error_message],
        errors=[f"[HEALTH_REPORT_BUILD_FAILED] {error_message}"],
    )


# =========================================================================
# Module-level convenience function
# =========================================================================

def get_data_state(force_refresh: bool = False) -> DataState:
    """Build a DataState using default settings. Convenience for scripting."""
    return DataOrchestrator(get_settings()).run(force_refresh=force_refresh)
