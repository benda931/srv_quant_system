"""
data_ops — Data health, coverage, validation and orchestration layer.

Public API
----------
Primary entry point:
    DataOrchestrator        — health-aware wrapper around DataLakeManager
    DataState               — result of orchestrator.run() (artifacts + health)
    get_data_state()        — module-level convenience function

Health report:
    DataHealthReport        — aggregated report from freshness + coverage + validation
    build_data_health_report()  — build report from paths + settings params

Sub-layer types (exposed for UI / testing):
    FreshnessSummary, assess_freshness, FRESH, STALE, MISSING
    CoverageReport, assess_coverage
    ValidationReport, ValidationIssue, validate_all, ERROR, WARNING, INFO

UI:
    build_health_tab(report)  — returns a Dash dbc.Container for the Data Health tab
"""

from data_ops.freshness import (
    FreshnessSummary,
    assess_freshness,
    FRESH,
    STALE,
    MISSING,
)
from data_ops.quality import (
    CoverageReport,
    assess_coverage,
)
from data_ops.validators import (
    ValidationReport,
    ValidationIssue,
    validate_all,
    ERROR,
    WARNING,
    INFO,
)
from data_ops.status_report import (
    DataHealthReport,
    build_data_health_report,
)
from data_ops.orchestrator import (
    DataOrchestrator,
    DataState,
    get_data_state,
)
from data_ops.health_panel import build_health_tab

__all__ = [
    # Freshness
    "FreshnessSummary", "assess_freshness", "FRESH", "STALE", "MISSING",
    # Quality
    "CoverageReport", "assess_coverage",
    # Validation
    "ValidationReport", "ValidationIssue", "validate_all", "ERROR", "WARNING", "INFO",
    # Status report
    "DataHealthReport", "build_data_health_report",
    # Orchestrator
    "DataOrchestrator", "DataState", "get_data_state",
    # UI
    "build_health_tab",
]
